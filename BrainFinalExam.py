"""Cerebro pyrobot para la DEMO FINAL de Robotica y Percepcion Computacional.

Es la EVOLUCION DIRECTA de nuestro brain `BrainFollowLine` de las practicas:
  * Se conserva su navegacion: maquina de estados LINEA / EVADIR / RODEAR, los
    sensores (get_sensores), la capa de emergencia, la evasion de obstaculos con
    sonar y el barrido de busqueda de linea, con las mismas constantes y ajustes.
  * Se SUSTITUYE la fuente del seguimiento: en lugar de `findLineDeviation` sobre
    la imagen en gris, el error de linea se calcula sobre la mascara AZUL
    (vision.py), mas robusto en la ruta final (linea azul + marcas rojas).
  * Se ANADE lo que pedia la integracion final:
      - modo CRUCE: detecta intersecciones Y/T/X y gira a la rama que indica la
        flecha roja;
      - reconocimiento y ANUNCIO de las marcas de senalizacion con su localizacion.

Lanzamiento en Stage (desde la raiz del repositorio):
  ./run.sh   (equivale a:
  pyrobot -s StageSimulator.py -w ./mundo/finalExam-202526-line1-v01.world -r Stage.py -b ./BrainFinalExam.py)

Requiere `vision.py` en la MISMA carpeta que este fichero.
"""

import os
import sys

import cv2
from numpy import mean

# Asegura que `vision.py` (en la misma carpeta) sea importable sea cual sea el
# directorio de trabajo desde el que pyrobot lance el cerebro.
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

# Permite importar este modulo (y probar la navegacion en el banco de pruebas)
# aunque pyrobot no este instalado en el equipo de desarrollo.
try:
    from pyrobot.brain import Brain
    _PYROBOT = True
except Exception:                                    # pragma: no cover
    _PYROBOT = False

    class Brain:                                     # stub minimo para el banco
        def __init__(self, name="", engine=None):
            self.robot = engine

        def move(self, translate, rotate):
            self._last_move = (translate, rotate)

from vision import Percepcion


def _primer_dir(candidatos):
    """Devuelve el primer directorio existente de la lista (o el ultimo)."""
    for c in candidatos:
        if os.path.isdir(c):
            return c
    return candidatos[-1]


# Dataset de marcas e imagenes del examen, buscados de forma robusta relativos a
# este fichero (funciona desde la carpeta del examen, la raiz o demo_final).
MARKS_DIR = _primer_dir([
    os.path.join(_HERE, "marcas-capturasStage"),
    os.path.join(_HERE, "..", "entrega4", "marcas-capturasStage", "marcas-capturasStage"),
    os.path.join(_HERE, "entrega4", "marcas-capturasStage", "marcas-capturasStage"),
    os.path.join(_HERE, "..", "..", "entrega4", "marcas-capturasStage", "marcas-capturasStage"),
])
EXAM_IMGS = _primer_dir([
    os.path.join(_HERE, "mundo", "images"),
    os.path.join(_HERE, "images"),
    os.path.join(_HERE, "..", "finalExam-202526-knownWorld-v01", "images"),
    os.path.join(_HERE, "finalExam-202526-knownWorld-v01", "images"),
])


class BrainFollowLine(Brain):

    # ===================== Constantes de BrainFollowLine =====================
    NO_FORWARD = 0
    SLOW_FORWARD = 0.1
    MED_FORWARD = 0.5
    FULL_FORWARD = 1.0

    NO_TURN = 0
    SOFT_LEFT = 0.25
    SOFT_RIGHT = 0.25
    MED_LEFT = 0.5
    HARD_LEFT = 1.0
    MED_RIGHT = -0.5
    HARD_RIGHT = -1.0

    NO_ERROR = 0
    DIST_SEGURA = 0.4

    KP_LINEA = 0.8        # ganancia del seguimiento (turn = -KP*error), como el original
    CRUISE = 0.5          # velocidad de crucero en recta (en lugar de 1.0): da margen
                          # para detectar y tomar los cruces con limpieza. Subir si se
                          # quiere mas velocidad una vez ajustado en Stage.

    # ===================== Constantes NUEVAS: cruces y marcas ================
    CRUCE_DEBOUNCE = 2     # frames de cruce seguidos antes de entrar (anti-parpadeo)
    CRUCE_FWD = 0.22       # avance durante el arco de giro
    CRUCE_TURN = 0.7       # giro firme hacia la rama elegida
    CRUCE_GIRO_MIN = 15    # pasos minimos de giro (~90 grados); AJUSTAR en Stage
    CRUCE_MAX_PASOS = 45   # tope de seguridad del giro
    CRUCE_SALIR_ERR = 0.55 # |error| para reenganchar la linea de la nueva rama
    CRUCE_COOLDOWN = 25    # pasos tras un cruce sin admitir uno nuevo (solo para
                           # alejarse del nodo actual; corto para no perderse el siguiente)
    ARROW_MEMORIA = 80     # ventana para recordar la flecha vista al acercarse

    MARK_COOLDOWN = 180  # pasos antes de poder reanunciar la MISMA marca
    MARK_ZONE_COOLDOWN = 40

    # ------------------------------------------------------------------------
    def setup(self):
        # --- estado original de BrainFollowLine ---
        self.modo = "LINEA"
        self.direccion_esquiva = None
        self.ultimo_giro = None
        self.busqueda_dir = 1
        self.busqueda_timer = 0

        # --- percepcion (vision) ---
        marks_dir = os.environ.get("MARKS_DIR", MARKS_DIR)
        # Imagenes oficiales de los simbolos del examen (si estan disponibles):
        # se anaden al entrenamiento porque son las marcas exactas de la prueba.
        try:
            self.percepcion = Percepcion(marks_dir=marks_dir, extra_dirs=[EXAM_IMGS])
            print("[setup] Modelo de marcas: %d muestras, accuracy %.1f%%"
                  % (self.percepcion.model.num_samples,
                     self.percepcion.model.training_accuracy * 100))
        except Exception as exc:
            print("[setup] AVISO: sin modelo de marcas:", exc)
            self.percepcion = Percepcion(model=None)

        # --- estado NUEVO: cruces, flecha y marcas ---
        self.paso = 0
        self._cruce_dir = None
        self._cruce_fase = "APROX"
        self._cruce_giro_pasos = 0
        self._cruce_candidato = 0
        self._cruce_cooldown = 0
        self._aprox_sin_cruce = 0
        self._aprox_max_by = 0.0
        self._arrow_votos = {}
        self._last_arrow_paso = -10 ** 9
        self._last_report_paso = {}      # {marca: paso del último anuncio}
        self._mark_zone_cooldown = 0
        self.reportes = []
        self.mostrar = os.environ.get("DEMO_HEADLESS", "0") != "1"

    # ------------------------------------------------------------------------
    def destroy(self):
        print("\n==================== RESUMEN DE LA DEMO ====================")
        if not self.reportes:
            print("No se reporto ninguna marca.")
        for r in self.reportes:
            print(f"  - paso {r['paso']}: {r['marca']} ({r['lado']})")
        print("===========================================================")
        cv2.destroyAllWindows()

    # ===================== Sensores (de BrainFollowLine) =====================
    def _sensores(self, lecturas):
        return {
            'izq_total': lecturas[0],
            'izq_diagonal': mean(lecturas[1:3]),
            'frontal': min(lecturas[3], lecturas[4]),
            'der_diagonal': mean(lecturas[5:7]),
            'der_total': lecturas[7],
        }

    def get_sensores(self):
        return self._sensores(list(self.robot.range.values()))

    def _leer_sonares(self):
        try:
            return [float(v) for v in list(self.robot.range.values())[:8]]
        except Exception:
            return [9.9] * 8

    # ===================== NUEVO: gestion de marcas ==========================
    def _gestionar_marcas(self, res):
        etiqueta = res.mark if res.mark not in ("NINGUNA", "DESCONOCIDA") else None
        if etiqueta is None:
            return

        # Cooldown: no reanunciar la misma marca demasiado pronto.
        ultimo = self._last_report_paso.get(etiqueta, -10 ** 9)
        if self.paso - ultimo < self.MARK_COOLDOWN:
            return

        self._last_report_paso[etiqueta] = self.paso
        lado = res.mark_side or "?"
        rep = {
            "paso": self.paso,
            "marca": etiqueta,
            "lado": lado,
            "punto": res.mark_point,
            "motivo": "ultima deteccion",
        }
        self.reportes.append(rep)

        print(f"\n{'='*45}")
        print(f"  MARCA DETECTADA: {etiqueta}")
        #print(f"  Motivo   : ultima deteccion")
        #print(f"  Posición : {lado}")
        print(f"  Paso     : {self.paso}")
        print(f"{'='*45}\n")

    # ===================== NUEVO: flecha de direccion ========================
    def _arrow_efectiva(self):
        if self.paso - self._last_arrow_paso > self.ARROW_MEMORIA or not self._arrow_votos:
            return None
        return max(self._arrow_votos.items(), key=lambda kv: kv[1])[0]

    # ===================== Decision de movimiento ============================
    def decidir(self, res, sonars):
        """Devuelve (forward, turn). Contiene la maquina de estados completa.

        Separado de step() para poder ejecutar EXACTAMENTE esta logica en el banco
        de pruebas (camera_sim) sin pyrobot.
        """
        self.paso += 1
        s = self._sensores(sonars)
        if self._mark_zone_cooldown > 0:
            self._mark_zone_cooldown -= 1
        if self.modo != "CRUCE" and self._cruce_candidato == 0 and self._mark_zone_cooldown == 0:
            self._gestionar_marcas(res)

        # Memoria de la flecha vista al acercarse (voto ponderado por area).
        if res.arrow_dir in ("izquierda", "derecha", "arriba") and res.arrow_area > 0:
            if self.paso - self._last_arrow_paso > self.ARROW_MEMORIA:
                self._arrow_votos = {}
            self._arrow_votos[res.arrow_dir] = (
                self._arrow_votos.get(res.arrow_dir, 0.0) + res.arrow_area)
            self._last_arrow_paso = self.paso

        foundLine, error = res.found_line, res.error_norm

        # --- Capa de EMERGENCIA (identica a BrainFollowLine) ---
        if min(s['frontal'], s['izq_diagonal'], s['der_diagonal']) < 0.3:
            if self.ultimo_giro is None:
                giro = self.HARD_RIGHT
            else:
                giro = self.HARD_LEFT if self.ultimo_giro < 0 else self.HARD_RIGHT
            return 0.05, giro

        if self.modo == "LINEA":
            return self._modo_linea(res, s, foundLine, error)
        if self.modo == "CRUCE":
            return self._modo_cruce(res, s)
        if self.modo == "EVADIR":
            return self._modo_evadir(res, s, foundLine)
        if self.modo == "RODEAR":
            return self._modo_rodear(res, s, foundLine)
        return self.NO_FORWARD, self.NO_TURN

    # --------------------- Modo LINEA (original + cruce) --------------------
    def _modo_linea(self, res, s, foundLine, error):
        # 1) Obstaculo -> EVADIR (logica original de BrainFollowLine)
        if (s['frontal'] < 1.0 or min(s['izq_diagonal'], s['der_diagonal']) < 0.5
                or min(s['izq_total'], s['der_total']) < 0.3):
            self.modo = "EVADIR"
            self.direccion_esquiva = ("IZQUIERDA" if s['izq_diagonal'] > s['der_diagonal']
                                      else "DERECHA")
            return self._modo_evadir(res, s, foundLine)

        # 2) NUEVO: deteccion de cruce (con cooldown y anti-parpadeo)
        if self._cruce_cooldown > 0:
            self._cruce_cooldown -= 1
        elif res.escena.startswith("Cruce"):
            self._cruce_candidato += 1
            if self._cruce_candidato >= self.CRUCE_DEBOUNCE:
                self._entrar_cruce(res)
                return self._modo_cruce(res, s)
        else:
            self._cruce_candidato = 0

        # 3) Seguir la LINEA AZUL (control proporcional original, con crucero capado)
        if foundLine:
            turn = -self.KP_LINEA * error
            forward = max(0.1, min(s['frontal'], self.CRUISE) * max(0.2, 1 - abs(turn)))
            self.ultimo_giro = 1 if turn > 0 else -1
            return forward, turn

        # 4) Linea perdida: recuperacion / barrido (original)
        if self.ultimo_giro is not None:
            return self.SLOW_FORWARD, self.ultimo_giro
        self.busqueda_timer += 1
        if self.busqueda_timer < 80:
            return self.FULL_FORWARD, self.NO_TURN
        elif self.busqueda_timer < 130:
            return self.NO_FORWARD, self.busqueda_dir
        else:
            self.busqueda_dir *= -1
            self.busqueda_timer = 0
            return self.NO_FORWARD, self.busqueda_dir

    # --------------------- NUEVO: Modo CRUCE --------------------------------
    def _entrar_cruce(self, res):
        self.modo = "CRUCE"
        self._cruce_fase = "APROX"
        self._cruce_giro_pasos = 0
        self._cruce_candidato = 0
        self._aprox_sin_cruce = 0
        self._aprox_max_by = 0.0
        self._cruce_dir = self._arrow_efectiva()  # inicial; se sigue actualizando en APROX
        #print(f"[paso {self.paso}] CRUCE detectado ({res.escena}); "
        #      f"flecha (provisional)={self._cruce_dir or 'aun no vista'}")

    def _es_cruce(self, res):
        return res.escena.startswith("Cruce")

    def _modo_cruce(self, res, s):
        # FASE APROX: seguir la linea de frente HASTA llegar al nodo, SIGUIENDO
        # leyendo la flecha (a menudo el cruce se ve antes que la flecha). La
        # direccion se decide aqui y se congela solo al empezar el giro.
        if self._cruce_fase == "APROX":
            self._cruce_dir = self._arrow_efectiva() or self._cruce_dir
            self._aprox_max_by = max(self._aprox_max_by, res.branch_y_norm)
            if self._es_cruce(res):
                self._aprox_sin_cruce = 0
            else:
                self._aprox_sin_cruce += 1
            cerca = ((self._aprox_sin_cruce >= 3 and self._aprox_max_by >= 0.65)
                     or res.branch_y_norm >= 0.90)
            if not cerca:
                turn = max(-1.0, min(1.0, -self.KP_LINEA * res.error_norm))
                return self.MED_FORWARD, turn
            self._cruce_fase = "GIRO"   # nodo alcanzado: se congela la direccion
            print(f"[paso {self.paso}] Nodo alcanzado, "
                f"{'giro a la ' + self._cruce_dir if self._cruce_dir else 'continúo recto'}")

        d = self._cruce_dir

        # Flecha hacia delante / sin flecha: cruzar el nodo de frente
        if d not in ("izquierda", "derecha"):
            turn = max(-1.0, min(1.0, -self.KP_LINEA * res.error_norm))
            if not self._es_cruce(res):
                self._salir_cruce()
            return self.CRUCE_FWD * 1.5, turn

        # FASE GIRO: arco comprometido hacia la rama elegida hasta reenganchar
        self._cruce_giro_pasos += 1
        turn = -self.CRUCE_TURN if d == "derecha" else self.CRUCE_TURN
        recentrada = res.found_line and abs(res.error_norm) < self.CRUCE_SALIR_ERR
        if self._cruce_giro_pasos >= self.CRUCE_GIRO_MIN and (
                recentrada or self._cruce_giro_pasos >= self.CRUCE_MAX_PASOS):
            print(f"[paso {self.paso}] Cruce superado por {d}")
            self.ultimo_giro = -1 if d == "derecha" else 1
            self._salir_cruce()
        return self.CRUCE_FWD, turn

    def _salir_cruce(self):
        """Vuelve a LINEA tras un cruce y REINICIA la votacion de flecha, para que
        el siguiente cruce no herede los votos (de area grande) del nodo anterior."""
        self.modo = "LINEA"
        self._cruce_cooldown = self.CRUCE_COOLDOWN
        self._mark_zone_cooldown = self.MARK_ZONE_COOLDOWN
        self._arrow_votos = {}
        self._last_arrow_paso = -10 ** 9

    # --------------------- Modo EVADIR (original) ---------------------------
    def _modo_evadir(self, res, s, foundLine):
        turn = 0.8 * (s['izq_diagonal'] - s['der_diagonal'])
        turn += 0.5 * (s['izq_total'] - s['der_total'])
        turn = max(-1, min(1, turn))
        forward = 0.1 if s['frontal'] < 0.5 else 0.3
        if s['frontal'] > 1.0 and s['izq_diagonal'] > 1.0 and s['der_diagonal'] > 1.0:
            self.modo = "LINEA"
        return forward, turn

    # --------------------- Modo RODEAR (original) ---------------------------
    def _modo_rodear(self, res, s, foundLine):
        if self.direccion_esquiva == "DERECHA":
            dist_lateral = s['izq_total']
            dist_diagonal = s['izq_diagonal']
            turn_sign = -1
        else:
            dist_lateral = s['der_total']
            dist_diagonal = s['der_diagonal']
            turn_sign = 1
        if min(s['frontal'], s['izq_diagonal'], s['der_diagonal']) < 0.6:
            self.modo = "EVADIR"
            return self.NO_FORWARD, self.NO_TURN
        if foundLine and s['frontal'] > 1.0 and min(dist_lateral, dist_diagonal) > self.DIST_SEGURA:
            self.modo = "LINEA"
            return self.NO_FORWARD, self.NO_TURN
        error_dist = self.DIST_SEGURA - min(dist_lateral, dist_diagonal)
        turn = max(-1, min(1, turn_sign * error_dist * 2.0))
        forward = max(0.1, min(s['frontal'], 0.5))
        if abs(turn) > 0.3:
            forward *= 0.5
        return forward, turn

    # ===================== Bucle de pyrobot ==================================
    def step(self):
        cv_image = self.robot.getImage()
        res = self.percepcion.procesar(cv_image)
        sonars = self._leer_sonares()
        forward, turn = self.decidir(res, sonars)
        self.move(forward, turn)
        if self.mostrar:
            self._overlay(cv_image, res, forward, turn)

    def _overlay(self, frame, res, forward, turn):
        img = frame.copy()
        if res.chosen_exit_point is not None:
            cv2.circle(img, res.chosen_exit_point, 8, (0, 255, 255), 2)
        if res.mark_point is not None:
            cv2.circle(img, res.mark_point, 7, (0, 0, 255), -1)
        for i, t in enumerate([
            f"Modo: {self.modo}", f"Escena: {res.escena}", f"Marca: {res.mark}",
            f"Flecha: {res.arrow_dir}",
            f"err={res.error_norm:+.2f} f={forward:.2f} t={turn:+.2f}",
        ]):
            cv2.putText(img, t, (8, 18 + 18 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 255, 0), 1, cv2.LINE_AA)
        cv2.imshow("BrainFollowLine - Demo Final", img)
        cv2.waitKey(1)


def INIT(engine):
    assert (engine.robot.requires("range-sensor") and
            engine.robot.requires("continuous-movement"))
    return BrainFollowLine('BrainFollowLine', engine)
