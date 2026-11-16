"""Modulo de percepcion para la demo final de Robotica y Percepcion Computacional.

Este modulo concentra TODO el procesamiento de imagen del proyecto en una unica
clase reutilizable, `Percepcion`, pensada para ser llamada una vez por cada paso
del bucle de control del robot (`step()` en el cerebro de pyrobot) o, en banco de
pruebas, una vez por frame de un video.

A partir de un frame BGR de la camara frontal devuelve un objeto `Resultado` con:
  * found_line / error_norm  -> variable de control de navegacion
  * escena                   -> recta / curva / cruce (Y, T, X)
  * salidas y salida elegida -> rama a seguir en un cruce segun la flecha roja
  * flecha (angulo, sentido) -> orientacion de la flecha de direccion
  * marca y posicion         -> simbolo reconocido y su localizacion en la imagen

El codigo de segmentacion, clasificacion de escena, orientacion de flecha y
reconocimiento de marcas reutiliza el pipeline ya validado en
`entrega4/robentrega4mayo.py` sobre el video real "video2017-3".
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Etiquetas de las marcas de senalizacion conocidas
# ---------------------------------------------------------------------------
MARK_LABELS = {
    "man": "HOMBRE",
    "woman": "MUJER",
    "stairs": "ESCALERA",
    "telephone": "TELEFONO",
    "arrow": "FLECHA"
}




@dataclass
class ModeloMarcas:
    """Clasificador de marcas por vecino mas cercano sobre descriptores de contorno."""

    features: np.ndarray
    labels: np.ndarray
    mean: np.ndarray
    std: np.ndarray
    training_accuracy: float
    confusion: dict
    threshold: float
    num_samples: int


@dataclass
class Salida:
    name: str
    point: tuple
    side: str  # "top" | "left" | "right"


@dataclass
class Resultado:
    # --- navegacion ---
    found_line: bool
    error_px: float           # error lateral en pixeles (centro - centroide linea)
    error_norm: float         # error normalizado a [-1, 1] (variable de control)
    # --- escena ---
    escena: str
    blue_ratio: float
    red_ratio: float
    branch_y_norm: float   # cercania de la bifurcacion (0=lejos/arriba, 1=cerca/abajo)
    # --- cruces ---
    entry_name: str
    entry_point: tuple | None
    exits: list = field(default_factory=list)
    chosen_exit: str = "sin salida"
    chosen_exit_point: tuple | None = None
    # --- flecha ---
    arrow_dir: str | None = None
    arrow_angle_deg: float | None = None
    arrow_area: float = 0.0
    # --- marcas ---
    mark: str = "NINGUNA"
    mark_dist: float = 0.0
    mark_point: tuple | None = None
    mark_side: str | None = None
    # --- diagnostico ---
    processing_ms: float = 0.0


# ---------------------------------------------------------------------------
# Segmentacion en espacio HSV + RGB normalizado (rg)
# ---------------------------------------------------------------------------
def normalizacion_rg(img_bgr: np.ndarray) -> np.ndarray:
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
    suma = rgb[:, :, 0] + rgb[:, :, 1] + rgb[:, :, 2] + 1e-6
    return np.stack(
        (rgb[:, :, 0] / suma, rgb[:, :, 1] / suma, rgb[:, :, 2] / suma), axis=2
    )


def mask_red_hsv(hsv: np.ndarray) -> np.ndarray:
    return cv2.bitwise_or(
        cv2.inRange(hsv, (0, 70, 40), (12, 255, 255)),
        cv2.inRange(hsv, (165, 70, 40), (180, 255, 255)),
    )


def segmentar_frame(frame_bgr: np.ndarray):
    """Devuelve (etiquetas, mascaras) con clases 1=suelo, 2=azul, 3=rojo."""
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    norm = normalizacion_rg(frame_bgr)

    blue_hsv = cv2.inRange(hsv, (95, 60, 40), (140, 255, 255))
    red_hsv = mask_red_hsv(hsv)

    blue_rg = (
        (norm[:, :, 2] > 0.38)
        & (norm[:, :, 2] > norm[:, :, 0] + 0.06)
        & (norm[:, :, 2] > norm[:, :, 1] + 0.06)
    ).astype(np.uint8) * 255
    red_rg = (
        (norm[:, :, 0] > 0.38)
        & (norm[:, :, 0] > norm[:, :, 1] + 0.06)
        & (norm[:, :, 0] > norm[:, :, 2] + 0.06)
    ).astype(np.uint8) * 255

    blue = cv2.bitwise_and(blue_hsv, blue_rg)
    red = cv2.bitwise_or(red_hsv, red_rg)

    kernel = np.ones((3, 3), np.uint8)
    blue = cv2.morphologyEx(blue, cv2.MORPH_OPEN, kernel)
    blue = cv2.morphologyEx(blue, cv2.MORPH_CLOSE, kernel)
    red = cv2.morphologyEx(red, cv2.MORPH_OPEN, kernel)
    red = cv2.morphologyEx(red, cv2.MORPH_CLOSE, kernel)

    h, w = frame_bgr.shape[:2]
    mask_dark = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY) < 15
    ground = np.ones((h, w), dtype=np.uint8) * 255
    ground[blue > 0] = 0
    ground[red > 0] = 0
    ground[mask_dark] = 0

    labels = np.zeros((h, w), dtype=np.uint8)
    labels[ground > 0] = 1
    labels[blue > 0] = 2
    labels[red > 0] = 3
    return labels, {"ground": ground, "blue": blue, "red": red}


# ---------------------------------------------------------------------------
# Descriptores de contorno y clasificador de marcas (momentos de Hu + kNN)
# ---------------------------------------------------------------------------
def extraer_contorno_mayor(mask: np.ndarray, min_area: float = 80.0):
    contornos, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contornos = [c for c in contornos if cv2.contourArea(c) >= min_area]
    if not contornos:
        return None
    return max(contornos, key=cv2.contourArea)


def caracteristicas_contorno(contorno: np.ndarray) -> dict:
    area = cv2.contourArea(contorno)
    peri = cv2.arcLength(contorno, True)
    x, y, w, h = cv2.boundingRect(contorno)
    hull = cv2.convexHull(contorno)
    hull_area = max(cv2.contourArea(hull), 1e-6)
    return {
        "area": area,
        "perimeter": peri,
        "aspect_ratio": w / max(h, 1),
        "circularity": 4.0 * math.pi * area / max(peri * peri, 1e-6),
        "solidity": area / hull_area,
        "extent": area / max(w * h, 1),
        "bbox_w": float(w),
        "bbox_h": float(h),
    }

def contar_defectos_convexidad(contorno: np.ndarray, min_profundidad: float = 5.0) -> int:
    """Cuenta defectos de convexidad significativos (distingue flechas de humanos)."""
    hull = cv2.convexHull(contorno, returnPoints=False)
    if hull is None or len(hull) < 3 or len(contorno) < 5:
        return 0
    defects = cv2.convexityDefects(contorno, hull)
    if defects is None:
        return 0
    # depth está en unidades de 1/256 de pixel
    return int(np.sum(defects[:, 0, 3] / 256.0 >= min_profundidad))


def vector_caracteristicas_contorno(contorno: np.ndarray) -> np.ndarray:
    feats = caracteristicas_contorno(contorno)
    hu = cv2.HuMoments(cv2.moments(contorno)).flatten()
    hu = -np.sign(hu) * np.log10(np.abs(hu) + 1e-12)
    
    n_defectos = contar_defectos_convexidad(contorno, min_profundidad=5.0)

    vec = np.concatenate(
        [
            hu,
            np.array(
                [
                    feats["aspect_ratio"],
                    feats["area"],
                    feats["circularity"],
                    feats["solidity"],
                    feats["extent"],
                    float(n_defectos)
                ],
                dtype=np.float32,
            ),
        ]
    )
    return vec.astype(np.float32)


def _muestras_flecha_sinteticas():
    """Genera contornos de flechas realistas (cabeza ancha + cola fina)."""
    # Flecha estilo Stage: rectángulo (cola) + triángulo (cabeza)
    arrow_shapes = [
        # Flecha clásica apuntando derecha
        np.array([(0,20),(50,20),(50,10),(80,30),(50,50),(50,40),(0,40)], np.float32),
        # Flecha más ancha (como en el suelo)
        np.array([(0,25),(45,25),(45,8),(90,35),(45,62),(45,45),(0,45)], np.float32),
    ]
    contornos = []
    for base in arrow_shapes:
        base -= base.mean(axis=0)
        for ang in range(0, 360, 20):
            for sc in (0.6, 1.0, 1.5):
                rad = math.radians(ang)
                R = np.array([[math.cos(rad), -math.sin(rad)],
                              [math.sin(rad), math.cos(rad)]], np.float32) * sc
                pts = (base @ R.T)
                pts -= pts.min(axis=0)
                canvas = np.zeros(
                    (int(pts[:, 1].max()) + 10, int(pts[:, 0].max()) + 10), np.uint8
                )
                cv2.fillPoly(canvas, [(pts + 5).astype(np.int32)], 255)
                c = extraer_contorno_mayor(canvas, min_area=30)
                if c is not None:
                    contornos.append(c)
    return contornos


def cargar_modelo_marcas(base_dir: Path, threshold: float = 4.75,
                         incluir_flechas: bool = True,
                         extra_dirs=None) -> ModeloMarcas:
    feats, labels = [], []
    dirs = [Path(base_dir)] + [Path(d) for d in (extra_dirs or [])]
    for d in dirs:
        if not d.exists():
            continue
        for path in sorted(d.glob("*.png")):
            img = cv2.imread(str(path))
            if img is None:
                continue

            # raw_label se extrae PRIMERO para poder filtrar antes de procesar
            raw_label = path.stem.split("-")[0]
            if raw_label not in MARK_LABELS:
                continue

            # Entrenar con la MISMA segmentacion que se usa en runtime evita que
            # el kNN vea contornos ligeramente distintos a los de la camara.
            _, masks = segmentar_frame(img)
            red = masks["red"]
            if int((red > 0).sum()) < 40:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                red = (gray < 128).astype(np.uint8) * 255

            contorno = extraer_contorno_mayor(red, min_area=40)
            if contorno is None:
                continue

            feats_dbg = caracteristicas_contorno(contorno)
            print(f"[TRAIN] {path.name} → label={MARK_LABELS[raw_label]}, "
                  f"area={feats_dbg['area']:.0f}, solidity={feats_dbg['solidity']:.2f}, "
                  f"aspect={feats_dbg['aspect_ratio']:.2f}, circ={feats_dbg['circularity']:.2f}")

            feats.append(vector_caracteristicas_contorno(contorno))
            labels.append(MARK_LABELS[raw_label])

    if not feats:
        raise RuntimeError(f"No se cargaron marcas desde {base_dir}")

    if incluir_flechas:
        for c in _muestras_flecha_sinteticas():
            feats.append(vector_caracteristicas_contorno(c))
            labels.append("FLECHA")

    X = np.array(feats, dtype=np.float32)
    y = np.array(labels)
    mu = X.mean(axis=0)
    sd = X.std(axis=0) + 1e-6
    Z = (X - mu) / sd

    preds = []
    confusion = defaultdict(Counter)
    for i in range(len(Z)):
        dists = np.sqrt(((Z - Z[i]) ** 2).sum(axis=1))
        dists[i] = 1e9
        idx = int(np.argmin(dists))
        preds.append(y[idx])
        confusion[y[i]][y[idx]] += 1
    acc = float(np.mean(y == np.array(preds)))

    print(f"[TRAIN] Modelo listo: {len(y)} muestras, accuracy={acc*100:.1f}%")
    print(f"[TRAIN] Confusion: { {k: dict(v) for k, v in confusion.items()} }")

    return ModeloMarcas(
        features=X,
        labels=y,
        mean=mu,
        std=sd,
        training_accuracy=acc,
        confusion={k: dict(v) for k, v in confusion.items()},
        threshold=threshold,
        num_samples=int(len(y)),
    )


def predecir_marca(contorno: np.ndarray, model: ModeloMarcas):
    vec = vector_caracteristicas_contorno(contorno)
    z = (vec - model.mean) / model.std
    Z = (model.features - model.mean) / model.std
    dists = np.sqrt(((Z - z) ** 2).sum(axis=1))
    idx = int(np.argmin(dists))
    best = float(dists[idx])
    if best > model.threshold:
        return "DESCONOCIDA", best
    return str(model.labels[idx]), best


# ---------------------------------------------------------------------------
# Flecha de direccion
# ---------------------------------------------------------------------------
def elongacion(contorno: np.ndarray) -> float:
    """Relacion lado largo/corto del rectangulo de area minima (invariante a giro)."""
    (_, _), (w, h), _ = cv2.minAreaRect(contorno)
    lado_l, lado_c = max(w, h), max(min(w, h), 1e-3)
    return lado_l / lado_c


def es_flecha(contorno: np.ndarray) -> bool:
    feats = caracteristicas_contorno(contorno)
    return (
        feats["area"] > 100
        and feats["aspect_ratio"] >= 1.45
        and feats["circularity"] <= 0.48
        and feats["solidity"] <= 0.88
    )


def estimar_orientacion_flecha(contorno: np.ndarray):
    """Estima la orientacion de una flecha.

    La punta de la flecha es el punto del contorno mas alejado del centroide.
    La direccion centroide -> punta da el sentido al que apunta la flecha, de
    forma robusta frente a rotaciones (mas fiable que medir la dispersion en los
    extremos del eje principal, que se confunde cuando la cabeza esta retranqueada
    respecto a la punta).
    """
    pts = contorno.reshape(-1, 2).astype(np.float32)
    if len(pts) < 5:
        return None, "indefinida"

    # Eje principal de la flecha (PCA sobre el contorno relleno)
    x, y, w, h = cv2.boundingRect(contorno)
    mask = np.zeros((h, w), np.uint8)
    cv2.drawContours(mask, [contorno - [x, y]], -1, 255, -1)
    ys, xs = np.where(mask > 0)
    if len(xs) < 10:
        return None, "indefinida"
    px = np.stack([xs.astype(np.float32), ys.astype(np.float32)], axis=1)
    center = px.mean(axis=0)
    cov = np.cov((px - center).T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    axis = eigvecs[:, int(np.argmax(eigvals))]  # direccion de mayor varianza
    axis = axis / (np.linalg.norm(axis) + 1e-6)

    # La cabeza (con el triangulo) acumula MAS area que la cola fina.
    # El lado del centroide con mas pixeles indica hacia donde apunta la flecha.
    proj = (px - center) @ axis
    masa_pos = float((proj > 0).sum())
    masa_neg = float((proj < 0).sum())
    vec = axis if masa_pos >= masa_neg else -axis
    if np.linalg.norm(vec) < 1e-3:
        return None, "indefinida"

    # angulo en convencion matematica (y de imagen hacia abajo -> se invierte)
    angle = math.degrees(math.atan2(float(-vec[1]), float(vec[0])))

    if -45 <= angle <= 45:
        salida = "derecha"
    elif angle >= 135 or angle <= -135:
        salida = "izquierda"
    elif 45 < angle < 135:
        salida = "arriba"
    else:
        salida = "abajo"
    return angle, salida


# ---------------------------------------------------------------------------
# Deteccion de entrada / salidas y clasificacion de escena
# ---------------------------------------------------------------------------
def centroide_binario(mask: np.ndarray):
    moments = cv2.moments(mask, binaryImage=True)
    if moments["m00"] <= 1e-6:
        return None, 0.0
    return moments["m10"] / moments["m00"], moments["m00"]


def agrupar_indices(indices: np.ndarray, min_gap: int = 8):
    if len(indices) == 0:
        return []
    groups = []
    start = prev = int(indices[0])
    for idx in indices[1:]:
        idx = int(idx)
        if idx - prev > min_gap:
            groups.append((start, prev))
            start = idx
        prev = idx
    groups.append((start, prev))
    return groups


def detectar_entrada_y_salidas(blue_mask: np.ndarray):
    h, w = blue_mask.shape
    bottom_roi = blue_mask[int(h * 0.78):, :]
    entry_x, area = centroide_binario(bottom_roi)
    entry_point = None
    entry_name = "entrada inferior"
    if entry_x is not None and area > 20:
        entry_point = (int(entry_x), h - 14)
        if entry_x < w * 0.4:
            entry_name = "entrada inferior izquierda"
        elif entry_x > w * 0.6:
            entry_name = "entrada inferior derecha"
        else:
            entry_name = "entrada inferior centro"

    exits = []
    bw = 10  # grosor de banda de borde
    # Borde superior (ramas que continuan/salen hacia delante)
    top_band = blue_mask[:bw, :]
    for a, b in agrupar_indices(np.where(top_band.max(axis=0) > 0)[0]):
        exits.append(Salida("", (int((a + b) / 2), bw), "top"))
    # Bordes laterales: se escanea casi toda la altura (excepto el fondo, que es
    # la entrada) para capturar ramas PERPENDICULARES que aparecen bajas.
    left_band = blue_mask[: int(h * 0.90), :bw]
    for a, b in agrupar_indices(np.where(left_band.max(axis=1) > 0)[0]):
        exits.append(Salida("", (bw, int((a + b) / 2)), "left"))
    right_band = blue_mask[: int(h * 0.90), w - bw:]
    for a, b in agrupar_indices(np.where(right_band.max(axis=1) > 0)[0]):
        exits.append(Salida("", (w - bw, int((a + b) / 2)), "right"))

    filtered = []
    for ex in exits:
        if entry_point is not None and np.linalg.norm(
            np.array(ex.point) - np.array(entry_point)
        ) < 40:
            continue
        if all(
            np.linalg.norm(np.array(ex.point) - np.array(prev.point)) >= 35
            for prev in filtered
        ):
            filtered.append(ex)

    filtered.sort(key=lambda e: (e.point[0], e.point[1]))
    for i, ex in enumerate(filtered, start=1):
        ex.name = f"S{i}"
    return entry_point, entry_name, filtered[:3]


def _segmentos_fila(fila_bin: np.ndarray, min_gap: int = 6, min_len: int = 4):
    """Devuelve los segmentos contiguos de pixeles activos de una fila."""
    idx = np.where(fila_bin > 0)[0]
    segs = [(a, b) for a, b in agrupar_indices(idx, min_gap) if b - a + 1 >= min_len]
    return segs


def detectar_bifurcacion(blue_mask: np.ndarray):
    """Detecta una interseccion de forma robusta, independiente de bordes.

    Devuelve (es_cruce, n_ramas) analizando la estructura de la linea:
      * Bifurcacion (Y): alguna fila de la mitad superior tiene >= 2 segmentos.
      * Barra perpendicular (T/X): alguna fila tiene un segmento mucho mas ancho
        que el grosor tipico de la linea.
    Ambos casos delatan que la linea deja de ser un unico trazo.
    """
    h, w = blue_mask.shape
    # grosor tipico de la linea: anchura mediana de los segmentos del tercio inferior
    anchos = []
    for y in range(int(h * 0.7), h, 3):
        for a, b in _segmentos_fila(blue_mask[y]):
            anchos.append(b - a + 1)
    ancho_linea = float(np.median(anchos)) if anchos else max(6.0, w * 0.04)

    n_multi = 0
    barra = False
    max_ramas = 1
    y_branch = 0  # fila (px) mas baja = mas cercana donde se aprecia la ramificacion
    #separación mínima entre segmentos para considerar que son ramas distintas 
    # y no esl mismo trazo visto en perspectiva durante una curva pronunciada
    min_sep = max(ancho_linea * 0.8, 6.0)
    for y in range(int(h * 0.05), int(h * 0.88), 2):
        segs = _segmentos_fila(blue_mask[y])
        if len(segs) >= 2:
            hay_hueco_real = any(segs[i+1][0] - segs[i][1] >= min_sep for i in range(len(segs) - 1))
            if hay_hueco_real:
                n_multi += 1
                max_ramas = max(max_ramas, len(segs))
                y_branch = max(y_branch, y)
        for a, b in segs:
            if (b - a + 1) > max(3.0 * ancho_linea, w * 0.40):
                barra = True
                y_branch = max(y_branch, y)
                
    es_cruce = (n_multi >= 6) or barra
    if barra:
        max_ramas = max(max_ramas, 3)
    branch_y_norm = y_branch / float(h) if y_branch else 0.0
    return es_cruce, max_ramas, branch_y_norm


def clasificar_escena(labels: np.ndarray, exits: list):
    h, w = labels.shape
    line = (labels == 2).astype(np.uint8) * 255
    roi_inf = line[int(h * 0.68):, :]
    roi_sup = line[: int(h * 0.38), :]

    cx_inf, area_inf = centroide_binario(roi_inf)
    cx_sup, _ = centroide_binario(roi_sup)
    # Convencion: error > 0 => linea a la DERECHA del centro (como findLineDeviation),
    # de modo que el control turn = -(KP*error) gire hacia la linea.
    error = 0.0 if cx_inf is None else cx_inf - (w / 2.0)
    blue_ratio = float((line > 0).mean())

    es_cruce, n_ramas, branch_y = detectar_bifurcacion(line)

    if es_cruce and (len(exits) >= 2 or n_ramas >= 3):
        escena = "Cruce X/T (3 salidas)" if (len(exits) >= 3 or n_ramas >= 3) \
            else "Cruce Y/T (2 salidas)"
    elif es_cruce:
        escena = "Cruce Y/T (2 salidas)"
    elif cx_inf is None or cx_sup is None or area_inf < 50:
        escena = "Linea recta"
    else:
        delta = cx_sup - cx_inf
        if delta < -18:
            escena = "Curva izquierda"
        elif delta > 18:
            escena = "Curva derecha"
        else:
            escena = "Linea recta"
    return error, escena, blue_ratio, (cx_inf is not None and area_inf >= 50), branch_y


def escoger_salida(exits: list, arrow_dir, scene: str, width: int):
    if not exits:
        return "sin salida detectada", None
    exits_sorted = sorted(exits, key=lambda e: e.point[0])
    center_exit = min(exits_sorted, key=lambda e: abs(e.point[0] - width / 2.0))
    if scene.startswith("Cruce"):
        if arrow_dir == "izquierda":
            chosen = exits_sorted[0]
        elif arrow_dir == "derecha":
            chosen = exits_sorted[-1]
        elif arrow_dir == "arriba":
            chosen = center_exit
        else:
            chosen = center_exit
    else:
        chosen = center_exit
    side = "centro"
    if chosen.point[0] < width * 0.4:
        side = "izquierda"
    elif chosen.point[0] > width * 0.6:
        side = "derecha"
    return f"{chosen.name} ({side})", chosen.point

def clasificar_por_geometria_humanos(contorno: np.ndarray, label_knn: str | None = None,
                                     dist_knn: float = 99.0) -> str | None:
    """
    Clasificador geometrico SOLO para HOMBRE y MUJER.

    La regla anterior usaba casi solo defectos de convexidad y convertia muchas
    mujeres en hombres. Aqui la geometria solo decide si el contorno parece
    humano; la etiqueta fina se conserva del kNN cuando esta razonablemente
    cerca de las muestras de entrenamiento.
    """
    feats = caracteristicas_contorno(contorno)
    solidity = feats["solidity"]
    circularity = feats["circularity"]
    extent = feats["extent"]

    parece_humano = (
        feats["area"] >= 250
        and solidity >= 0.74
        and 0.12 <= circularity <= 0.30
        and extent >= 0.36
    )
    if parece_humano and label_knn in ("HOMBRE", "MUJER") and dist_knn <= 5.25:
        return label_knn
    return None


def clasificar_por_geometria_marca(contorno: np.ndarray, label_knn: str,
                                   dist_knn: float, en_cruce: bool) -> tuple[str, float]:
    """Refina la prediccion por familias de forma conservadora."""
    feats = caracteristicas_contorno(contorno)
    n_def = contar_defectos_convexidad(contorno)
    solidity = feats["solidity"]
    circularity = feats["circularity"]
    extent = feats["extent"]
    area = feats["area"]

    if area < 120:
        return "DESCONOCIDA", dist_knn

    # Escalera: contorno dentado, muy concavo y poco circular.
    if solidity <= 0.64 and circularity <= 0.22 and n_def >= 2:
        return "ESCALERA", min(dist_knn, 0.0 if label_knn == "ESCALERA" else 1.5)

    # Telefono: pieza compacta en forma de auricular, con pocas concavidades.
    if (0.66 <= solidity <= 0.78 and circularity >= 0.28
            and extent >= 0.35 and n_def <= 2):
        return "TELEFONO", min(dist_knn, 0.0 if label_knn == "TELEFONO" else 1.5)

    humano = clasificar_por_geometria_humanos(contorno, label_knn, dist_knn)
    if humano is not None:
        return humano, dist_knn

    # Las flechas solo son marcas validas dentro de cruces.
    if label_knn == "FLECHA" and not en_cruce:
        return "DESCONOCIDA", dist_knn

    return label_knn, dist_knn


# ---------------------------------------------------------------------------
# Clase principal de percepcion
# ---------------------------------------------------------------------------
class Percepcion:
    """Encapsula todo el procesamiento de imagen del proyecto.

    Uso tipico (en el cerebro de pyrobot)::

        self.percepcion = Percepcion(marks_dir="...")   # en setup()
        res = self.percepcion.procesar(cv_image)         # en cada step()
    """

    def __init__(self, marks_dir=None, model: ModeloMarcas | None = None,
                 extra_dirs=None):
        if model is not None:
            self.model = model
        elif marks_dir is not None:
            self.model = cargar_modelo_marcas(Path(marks_dir), extra_dirs=extra_dirs)
        else:
            self.model = None

    def procesar(self, frame_bgr: np.ndarray) -> Resultado:
        import time

        start = time.perf_counter()
        h, w = frame_bgr.shape[:2]
        labels, masks = segmentar_frame(frame_bgr)
        entry_point, entry_name, exits = detectar_entrada_y_salidas(masks["blue"])
        error_px, escena, blue_ratio, found_line, branch_y = clasificar_escena(labels, exits)
        red_ratio = float((labels == 3).mean())

        mark, mark_dist, mark_point, mark_side = "NINGUNA", 0.0, None, None
        arrow_dir, arrow_angle, arrow_area = None, None, 0.0

        red_contour = extraer_contorno_mayor(masks["red"], min_area=20) #antes 70

        # ── DIAGNÓSTICO: print cada vez que hay píxeles rojos ──────────────────
        _px_rojos = int((masks["red"] > 0).sum())
        if _px_rojos > 0:
            _info = f"[ROJO] {_px_rojos} px"
            if red_contour is not None:
                _f = caracteristicas_contorno(red_contour)
                _elon = elongacion(red_contour)
                _xd, _yd, _cw, _ch = cv2.boundingRect(red_contour)
                _toca = _xd <= 1 or _yd <= 1 or (_xd + _cw) >= w - 1 or (_yd + _ch) >= h - 1
                _info += (f" | contorno: area={_f['area']:.0f}  solidity={_f['solidity']:.2f}"
                          f"  elon={_elon:.2f}  toca_borde={_toca}")
            #print(_info)
        # ───────────────────────────────────────────────────────────────────────

        if red_contour is not None:
            en_cruce = escena.startswith("Cruce")

            if self.model is not None:
                label, dist = predecir_marca(red_contour, self.model)
            else:
                label, dist = "DESCONOCIDA", 0.0

            label, dist = clasificar_por_geometria_marca(
                red_contour, label, dist, en_cruce
            )

            feats = caracteristicas_contorno(red_contour)
            elon = elongacion(red_contour)

            #print(f"[KNN] label={label}, dist={dist:.2f}, area={cv2.contourArea(red_contour):.0f}, "
            #      f"solidity={feats['solidity']:.2f}, elon={elon:.2f}, en_cruce={en_cruce}")

            # Flecha: SOLO en cruces
            es_arrow = en_cruce and (
                label == "FLECHA" or (
                    label == "DESCONOCIDA"
                    and elon >= 2.0
                    and feats["solidity"] <= 0.80
                )
            )
            
            if not en_cruce and (
                label == "FLECHA"
                or (
                    label == "DESCONOCIDA"
                    and elon >= 2.0
                    and feats["solidity"] <= 0.80
                )
            ):
                #print(f"[DESCARTADO] Posible flecha ignorada fuera de cruce "
                #        f"(label={label}, elon={elon:.2f}, solidity={feats['solidity']:.2f})")
                label = "DESCONOCIDA"
                es_arrow = False
            
            #print(f"[CLASIF] label={label}, dist={dist:.2f}, "
            #    f"area={cv2.contourArea(red_contour):.0f}, "
            #    f"solidity={feats['solidity']:.2f}, elon={elon:.2f}, "
            #    f"en_cruce={en_cruce}, es_arrow={es_arrow}")

            x_, y_, w_, h_ = cv2.boundingRect(red_contour)
            recortada_abajo = (y_ + h_) >= h - 2
            toca_borde = x_ <= 1 or y_ <= 1 or (x_ + w_) >= w - 1 or (y_ + h_) >= h - 1
            area_red = cv2.contourArea(red_contour)

            if es_arrow and not recortada_abajo:
                arrow_angle, arrow_dir = estimar_orientacion_flecha(red_contour)
                arrow_area = float(area_red)
            elif (label not in ("DESCONOCIDA", "FLECHA")
                  and not toca_borde and area_red >= 300):
                mark, mark_dist = label, dist
                M = cv2.moments(red_contour)
                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    mark_point = (cx, cy)
                    if cx < w * 0.4:
                        mark_side = "a la izquierda"
                    elif cx > w * 0.6:
                        mark_side = "a la derecha"
                    else:
                        mark_side = "al frente"

        chosen_exit, chosen_point = escoger_salida(exits, arrow_dir, escena, w)
        error_norm = float(np.clip(error_px / (w / 2.0), -1.0, 1.0))

        return Resultado(
            found_line=found_line,
            error_px=error_px,
            error_norm=error_norm,
            escena=escena,
            blue_ratio=blue_ratio,
            red_ratio=red_ratio,
            branch_y_norm=branch_y,
            entry_name=entry_name,
            entry_point=entry_point,
            exits=exits,
            chosen_exit=chosen_exit,
            chosen_exit_point=chosen_point,
            arrow_dir=arrow_dir,
            arrow_angle_deg=arrow_angle,
            arrow_area=arrow_area,
            mark=mark,
            mark_dist=mark_dist,
            mark_point=mark_point,
            mark_side=mark_side,
            processing_ms=(time.perf_counter() - start) * 1000.0,
        )
