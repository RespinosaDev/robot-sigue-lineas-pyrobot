# Robot sigue-líneas con visión · Pyrobot + Stage

Es el controlador de un robot móvil, hecho en **Python** con **Pyrobot** y **OpenCV**, para el simulador **Stage**. Con la cámara frontal el robot sigue una línea azul, detecta los cruces y gira hacia donde le indica una **flecha roja**. Además esquiva obstáculos con los **sonares** y **reconoce las señales** que se va encontrando (hombre, mujer, escalera y teléfono).

Fue la demo final de la asignatura *Robótica y Percepción Computacional* (Grado en Ingeniería Informática, Universidad Politécnica de Madrid). La hicimos en un grupo de 3 personas.

---

## Arquitectura

El código está dividido en dos partes, **percepción** y **control**:

```
cámara ──► vision.py (Percepcion.procesar) ──► Resultado ──► BrainFinalExam.py (decidir) ──► move(avance, giro)
sonares ─────────────────────────────────────────────────────────────┘
```

### Percepción — [`vision.py`](vision.py)

Cada frame produce un objeto `Resultado` con todo lo que necesita el controlador:

- **Segmentación de color** combinando **HSV** y **RGB normalizado (rg)**, con apertura y cierre morfológicos. Separa suelo, línea azul y marcas rojas.
- **Error de seguimiento**: desplazamiento lateral del centroide de la línea, normalizado a `[-1, 1]`.
- **Clasificación de la escena**: recta, curva o cruce en **Y / T / X**, detectando las ramas de salida y lo cerca que está la bifurcación.
- **Orientación de la flecha** (izquierda, derecha o recto) a partir de su contorno.
- **Reconocimiento de señales** con un clasificador **k-NN** sobre descriptores de contorno: **momentos de Hu**, circularidad, solidez, *extent*, relación de aspecto y defectos de convexidad. Se complementa con reglas geométricas para separar flechas de figuras humanas.

### Control — [`BrainFinalExam.py`](BrainFinalExam.py)

Máquina de estados con una capa de emergencia por encima:

| Modo | Comportamiento |
|------|----------------|
| `LINEA` | Control **proporcional** sobre el error de la línea, con la velocidad de crucero limitada. Si pierde la línea, intenta recuperarla y después hace un barrido de búsqueda. |
| `CRUCE` | Se **aproxima** al nodo siguiendo la línea mientras vota la dirección de la flecha (votos ponderados por área). Al llegar, fija la dirección y **gira** hasta reengancharse a la nueva rama. |
| `EVADIR` | Se aleja del obstáculo con un giro proporcional a la diferencia entre los sonares izquierdos y derechos. |
| `RODEAR` | Bordea el obstáculo a una distancia segura hasta reencontrar la línea. |
| *Emergencia* | Si algún sonar frontal marca menos de 0,3 m, frena y gira, sea cual sea el modo. |

Usa **anti-rebote** (varios frames seguidos antes de dar un cruce por bueno) y **tiempos de espera** para no volver a detectar el mismo cruce ni anunciar dos veces la misma señal. Al terminar, muestra un resumen con las señales detectadas.

## Estructura

```
├── BrainFinalExam.py     # Cerebro del robot (control)
├── vision.py             # Módulo de percepción
├── run.sh                # Lanza el simulador
└── mundo/                # Escenario del examen (proporcionado por la asignatura)
    ├── finalExam-202526-line1-v01.world
    └── images/           # Circuito y señales; también se usan para entrenar el k-NN
```

## Ejecución

Requiere el entorno de la asignatura: **Pyrobot** con el simulador **Stage** (Linux), **OpenCV** y **NumPy**.

```bash
./run.sh
```

Con `DEMO_HEADLESS=1` no se abre la ventana de depuración, que muestra el modo, la escena, la marca y la flecha sobre la imagen de la cámara. Opcionalmente, `MARKS_DIR` puede apuntar a una carpeta con más capturas de señales para entrenar el clasificador.
