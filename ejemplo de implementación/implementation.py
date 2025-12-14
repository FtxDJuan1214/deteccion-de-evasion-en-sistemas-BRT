import os
import sys
from ultralytics import YOLO
import evasion_detection as evasionModel

"""
    Ejemplo de implementación del método de detección de evasión.
    Este método procesa un video para detectar evasiones de objetos basadas en configuraciones
    de modelos YOLO entrenados al contexto y umbrales.

    Parámetros:
    config:
        "person_model": [
            "yolo11s.pt",
            "yolo11m.pt",
            "yolo11...pt",

            "yolo11tm.pt" - modelo entrenado con dataset WiderPerson
        ],
        "station_model": [
            "stationv1.pt", - toma todo el poligono de la estación
            "stationv2.pt" - toma el poligono de la estación sin el borde
        ],
        "person_conf": [0.35, 0.40, 0.50], - confianza mínima para que YOLO detecte personas
        "person_iou": [0.4, 0.5], - umbral de intersección sobre unión para la detección de personas
        "imgsz": [1280, 1920], - tamaño de la imagen para la detección de personas
        "PROXIMITY_THRESHOLD": [100, 150, 200], - umbral de proximidad a la estación para detectar evasiones
        "MIN_TIME_OUTSIDE": [1, 1.5, 2.0], - tiempo mínimo fuera de la estación para considerar que entró
        "tracker": [
            "bytetrack", - tracker de ByteTrack
            "botsort" - tracker de BoTSORT (mejor para múltitudes)
        ]

    Retorna:
    - int (conteo de evasiones), o
    - diccionario con métricas si return_detailed_metrics=True
"""

# Ruta de videos de ejemplo
src_videos = r'../videos ejemplos'

# ejemplos relevantes - descomentar para probar

# video_name = 'Ch2_20181120094414-02.mkv' #1 colados ✅
# video_name = 'Ch1_20181120194059-04.mkv' #16 colados ✅
# video_name = 'Ch2_20181118175143-09.mkv' #8 colados ✅
# video_name = 'Ch2_20181118193941-03.mkv' # 2 colados ✅
# video_name = 'Ch1_20181023064348-00.mkv' #1 colados - complejo ✅
# video_name = '0-Ch2_20181118132051-04.mkv' #0 colados ✅
# video_name = 'Ch2_20181118071855-02.mkv' #persona compleja ✅
# video_name = '0-Ch2_20181019073619-00.mkv' #No colado pero complejo ✅

# Ruta base donde guardar resultados
OUT_DIR = 'procesados'
os.makedirs(OUT_DIR, exist_ok=True)


src_yolo_models = r"../modelos de deteción de personas"
src_station_models = r"../modelos de detección de estación"
src_trackers = r"../trackers"

# Configuración por defecto (se puede jugar con los parámetros descritos al inicio)
config = {
    "models": {
        "person": {
            "iou": 0.4,
            "conf": 0.35,
            "imgsz": 1280,
            "model": os.path.join(src_yolo_models, "yolo11m.pt"),
            "tracker": os.path.join(src_trackers, "botsort.yaml"),
        },
        "station": os.path.join(src_station_models, "stationv1.pt"),
    },
    "PROXIMITY_THRESHOLD": 50,
    "MIN_TIME_OUTSIDE": 1,
}

# Cargar modelos (esto en caso de que se quiera implementar para varios videos -  se pueden enviar como None)
person_model_path = config["models"]["person"]["model"]
station_model_path = config["models"]["station"]

person_model = YOLO(person_model_path)
station_model = YOLO(station_model_path)

# Ruta completa del video a procesar
video_path = os.path.join(src_videos, video_name)

try:
    # Llamar la función con la configuración por defecto
    metrics = evasionModel.count_evasion(
        video_path,
        config,
        person_model=person_model,
        station_model=station_model,
        headless=False,
        output_video_path=os.path.join(OUT_DIR, video_name),
        return_detailed_metrics=False
    )

except Exception as e:
    print(f"\n      Error en {video_name}: {e} ⛔")

# liberar modelos
try:
    person_model.close()
    station_model.close()
except Exception:
    pass


        