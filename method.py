import av
import cv2
import random
import numpy as np
from ultralytics import YOLO
from collections import defaultdict

"""
    Esta es el código final del método desarrollado para detectar la evasión por puertas de vagones en sistemas BRT.	
    Procesa un video y cuenta las evasiones detectadas.

    Parámetros:
    - video_path: ruta al video a procesar.
    - conf: diccionario de configuración con modelos y thresholds.
    - person_model: modelo YOLO ya cargado.
    - station_model: modelo YOLO ya cargado.
    - headless: si True, no muestra ventanas ni usa cv2.imshow().
    - output_video_path: ruta para guardar el video procesado (None = no guardar).
    - return_detailed_metrics: si True, devuelve un diccionario de métricas detalladas.

    Retorna:
    - int (conteo de evasiones), o
    - diccionario con métricas si return_detailed_metrics=True
"""

def count_evasion(
    video_path,
    conf,
    person_model=None,
    station_model=None,
    headless=False,
    output_video_path=None,
    return_detailed_metrics=False
):    
    # -------------------  Config visual -------------------
    color_station = (105, 255, 105) # polígono estación
    font_scale = 0.8
    thickness = 2


    # -------------------  Cargar modelos -------------------
    # Cargar modelos si no se pasaron desde fuera
    local_person = False
    local_station = False

    if person_model is None:
        person_model = YOLO(conf["models"]["person"]["model"])
        local_person = True

    if station_model is None:
        station_model = YOLO(conf["models"]["station"])
        local_station = True

    # -------------------  Obtener polígono de estación -------------------

    def getStationPolygon(frame):
        frame_polygon = None
        img = frame.to_ndarray(format="bgr24")
        results_station = station_model.predict(img, conf=0.90, verbose=False) # conf 0.90 para detectar la estación funciona siempre
        if(results_station[0].masks is not None):
            if len(results_station[0].masks) > 0:
                mask = results_station[0].masks.xy[0]
                frame_polygon = np.array(mask, dtype=np.int32)
        return frame_polygon

    container = av.open(video_path)
    first_frame = next(container.decode(video=0))
    first_frame_polygon = getStationPolygon(first_frame)

    container = av.open(video_path)
    container.seek(int(container.duration * av.time_base))
    last_frame = None
    for frame in container.decode(video=0):
        last_frame = frame
    last_frame_polygon = getStationPolygon(last_frame)

    station_polygon = None

    # -------------------  Obtener polígono de estación -------------------
    # si hay polígono en el primer frame y en el último, se elige el más ajustado
    if(first_frame_polygon is not None and last_frame_polygon is not None):
        first_frame_ratio = cv2.arcLength(first_frame_polygon, True) / cv2.contourArea(first_frame_polygon)
        last_frame_ratio = cv2.arcLength(last_frame_polygon, True) / cv2.contourArea(last_frame_polygon)

        # a menor ratio, mejor es el polígono
        station_polygon = first_frame_polygon if first_frame_ratio < last_frame_ratio else last_frame_polygon
    else:
        # si no hay polígono en el primer frame, se elige el último
        station_polygon = first_frame_polygon if first_frame_polygon is not None else None
        if(station_polygon is None):
            station_polygon = last_frame_polygon if last_frame_polygon is not None else None

    # ------------------- Inicializar tracker con la configuración -------------------
    tracker = person_model.track(
        video_path,
        classes=[0],
        conf=conf["models"]["person"]["conf"],
        iou=conf["models"]["person"]["iou"],
        persist=True,
        stream=True,
        verbose=False,
        tracker=conf["models"]["person"]["tracker"],
        imgsz=conf["models"]["person"]["imgsz"],
    )

    # Diccionarios
    trajectories = defaultdict(list) #{id: [(x,y), (x,y), ...]} '-- historial de puntos
    track_colors = {} #color único para cada track


    # -------------------  Configuración de salida -------------------
    output_path = "processed.mp4"
    container_in = av.open(video_path)
    width = container_in.streams.video[0].width
    height = container_in.streams.video[0].height
    fps = float(container_in.streams.video[0].average_rate)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = None
    if output_video_path is not None:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))


    # -------------------  Variables de detección de evasión -------------------
    entered_ids = set()         # guarda IDs que han entrado al polígono=
    previous_positions = {}     # guarda la última posición conocida de cada track
    first_entry_point = {}      # guarda el punto de entrada al polígono
    entry_frame = {}            # frame en que entró al polígono
    first_seen_frame = {}       # frame en que se detectó por primera vez el track
    last_seen_position = {}     # ultima posición conocida del track
    last_seen_frame = {}        # ultimo frame en que se vio el track
    evasion_marks = {}          # {track_id: (x, y, tipo)} donde tipo es 'entrada' o 'desaparición'
    evasion_detected = set()    # Ids que ya fueron marcados como evasión
    ignored_ids = set()         # Ids que aparecieron dentro del polígono (ignorar)

    PROXIMITY_THRESHOLD = conf["PROXIMITY_THRESHOLD"]   # Distancia en píxeles para considerar "cerca del polígono"
    MIN_TIME_OUTSIDE = conf["MIN_TIME_OUTSIDE"]      # Segundos mínimos de vida del track antes de entrar
    frame_count = 0


    # -------------------  sacar distancia de un punto al polígono de la estación -------------------
    def distance_to_polygon(point, polygon):
        return abs(cv2.pointPolygonTest(polygon, point, True))


    # -------------------  procesar frames del video -------------------
    for result in tracker:
        img = result.orig_img.copy()
        annotated_frame = img.copy()
        frame_count += 1

        # Mostrar poligono de la estación
        if station_polygon is not None:
            overlay = annotated_frame.copy()
            cv2.fillPoly(overlay, [station_polygon], color=color_station)
            alpha = 0.3
            cv2.addWeighted(overlay, alpha, annotated_frame, 1 - alpha, 0, annotated_frame)
            cv2.polylines(annotated_frame, [station_polygon], isClosed=True, color=color_station, thickness=2)

        # IDs activos en este frame
        active_ids = set()

        # Detecciones con IDs
        if result.boxes.id is not None:
            boxes = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            ids = result.boxes.id.cpu().numpy().astype(int)

            for (x1, y1, x2, y2, conf, track_id) in zip(boxes[:,0], boxes[:,1], boxes[:,2], boxes[:,3], confs, ids):
                active_ids.add(track_id)

                # Asignar color
                if track_id not in track_colors:
                    track_colors[track_id] = (
                        random.randint(0, 255),
                        random.randint(0, 255),
                        random.randint(0, 255)
                    )

                color_person = track_colors[track_id]
                cx, cy = int((x1+x2)/2), int((y1+y2)/2)

                # Verificar si está dentro del polígono
                inside = False
                if station_polygon is not None:
                    inside = cv2.pointPolygonTest(station_polygon, (cx, cy), False) >= 0

                # Si es la primera vez que vemos este track, verificar si aparece dentro
                if track_id not in previous_positions:
                    first_seen_frame[track_id] = frame_count
                    if inside:
                        ignored_ids.add(track_id)

                # Ignorar este track si apareció dentro del polígono
                if track_id in ignored_ids:
                    previous_positions[track_id] = inside
                    continue

                # Actualizar última posición conocida
                last_seen_position[track_id] = (cx, cy)
                last_seen_frame[track_id] = frame_count

                # Detectar entrada al polígono
                was_inside = previous_positions.get(track_id, False)
                if not was_inside and inside and track_id not in entered_ids:
                    # Verificar que el track estuvo fuera por más de MIN_TIME_OUTSIDE segundos
                    frames_outside = frame_count - first_seen_frame[track_id]
                    time_outside = frames_outside / fps
                    
                    if time_outside >= MIN_TIME_OUTSIDE:
                        entered_ids.add(track_id)
                        first_entry_point[track_id] = (cx, cy)
                        entry_frame[track_id] = frame_count
                        
                        # CASO 1: Marcar como evasión inmediatamente al entrar
                        evasion_marks[track_id] = (cx, cy, 'entrada')
                        evasion_detected.add(track_id)
                        # print(f"EVASIÓN DETECTADA (Caso 1): ID {track_id} estuvo fuera {time_outside:.1f}s y entró en frame {frame_count}")

                previous_positions[track_id] = inside

                # Guardar trayectoria completa (dentro y fuera)
                trajectories[track_id].append((cx, cy))

                # Dibujar boundingbox
                cv2.rectangle(annotated_frame,
                            (int(x1), int(y1)),
                            (int(x2), int(y2)),
                            color_person, thickness=thickness)

                label = f"ID {track_id} ({conf:.2f})"
                cv2.putText(annotated_frame, label,
                            (int(x1), int(y1) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=font_scale,
                            color=color_person,
                            thickness=thickness)

                # Dibujar trayectoria
                pts = np.array(trajectories[track_id], np.int32)
                if len(pts) > 1:
                    cv2.polylines(annotated_frame, [pts], False, color_person, 2)

        # CASO 2: Detectar tracks que desaparecieron cerca del polígono
        all_known_ids = set(last_seen_frame.keys())
        disappeared_ids = all_known_ids - active_ids

        for track_id in disappeared_ids:
            # Ignorar si está en la lista de ignorados
            if track_id in ignored_ids:
                continue
                
            if track_id in evasion_detected or track_id in entered_ids:
                continue
            
            if track_id in last_seen_position and track_id in first_seen_frame and station_polygon is not None:
                # Verificar que la trayectoria existió por más de MIN_TIME_OUTSIDE segundos
                total_frames = last_seen_frame[track_id] - first_seen_frame[track_id]
                total_time = total_frames / fps
                
                if total_time >= MIN_TIME_OUTSIDE:
                    last_pos = last_seen_position[track_id]
                    dist = distance_to_polygon(last_pos, station_polygon)
                    
                    # Verificar que el último punto está cerca del polígono (PROXIMITY_THRESHOLD)
                    if dist <= PROXIMITY_THRESHOLD and len(trajectories[track_id]) >= 3:
                        # Analizar dirección: últimos 3 puntos de la trayectoria
                        recent_points = trajectories[track_id][-3:]
                        distances = [distance_to_polygon(pt, station_polygon) for pt in recent_points]
                        
                        # Si las distancias están decreciendo, se dirigía hacia la estación
                        if distances[0] > distances[1] > distances[2]:
                            evasion_marks[track_id] = (last_pos[0], last_pos[1], 'desaparición')
                            evasion_detected.add(track_id)
                            # print(f"EVASIÓN DETECTADA (Caso 2): ID {track_id} existió {total_time:.1f}s y desapareció a {dist:.1f} px del polígono")

        # Dibujar marcas de evasión
        for track_id, (ex, ey, tipo) in evasion_marks.items():
            color = track_colors.get(track_id, (0, 0, 255))
            
            # Círculo de marca
            cv2.circle(annotated_frame, (ex, ey), 5, color, -1)
            cv2.circle(annotated_frame, (ex, ey), 7, (0, 0, 255), 2)
            
            # Texto
            text = f"[ID:{track_id}]"
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            text_x = ex - text_size[0] // 2
            text_y = ey - 25
            
            # Fondo para el texto
            cv2.rectangle(annotated_frame, 
                        (text_x - 5, text_y - text_size[1] - 5),
                        (text_x + text_size[0] + 5, text_y + 5),
                        (0, 0, 0), -1)
            
            cv2.putText(annotated_frame, text, (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Mostrar contadores en pantalla
        count_text = f"Entradas ilegales: {len(evasion_detected)}"
        cv2.putText(annotated_frame, count_text, (30, height - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)

        # Mostrar y guardar frame
        if not headless:
            cv2.imshow("Evasion TM", annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        if out is not None:
            out.write(annotated_frame)


    if not headless:
        cv2.destroyAllWindows()

    if out is not None:
        out.release()


    print(f"\n--- RESUMEN ---")
    print(f"\n  Total de evasiones detectadas: {len(evasion_detected)}")
    print(f"\n    - Caso 1 (Entrada a la estación): {sum(1 for t in evasion_marks.values() if t[2] == 'entrada')}")
    print(f"\n    - Caso 2 (desaparición cerca): {sum(1 for t in evasion_marks.values() if t[2] == 'desaparición')}")

    case1_count = sum(1 for t in evasion_marks.values() if t[2] == 'entrada')
    case2_count = sum(1 for t in evasion_marks.values() if t[2] == 'desaparición')
    entered_count = len(entered_ids)
    ignored_count = len(ignored_ids)
    evasion_count = len(evasion_detected)

    metrics = {
        "evasion_count": evasion_count,
        "case1_count": case1_count,
        "case2_count": case2_count,
        "entered_count": entered_count,
        "ignored_count": ignored_count
    }

    
    # Liberar modelos si se cargaron dentro de la función
    if local_person:
        del person_model
    if local_station:
        del station_model

    if return_detailed_metrics:
        return metrics
    else:
        return evasion_count