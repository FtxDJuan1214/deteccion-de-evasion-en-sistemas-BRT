# 🚨🚪🏃🏻‍♂️ Detección de evasión por puerta de vagón en sistemas BRT

Este repositorio presenta un **método basado en visión por computador para la detección automática de evasión de pago en puertas laterales de vagones en sistemas BRT**, desarrollado y validado utilizando videos reales del sistema TransMilenio en Bogotá, Colombia.

El proyecto propone una solución integral que combina **modelos de deep learning para percepción visual** con **análisis de trayectorias y reglas espaciales**, permitiendo identificar ingresos ilegales en un escenario altamente complejo debido a la congestión, oclusiones y comportamientos humanos no estructurados.

---

## 🎯 Contexto del problema

La evasión del pago del pasaje representa un problema operativo y económico significativo en los sistemas de transporte masivo tipo BRT. En TransMilenio, una proporción considerable de usuarios ingresa al sistema de manera ilegal, particularmente a través de las puertas laterales de los vagones, donde el control es limitado.

Las metodologías tradicionales para medir este fenómeno dependen en gran medida de conteos manuales o de sistemas propietarios cuyos detalles técnicos no son públicos, lo que dificulta su análisis, comparación y mejora.

<img width="1137" height="524" alt="image" src="https://github.com/user-attachments/assets/25c92c50-aa8c-4c77-86c8-c4f9c8079a9a" />

Este trabajo aborda la problematica desde un enfoque académico.

---

## 🧠 Método propuesto

El método desarrollado fue probado con clips de video de corta duración (20–24 segundos) y videos completos de hasta 20 minutos capturados por cámaras fijas ubicadas en las estaciones, con vista hacia las puertas de los vagones. A diferencia de enfoques basados exclusivamente en reconocimiento de acciones o poses, esta propuesta se centra en el **análisis espacio-temporal de las trayectorias de las personas en relación con la geometría de la estación**.

### Flujo general del método

1. **Segmentación de la estación**
   Se utiliza un modelo YOLO entrenado para segmentación de instancias con el fin de detectar el polígono que define la estación. Se consideran dos variantes del polígono (con y sin plataforma) para adaptarse a diferentes configuraciones y reducir falsos positivos.

2. **Detección de personas**
   Las personas son detectadas cuadro a cuadro mediante modelos YOLO preentrenados y ajustados al contexto del problema. Únicamente se considera la clase *persona*.

3. **Seguimiento de personas (tracking)**
   A partir de las detecciones, se asignan identificadores únicos a cada individuo utilizando algoritmos de tracking-by-detection como **ByteTrack** y **BoT-SORT**, permitiendo reconstruir trayectorias completas a lo largo del video.

4. **Análisis de trayectorias y reglas de decisión**
   En lugar de emplear reconocimiento explícito de poses debido a la poca resolución de muchos casos, el método aplica un conjunto de **reglas basadas en el dominio** sobre las trayectorias:

   * Una trayectoria que inicia fuera de la estación y luego ingresa al polígono de la estación, cumpliendo un tiempo mínimo fuera, se clasifica como evasión.
   * Una trayectoria que desaparece cerca del polígono mientras se dirige hacia la estación también se considera evasión, capturando casos de oclusión o saltos.
   * Se incorporan mecanismos para evitar dobles conteos y reducir falsos positivos/negativos.

5. **Conteo de eventos de evasión**
   Cada trayectoria puede generar como máximo un evento de evasión, permitiendo estimar de forma confiable la cantidad de ingresos ilegales por clip.

Este enfoque híbrido combina la robustez del deep learning con la interpretabilidad de reglas explícitas, lo que resulta adecuado para escenarios reales de videovigilancia.

<img width="1147" height="681" alt="image" src="https://github.com/user-attachments/assets/a1c4ddc8-c528-46e9-8cbc-d9667c34ea4e" />

---

## 📊 Dataset y evaluación

* **1.800 clips de video reales** capturados en estaciones de TransMilenio

  * 1.200 clips con al menos un evento de evasión
  * 600 clips sin evasión
* Escenarios diversos: horas pico, alta densidad de personas, oclusiones, distintas orientaciones de estación
* Cada video tiene una duración entre 20-24 segundos

El método fue evaluado mediante una **exploración sistemática de parámetros (grid search)** sobre 300 videos representativos aplicando 40 configuraciones diferentes, variando:

* Modelos y umbrales de detección
* Algoritmos de tracking
* Parámetros espaciales y temporales del análisis de trayectorias

La métrica principal de evaluación fue el **F1-score**, ya que permite balancear precisión (reducción de falsas alarmas) y recall (detección de eventos reales), aspecto crítico en este contexto.

A continuación se presenta el ranking de las 20 mejores configuraciones
<img width="4727" height="2392" alt="image" src="https://github.com/user-attachments/assets/18dcd688-b9a7-49a7-94c8-51851d04de3f" />

La mejor configuración encontrada consta de los siguientes parámetros:
```python
config = {
    "id": 30,
    "models": {
        "person": {
            "model": "yolo11m.pt",
            "conf": 0.35,
            "iou": 0.4,
            "tracker": "botsort.yaml",
            "imgsz": 1280,
        },
        "station": "stationv1.pt",
    },
    "PROXIMITY_THRESHOLD": 50,
    "MIN_TIME_OUTSIDE": 1,
}
```

Esta configuración se toma como la configuración por defecto en la demostración de este repositorio.

## Validación final

Luego de tener la mejor configuración para los parámetros del método, se realizó la validación con 1500 videos, donde 1000 videos presentaban casos de evasión y 500 no.
A continuación se presentan dos matrices, la primera basada en los conteos exactos y la segunda basada en eventos de evasión, esta segunda matriz se calcula con las diferencias de conteos predichos y reales en cada video.

<img width="772" height="364" alt="image" src="https://github.com/user-attachments/assets/dc4c7514-4824-468a-99c9-03e283fddb2e" />

|  Precision |     Recall |   F1-Score |   Accuracy | MAE       |
| ---------: | ---------: | ---------: | ---------: | --------: |
| **0.9270** | **0.7828** | **0.8488** | **0.7866** | **0.341** |

Con estos resultados se concluyó que:
* El alto valor de precisión indica una baja tasa de falsas alarmas en la detección de evasión.
* El recall refleja una buena capacidad del método para identificar eventos reales de evasión, incluso en escenarios congestionados.
* El F1-Score evidencia un balance adecuado entre precisión y recall.
* El MAE muestra que el error promedio en el conteo de evasiones por video es reducido.
* El conteo exacto indica que el método logra estimar correctamente el número de evasiones en un 75% de los casos.
* El método tiene métricas destacables teniendo en cuenta la alta complejidad que presentan los videos.

---

## 🎥 Ejemplos

A continuación se muestran ejemplos de videos procesados por el método, donde se visualizan las personas detectadas, sus trayectorias y los eventos de evasión identificados.

<details>
  
  <summary><strong>Ejemplos complejos de evasión 📽️</strong></summary>
  
  <br>
  
  https://github.com/user-attachments/assets/235bc1b1-b455-47f2-ad22-f318bc1217b2

  https://github.com/user-attachments/assets/55f725c8-4d94-4826-83ef-925a5a80ec28

</details>

<details>
  
  <summary><strong>Ejemplos regulares de evasión 📽️</strong></summary>
  
  <br>
  
  https://github.com/user-attachments/assets/7c1b3d60-0d6b-4e25-a0e5-2dd6ddb5bdca

  https://github.com/user-attachments/assets/64bcc0de-08ca-4937-a94d-634037ca623d

</details>


<details>
  
  <summary><strong>Ejemplos de no evasión 📽️</strong></summary>
  
  <br>
  
  https://github.com/user-attachments/assets/e008d617-310e-4815-bdd4-283896ace219

  https://github.com/user-attachments/assets/2b147f02-0e1c-4a0f-95d1-1536de8562bc
  
  https://github.com/user-attachments/assets/3b1104c4-5b56-4077-9309-c3bc351333cd

</details>


<details>
  
  <summary><strong>Ejemplos de casos mixtos 📽️</strong></summary>
  
  <br>
  
  https://github.com/user-attachments/assets/1afe256b-a6d7-4dc2-8a1e-3a656b08fba7

</details>

---

## 🧪 Uso del método

* En este reopositorio se encuentra un [`ejemplo de implementación `](https://github.com/FtxDJuan1214/deteccion-de-evasion-en-sistemas-BRT/blob/main/ejemplo%20de%20implementaci%C3%B3n/implementation.py)
 este se complementa con los videos de la [`carpeta de ejemplos `](https://github.com/FtxDJuan1214/deteccion-de-evasion-en-sistemas-BRT/tree/main/videos%20ejemplos) que serviran para hacer multiples pruebas localmente.

---

## ✨ Aportes principales

* Propuesta de un **método automático y explicable** para la detección de evasión por puerta de vagón
* Integración de **segmentación de la estación, detección de personas, tracking multiobjeto y análisis de trayectorias**
* Validación experimental extensiva con datos reales de operación
* Pipeline reproducible y adaptable a otros sistemas BRT

---

## 📌 Consideraciones

* Este repositorio tiene un **propósito académico y de investigación**.
* No representa un sistema de vigilancia en producción.
* Los videos se comparten únicamente con fines demostrativos y científicos.

---

## 👤 Autor

**Juan Camilo Hernández Ortiz**

Maestría en Ingeniería – Ingeniería de Sistemas y Computación

Universidad Nacional de Colombia

2025

---

## 📄 Licencia

Este proyecto se comparte para uso académico y de investigación. Revisa el archivo [`LICENSE`](https://github.com/FtxDJuan1214/deteccion-de-evasion-en-sistemas-BRT/blob/main/LICENSE) para más detalles.
