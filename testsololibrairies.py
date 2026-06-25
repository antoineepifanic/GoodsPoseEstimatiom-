"""
Lectura de distancia central con OAK-D Lite
Solo profundidad estereo nativa - sin calculos adicionales
Salir: tecla Q
"""

import cv2
import depthai as dai
import numpy as np

# --- Pipeline minimalista ---

pipeline = dai.Pipeline()

mono_izq = pipeline.create(dai.node.MonoCamera)
mono_der = pipeline.create(dai.node.MonoCamera)
estereo  = pipeline.create(dai.node.StereoDepth)
salida   = pipeline.create(dai.node.XLinkOut)

salida.setStreamName("profundidad")

mono_izq.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
mono_der.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
mono_izq.setCamera("left")
mono_der.setCamera("right")

# HIGH_DENSITY = mapa mas lleno, menos zonas sin dato
estereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
estereo.initialConfig.setMedianFilter(dai.MedianFilter.KERNEL_7x7)

mono_izq.out.link(estereo.left)
mono_der.out.link(estereo.right)
estereo.depth.link(salida.input)

# --- Bucle principal ---

print("Iniciando OAK-D Lite...")
print("Presione Q para salir\n")

with dai.Device(pipeline) as dispositivo:

    cola = dispositivo.getOutputQueue(
        name="profundidad",
        maxSize=1,          # Solo conserva el frame mas reciente
        blocking=False      # No bloquea si la cola esta llena
    )

    while True:

        paquete = cola.tryGet()

        if paquete is None:
            # Sin frame disponible todavia, esperar sin bloquear CPU
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue

        frame_prof = paquete.getFrame()   # uint16, valores en milimetros
        h, w = frame_prof.shape
        cx, cy = w // 2, h // 2

        # Lectura en zona 5x5 centrada (mediana para evitar pixel ruidoso)
        zona = frame_prof[cy-2:cy+3, cx-2:cx+3]
        pixeles_validos = zona[zona > 0]

        if pixeles_validos.size > 0:
            distancia_mm = int(np.median(pixeles_validos))
            distancia_cm = distancia_mm / 10.0
            texto_dist   = f"Distancia: {distancia_mm} mm  ({distancia_cm:.1f} cm)"
            color_texto  = (0, 255, 80)
        else:
            distancia_mm = 0
            texto_dist   = "Sin medida valida"
            color_texto  = (0, 80, 255)

        # --- Visualizacion ---

        # Normalizar mapa de profundidad a escala de grises
        frame_vis = cv2.normalize(frame_prof, None, 0, 255,
                                  cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        frame_vis = cv2.applyColorMap(frame_vis, cv2.COLORMAP_INFERNO)

        # Mira central
        cv2.line(frame_vis, (cx - 20, cy), (cx + 20, cy), (255,255,255), 1)
        cv2.line(frame_vis, (cx, cy - 20), (cx, cy + 20), (255,255,255), 1)
        cv2.circle(frame_vis, (cx, cy), 5, (255, 255, 255), 1)

        # Fondo semitransparente para el texto
        overlay = frame_vis.copy()
        cv2.rectangle(overlay, (0, 0), (w, 50), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.55, frame_vis, 0.45, 0, frame_vis)

        cv2.putText(frame_vis, texto_dist,
                    (10, 33), cv2.FONT_HERSHEY_SIMPLEX,
                    0.85, color_texto, 2, cv2.LINE_AA)

        cv2.imshow("OAK-D Lite - Distancia central", frame_vis)

        print(f"\r{texto_dist}          ", end="", flush=True)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cv2.destroyAllWindows()
print("\nCerrado.")