import cv2
import numpy as np
import depthai as dai
import time

# --------------------------------------------------
# CONFIGURACIÓN
# --------------------------------------------------
DISTANCIA_CAMARA_CINTA_CM = 110.0   # distancia fija cámara -> cinta
RGB_W = 640
RGB_H = 400


def crear_pipeline():
    pipeline = dai.Pipeline()

    # ---------------- RGB ----------------
    cam_rgb = pipeline.create(dai.node.ColorCamera)
    cam_rgb.setBoardSocket(dai.CameraBoardSocket.CAM_A)
    cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    cam_rgb.setPreviewSize(RGB_W, RGB_H)
    cam_rgb.setInterleaved(False)
    cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)

    # ---------------- MONO LEFT / RIGHT ----------------
    mono_left = pipeline.create(dai.node.MonoCamera)
    mono_left.setBoardSocket(dai.CameraBoardSocket.CAM_B)
    mono_left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_480_P)

    mono_right = pipeline.create(dai.node.MonoCamera)
    mono_right.setBoardSocket(dai.CameraBoardSocket.CAM_C)
    mono_right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_480_P)

    # ---------------- STEREO DEPTH ----------------
    stereo = pipeline.create(dai.node.StereoDepth)
    stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.DEFAULT)

    # Alinear profundidad con la cámara RGB
    stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)
    stereo.setOutputSize(RGB_W, RGB_H)

    stereo.setLeftRightCheck(True)
    stereo.setSubpixel(False)

    mono_left.out.link(stereo.left)
    mono_right.out.link(stereo.right)

    # ---------------- SALIDA RGB ----------------
    xout_rgb = pipeline.create(dai.node.XLinkOut)
    xout_rgb.setStreamName("rgb")
    cam_rgb.preview.link(xout_rgb.input)

    # ---------------- SALIDA DEPTH ----------------
    xout_depth = pipeline.create(dai.node.XLinkOut)
    xout_depth.setStreamName("depth")
    stereo.depth.link(xout_depth.input)

    return pipeline


def obtener_distancia_centro(depth_frame, cx, cy, ventana=7):
    """
    Devuelve la distancia en mm alrededor del punto central usando la mediana
    de una pequeña ventana para reducir ruido.
    """
    h, w = depth_frame.shape[:2]

    x1 = max(0, cx - ventana)
    x2 = min(w, cx + ventana + 1)
    y1 = max(0, cy - ventana)
    y2 = min(h, cy + ventana + 1)

    roi = depth_frame[y1:y2, x1:x2]

    # Solo valores válidos (>0)
    validos = roi[roi > 0]
    if len(validos) == 0:
        return None

    return float(np.median(validos))  # mm


def main():
    pipeline = crear_pipeline()

    with dai.Device(pipeline) as device:
        rgb_queue = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
        depth_queue = device.getOutputQueue(name="depth", maxSize=4, blocking=False)

        print("Iniciando medición del punto central...")
        print(f"Distancia fija cámara -> cinta: {DISTANCIA_CAMARA_CINTA_CM:.1f} cm")

        while True:
            in_rgb = rgb_queue.tryGet()
            in_depth = depth_queue.tryGet()

            if in_rgb is None or in_depth is None:
                time.sleep(0.005)
                continue

            frame_rgb = in_rgb.getCvFrame()
            frame_depth = in_depth.getFrame()   # uint16 en mm

            # Centro de la imagen
            h, w = frame_rgb.shape[:2]
            cx = w // 2
            cy = h // 2

            # Distancia medida en el centro
            distancia_mm = obtener_distancia_centro(frame_depth, cx, cy, ventana=7)

            # Dibujar centro
            cv2.drawMarker(
                frame_rgb,
                (cx, cy),
                (0, 255, 255),
                markerType=cv2.MARKER_CROSS,
                markerSize=20,
                thickness=2
            )

            if distancia_mm is not None:
                distancia_cm = distancia_mm / 10.0
                altura_paquete_cm = DISTANCIA_CAMARA_CINTA_CM - distancia_cm

                # Evitar negativos
                if altura_paquete_cm < 0:
                    altura_paquete_cm = 0.0

                texto1 = f"Distancia centro: {distancia_cm:.1f} cm"
                texto2 = f"Altura paquete: {altura_paquete_cm:.1f} cm"

                cv2.putText(frame_rgb, texto1, (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.putText(frame_rgb, texto2, (20, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                print(f"Distancia centro = {distancia_cm:.1f} cm | Altura paquete = {altura_paquete_cm:.1f} cm")

            else:
                cv2.putText(frame_rgb, "No hay profundidad valida en el centro", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            # SOLO mostrar la cámara RGB
            cv2.imshow("Camara RGB - Altura paquete", frame_rgb)

            tecla = cv2.waitKey(1) & 0xFF
            if tecla == ord('q'):
                break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()