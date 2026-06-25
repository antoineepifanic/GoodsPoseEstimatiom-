import cv2
import depthai as dai
import blobconverter
import numpy as np

# ─── 1. Initialisation du Pipeline ──────────────────────────────────────────
pipeline = dai.Pipeline()

# ─── 2. Configuration de la Caméra Couleur ──────────────────────────────────
color_cam = pipeline.create(dai.node.ColorCamera)
# 416x416 est la résolution d'entrée standard pour les modèles YOLOv4
color_cam.setPreviewSize(416, 416)
color_cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
color_cam.setInterleaved(False)
color_cam.setFps(15)

# ─── 3. Configuration de la Stéréovision (Tes fixes sont conservés) ─────────
mono_izq = pipeline.create(dai.node.MonoCamera)
mono_izq.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
mono_izq.setCamera("left")

mono_der = pipeline.create(dai.node.MonoCamera)
mono_der.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
mono_der.setCamera("right")

estereo = pipeline.create(dai.node.StereoDepth)
estereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.DEFAULT)
estereo.setSubpixel(True)
estereo.setLeftRightCheck(True)
# CRUCIAL : On aligne la carte de profondeur sur la caméra couleur
estereo.setDepthAlign(dai.CameraBoardSocket.CAM_A) 

# ─── 4. Le cœur du système : Réseau de Détection Spatial ────────────────────
spatial_nn = pipeline.create(dai.node.YoloSpatialDetectionNetwork)

# Chargement d'un modèle YOLO pré-entraîné depuis le zoo DepthAI
spatial_nn.setBlobPath(blobconverter.from_zoo(name="yolov4_tiny_coco_416x416", shaves=6))
spatial_nn.setConfidenceThreshold(0.5)
spatial_nn.input.setBlocking(False)

# C'est ici que la magie opère pour la robustesse de la profondeur :
# On réduit la zone de calcul à 50% du centre de la Bounding Box pour 
# éviter d'attraper les bords du colis et le tapis de fond.
spatial_nn.setBoundingBoxScaleFactor(0.5) 
spatial_nn.setDepthLowerThreshold(100)  # Ignorer les pixels aberrants < 10 cm
spatial_nn.setDepthUpperThreshold(5000) # Ignorer les pixels aberrants > 5 m

# ─── 5. Liaisons (Routing des flux) ─────────────────────────────────────────
mono_izq.out.link(estereo.left)
mono_der.out.link(estereo.right)

color_cam.preview.link(spatial_nn.input)
estereo.depth.link(spatial_nn.inputDepth)

# ─── 6. Création des files de sortie ────────────────────────────────────────
xout_rgb = pipeline.create(dai.node.XLinkOut)
xout_rgb.setStreamName("rgb")
spatial_nn.passthrough.link(xout_rgb.input)

xout_nn = pipeline.create(dai.node.XLinkOut)
xout_nn.setStreamName("detections")
spatial_nn.out.link(xout_nn.input)

# ─── 7. Boucle principale ───────────────────────────────────────────────────
print("Démarrage OAK-D Lite (Spatial AI)...")

with dai.Device(pipeline) as device:
    q_rgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
    q_nn = device.getOutputQueue(name="detections", maxSize=4, blocking=False)

    while True:
        in_rgb = q_rgb.tryGet()
        in_nn = q_nn.tryGet()

        if in_rgb is not None:
            frame = in_rgb.getCvFrame()

            if in_nn is not None:
                detections = in_nn.detections

                for det in detections:
                    # Extraction des coordonnées de la Bounding Box
                    x1 = int(det.xmin * frame.shape[1])
                    y1 = int(det.ymin * frame.shape[0])
                    x2 = int(det.xmax * frame.shape[1])
                    y2 = int(det.ymax * frame.shape[0])

                    # La distance calculée en millimètres sur l'axe Z
                    dist_mm = int(det.spatialCoordinates.z)
                    dist_cm = dist_mm / 10.0

                    # Affichage
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 80), 2)
                    texte = f"Colis: {dist_cm:.1f} cm"
                    cv2.putText(frame, texte, (x1 + 5, y1 + 25), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 80), 2)

            cv2.imshow("OAK-D Spatial Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cv2.destroyAllWindows()