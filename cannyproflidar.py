"""
OAK-D Lite — Suivi de colis (Canny) + Hauteur via LiDAR TF-Luna
- Dimensions 2D (largeur × longueur) depuis le flux RGB + Canny
- Hauteur = HAUTEUR_CONVOYEUR_CM − distance LiDAR
- Lecture LiDAR dans un thread séparé (non bloquant)
Quitter : touche Q
"""

import cv2
import depthai as dai
import numpy as np
import serial
import threading

# ─── Paramètres LiDAR TF-Luna ────────────────────────────────────────────────
PORT_LIDAR          = "COM4"
BAUD_LIDAR          = 115200
HAUTEUR_CONVOYEUR_CM = 110.0          # Distance caméra → convoyeur vide (cm)
SIGNAL_MIN          = 100             # Force minimale du signal LiDAR (fiabilité)

# ─── Paramètres physiques (Zone du convoyeur et étalonnage) ──────────────────
MARGE_X = int((490 - 150) * 0.05)
MARGE_Y = int((350 - 50)  * 0.05)

X_MIN = (150 + MARGE_X) + 51
X_MAX = 490 - MARGE_X
Y_MIN, Y_MAX = 50 + MARGE_Y, 350 - MARGE_Y

CENTRE_CROP_X = (X_MAX - X_MIN) // 2
CENTRE_CROP_Y = (Y_MAX - Y_MIN) // 2

AREA_MIN_COLIS = 5000
RATIO_CM_PX    = (21.0 / 114.0) * 1.02533 # Correction empirique pour compenser la perspective (le dessus du colis est plus proche de la caméra que le sol)

# ─── Thread LiDAR TF-Luna ────────────────────────────────────────────────────
# Protocole UART TF-Luna : trame de 9 octets
#   [0x59][0x59][Dist_L][Dist_H][Str_L][Str_H][Temp_L][Temp_H][Checksum]
#   distance (cm) = Dist_L + Dist_H * 256
#   checksum      = somme des 8 premiers octets & 0xFF

_dist_lidar_cm  = None   # dernière mesure valide
_lidar_strength = 0      # force du signal (fiabilité)
_lidar_ok       = False  # True si le port est ouvert
_lidar_lock     = threading.Lock()


def _thread_lidar():
    global _dist_lidar_cm, _lidar_strength, _lidar_ok

    try:
        ser = serial.Serial(PORT_LIDAR, BAUD_LIDAR, timeout=1)
        _lidar_ok = True
        print(f"[LiDAR] Connecté sur {PORT_LIDAR}")
    except serial.SerialException as e:
        print(f"[LiDAR] Impossible d'ouvrir le port : {e}")
        print("[LiDAR] La hauteur ne sera pas disponible.")
        return

    while True:
        try:
            # Synchronisation sur l'en-tête 0x59 0x59
            if ser.read(1) != b'\x59':
                continue
            if ser.read(1) != b'\x59':
                continue

            data = ser.read(7)   # [Dist_L, Dist_H, Str_L, Str_H, Temp_L, Temp_H, CS]
            if len(data) < 7:
                continue

            # Vérification du checksum
            cs_calc = (0x59 + 0x59 + sum(data[:6])) & 0xFF
            if cs_calc != data[6]:
                continue

            dist     = data[0] + data[1] * 256
            strength = data[2] + data[3] * 256

            with _lidar_lock:
                _dist_lidar_cm  = dist
                _lidar_strength = strength

        except serial.SerialException:
            break   # Port perdu, on arrête silencieusement


_thread = threading.Thread(target=_thread_lidar, daemon=True)
_thread.start()


def get_hauteur_colis():
    """Retourne la hauteur du colis (cm) ou None si mesure indisponible."""
    with _lidar_lock:
        dist     = _dist_lidar_cm
        strength = _lidar_strength

    if dist is None or strength < SIGNAL_MIN:
        return None

    hauteur = HAUTEUR_CONVOYEUR_CM - dist
    return round(hauteur, 1) if hauteur > 0 else None


# ─── Pipeline OAK-D (RGB uniquement) ─────────────────────────────────────────
pipeline  = dai.Pipeline()
color_cam = pipeline.create(dai.node.ColorCamera)
xout_rgb  = pipeline.create(dai.node.XLinkOut)

xout_rgb.setStreamName("couleur")
color_cam.setPreviewSize(640, 400)
color_cam.setInterleaved(False)
color_cam.setFps(30)
color_cam.preview.link(xout_rgb.input)

# ─── Interface curseurs Canny ─────────────────────────────────────────────────
def nothing(x):
    pass

cv2.namedWindow("Masque Canny (Crop)")
cv2.createTrackbar("Seuil Min", "Masque Canny (Crop)", 50,  255, nothing)
cv2.createTrackbar("Seuil Max", "Masque Canny (Crop)", 150, 255, nothing)

# ─── Boucle principale ────────────────────────────────────────────────────────
print("Démarrage OAK-D Lite (Canny + LiDAR TF-Luna)...")
print("Appuyer sur Q pour quitter.\n")

with dai.Device(pipeline) as dispositif:

    cola_color = dispositif.getOutputQueue(name="couleur", maxSize=4, blocking=False)

    while True:
        paq_color = cola_color.tryGet()

        if paq_color is not None:
            vis_color = paq_color.getCvFrame()

            # ── A. Crop ───────────────────────────────────────────────────────
            crop = vis_color[Y_MIN:Y_MAX, X_MIN:X_MAX]

            # ── B. Canny ──────────────────────────────────────────────────────
            gray    = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (7, 7), 0)

            seuil_min = cv2.getTrackbarPos("Seuil Min", "Masque Canny (Crop)")
            seuil_max = cv2.getTrackbarPos("Seuil Max", "Masque Canny (Crop)")

            edges  = cv2.Canny(blurred, seuil_min, seuil_max)
            kernel = np.ones((5, 5), np.uint8)
            edges  = cv2.dilate(edges, kernel, iterations=1)

            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)

            # ── C. Sélection du contour central ───────────────────────────────
            meilleur_contour = None
            min_dist = float('inf')

            for cnt in contours:
                if cv2.contourArea(cnt) > AREA_MIN_COLIS:
                    M = cv2.moments(cnt)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        dist = (cx - CENTRE_CROP_X)**2 + (cy - CENTRE_CROP_Y)**2
                        if dist < min_dist:
                            min_dist = dist
                            meilleur_contour = cnt

            # ── D. Dessin + dimensions ─────────────────────────────────────────
            hauteur_cm = get_hauteur_colis()   # None si LiDAR absent ou faible

            if meilleur_contour is not None:
                rect = cv2.minAreaRect(meilleur_contour)
                larg_px, haut_px = rect[1]

                # Correction perspective : le dessus du colis est plus proche
                # de la caméra que le sol → les pixels couvrent moins de cm
                # facteur = dist_lidar / hauteur_convoyeur  (< 1 si colis présent)
                if hauteur_cm is not None:
                    dist_lidar_cm = HAUTEUR_CONVOYEUR_CM - hauteur_cm
                    facteur = dist_lidar_cm / HAUTEUR_CONVOYEUR_CM
                else:
                    facteur = 1.0   # pas de LiDAR → ratio fixe comme avant

                larg_cm = larg_px * RATIO_CM_PX * facteur
                long_cm = haut_px * RATIO_CM_PX * facteur

                box        = cv2.boxPoints(rect)
                box        = np.int32(box)
                box_global = box + [X_MIN, Y_MIN]

                cv2.drawContours(vis_color, [box_global], 0, (0, 255, 80), 2)

                M  = cv2.moments(meilleur_contour)
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                cv2.circle(vis_color, (cx + X_MIN, cy + Y_MIN), 4, (0, 0, 255), -1)

                cv2.putText(vis_color, "Colis verrouille",
                            (X_MIN, Y_MIN - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 80), 2)

                # Texte dimensions : L × l × H  (H = "?" si LiDAR indisponible)
                if hauteur_cm is not None:
                    dim_texte = (f"L:{larg_cm:.1f}  l:{long_cm:.1f}  H:{hauteur_cm:.1f} cm")
                else:
                    dim_texte = (f"L:{larg_cm:.1f}  l:{long_cm:.1f}  H:-- cm")

            else:
                cv2.putText(vis_color, "En attente...",
                            (X_MIN, Y_MIN - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 100, 255), 2)
                if hauteur_cm is not None:
                    dim_texte = f"L:--  l:--  H:{hauteur_cm:.1f} cm"
                else:
                    dim_texte = "L:--  l:--  H:-- cm"

            # Bandeau dimensions
            cv2.rectangle(vis_color, (10, 10), (430, 50), (0, 0, 0), -1)
            cv2.putText(vis_color, dim_texte, (20, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # Indicateur état LiDAR (petit point en haut à droite)
            couleur_lidar = (0, 255, 0) if hauteur_cm is not None else (0, 0, 255)
            cv2.circle(vis_color, (620, 20), 8, couleur_lidar, -1)
            cv2.putText(vis_color, "LiDAR", (590, 48),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, couleur_lidar, 1)

            # ── E. Zone de sécurité ────────────────────────────────────────────
            cv2.rectangle(vis_color, (X_MIN, Y_MIN), (X_MAX, Y_MAX), (0, 0, 255), 1)

            cv2.imshow("OAK-D Lite — Flux Principal", vis_color)
            cv2.imshow("Masque Canny (Crop)", edges)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cv2.destroyAllWindows()
print("\nProgramme terminé.")