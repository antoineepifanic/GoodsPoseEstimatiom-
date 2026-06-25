"""
OAK-D Lite — Mesure de distance centrale
- Fixes précision : subpixel + left-right check
- Algorithme : médiane des N% pixels les plus proches
- Double affichage : caméra couleur + carte de profondeur
- Stabilisation temporelle de la distance
Quitter : touche Q
"""

import cv2
import depthai as dai
import numpy as np
from collections import deque

# ─── Pipeline ───────────────────────────────────────────────────────────────

pipeline = dai.Pipeline()

mono_izq   = pipeline.create(dai.node.MonoCamera)
mono_der   = pipeline.create(dai.node.MonoCamera)
estereo    = pipeline.create(dai.node.StereoDepth)
color_cam  = pipeline.create(dai.node.ColorCamera)
xout_prof  = pipeline.create(dai.node.XLinkOut)
xout_color = pipeline.create(dai.node.XLinkOut)

xout_prof.setStreamName("profondeur")
xout_color.setStreamName("couleur")

# Caméras mono stéréo
mono_izq.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
mono_der.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
mono_izq.setCamera("left")
mono_der.setCamera("right")

# Caméra couleur
color_cam.setPreviewSize(640, 400)
color_cam.setInterleaved(False)
color_cam.setFps(15)

# ─── Stéréo depth : configuration corrigée ──────────────────────────────────

# HIGH_DENSITY sigue funcionando pero está deprecado.
# Si tu versión acepta DEFAULT, puedes cambiarlo luego.
estereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)

# Filtro mediano
estereo.initialConfig.setMedianFilter(dai.MedianFilter.KERNEL_7x7)

# FIX 1 — Subpixel
# En DepthAI v3 se activa así:
estereo.setSubpixel(True)

# Para mantener el filtro mediano activo, usamos 3 bits y no 5
estereo.initialConfig.setSubpixelFractionalBits(3)

# FIX 2 — Left-right check
estereo.setLeftRightCheck(True)

# El threshold LR se configura en initialConfig, NO en estereo
estereo.initialConfig.setLeftRightCheckThreshold(5)

# (Opcional) threshold de confianza algo más estricto
# Si ves demasiados ceros, prueba 180 o comenta esta línea
estereo.initialConfig.setConfidenceThreshold(150)

# Enlaces
mono_izq.out.link(estereo.left)
mono_der.out.link(estereo.right)
estereo.depth.link(xout_prof.input)
color_cam.preview.link(xout_color.input)

# ─── Paramètres de mesure ────────────────────────────────────────────────────

ROI_TAM = 21                   # ROI central 21x21
PORCENTAJE_CERCANOS = 0.30     # 30% de los píxeles válidos más cercanos

# Suavizado temporal
N_HIST = 7                     # mediana de las últimas N medidas
historial = deque(maxlen=N_HIST)

# ─── Helpers ─────────────────────────────────────────────────────────────────

def dessiner_bandeau(frame, texte1, texte2, couleur_texte):
    """Bandeau semi-transparent en haut du frame avec 2 lignes de texte."""
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 75), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    cv2.putText(frame, texte1,
                (10, 28), cv2.FONT_HERSHEY_SIMPLEX,
                0.75, couleur_texte, 2, cv2.LINE_AA)

    cv2.putText(frame, texte2,
                (10, 58), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (255, 255, 255), 1, cv2.LINE_AA)


def dessiner_viseur(frame, cx, cy):
    """Croix centrale identique sur les deux fenêtres."""
    cv2.line(frame,  (cx - 20, cy), (cx + 20, cy), (255, 255, 255), 1)
    cv2.line(frame,  (cx, cy - 20), (cx, cy + 20), (255, 255, 255), 1)
    cv2.circle(frame, (cx, cy), 5, (255, 255, 255), 1)

# ─── Boucle principale ───────────────────────────────────────────────────────

print("Démarrage OAK-D Lite…")
print("Appuyer sur Q pour quitter\n")

with dai.Device(pipeline) as dispositif:

    cola_prof  = dispositif.getOutputQueue(name="profondeur", maxSize=1, blocking=False)
    cola_color = dispositif.getOutputQueue(name="couleur",    maxSize=1, blocking=False)

    dernier_frame_color = None
    texte_dist = "En attente…"
    texte_info = ""
    couleur_texte = (200, 200, 200)

    while True:

        # ── Lecture caméra couleur ──────────────────────────────────────────
        paq_color = cola_color.tryGet()
        if paq_color is not None:
            dernier_frame_color = paq_color.getCvFrame()

        # ── Lecture profondeur ──────────────────────────────────────────────
        paq_prof = cola_prof.tryGet()

        if paq_prof is not None:

            frame_prof = paq_prof.getFrame()    # uint16, en mm
            h, w = frame_prof.shape
            cx, cy = w // 2, h // 2

            # ROI centrale
            demi = ROI_TAM // 2
            x1 = max(0, cx - demi)
            x2 = min(w, cx + demi + 1)
            y1 = max(0, cy - demi)
            y2 = min(h, cy + demi + 1)

            zone = frame_prof[y1:y2, x1:x2]
            pixels_valides = zone[zone > 0]

            if pixels_valides.size > 0:
                # Trier du plus proche au plus loin
                pixels_tries = np.sort(pixels_valides)

                # Garder les N% les plus proches
                n_proches = max(1, int(len(pixels_tries) * PORCENTAJE_CERCANOS))
                pixels_proches = pixels_tries[:n_proches]

                # Mesure brute de cette frame
                dist_raw_mm = int(np.median(pixels_proches))

                # Stabilisation temporelle
                historial.append(dist_raw_mm)
                dist_mm = int(np.median(historial))
                dist_cm = dist_mm / 10.0

                # Stats utiles
                n_valides = int(pixels_valides.size)
                std_mm = float(np.std(pixels_valides))
                min_mm = int(np.min(pixels_valides))
                max_mm = int(np.max(pixels_valides))

                texte_dist = f"Distance : {dist_mm} mm ({dist_cm:.1f} cm)"
                texte_info = f"raw:{dist_raw_mm}  val:{n_valides}  std:{std_mm:.1f}  min:{min_mm}  max:{max_mm}"
                couleur_texte = (0, 255, 80)
            else:
                texte_dist = "Aucune mesure valide"
                texte_info = "ROI sans pixels depth valides"
                couleur_texte = (0, 80, 255)

            # ── Visualisation profondeur ────────────────────────────────────
            vis_prof = cv2.normalize(frame_prof, None, 0, 255,
                                     cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            vis_prof = cv2.applyColorMap(vis_prof, cv2.COLORMAP_INFERNO)

            dessiner_viseur(vis_prof, cx, cy)
            cv2.rectangle(vis_prof, (x1, y1), (x2 - 1, y2 - 1), (255, 255, 255), 1)
            dessiner_bandeau(vis_prof, texte_dist, texte_info, couleur_texte)

            cv2.imshow("OAK-D Lite — Profondeur", vis_prof)

            print(f"\r{texte_dist} | {texte_info}            ", end="", flush=True)

        # ── Affichage caméra couleur ────────────────────────────────────────
        if dernier_frame_color is not None:
            vis_color = dernier_frame_color.copy()
            hc, wc = vis_color.shape[:2]
            cxc, cyc = wc // 2, hc // 2

            dessiner_viseur(vis_color, cxc, cyc)
            dessiner_bandeau(vis_color, texte_dist, texte_info, couleur_texte)

            cv2.imshow("OAK-D Lite — Caméra", vis_color)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cv2.destroyAllWindows()
print("\nFermé.")