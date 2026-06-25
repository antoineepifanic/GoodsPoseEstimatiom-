"""
OAK-D Lite — Suivi d'orientation et dimensions physiques (Sans Profondeur)
- Flux RGB à haute vitesse (30 FPS)
- Filtrage spatial : Crop réduit (-10%) sur la zone du convoyeur
- Segmentation : Filtre de Canny avec curseurs dynamiques
- Sélection : Heuristique du centre de gravité
- Visuel : Tracé du rectangle orienté et conversion des pixels en centimètres
Quitter : touche Q
"""

import cv2
import depthai as dai
import numpy as np

# ─── Paramètres physiques (Zone du convoyeur et Étalonnage) ─────────────────
# Marge de sécurité de 10% sur la zone utile
MARGE_X = int((490 - 150) * 0.05) 
MARGE_Y = int((350 - 50) * 0.05)

X_MIN = (150 + MARGE_X) + 51 # Ajout de 15% de réduction à gauche
X_MAX = 490 - MARGE_X

Y_MIN, Y_MAX = 50 + MARGE_Y,  350 - MARGE_Y
CENTRE_CROP_X = (X_MAX - X_MIN) // 2
CENTRE_CROP_Y = (Y_MAX - Y_MIN) // 2

AREA_MIN_COLIS = 5000  # Surface minimale (en pixels) pour ignorer le bruit

# Nouveau ratio de calibration (Feuille A4 à la nouvelle distance)
# 21 cm correspond désormais à environ 114 pixels
RATIO_CM_PX = 21.0 / 114.0 

# ─── 1. Création du Pipeline (Léger, uniquement RGB) ────────────────────────
pipeline = dai.Pipeline()

color_cam = pipeline.create(dai.node.ColorCamera)
xout_rgb  = pipeline.create(dai.node.XLinkOut)

xout_rgb.setStreamName("couleur")

# Configuration de la caméra couleur
color_cam.setPreviewSize(640, 400)
color_cam.setInterleaved(False)
color_cam.setFps(30)  # Pleine fluidité

# Liaison
color_cam.preview.link(xout_rgb.input)

# ─── 2. Interface OpenCV (Curseurs de réglage) ──────────────────────────────
def nothing(x):
    pass

cv2.namedWindow("Masque Canny (Crop)")
cv2.createTrackbar("Seuil Min", "Masque Canny (Crop)", 50, 255, nothing)
cv2.createTrackbar("Seuil Max", "Masque Canny (Crop)", 150, 255, nothing)

# ─── 3. Boucle principale ───────────────────────────────────────────────────
print("Démarrage OAK-D Lite (Suivi et Dimensions 2D en cm)...")
print("Ajuste les curseurs pour isoler le colis. Appuie sur Q pour quitter.\n")

with dai.Device(pipeline) as dispositif:

    cola_color = dispositif.getOutputQueue(name="couleur", maxSize=4, blocking=False)

    while True:
        paq_color = cola_color.tryGet()

        if paq_color is not None:
            vis_color = paq_color.getCvFrame()

            # ─── A. Découpage sur la zone d'intérêt (Crop restreint) ────────
            vis_color_crop = vis_color[Y_MIN:Y_MAX, X_MIN:X_MAX]
            
            # ─── B. Traitement d'image ──────────────────────────────────────
            gray = cv2.cvtColor(vis_color_crop, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (7, 7), 0)
            
            seuil_min = cv2.getTrackbarPos("Seuil Min", "Masque Canny (Crop)")
            seuil_max = cv2.getTrackbarPos("Seuil Max", "Masque Canny (Crop)")
            
            edges = cv2.Canny(blurred, seuil_min, seuil_max)
            kernel = np.ones((5,5), np.uint8)
            edges = cv2.dilate(edges, kernel, iterations=1)

            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            meilleur_contour = None
            min_dist = float('inf')
            
            # ─── C. Trouver le colis au centre (Heuristique) ────────────────
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
            
            # ─── D. Dessin du cadre orienté et affichage des dimensions ─────
            if meilleur_contour is not None:
                # minAreaRect renvoie : (centre(x,y), (largeur, hauteur), angle)
                rect = cv2.minAreaRect(meilleur_contour)
                largeur_px, hauteur_px = rect[1]
                
                # Conversion en centimètres avec le nouveau ratio
                largeur_cm = largeur_px * RATIO_CM_PX
                hauteur_cm = hauteur_px * RATIO_CM_PX
                
                box = cv2.boxPoints(rect)
                box = np.int32(box)
                
                # Translation des points vers le repère Global
                box_global = box + [X_MIN, Y_MIN]
                
                # Tracé du polygone orienté
                cv2.drawContours(vis_color, [box_global], 0, (0, 255, 80), 2)
                
                # Tracé du centre de gravité
                M = cv2.moments(meilleur_contour)
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                cv2.circle(vis_color, (cx + X_MIN, cy + Y_MIN), 4, (0, 0, 255), -1)
                
                # Statut du verrouillage
                cv2.putText(vis_color, "Colis verrouille", (X_MIN, Y_MIN - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 80), 2)
                            
                # Affichage des dimensions réelles en haut à gauche de l'écran
                dim_texte = f"Dimensions: {largeur_cm:.1f} x {hauteur_cm:.1f} cm"
                cv2.rectangle(vis_color, (10, 10), (350, 50), (0, 0, 0), -1) 
                cv2.putText(vis_color, dim_texte, (20, 35), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            else:
                cv2.putText(vis_color, "En attente...", (X_MIN, Y_MIN - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 100, 255), 2)
                cv2.rectangle(vis_color, (10, 10), (350, 50), (0, 0, 0), -1)
                cv2.putText(vis_color, "Dimensions: -- x -- cm", (20, 35), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # ─── E. Visualisation de la zone de sécurité ────────────────────
            cv2.rectangle(vis_color, (X_MIN, Y_MIN), (X_MAX, Y_MAX), (0, 0, 255), 1)

            # ─── Affichage final ────────────────────────────────────────────
            cv2.imshow("OAK-D Lite — Flux Principal", vis_color)
            cv2.imshow("Masque Canny (Crop)", edges)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cv2.destroyAllWindows()
print("\nProgramme terminé.")