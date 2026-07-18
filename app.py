"""
Application Streamlit — Détection et segmentation de défauts sur bâtiments.

Projet de fin d'année (PFA) — Ingénorat 1, Génie Informatique, EPO.
Auteurs      : ALI Hamidou, SEREME Djamilou Papy, ZALLE Théophile P.
Encadrants   : SANOGO Kassoum (directeur), SANOGO Satafa (co-directeur).

Utilisation :
    streamlit run app.py

Structure du dossier attendue :
    app.py
    models/
        detection.pt       (modèle YOLO11 détection multi-classes)
        segmentation.pt    (modèle YOLO11-seg segmentation fissures)
    assets/
        demo.jpg           (image de démonstration, facultatif)
"""

import io
from pathlib import Path
from typing import Optional

import numpy as np
import streamlit as st
from PIL import Image
from ultralytics import YOLO


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Détection de défauts — PFA EPO",
    page_icon="🏗️",
    layout="wide",
)

# Chemins vers les modèles entraînés
MODELE_DETECTION = Path("models/detection.pt")
MODELE_SEGMENTATION = Path("models/segmentation.pt")

# Chemin vers l'image de démonstration (facultatif)
IMAGE_DEMO = Path("assets/demo.jpg")

# Correspondance identifiant → libellé lisible pour le tableau récapitulatif
NOMS_CLASSES_FR = {
    "corrosion": "Corrosion",
    "crack": "Fissure",
    "delamination": "Délamination",
    "dirt_mold": "Salissure / moisissure",
    "paint_defect": "Défaut de peinture",
}


# ---------------------------------------------------------------------------
# Chargement des modèles
# ---------------------------------------------------------------------------

@st.cache_resource
def charger_modele(chemin: Path) -> Optional[YOLO]:
    """Charge un modèle YOLO depuis le disque.
    
    Le décorateur @st.cache_resource garantit que le modèle n'est chargé
    qu'une seule fois par session, ce qui accélère considérablement les
    analyses successives.
    
    Retourne None si le fichier est absent, pour que l'appelant puisse
    afficher un message d'erreur clair.
    """
    if not chemin.exists():
        return None
    return YOLO(str(chemin))


# ---------------------------------------------------------------------------
# Fonctions de traitement
# ---------------------------------------------------------------------------

def executer_detection(modele: YOLO, image: Image.Image, seuil: float):
    """Exécute la détection multi-classes sur une image.
    
    Retourne un tuple (image annotée, dictionnaire de comptage par classe).
    Le dictionnaire est vide si aucun défaut n'est détecté au-dessus du seuil.
    """
    resultats = modele.predict(image, conf=seuil, verbose=False)
    resultat = resultats[0]

    # L'image annotée renvoyée par plot() est au format BGR (OpenCV) ;
    # on la convertit en RGB pour Streamlit.
    image_annotee = Image.fromarray(resultat.plot()[..., ::-1])

    # Comptage des défauts détectés par classe
    comptage = {}
    if resultat.boxes is not None and len(resultat.boxes) > 0:
        classes = resultat.boxes.cls.cpu().numpy()
        confiances = resultat.boxes.conf.cpu().numpy()
        for classe_id, confiance in zip(classes, confiances):
            nom_classe = modele.names[int(classe_id)]
            if nom_classe not in comptage:
                comptage[nom_classe] = {"nombre": 0, "confiances": []}
            comptage[nom_classe]["nombre"] += 1
            comptage[nom_classe]["confiances"].append(float(confiance))

    return image_annotee, comptage


def executer_segmentation(modele: YOLO, image: Image.Image, seuil: float) -> Image.Image:
    """Exécute la segmentation des fissures sur une image.
    
    Retourne l'image annotée avec les masques superposés.
    """
    resultats = modele.predict(image, conf=seuil, verbose=False)
    resultat = resultats[0]
    return Image.fromarray(resultat.plot()[..., ::-1])


def image_vers_octets(image: Image.Image) -> bytes:
    """Convertit une image PIL en flux d'octets JPEG pour téléchargement."""
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=90)
    return buffer.getvalue()


def afficher_tableau_recapitulatif(comptage: dict) -> None:
    """Affiche un tableau récapitulatif des défauts détectés par classe."""
    if not comptage:
        st.info("Aucun défaut détecté au-dessus du seuil de confiance choisi.")
        return

    donnees = []
    for nom_classe, infos in sorted(comptage.items()):
        donnees.append({
            "Type de défaut": NOMS_CLASSES_FR.get(nom_classe, nom_classe),
            "Nombre détecté": infos["nombre"],
            "Confiance moyenne": f"{np.mean(infos['confiances']):.1%}",
            "Confiance maximale": f"{np.max(infos['confiances']):.1%}",
        })
    st.dataframe(donnees, use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# Interface utilisateur
# ---------------------------------------------------------------------------

st.title("🏗️ Détection et segmentation de défauts sur bâtiments")
st.markdown(
    "**Projet de fin d'année — Ingénorat 1, Génie Informatique, EPO (2026)**  \n"
    "Auteurs : ALI Hamidou · SEREME Djamilou Papy · ZALLE Théophile P."
)

st.divider()

# --- Panneau latéral : paramètres d'analyse ---
with st.sidebar:
    st.header("Paramètres d'analyse")

    tache = st.selectbox(
        "Analyse à effectuer",
        [
            "Détection multi-classes",
            "Segmentation des fissures",
            "Détection + segmentation",
        ],
        index=0,
    )

    seuil_confiance = st.slider(
        "Seuil de confiance",
        min_value=0.10,
        max_value=0.90,
        value=0.25,
        step=0.05,
        help=(
            "Score minimum pour qu'une prédiction soit conservée. "
            "Un seuil plus élevé réduit les faux positifs mais peut faire "
            "passer des défauts réels sous silence."
        ),
    )

    st.markdown("---")
    st.markdown(
        "**Modèles utilisés**  \n"
        "- Détection : YOLO11 (5 classes)  \n"
        "- Segmentation : YOLO11-seg (fissures)"
    )

# --- Chargement conditionnel des modèles ---
besoin_detection = "Détection" in tache
besoin_segmentation = "segmentation" in tache.lower()

modele_det = charger_modele(MODELE_DETECTION) if besoin_detection else None
modele_seg = charger_modele(MODELE_SEGMENTATION) if besoin_segmentation else None

erreur_modele = False
if besoin_detection and modele_det is None:
    st.error(
        f"❌ Modèle de détection introuvable à l'emplacement `{MODELE_DETECTION}`. "
        "Placez le fichier `best.pt` du meilleur run de détection dans le dossier `models/` "
        "et renommez-le `detection.pt`."
    )
    erreur_modele = True
if besoin_segmentation and modele_seg is None:
    st.error(
        f"❌ Modèle de segmentation introuvable à l'emplacement `{MODELE_SEGMENTATION}`. "
        "Placez le fichier `best.pt` du meilleur run de segmentation dans le dossier `models/` "
        "et renommez-le `segmentation.pt`."
    )
    erreur_modele = True

# --- Chargement des images ---
fichiers = st.file_uploader(
    "Chargez une ou plusieurs images de bâtiments à analyser",
    type=["jpg", "jpeg", "png", "bmp", "webp"],
    accept_multiple_files=True,
)

if "demo_actif" not in st.session_state:
    st.session_state.demo_actif = False

# Bouton de démonstration (utile en soutenance si aucune image n'est fournie)
if not fichiers and IMAGE_DEMO.exists() and not erreur_modele:
    if st.button("📸 Utiliser une image de démonstration"):
        st.session_state.demo_actif = True

# Construction de la liste des images à traiter
images_a_traiter = []
if fichiers:
    # L'utilisateur a chargé ses propres images : on désactive la démo
    st.session_state.demo_actif = False
    for f in fichiers:
        try:
            img = Image.open(f).convert("RGB")
            images_a_traiter.append((f.name, img))
        except Exception as e:
            st.error(f"Impossible de lire {f.name} : {e}")
elif st.session_state.demo_actif and IMAGE_DEMO.exists():
    images_a_traiter.append((IMAGE_DEMO.name, Image.open(IMAGE_DEMO).convert("RGB")))

# --- Traitement et affichage des résultats ---
if images_a_traiter and not erreur_modele:
    st.divider()

    for idx, (nom_fichier, image) in enumerate(images_a_traiter):
        st.subheader(f"📷 {nom_fichier}")

        colonne_originale, colonne_resultat = st.columns(2)

        with colonne_originale:
            st.markdown("**Image d'origine**")
            st.image(image, use_container_width=True)

        with colonne_resultat:
            st.markdown("**Résultat de l'analyse**")
            with st.spinner("Analyse en cours…"):
                comptage = None

                if tache == "Détection multi-classes":
                    image_annotee, comptage = executer_detection(
                        modele_det, image, seuil_confiance
                    )
                    st.image(image_annotee, use_container_width=True)
                    st.download_button(
                        "⬇️ Télécharger l'image annotée",
                        data=image_vers_octets(image_annotee),
                        file_name=f"detection_{nom_fichier}",
                        mime="image/jpeg",
                        key=f"dl_det_{idx}",
                    )

                elif tache == "Segmentation des fissures":
                    image_annotee = executer_segmentation(
                        modele_seg, image, seuil_confiance
                    )
                    st.image(image_annotee, use_container_width=True)
                    st.download_button(
                        "⬇️ Télécharger l'image annotée",
                        data=image_vers_octets(image_annotee),
                        file_name=f"segmentation_{nom_fichier}",
                        mime="image/jpeg",
                        key=f"dl_seg_{idx}",
                    )

                else:  # Détection + segmentation
                    image_det, comptage = executer_detection(
                        modele_det, image, seuil_confiance
                    )
                    image_seg = executer_segmentation(
                        modele_seg, image, seuil_confiance
                    )
                    st.markdown("*Détection multi-classes*")
                    st.image(image_det, use_container_width=True)
                    st.download_button(
                        "⬇️ Image annotée (détection)",
                        data=image_vers_octets(image_det),
                        file_name=f"detection_{nom_fichier}",
                        mime="image/jpeg",
                        key=f"dl_det_combo_{idx}",
                    )
                    st.markdown("*Segmentation des fissures*")
                    st.image(image_seg, use_container_width=True)
                    st.download_button(
                        "⬇️ Image annotée (segmentation)",
                        data=image_vers_octets(image_seg),
                        file_name=f"segmentation_{nom_fichier}",
                        mime="image/jpeg",
                        key=f"dl_seg_combo_{idx}",
                    )

        # Tableau récapitulatif (uniquement si la détection a été exécutée)
        if comptage is not None:
            st.markdown("**Récapitulatif des défauts détectés**")
            afficher_tableau_recapitulatif(comptage)

        st.divider()

elif not erreur_modele:
    st.info(
        "💡 Chargez une ou plusieurs images pour lancer l'analyse, "
        "ou utilisez le bouton de démonstration si disponible."
    )

# --- Pied de page ---
st.markdown(
    "<div style='text-align:center; color:gray; padding-top:2em; font-size:0.85em;'>"
    "PFA 2026 · École Polytechnique de Ouagadougou · Ingénorat 1 Génie Informatique"
    "</div>",
    unsafe_allow_html=True,
)
