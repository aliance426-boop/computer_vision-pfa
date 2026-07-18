<<<<<<< HEAD
# Application de détection et segmentation de défauts

Prototype d'application développé dans le cadre du Projet de Fin d'Année (PFA)
en Ingénorat 1 Génie Informatique à l'École Polytechnique de Ouagadougou.

## Fonctionnalités

- Chargement d'une ou plusieurs images de bâtiments (glisser-déposer).
- Trois modes d'analyse : détection multi-classes, segmentation des fissures, ou les deux.
- Réglage du seuil de confiance en direct.
- Affichage côte-à-côte de l'image d'origine et du résultat annoté.
- Tableau récapitulatif : nombre de défauts par classe, confiance moyenne et maximale.
- Téléchargement des images annotées.
- Bouton de démonstration avec image pré-chargée (utile en soutenance).

## Installation

### 1. Cloner ou copier le dossier

```bash
cd streamlit_app/
```

### 2. Créer un environnement virtuel Python (recommandé)


```bash
# LANCER UNE SEULE FOIS
python3 -m venv .venv

# ACTIVER L'ENVIRONNEMENT VIRTUEL
source .venv/bin/activate      # Linux / macOS
# ou
.venv\Scripts\activate         # Windows
```

### 3. Installer les dépendances

```bash
pip install -r requirements.txt
```

### 4. Placer les modèles entraînés

Créez un sous-dossier `models/` et déposez-y les deux fichiers `.pt`
issus des entraînements Ultralytics :

```
streamlit_app/
├── app.py
├── requirements.txt
├── README.md
├── models/
│   ├── detection.pt        ← best.pt du meilleur run de détection (YOLO26)
│   └── segmentation.pt     ← best.pt du meilleur run de segmentation (YOLO26-seg)
└── assets/
    └── demo.jpg            ← image de démonstration (facultatif)
```

Récupérez les fichiers `best.pt` depuis le Drive du projet et renommez-les
`detection.pt` et `segmentation.pt` avant de les placer dans `models/`.

### 5. Lancer l'application

```bash
streamlit run app.py
```

L'interface s'ouvre automatiquement dans le navigateur à l'adresse
`http://localhost:8501`.

## Déploiement en ligne

Pour déployer sur **Streamlit Community Cloud** (gratuit) :

1. Créez un dépôt GitHub public contenant `app.py`, `requirements.txt` et `models/`.
2. Attention à la taille : les fichiers `.pt` de plusieurs Mo sont acceptés, mais
   au-delà de 100 Mo par fichier il faut utiliser Git LFS.
3. Rendez-vous sur https://share.streamlit.io et connectez le dépôt.
4. L'application est accessible à une URL publique en quelques minutes.

## Structure du code (`app.py`)

Le fichier est organisé en cinq sections :

1. **Configuration** : chemins des modèles, libellés en français.
2. **Chargement des modèles** : fonction `charger_modele` avec mise en cache
   par Streamlit pour éviter les rechargements coûteux.
3. **Fonctions de traitement** : `executer_detection`, `executer_segmentation`,
   `image_vers_octets`, `afficher_tableau_recapitulatif`.
4. **Interface utilisateur** : titre, panneau latéral, chargeur de fichiers,
   affichage des résultats.
5. **Pied de page**.

## Auteurs

- ALI Hamidou
- SEREME Djamilou Papy
- ZALLE Théophile P.

## Encadrants

- SANOGO Kassoum (directeur)
- SANOGO Satafa (co-directeur)

## Licence

Projet académique — usage réservé au cadre du PFA 2026.
=======
# computer_vision-pfa
Système d'analyse de façades de bâtiments par deep learning basé sur YOLO26. Détecte 5 types de défauts (corrosion, fissures, délamination, moisissures, défauts de peinture) et effectue la segmentation d'instance spécifiquement pour les fissures. Inclut un pipeline d'augmentation avec Albumentations et une interface Streamlit.
>>>>>>> 31aa710d66858a5a28eef96573f3999998553dbd
