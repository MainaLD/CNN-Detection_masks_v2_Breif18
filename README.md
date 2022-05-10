# 18_Model_IA_Detection_Masks_2
Application avec streamlit et open CV de Transfer learning pour la détection des masques

**************************************************************************************************************
## Contexte du projet
Nous cherchons à améliorer l’application, développée au projet [14_Modele_detection_masques](https://github.com/MainaLD/14_Modele_intelligent_detection_masques).</br>
Développer une application Streamlit capable à détecter/localiser le ou les visages dans une image, puis la présence ou l’absence du masque pour chaque visage détecté.

**************************************************************************************************************
## Cahier de charge de l’application :
- Charger une image
- Lancer la webcam (facultatif)
- Détection du masque.
- Comptage (personne avec masque et personne sans masque).
- Un historique sous forme un tableau (personne, date/heure de détection et statut).

**************************************************************************************************************
# Livrables :
- Ce readme décrivant le projet
- Application streamlit : app.py
- détection de masques : detect_mask.py
- les documents annexes au bon fonctionnements de l'application streamlit :
    - modèle IA : [model_mask.h5](https://drive.google.com/file/d/1-4OFNg_QGEQLkae8Xg1roJzZnRm1SuJw/view?usp=sharing)
    - image par défaut : 004.jpg
    - base de données de reconnaissance des visage openCV : cascades\haarcascade_frontalface_default.xml
    - l'environnement : requirements.txt

**************************************************************************************************************
## Application streamlit : "Détection des masques "
lien : [site](https://share.streamlit.io/mainald/18_model_ia_detection_masks_2/main/app.py)

Dans l'application je propose de détecter les visages sur les images, puis de détecter si la personne porte une masque ou non.
Je donne le résultats obtenus et la possibilité d'importer ce résultat. 

### Détection par import d'une image, avec image type pré-importé
![1](images\Capture01.JPG)
![2](images\Capture02.JPG)
### Détection par prise d'une image avec la webcam
![3](images\Capture03.JPG)
![4](images\Capture04.JPG)


