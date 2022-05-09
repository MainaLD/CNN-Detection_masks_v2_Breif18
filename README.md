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
