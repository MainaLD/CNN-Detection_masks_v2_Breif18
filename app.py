import streamlit as st 
from PIL import Image
import cv2
from detect_mask import afficher_visage

image_defaut = "./images/004.jpg"

@st.cache
def load_image(img):
    im = Image.open(img)
    return im

@st.cache
def convert_df(df):
    return df.to_csv().encode('utf-8')

def detect_mask(image):
    visages, list_nom_img, tableau = afficher_visage(image)

    st.image(visages, use_column_width='auto', caption= list_nom_img, width=10)
    for idx,name in enumerate(tableau['Masque'].value_counts().index.tolist()):
        st.write('Nombre de visage', name, ':', tableau['Masque'].value_counts()[idx])

    st.dataframe(tableau)
    csv = convert_df(tableau)
    st.download_button("Télécharger tableau", csv, "file.csv", "text/csv", key='download-csv')


def main():
    """Face Detection App"""
    st.title("Application de détection de masques")
    st.text("Avec Streamlit et OpenCV")
    
    activities = ["Importer image","Image par webcam","Autre"]
    choice = st.sidebar.selectbox("Choix",activities)
    
    if choice == 'Importer image':
        st.subheader("Détection de masques sur 1 image")      
        image_file = st.file_uploader("Charger une image",type=['jpg','png','jpeg'])

        if image_file is not None:    
            st.image(image_file, use_column_width='auto', caption= "Image originale")
            if st.button("Détecter les masques"):
                detect_mask(image_file)
        
        else:
            img_defaut = Image.open(image_defaut)
            st.text("Image chargée par défaut")
            st.image(img_defaut, use_column_width='auto')
            if st.button("Détecter les masques"):
                detect_mask(image_defaut)


    elif choice == 'Image par webcam':
        st.subheader("Image par webcam")
        picture = st.camera_input("Prendre une capture")

        if st.button("Détecter les masques"):
            detect_mask(picture)


    elif choice == 'Autre':
        st.subheader("Autre")
		

if __name__ == '__main__':
		main()	
