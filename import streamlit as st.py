import streamlit as st

st.title("Ma Première App Streamlit 🚀")
st.write("Bonjour, voici une application simple avec Streamlit!")

# Slider interactif
age = st.slider("Quel est votre âge ?", 0, 100, 25)
st.write(f"Vous avez {age} ans.")

# Bouton
if st.button("Cliquez-moi"):
    st.success("Bravo, vous avez cliqué !")