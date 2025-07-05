import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from PIL import Image
import os

st.set_page_config(page_title="Prédiction de style artistique", layout="wide")

dirs = os.listdir( path )

# Print all the files and directories
for file in dirs:
   print(file)

# === 1. Chargement du modèle ===

def load_model():
    return tf.keras.models.load_model("MNV2_cafonctionne.keras")
    #return tf.keras.models.load_model("MobileNetV2_Final_model.keras")

model = load_model()

model.summary()


st.title("Art and facts Virtual Visit")

# === 2. Noms des classes et contexte associé ===
class_context = {
    'Abstract_Expressionism': {
        'période': '1940–1960',
        'description': "Mouvement américain centré sur l'expression libre, gestuelle et émotionnelle.",
        'artistes': ['Jackson Pollock', 'Willem de Kooning']
    },
    'Art_Nouveau_Modern': {
        'période': '1890–1910',
        'description': "Style décoratif fluide inspiré de la nature, très ornemental.",
        'artistes': ['Alphonse Mucha', 'Gustav Klimt']
    },
    'Baroque': {
        'période': '1600–1750',
        'description': "Art théâtral, riche, dynamique, souvent religieux.",
        'artistes': ['Caravage', 'Rubens']
    },
    'Color_Field_Painting': {
        'période': '1950–1970',
        'description': "Abstraction américaine par larges aplats de couleurs vibrantes.",
        'artistes': ['Mark Rothko', 'Barnett Newman']
    },
    'Cubism': {
        'période': '1907–1914',
        'description': "Fragmentation des formes, multi-perspectives.",
        'artistes': ['Pablo Picasso', 'Georges Braque']
    },
    'Early_Renaissance': {
        'période': '1400–1490',
        'description': "Redécouverte de l’antiquité, équilibre entre nature et perspective.",
        'artistes': ['Masaccio', 'Fra Angelico']
    },
    'Expressionism': {
        'période': '1905–1930',
        'description': "Déformation expressive de la réalité au service des émotions.",
        'artistes': ['Edvard Munch', 'Egon Schiele']
    },
    'Fauvism': {
        'période': '1905–1910',
        'description': "Couleurs vives et non naturalistes, spontanéité.",
        'artistes': ['Henri Matisse', 'André Derain']
    },
    'High_Renaissance': {
        'période': '1490–1527',
        'description': "Apogée de l’harmonie classique et de la virtuosité technique.",
        'artistes': ['Léonard de Vinci', 'Raphaël', 'Michel-Ange']
    },
    'Impressionism': {
        'période': '1860–1890',
        'description': "Jeu de lumière, instantanéité, paysages en plein air.",
        'artistes': ['Claude Monet', 'Pierre-Auguste Renoir']
    },
    'Mannerism_Late_Renaissance': {
        'période': '1520–1600',
        'description': "Figures allongées, compositions complexes, art maniériste.",
        'artistes': ['Le Parmesan', 'Le Greco']
    },
    'Minimalism': {
        'période': '1960–1980',
        'description': "Formes simples, réduction maximale, neutralité expressive.",
        'artistes': ['Donald Judd', 'Frank Stella']
    },
    'Naive_Art_Primitivism': {
        'période': 'XIXe–XXe siècle',
        'description': "Simplicité volontaire, absence de perspective académique.",
        'artistes': ['Henri Rousseau', 'Grandma Moses']
    },
    'Northern_Renaissance': {
        'période': '1430–1580',
        'description': "Précision du détail, symbolisme, techniques à l’huile.",
        'artistes': ['Jan van Eyck', 'Hieronymus Bosch']
    },
    'Pop_Art': {
        'période': '1955–1975',
        'description': "Culture populaire, publicité, couleurs vives.",
        'artistes': ['Andy Warhol', 'Roy Lichtenstein']
    },
    'Post_Impressionism': {
        'période': '1885–1910',
        'description': "Exploration personnelle du réel, au-delà de la lumière.",
        'artistes': ['Van Gogh', 'Paul Cézanne']
    },
    'Realism': {
        'période': '1840–1870',
        'description': "Représentation fidèle du quotidien, sans idéalisme.",
        'artistes': ['Gustave Courbet', 'Jean-François Millet']
    },
    'Romanticism': {
        'période': '1800–1850',
        'description': "Émotion, sublime, nature sauvage, révolte.",
        'artistes': ['Eugène Delacroix', 'Caspar David Friedrich']
    },
    'Rococo': {
        'période': '1730–1780',
        'description': "Art décoratif léger, frivole, galant.",
        'artistes': ['Fragonard', 'Boucher']
    },
    'Symbolism': {
        'période': '1880–1910',
        'description': "Imagerie onirique, mythologique ou spirituelle.",
        'artistes': ['Odilon Redon', 'Gustave Moreau']
    },
    'Ukiyo_e': {
        'période': '1600–1900',
        'description': "Gravures japonaises sur bois illustrant la vie urbaine ou les paysages.",
        'artistes': ['Hokusai', 'Hiroshige']
    }
}

# === 3. Prédiction ===
def predict_style(img_pil):
    img_resized = img_pil.resize((96, 96))
    #img_array = image.img_to_array(img_resized) / 127.5 - 1.0
    img_array = tf.expand_dims(img_resized, axis=0)
    predictions = tf.nn.softmax(model.predict(img_array)[0]).numpy()
    top_indices = predictions.argsort()[::-1][:5]
    return [(list(class_context.keys())[i], predictions[i]) for i in top_indices]

# === 4. Interface Streamlit ===

st.title("🎨 Prédiction du style artistique d’une peinture")

uploaded_file = st.file_uploader("📷 Téléchargez une image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Image chargée", use_container_width=True)

    with st.spinner("🔍 Analyse en cours..."):
        results = predict_style(img)

    top1 = results[0][0]
    score1 = results[0][1]

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("🎯 Résultat Top 1")
        st.markdown(f"**{top1}** — `{100 * score1:.4f}`%")

        st.subheader("🔝 Top 5")
        for i, (style, score) in enumerate(results, start=1):
            st.write(f"**Top {i}: {style}**")#— score : `{score:.4f}`" 
                    

    with col2:
        st.subheader("🖼️ Contexte artistique")
        contexte = class_context[top1]
        st.markdown(f"**Période :** {contexte['période']}")
        st.markdown(f"**Description :** {contexte['description']}")
        st.markdown(f"**Artistes célèbres :** {', '.join(contexte['artistes'])}")
