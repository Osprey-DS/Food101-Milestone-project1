import numpy as np
import tensorflow as tf
import streamlit as st
from PIL import  Image

class_names = ['apple_pie',
 'baby_back_ribs',
 'baklava',
 'beef_carpaccio',
 'beef_tartare',
 'beet_salad',
 'beignets',
 'bibimbap',
 'bread_pudding',
 'breakfast_burrito',
 'bruschetta',
 'caesar_salad',
 'cannoli',
 'caprese_salad',
 'carrot_cake',
 'ceviche',
 'cheesecake',
 'cheese_plate',
 'chicken_curry',
 'chicken_quesadilla',
 'chicken_wings',
 'chocolate_cake',
 'chocolate_mousse',
 'churros',
 'clam_chowder',
 'club_sandwich',
 'crab_cakes',
 'creme_brulee',
 'croque_madame',
 'cup_cakes',
 'deviled_eggs',
 'donuts',
 'dumplings',
 'edamame',
 'eggs_benedict',
 'escargots',
 'falafel',
 'filet_mignon',
 'fish_and_chips',
 'foie_gras',
 'french_fries',
 'french_onion_soup',
 'french_toast',
 'fried_calamari',
 'fried_rice',
 'frozen_yogurt',
 'garlic_bread',
 'gnocchi',
 'greek_salad',
 'grilled_cheese_sandwich',
 'grilled_salmon',
 'guacamole',
 'gyoza',
 'hamburger',
 'hot_and_sour_soup',
 'hot_dog',
 'huevos_rancheros',
 'hummus',
 'ice_cream',
 'lasagna',
 'lobster_bisque',
 'lobster_roll_sandwich',
 'macaroni_and_cheese',
 'macarons',
 'miso_soup',
 'mussels',
 'nachos',
 'omelette',
 'onion_rings',
 'oysters',
 'pad_thai',
 'paella',
 'pancakes',
 'panna_cotta',
 'peking_duck',
 'pho',
 'pizza',
 'pork_chop',
 'poutine',
 'prime_rib',
 'pulled_pork_sandwich',
 'ramen',
 'ravioli',
 'red_velvet_cake',
 'risotto',
 'samosa',
 'sashimi',
 'scallops',
 'seaweed_salad',
 'shrimp_and_grits',
 'spaghetti_bolognese',
 'spaghetti_carbonara',
 'spring_rolls',
 'steak',
 'strawberry_shortcake',
 'sushi',
 'tacos',
 'takoyaki',
 'tiramisu',
 'tuna_tartare',
 'waffles']

# loading the modle
@st.cache(allow_output_mutation=True)
def load_model():
    model = tf.keras.models.load_model(r'effi_080_second.h5')
    return model
model = load_model()

# showing a Header
st.title('Food 101 Classifier™')

col1, col2 = st.beta_columns(2)

# Asking for file
file = col2.file_uploader("Upload an image of food", type=["png", "jpg"])

if file is not None:
    image = Image.open(file)
    img2 = image.copy()
    img2.resize((300,300))
    col1.image(img2, caption=f"Looks Delicious!! ", use_column_width=True, width=300)

    img_array = np.array(image)
    img = tf.image.resize(img_array, size=(224,224))
    img = tf.expand_dims(img, axis=0)
    pred = model.predict(img)
    pred_cls = class_names[pred.argmax()]
    col2.success("Predicted: " + pred_cls)  # showing the prediction class name

note = """ 
This project is part of the Zero to Mastery Tensorflow Developer course (MileStone Project 1) \n
This project based on the [Food101](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/) Paper which used Convolutional Neuranetwork trained for 2 to 3 days to achieve 77.4% top-1 accuracy.
The project is made by download the food101 dataset from the [TensorFlow dataset](https://www.tensorflow.org/datasets/catalog/food101)(size: 4.6GB) which consists of 750 images x 101 training classes = 75750 training images.
I used the [EfficientNetB0](https://www.tensorflow.org/api_docs/python/tf/keras/applications/EfficientNetB0) model with fine-tune unfreeze all layers of the model. \n

Although this WebApp model accuracy is around 80% to 82%. I am also sharing the [notebook](https://colab.research.google.com/drive/15sJJhrZBo12CA3flnrX-NC4WwrP84z0D?usp=sharing) for this project.
"""
st.write(note)

with st.beta_expander('Food Names(Classes), The model will work better if you chose food from this list'):
 st.write(class_names)