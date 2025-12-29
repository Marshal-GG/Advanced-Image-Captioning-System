import gradio as gr
import numpy as np
import pickle
import tensorflow as tf
from keras.preprocessing.image import load_img, img_to_array
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.models import Model
from keras.preprocessing.sequence import pad_sequences

# 1. Load Resources
model = tf.keras.models.load_model('data/processed/trained_model.keras')
with open('data/vocab.pkl', 'rb') as f:
    ixtoword = pickle.load(f)
with open('data/wordtoix.pkl', 'rb') as f:
    wordtoix = pickle.load(f)

# 2. Setup Feature Extractor (InceptionV3)
base_model = InceptionV3(weights='imagenet')
feature_extractor = Model(base_model.input, base_model.layers[-2].output)

def extract_features(image):
    # Resize and preprocess image for InceptionV3
    image = image.resize((299, 299))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)
    feature = feature_extractor.predict(image)
    return np.reshape(feature, (feature.shape[1]))

def generate_caption(image):
    # Get image features
    photo = extract_features(image)
    
    # Generate caption
    in_text = 'startseq'
    max_length = 34  # Match your training max_length
    
    for i in range(max_length):
        sequence = [wordtoix[w] for w in in_text.split() if w in wordtoix]
        sequence = pad_sequences([sequence], maxlen=max_length)[0]
        sequence = np.expand_dims(sequence, axis=0)
        
        # Predict next word
        yhat = model.predict([np.array([photo]), sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = ixtoword.get(yhat)
        
        if word is None:
            break
            
        in_text += ' ' + word
        if word == 'endseq':
            break
            
    final_caption = in_text.replace('startseq', '').replace('endseq', '')
    return final_caption.strip()

# 3. Launch Interface
interface = gr.Interface(
    fn=generate_caption,
    inputs=gr.Image(type="pil"),
    outputs="text",
    title="Image Caption Generator",
    description="Upload an image and the AI will describe it."
)

interface.launch()