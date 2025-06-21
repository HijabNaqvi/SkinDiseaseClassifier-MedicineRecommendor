from flask import Flask, render_template, request
import tensorflow as tf
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
import pandas as pd
import os

app = Flask(__name__)
model = load_model("my_model.h5")  

# Define class names (same order as model training)
class_names = ['Eczema1677', 'Warts Molluscum and other Viral Infections -', 'Melanoma15.75k',
               'Atopic Dermatitis-1.25k', 'BasalCellCarcinoma(BCC)3323', 'MelanocyticNevi(NV)-7970',
               'BenignKeratosis-', 'Psoriasis pictures Lichen Planus and related diseases-2k',
               'Seborrheic Keratoses and other Benign Tumors - 1.8k',
               'Tinea Ringworm Candidiasis and other Fungal Infections - 1.7k']

# Mapping to clean names
mapping = {
    'Eczema1677': 'Eczema',
    'Warts Molluscum and other Viral Infections -': 'Warts Molluscum',
    'Melanoma15.75k': 'Melanoma',
    'Atopic Dermatitis-1.25k': 'Atopic Dermatitis',
    'BasalCellCarcinoma(BCC)3323': 'BasalCellCarcinoma',
    'MelanocyticNevi(NV)-7970': 'MelanocyticNevi(NV)',
    'BenignKeratosis-': 'BenignKeratosis',
    'Psoriasis pictures Lichen Planus and related diseases-2k': 'Psoriasis',
    'Seborrheic Keratoses and other Benign Tumors - 1.8k': 'Seborrheic Keratoses',
    'Tinea Ringworm Candidiasis and other Fungal Infections - 1.7k': 'Tinea Ringworm Candidiasis'
}

# OTC medicine data
otc_data = {
    'Disease': [
        'Eczema', 'Warts Molluscum', 'Melanoma', 'Atopic Dermatitis',
        'BasalCellCarcinoma', 'MelanocyticNevi(NV)', 'BenignKeratosis',
        'Psoriasis', 'Seborrheic Keratoses', 'Tinea Ringworm Candidiasis'
    ],
    'OTC_Medicine': [
        'Hydrocortisone cream, moisturizers, anti-itch lotions',
        'No FDA-approved OTC treatment — consult doctor',
        'No OTC — requires medical treatment',
        'Hydrocortisone cream, moisturizers, anti-itch lotions',
        'No OTC — requires medical treatment',
        'No OTC — monitor and consult doctor if changed',
        'Moisturizers, salicylic acid creams',
        'Coal tar, salicylic acid, moisturizers',
        'No OTC — dermatologist removal if necessary',
        'Clotrimazole, Miconazole (antifungal creams)'
    ],
    'Notes': [
        'Helps reduce itching and inflammation.',
        'Usually resolves over time, but dermatologist advice is best.',
        'Melanoma is dangerous and must be seen by a professional.',
        'Same as eczema — moisturize and reduce itching.',
        'Requires biopsy and clinical intervention.',
        'Benign but monitor for size/color change.',
        'Softens and sheds keratotic lesions.',
        'Helpful for scaling and itching. Maintain moisturization.',
        'Benign; usually cosmetic concern.',
        'Keep area dry, use antifungal creams consistently.'
    ]
}
medicine_df = pd.DataFrame(otc_data)

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    img_path = None
    if request.method == 'POST':
        file = request.files.get('image')
        action = request.form.get('action')
        if file:
            img_path = os.path.join("static", file.filename)
            file.save(img_path)

        if action == 'predict' and img_path:
            # Preprocess image
            img = load_img(img_path, target_size=(256, 256))
            img = img_to_array(img)
            img = np.expand_dims(img, axis=0)
            img = tf.keras.applications.efficientnet.preprocess_input(img)

            # Predict
            pred = model.predict(img)
            predicted_class = np.argmax(pred, axis=1)[0]
            predicted_label = class_names[predicted_class]
            clean_disease = mapping.get(predicted_label, predicted_label)

            # Get OTC info
            med_info = medicine_df[medicine_df['Disease'] == clean_disease].iloc[0]

            # Pass prediction to template
            prediction = {
                "disease": clean_disease,
                "otc": med_info["OTC_Medicine"],
                "notes": med_info["Notes"],
                "image_path": img_path
            }
        else:
            prediction = {"image_path": img_path}

    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
