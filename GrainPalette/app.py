from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)
model = load_model('rice_model_mobilenet.h5')
class_names = ['Arborio', 'Basmati', 'Ipsala', 'Jasmine', 'Karacadag']

UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def predict_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]

    info = {
        'Arborio': {
            'water': 'High (flooded fields)',
            'fertilizer': 'High nitrogen (urea), phosphorus, and potassium'
        },
        'Basmati': {
            'water': 'Moderate to high',
            'fertilizer': 'Balanced NPK; prefers organic fertilizers too'
        },
        'Ipsala': {
            'water': 'High (paddy cultivation)',
            'fertilizer': 'Nitrogen-rich; urea topdressing after 3–4 weeks'
        },
        'Jasmine': {
            'water': 'Moderate (needs moist soil)',
            'fertilizer': 'Moderate NPK; compost or slow-release fertilizers recommended'
        },
        'Karacadag': {
            'water': 'Low to moderate (drought-tolerant)',
            'fertilizer': 'Minimal fertilizer, prefers organic or slow-release types'
        }
    }

    return predicted_class, info[predicted_class]['water'], info[predicted_class]['fertilizer']

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    image_url = None
    water = None
    fertilizer = None

    if request.method == 'POST':
        file = request.files['file']
        if file.filename != '':
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)
            prediction, water, fertilizer = predict_image(filepath)
            image_url = filepath

    return render_template('index.html',
                           prediction=prediction,
                           image_url=image_url,
                           water=water,
                           fertilizer=fertilizer)

if __name__ == '__main__':
    app.run(debug=True)
