# from flask import Flask
# app = Flask(__name__)
# @app.route('/', methods= ['GET'])
# def hello_world():
#     return "Hello World"

# if __name__ == '__main__':
#     app.run(port=3000, debug=True)

from flask import Flask, request, jsonify
from flask_cors import cross_origin, CORS
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

app = Flask(__name__)
CORS(app)
model = load_model('my_model.h5')
classes = ['cats', 'dogs']

@app.route('/predict', methods=['POST'])
def predict():

    if 'image' not in request.files:
        return jsonify({'error': 'no image file found'})

    image_file = request.files['image']
    file_path = './temp/temp.jpg'  
    image_file.save(file_path)  

    image = load_img(file_path, target_size=(100, 100))
    image = img_to_array(image) / 255.0
    image = np.expand_dims(image, axis=0)

    predictions = model.predict(image)
    prediction_index = np.argmax(predictions)
    predicted_class = classes[prediction_index]

    

    return jsonify({'class': predicted_class})


@app.route('/save', methods=['POST'])
def save():
    data = request.json
    print(data)
    return jsonify({'status': 'success'})

if __name__ == '__main__':
    app.run(debug=True)
