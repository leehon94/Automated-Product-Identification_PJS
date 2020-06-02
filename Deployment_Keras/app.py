from google.colab import drive 
drive.mount('/content/gdrive')

# % cd '/content/gdrive/My Drive/'

"""#Laden der Packages"""

from keras import *

from keras.preprocessing import image
from keras import backend as K
import numpy as np
import os
import json

from flask import Flask, request, jsonify
from fastai.basic_train import load_learner
from fastai.vision import open_image
from flask_cors import CORS,cross_origin
app = Flask(__MVP__)
CORS(app, support_credentials=True)

"""#Deklarieren der globalen Variablen"""

img_width, img_height = 224, 224
model = models.load_model("/content/gdrive/My Drive/Keras Models/MobileNet5-1.h5")
optimizer = optimizers.RMSprop()
loss = 'categorical_crossentropy'
metrics = metrics.Accuracy()

"""#Funktion zur Vorhersage
Ausgabe ist eine json datei:
*   Artikelnummer: Art x
*   Wahrscheinlichkeit: y
"""

def Vorhersage(Artikelfoto, Speicherort_KI_Modell, Name_KI_Modell):

    model = models.load_model(Speicherort_KI_Modell + Name_KI_Modell)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['acc'])
    Artikelfoto = image.load_img(Artikelfoto, target_size=(img_width, img_height))
    Artikelfoto = image.img_to_array(Artikelfoto)
    Artikelfoto = np.expand_dims(Artikelfoto, axis=0)
    prediction = model.predict(Artikelfoto)
    prediction = prediction.flatten()
    artikelnummer = 'Art' + str(np.argmax(prediction)+1)
    wahrscheinlichkeit = str(prediction[np.argmax(prediction)])

    response = json.dumps({"Artikelnummer":artikelnummer, "Wahrscheinlichkeit":wahrscheinlichkeit})

    return response

"""#Beispielhafter Aufruf der Funktion"""

Vorhersage(base_dir + 'MVP Datensatz/Testbilder/P1040695.JPG',
           base_dir + 'Keras Models/',
           'MobileNet5-1.h5')

# route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    return jsonify(predict_single(request.files['image']))

if __MVP__ == '__main__':
    app.run()
