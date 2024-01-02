from flask import Flask, render_template,request,jsonify,url_for,redirect
from tensorflow.keras.preprocessing.image import load_img,img_to_array
from PIL import Image
import numpy as np
import os
import tensorflow as tf

app=Flask(__name__)
model = tf.keras.models.load_model('cnn_vgg16_Knee_95_final.h5')

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/predict')
def predict():
    return render_template("inner-page.html")

@app.route('/output', methods=['GET','POST'])
def output():
    if request.method =='POST':
        f=request.files['file']
        basepath=os.path.dirname(__file__)
        filepath=os.path.join(basepath,'uploads',f.filename)
        f.save(filepath)
        img=load_img(filepath,target_size=(224,224))
        # Resize the image to the required size
        # Convert the image to an array and normalize it
        image_array = np.array(img)
        # Add a batch dimension
        image_array = np.expand_dims(image_array, axis=0)
        # Use the pre-trained model to make a prediction
        pred=np.argmax(model.predict(image_array),axis=1)
        index=['arthritis','fracture','stage_1','stage_2','stage_3','stage_4']
        prediction = index[int(pred)]
        print("prediction")
        #predict = prediction
        return render_template("index.html", predict = prediction)


if __name__=='__main__':
    app.run(debug = True,port = 2222)


