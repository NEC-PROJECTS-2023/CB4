from flask import Flask, render_template, request, redirect, url_for
from flask_compress import Compress
compress = Compress()
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import io

app = Flask(__name__)
model = tf.keras.models.load_model("C:\\Users\Veera\Desktop\Traffic\\traffic_classifier.h5")

classes = { 
            1:'Speed limit (20km/h)',
            2:'Speed limit (30km/h)', 
            3:'Speed limit (50km/h)', 
            4:'Speed limit (60km/h)', 
            5:'Speed limit (70km/h)', 
            6:'Speed limit (80km/h)', 
            7:'End of speed limit (80km/h)', 
            8:'Speed limit (100km/h)', 
            9:'Speed limit (120km/h)', 
            10:'No passing',
            11:'No passing veh over 3.5 tons', 
            12:'Right-of-way at intersection', 
            13:'Priority road', 
            14:'Yield', 
            15:'Stop', 
            16:'No vehicles', 
            17:'Veh > 3.5 tons prohibited', 
            18:'No entry', 
            19:'General caution', 
            20:'Dangerous curve left', 
            21:'Dangerous curve right', 
            22:'Double curve', 
            23:'Bumpy road', 
            24:'Slippery road', 
            25:'Road narrows on the right', 
            26:'Road work', 
            27:'Traffic signals', 
            28:'Pedestrians', 
            29:'Children crossing', 
            30:'Bicycles crossing', 
            31:'Beware of ice/snow',
            32:'Wild animals crossing', 
            33:'End speed + passing limits', 
            34:'Turn right ahead', 
            35:'Turn left ahead', 
            36:'Ahead only', 
            37:'Go straight or right', 
            38:'Go straight or left', 
            39:'Keep right', 
            40:'Keep left', 
            41:'Roundabout mandatory', 
            42:'End of no passing', 
            43:'End no passing veh > 3.5 tons' }

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/info')
def info():
    return render_template("info.html")

@app.route('/predict', methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        image = request.files["image"].read()
        image = Image.open(io.BytesIO(image))
        image = image.resize((30, 30))
        image = np.array(image)
        image = np.expand_dims(image, axis=0)
        prediction = model.predict(image)
        class_id = np.argmax(prediction[0])
        return redirect(url_for('result', class_id=class_id + 1))
    else:
        return render_template("predict.html")

'''@app.route('/predict', methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        # check if uploaded file is an image
        if 'image' not in request.files or not request.files['image'].filename:
            return render_template('predict.html', error_message='Please upload an image')
        
        # read image and process it
        image = request.files["image"].read()
        image = Image.open(io.BytesIO(image))
        image = image.resize((30, 30))
        image = np.array(image)
        image = np.expand_dims(image, axis=0)
        
        # check if uploaded image is a traffic sign image
        prediction = model.predict(image)
        class_id = np.argmax(prediction[0])
        if class_id + 1 not in classes:
            return render_template('predict.html', error_message='Please upload a traffic sign image')
        
        # check if the predicted class is a valid traffic sign class
        predicted_class = classes[class_id+1]
        if "Speed limit" not in predicted_class and "No passing" not in predicted_class and "End" not in predicted_class and "Priority" not in predicted_class:
            return render_template('predict.html', error_message='Please upload a traffic sign image')
        
        return redirect(url_for('result', class_id=class_id + 1))
    else:
        return render_template("predict.html")'''




@app.route('/result')
def result():
    class_id = request.args.get("class_id")
    if class_id is not None:
        return render_template('result.html', prediction=classes[int(class_id)])
    else:
        return redirect(url_for('predict'))
    
@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404

if __name__ == '__main__':
    compress.init_app(app)
    app.run(debug=True)
