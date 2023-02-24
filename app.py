import numpy as np
import keras
from flask import Flask,request,render_template
import pickle
import cv2
from werkzeug.utils import secure_filename
app=Flask(__name__)
import pickle
# Load the model from the file
with open('model_gpu.pkl', 'rb') as file:
    model_r = pickle.load(file)
import tensorflow as tf
# Load the model to extract features
model = tf.keras.models.load_model('densenet121_gpu.h5')
@app.route('/')
def home():
    return render_template("index.html")
@app.route('/predict',methods=['POST','GET'])
def predict():
    if request.method=="POST":
        # img=request.files["image"]
        tu={
            "0":"GLIOMA",
            "1":"MENINGIOMA",
            "2":"NO TUMOUR",
            "3":"PITUTARY"
        }
        image = request.files["image"].read()
        image = cv2.imdecode(np.frombuffer(image, np.uint8), cv2.IMREAD_UNCHANGED)
        # filename = secure_filename(image.filename)
        # p=os.getcwd()
        # image.save(os.path.join(p, filename))
        # img = cv2.imread(os.path.join(p, filename))
        img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # print(image)
        img=cv2.resize(img,(224,224),interpolation=cv2.INTER_LANCZOS4)
        # print(img)
        # print(img.shape)
        img=img.astype('float32')
        img/=255.0
        img=np.expand_dims(img,axis=0)
        feature_extractor = keras.Model(inputs=model.inputs,outputs=model.get_layer(name="final").output,)
        features =list(feature_extractor(img))
        l=[]
        for i in list(features):
            for j in list(i): 
                l.append(np.float32(j))
        final_features=list(np.array(l).reshape(1,-1))        
        prediction=model_r.predict(final_features)
        prediction=list(prediction)[0].strip('[]').split().index('1.')
        output=tu[str(prediction)]
        return render_template("index.html",prediction_text="{}".format(output))
if __name__=="__main__":
    app.run(debug=True)