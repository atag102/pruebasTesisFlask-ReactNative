from flask import Flask, jsonify,request
import base64
import numpy as np
import cv2
from flask_cors import CORS, cross_origin
import matplotlib.image as mpimg
import matplotlib.pyplot as plt





app=Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'



cat_face_cascade = cv2.CascadeClassifier('./haarcascade_frontalcatface_extended.xml')

@app.route('/',methods=['GET'])
def get_articles():
    list=[
        {"id":1, "title":"a", "body":"jajaja"},
        {"id":2, "title":"b", "body":"jajaja2"},
        {"id":3, "title":"c", "body":"jajaja3"}
    ]
    return jsonify(list)

@app.route('/detect', methods=['POST'])
@cross_origin()
def detect():
    try:
        data = request.json
        img_data = base64.b64decode(data)
        np_arr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        cats = cat_face_cascade.detectMultiScale(img, 1.5, 4)
        print(cats)
        cat_faces = []
        for (x, y, w, h) in cats:
            cat_faces.append({"x": int(x), "y": int(y), "w": int(w), "h": int(h)})
        plt.imshow(img)
        plt.show()
        print('Detected cat faces:', cat_faces)
        return jsonify({"cat_faces": cat_faces})
    except Exception as e:
        print('Error:', str(e))
        return jsonify({"error": str(e)}), 500

@app.route('/detect/local', methods=['GET'])
@cross_origin()
def detectlocal():
    try:
        data = mpimg.imread('../Flask/0001_007.jpg')
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)

        cats = cat_face_cascade.detectMultiScale(img, 1.5, 4)
        print(cats)
        cat_faces = []
        for (x, y, w, h) in cats:
            cat_faces.append({"x": int(x), "y": int(y), "w": int(w), "h": int(h)})
        plt.imshow(img)
        plt.show()
        print('Detected cat faces:', cat_faces)
        return jsonify({"cat_faces": cat_faces})
    except Exception as e:
        print('Error:', str(e))
        return jsonify({"error": str(e)}), 500

if __name__=="__main__":
    app.run(host='0.0.0.0', port=5000,debug=True)