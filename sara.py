import os
from flask import Flask, request, redirect, render_template, flash
from werkzeug.utils import secure_filename
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.preprocessing import image

import numpy as np


classes = ['!Alert! covered face','Everything is ok']
image_size = 50

UPLOAD_FOLDER = "static/uploads"
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

app = Flask(__name__)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

model = load_model('./my_model.h5')#学習済みモデルをロード


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('ファイルがありません')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('ファイルがありません')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(UPLOAD_FOLDER, filename))
            filepath = os.path.join(UPLOAD_FOLDER, filename)

            #受け取った画像を読み込み、np形式に変換
            img = image.load_img(filepath, grayscale=False, target_size=(image_size, image_size))
            img = image.img_to_array(img)
            data = np.array([img])
            result = model.predict(data)[0]
            #変換したデータをモデルに渡して予測する
            predicted = result.argmax()
            pred_answer = classes[predicted]
            pred_proba = np.round(model.predict(data) *100,decimals=1)
            pred_proba_str = " ".join(str(i)+"%" for i in pred_proba[0])
            # return render_template("index.html",answer=pred_answer,filepath=filepath)
            return render_template("index.html",answer=pred_answer,proba=pred_proba_str,filepath=filepath)
    
    else:
        return render_template("index.html",answer="",proba="")

    


# if __name__ == "__main__":
#     app.run()

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 8080))
    app.run(host ='0.0.0.0',port = port)