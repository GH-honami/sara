import os
from flask import Flask, request, redirect, url_for, render_template, flash
from werkzeug.utils import secure_filename
from tensorflow.python.keras.models import Sequential, load_model
from keras.preprocessing import image
import tensorflow as tf
import numpy as np
from tensorflow import keras
import tensorflow.keras
import cv2
from tensorflow.keras.utils import load_img, img_to_array 

#分類したいクラスと学習に用いた画像のサイズ
classes = ['!Alert! covered face','Everything is ok']
img_size = 50

#アップロードされた画像を保存するフォルダ名とアップロードを許可する拡張子
# UPLOAD_FOLDER = "uploads"
UPLOAD_FOLDER = "static"
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

#Flaskクラスのインスタンスの作成
app = Flask(__name__)

#アップロードされたファイルの拡張子のチェックをする関数
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

#学習済みモデルをロード
model = load_model('./my_model.h5')


# こちらのコメントアウトしているところがエラーの原因になっていたところです。
# @app.route('/', methods=['GET', 'POST'])
# def upload_file():
#     if request.method == 'POST':

# こちらに変更しました。
#graph = tf.get_default_graph()
graph = tf.compat.v1.get_default_graph()
@app.route('/', methods=['GET', 'POST'])
def upload_file():
    global graph
    with graph.as_default():
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


                #img = image.load_img(filepath, grayscale=False, target_size=(img_size,img_size)) エラー
                img = keras.utils.load_img(filepath, grayscale=False, target_size=(img_size,img_size))
                #img = image.img_to_array(img)　エラー
                img = keras.utils.img_to_array(img)
                data = np.array([img])
                result = model.predict(data)[0]
                #変換したデータをモデルに渡して予測する
                predicted = result.argmax()
                pred_answer = "これは " + classes[predicted] + " です"

            return render_template("index.html",answer=pred_answer,filepath=filepath)

    return render_template("index.html",answer="")


# if __name__ == "__main__":
#     port = int(os.environ.get('PORT', 8080))
#     app.run(host ='0.0.0.0',port = port)
if __name__ == "__main__":
     app.run()
