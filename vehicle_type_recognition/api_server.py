import os
import argparse
import matplotlib.pyplot as plt
from flask import Flask, jsonify, request, redirect

app = Flask(__name__)
app.config["MODELS_PATH"] = os.path.join(app.root_path, "models")

from predict import predict


@app.route('/', methods=['GET', 'POST'])
def image_handler():
    if request.method == 'POST':
        if 'image' not in request.files:
            return redirect(request.url)
        image = request.files['image']
        if image.filename == "":
            return redirect(request.url)
        if image:
            numpy_arr_img = plt.imread(image)
            result = predict(numpy_arr_img)
            return jsonify({'result': result})
    return '''
        <!doctype html>
        <title>Upload new Image</title>
        <h1>Upload new Image</h1>
        <form method=post enctype=multipart/form-data>
        <input type=file name=image>
        <input type=submit value=Upload>
        </form>
        '''


if __name__ == "__main__":
    app.run()
