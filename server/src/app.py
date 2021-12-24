# 
import os

from numpy.lib.npyio import load
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
# make sure import os and os.environ... are at top

import random
from ml_model import predictArtist
from ml_model import predictArt
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
import numpy as np
from sklearn.neighbors import NearestNeighbors
from flask import Flask, flash, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename
import glob

import csv

app = Flask('app')
app.config['UPLOAD_FOLDER'] = "tmp"

app.config['SECRET_KEY'] = 'my super secret key'.encode('utf8')

images = []
data = []
folders = []

imageFolder = 'C:/Users/calvi/Desktop/smART_data2/images/*'
for name in glob.glob(imageFolder):
    # print("name", name)
    folders.append(f"{name}")

# print("folders", folders)

for a in range(len(folders)):
    for b in (os.listdir(folders[a])):
        images.append(b)


listImgFolders = []
for i in range(len(folders)):
    listImgFolders.append(len(os.listdir(folders[i])))

# print(listImgFolders[0],listImgFolders[1])
# get image path, assign to list

for a in range(len(images)):
    imgNum = listImgFolders[0]
    for b in range(len(folders)):
        if a < (imgNum):
            data.append(f'{folders[b]}/{images[a]}')
            break
        else:
            imgNum += listImgFolders[b+1]

print("len(images)",len(images))
print("len(data)",len(data))

@app.route('/')
def on_load():
    # rand = random.randint(1, len(os.listdir("dataset")))
    # startImage = os.listdir("dataset")[rand]
    # predict(f"dataset/{startImage}")
    return render_template('index.html')


# method that handles image_load
@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file:
            # INSERT LINK TO TENSORFLOW HERE
            filename = secure_filename(file.filename)

            # not sure if image needs to be saved or not
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

            # call tensorflow
            predictArt(filename)

            # stay on main page
            return redirect(request.referrer)


def load_image_data(id):
    print(id)
    with open('static/artists.csv', newline='',encoding="utf-8") as f:

        imgNum = len(os.listdir(folders[0]))
        for b in range(len(folders)):
            if id < (imgNum):
                id = b
                break
            else:
                imgNum += len(os.listdir(folders[b+1]))

        reader = csv.reader(f)
        data = []
        for i in reader:
            data.append(i)
        artist_info = data[id][:]
        print("artist_info",artist_info)
    return (artist_info)


@app.route("/home", methods=['GET', 'POST'])
def home():
    print(len(os.listdir("static/dataset")))
    rand = random.randint(0, len(os.listdir(""))-1)
    startImage = f"/dataset/{os.listdir('static/dataset')[rand]}"
    print("startImage",startImage)
    artistVals = predictArtist(startImage,data,images)
    print("artistVals",artistVals)
    artistDatas = load_image_data(artistVals[0]),load_image_data(artistVals[1]),load_image_data(artistVals[2])
    # fix predictArt
    print("ASDASDASD")
    return render_template('index.html', currentImage=f"{startImage}", predictions = predictArt(startImage,data,images), artistData = artistDatas)


app.run(host='0.0.0.0', port=8080, debug=True)
