import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}


import time
import glob
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from numpy import savez_compressed
import numpy as np
from math import floor
from random import shuffle
from numpy import asarray


# from PIL import Image
# import image

images = []
data = []

# folders ['C:/Users/calvi/Desktop/smART_data/dataset/training_set\\drawings', 'C:/Users/calvi/Desktop/smART_data/dataset/training_set\\engraving', 'C:/Users/calvi/Desktop/smART_data/dataset/training_set\\iconography', 'C:/Users/calvi/Desktop/smART_data/dataset/training_set\\painting', 'C:/Users/calvi/Desktop/smART_data/dataset/training_set\\sculpture']
def formatData():
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

def train():
    # model setup, only need result of feature extraction(after 7x7x512)
    model = VGG16(weights='imagenet', include_top=False)

    # process images
    img_paths = data
    img_vector_features = []
    for img_path in range(len(images)):
        print(img_path, img_paths[img_path])
        img = image.load_img(img_paths[img_path], target_size=(256, 256))
        img_data = image.img_to_array(img)
        img_data = np.expand_dims(img_data, axis=0)
        img_data = preprocess_input(img_data)

        # get features from images
        vgg16_feature = model.predict(img_data)
        vgg16_feature = np.array(vgg16_feature)
        vgg16_feature = vgg16_feature.flatten()
        img_vector_features.append(vgg16_feature)
    np.savez_compressed('img_vector_features3.npz', img_vector_features)
    print("finished training")


from sklearn.neighbors import NearestNeighbors
import numpy as np
from keras.preprocessing import image
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input

def predictArt(qImg,data,images):
    # reverse image search prediction using kNearestNeighbors

    model = VGG16(weights='imagenet', include_top=False)

    # inputted image
    query_path = f"static/{qImg}"

    # load image
    img = image.load_img(query_path, target_size=(256, 256))
    img_data = image.img_to_array(img)
    img_data = np.expand_dims(img_data, axis=0)
    img_data = preprocess_input(img_data)

    # parse features of inputted image
    vgg16_feature = model.predict(img_data)
    vgg16_feature = np.array(vgg16_feature)
    query_feature = vgg16_feature.flatten()
    # load features

    with np.load("C:/Users/calvi/Desktop/smART2/img_vector_features3.npz") as img_vector_features_1:
        img_vector_features = img_vector_features_1["arr_0"]

        # find similar images
        N_QUERY_RESULT = 3
        nbrs = NearestNeighbors(n_neighbors=N_QUERY_RESULT,
                                metric="cosine").fit(img_vector_features)

        # formatting
        distances, indices = nbrs.kneighbors([query_feature])
        similar_image_indices = indices.reshape(-1)

        print(similar_image_indices)
        print(data[similar_image_indices[0]], ",",images[similar_image_indices[0]], ",", data[similar_image_indices[0]].split("/")[-2:])
        print(data[similar_image_indices[1]], ",", images[similar_image_indices[1]], ",", data[similar_image_indices[1]].split("/")[-2:])
        print(data[similar_image_indices[2]], ",", images[similar_image_indices[2]], ",", data[similar_image_indices[2]].split("/")[-2:])
    return data[similar_image_indices[0]], data[similar_image_indices[1]], data[similar_image_indices[2]]

def predictArtist(qImg,data,images):
# reverse image search prediction using kNearestNeighbors

    model = VGG16(weights='imagenet', include_top=False)

    # inputted image
    query_path = f"static/{qImg}"

    # load image
    img = image.load_img(query_path, target_size=(256, 256))
    img_data = image.img_to_array(img)
    img_data = np.expand_dims(img_data, axis=0)
    img_data = preprocess_input(img_data)

    # parse features of inputted image
    vgg16_feature = model.predict(img_data)
    vgg16_feature = np.array(vgg16_feature)
    query_feature = vgg16_feature.flatten()
    # load features

    with np.load("C:/Users/calvi/Desktop/smART2/img_vector_features3.npz") as img_vector_features_1:
        img_vector_features = img_vector_features_1["arr_0"]

        # find similar images
        N_QUERY_RESULT = 3
        nbrs = NearestNeighbors(n_neighbors=N_QUERY_RESULT,
                                metric="cosine").fit(img_vector_features)

        # formatting
        distances, indices = nbrs.kneighbors([query_feature])
        similar_image_indices = indices.reshape(-1)

        print(similar_image_indices)
        print(data[similar_image_indices[0]], ",",images[similar_image_indices[0]], ",", data[similar_image_indices[0]].split("/")[-1])
        print(data[similar_image_indices[1]], ",", images[similar_image_indices[1]], ",", data[similar_image_indices[1]].split("/")[-1])
        print(data[similar_image_indices[2]], ",", images[similar_image_indices[2]], ",", data[similar_image_indices[2]].split("/")[-1])

    return similar_image_indices[0], similar_image_indices[1], similar_image_indices[2]
    
formatData()
# train()
predictArt("query1.jpg",data,images)
# predict("query2.jpg")
# predict("query3.jpg")
