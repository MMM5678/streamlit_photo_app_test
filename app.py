import streamlit as st #streamlit==1.22.0
import matplotlib.pyplot as plt #matplotlib==3.7.1
#from PIL import Image
import requests #requests==2.29.0
# import os.path
import numpy as np #numpy==1.23.5  
#from openvino.inference_engine import IECore
from torchvision.io import read_image   #torchvision==0.15.1

#
#####
import os
from keras.models import Sequential  #keras==2.12.0
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.utils import to_categorical
import cv2  #opencv-python==4.7.0.72
import tkinter as tk
from tkinter import filedialog
from PIL import ImageTk, Image
from pathlib import Path
import tempfile

# 画像ファイルのパス
from tensorflow.python.keras.layers import Activation #tensorflow==2.12.0
#####
#

st.set_option("deprecation.showfileUploaderEncoding", False)

st.sidebar.title("画像判定アプリ")
st.sidebar.write("画像より判定します。")

st.sidebar.write("")

img_source = st.sidebar.radio("画像ファイルを選択してください。",
                              ("画像をアップロード", "カメラで撮影"))
if img_source == "画像をアップロード":
    img_file = st.sidebar.file_uploader("画像を選択してください。", type=["png", "jpg"])
    
elif img_source == "カメラで撮影":
    img_file = st.camera_input("カメラで撮影")

###
###
###
if img_file is not None:
    IMG_PATH = 'imgs'
    img_path = os.path.join(IMG_PATH, img_file.name)
    st.image(img_file)
    print(img_file)
    print(img_path)
 
    IMG_SIZE = 64

    # モデルの作成とトレーニング
    data = []
    labels = []
    classes = ['lighton', 'lightoff']

    for c in classes:
        path = os.path.join(os.getcwd(), c)
        for img in os.listdir(path):
            try:
                img_path = os.path.join(path, img)
                image = cv2.imread(img_path)
                image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
                data.append(image)
                labels.append(classes.index(c))
                print(img_path)
            except Exception as e:
                print(e)

    data = np.array(data)
    labels = np.array(labels)

    # データをシャッフルする
    idx = np.arange(data.shape[0])
    np.random.shuffle(idx)
    data = data[idx]
    labels = labels[idx]

    # データをトレーニング用と検証用に分割する
    num_samples = len(data)
    num_train = int(num_samples * 0.8)
    x_train = data[:num_train]
    y_train = labels[:num_train]
    x_val = data[num_train:]
    y_val = labels[num_train:]

    # 画像データの正規化
    x_train = x_train / 255.0
    x_val = x_val / 255.0

    # ラベルをone-hotエンコーディングに変換する
    y_train = to_categorical(y_train)
    y_val = to_categorical(y_val)

    # モデルを構築する
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=x_train.shape[1:]))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dense(2))
    model.add(Activation('softmax'))

    # モデルをコンパイルする
    model.compile(loss='categorical_crossentropy',
                optimizer='adam',
                metrics=['accuracy'])

    # モデルをトレーニングする
    model.fit(x_train, y_train, epochs=10, batch_size=32, validation_data=(x_val, y_val))

    # GUIの作成
    root = tk.Tk()
    root.title("loghton or lightoff Classifier")
    root.geometry("500x500")

    # 画像を選択する
    file_path = img_path
    print(img_path)
    print(file_path)

    # 選択された画像をモデルに入力して予測結果を表示する
    image_selected = cv2.imread(file_path)
    image_selected = cv2.resize(image_selected, (IMG_SIZE, IMG_SIZE))
    image_selected = np.expand_dims(image_selected, axis=0)
    image_selected = image_selected / 255.0
    prediction = model.predict(image_selected)
    if np.argmax(prediction) == 0:
        st.write("LIGHT-ON!")
        print("LIGHT-ON")
    else:
        st.write("LIGHT-OFF!")
        print("LIGHT-OFF")

###