import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from PIL import Image
import PIL.ImageOps

X = np.load('image.npz')['arr_0']
y = pd.read_csv("https://raw.githubusercontent.com/whitehatjr/datasets/master/C%20122-123/labels.csv")["labels"]

x_train,x_test,y_train,y_test = train_test_split(X,y,test_size = 2500, train_size = 7500, random_state = 0)
x_train_scale = x_train/255 
x_test_sclae = x_test/255
clf = LogisticRegression(solver = "saga", multi_class = "multinomial")
clf.fit(x_train_scale,y_train)

def get_prediction(image):
    im_pil = Image.open(image)
    image_bw = im_pil.convert('L') 
    imgresize = image_bw.resize((28,28),Image.ANTIALIAS)
    pixelfilter = 20
    minpixel = np.percentile(imgresize,pixelfilter)
    imginvertscale = np.clip(imgresize-minpixel,0,255)
    maxpixel = np.max(imgresize)
    imginvertscale = np.asarray(imginvertscale/maxpixel,0,255)
    testSample = np.array(imginvertscale).reshape(1,784)
    test_pred = clf.predict(testSample)

    return test_pred[0]