import scipy.misc as im
import numpy as np
import pickle
from data_t import load_test_img
from hrnet_keras import hrnet_keras


model_weight = "../model_weight.pkl"
imgDir = "../Training400/imgs/n0094.jpg"
save_path = "../test_labl/"


img_test = load_test_img(imgDir)
img_test = np.reshape(img_test, (1, 256, 256, 3))

model = hrnet_keras()
with open(model_weight, 'rb') as fpkl:
    weight = pickle.load(fpkl)
    model.set_weights(weight)

labl_test = np.zeros((1, 256, 256), dtype='float32')

result_test = model.predict(img_test, batch_size=1)

for i in range(len(result_test)):
    for j in range(len(result_test[i])):
        for k in range(len(result_test[i][j])):
            if result_test[i][j][k][0] >= result_test[i][j][k][1] and result_test[i][j][k][0] >= result_test[i][j][k][2]:
                labl_test[i][j][k] = 255

            elif result_test[i][j][k][1] >= result_test[i][j][k][0] and result_test[i][j][k][1] >= result_test[i][j][k][2]:
                labl_test[i][j][k] = 127

            else:
                labl_test[i][j][k] = 0

im.imsave(save_path+"n0094.jpg", labl_test[0])
