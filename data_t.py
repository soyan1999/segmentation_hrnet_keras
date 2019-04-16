import scipy.misc as im
import numpy as np
import os


def load_test_img(imgFile):

    img0 = im.imread(imgFile)
    img1 = im.imresize(img0, (256, 256))
    img2 = np.asarray(img1, dtype="float32")

    return img2/255


def load_test_data(dataDir):
    imgs = os.listdir(dataDir)
    num = len(imgs)

    for i in range(num):
        imgs[i] = load_test_img(dataDir+'/'+imgs[i])

    data_test = np.asarray(imgs)

    return data_test


'''
dataDir="d:/my code/deep_learning/dis_cup/REFUGE-Validation400"

saveDir="d:/my code/deep_learning/dis_cup/"
data_test=load_test_data(dataDir)
np.save(saveDir+"data_t.npy",data_test)
'''
