import scipy.misc as im
import numpy as np
import os
import random


def load_img_pairs(imgFile, imgLabelFile):

    img0 = im.imread(imgFile)
    img1 = im.imresize(img0, (256, 256))
    img2 = np.asarray(img1, dtype="float32")
    img2 = img2/255

    labl0 = im.imread(imgLabelFile)
    labl1 = im.imresize(labl0, (256, 256))
    labl2 = np.asarray(labl1, dtype="float32")

    labl3, labl4 = np.unique(labl2, return_inverse=True)
    labl4 = np.reshape(labl4, (256, 256))
    labl5 = np.zeros((256, 256, 3), dtype="float32")
    for i in range(256):
        for j in range(256):
            if(labl4[i, j] < 40):
                labl5[i, j] = [0, 0, 1]
            elif(labl4[i, j] < 130):
                labl5[i, j] = [0, 1, 0]
            else:
                labl5[i, j] = [1, 0, 0]
    return img2, labl5


def load_data_gen(batch_size, begin, end, dataDir, lablDir):

    imgs = os.listdir(dataDir)
    labls = os.listdir(lablDir)
    i = 0
    lis = []
    while i < end-begin:
        lis.append((imgs[i+begin], labls[i+begin]))
        i += 1

    random.shuffle(lis)

    i = 0
    img_gen = np.zeros((batch_size, 256, 256, 3), dtype='float32')
    labl_gen = np.zeros((batch_size, 256, 256, 3), dtype='float32')
    while True:

        img_gen[i % batch_size], labl_gen[i % batch_size] = load_img_pairs(
            dataDir+'/'+lis[i][0], lablDir+'/'+lis[i][1])

        if (i % batch_size) == (batch_size-1):

            yield (img_gen, labl_gen)
        i += 1
        if i == end - begin:
            i = 0
            random.shuffle(lis)
