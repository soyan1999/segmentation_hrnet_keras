import scipy.misc as im
import numpy as np
from data_t import load_test_img
from keras.models import load_model
from model import *
from keras.optimizers import Adam
from dice_loss import dice_coef,dice_loss

model_weight="../model_weight.hdf5"
imgDir="../Training400/imgs/n0002.jpg"
save_path="../test_labl/"


img_test=load_test_img(imgDir)
img_test=np.reshape(img_test,(1,256,256,3))
print(img_test[0][100])
model=Deeplabv3(weights=None,input_shape=(256,256,3),classes=3,backbone="xception")

model.compile(optimizer = Adam(lr = 1e-5), loss = dice_loss, metrics = [dice_coef])
model.load_weights(model_weight)
#pec=np.array([0.7091276,0.609758,0.38409415],dtype="float32")

labl_test=np.zeros((1,256,256),dtype='float32')

result_test=model.predict(img_test,batch_size=1)

for i in range(len(result_test)):
    for j in range(len(result_test[i])):
        for k in range(len(result_test[i][j])) :
            if result_test[i][j][k][0]>=result_test[i][j][k][1] and result_test[i][j][k][0]>=result_test[i][j][k][2] : 
                labl_test[i][j][k]=255
                
            elif result_test[i][j][k][1]>=result_test[i][j][k][0] and result_test[i][j][k][1]>=result_test[i][j][k][2] : 
                labl_test[i][j][k]=127
                
            else : labl_test[i][j][k]=0

im.imsave(save_path+"p.jpg",labl_test[0])
