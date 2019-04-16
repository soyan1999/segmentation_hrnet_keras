import numpy as np
import pickle
from data import load_data_gen
from dice_loss import dice_coef, dice_loss
from keras.callbacks import TensorBoard
from keras.models import load_model
from hrnet_keras import hrnet_keras


modelFile = "../model.hdf5"
model_weight_path = "../model_weight.pkl"
lablDir = "../Annotation-Training400/Disc_Cup_Masks/label"
dataDir = "../Training400/imgs"
batch_size = 1
train_begin = 40
train_end = 120
val_begin = 200
val_end = 230
LoadWeight = False
train_gen = load_data_gen(batch_size, train_begin, train_end, dataDir, lablDir)
val_gen = load_data_gen(batch_size, val_begin, val_end, dataDir, lablDir)
train_steps = (train_end-train_begin)/batch_size
val_steps = (val_end-val_begin)/batch_size
callback = [TensorBoard(log_dir='../train_logs')]


model = hrnet_keras()

if LoadWeight:
    with open(model_weight_path, 'rb') as fpkl:
        weight = pickle.load(fpkl)
        model.set_weights(weight)

model.fit_generator(train_gen, steps_per_epoch=train_steps, callbacks=callback, epochs=100,
                    validation_data=val_gen, validation_steps=val_steps)


with open(model_weight_path, 'wb') as fpkl:
    weight = model.get_weights()
    pickle.dump(weight, fpkl, protocol=pickle.HIGHEST_PROTOCOL)
