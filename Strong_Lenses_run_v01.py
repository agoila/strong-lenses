#import keras
#import numpy
#import utils
#import metrics
#import models_preprocessing

#auroc = metrics.auroc
#data = numpy.load('Simulated_Data/wiener2.npy')
#labels = numpy.load('Simulated_Data/classification.npy')

#utils.epoch_curve(models_preprocessing.compiledConvnet, data, labels, 0.3, range(1,30), auroc)

import utils, models_preprocessing, metrics
import numpy
from keras.preprocessing.image import ImageDataGenerator

model_function = models_preprocessing.compiledRegularizedConvnet
auroc = metrics.auroc
accuracy = metrics.accuracy
text = metrics.basicTextMetrics

data = numpy.load('data/imadjust.npy')
labels = numpy.load('labels/classification.npy')

generator = ImageDataGenerator(horizontal_flip=True, vertical_flip=True)

utils.epoch_curve_generator(model_function, data, labels, generator, 32, 0.3, range(1, 41), [auroc, accuracy])
