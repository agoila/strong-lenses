import keras
import numpy
import utils
import metrics
import models_preprocessing

auroc = metrics.auroc
data = numpy.load('Simulated_Data/wiener2.npy')
labels = numpy.load('Simulated_Data/classification.npy')

utils.epoch_curve(models_preprocessing.compiledConvnet, data, labels, 0.3, range(1,30), auroc)