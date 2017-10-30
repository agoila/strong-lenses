import utils, models_preprocessing, metrics
import numpy
from keras.preprocessing.image import ImageDataGenerator

model_function = models_preprocessing.compiledConvnet
# model_function = models_preprocessing.compiledRegularizedConvnet
auroc = metrics.auroc
accuracy = metrics.accuracy
text = metrics.basicTextMetrics

data = numpy.load('data/source.npy')
labels = numpy.load('labels/classification.npy')

## Run 1: Compiled ConvNet
# utils.epoch_curve(model_function, data, labels, 0.3, range(1,41), [auroc, accuracy])


## Run 2: Compiled Regularized ConvNet
#generator = ImageDataGenerator(horizontal_flip=True, vertical_flip=True)
#utils.epoch_curve_generator(model_function, data, labels, generator, 32, 0.3, range(1, 41), [auroc, accuracy])


## Run 3: Complied Convnet and Compiled Regularized Convnet with 5-fold cross-validation
utils.cross_validation(model_function, data, labels, 5, 20, metrics_array=text)
