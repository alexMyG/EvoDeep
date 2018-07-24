import os
import pprint
import time
from math import sqrt, ceil, trunc

import pandas as pd
from numpy.ma import true_divide
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split

# os.environ['KERAS_BACKEND'] = "tensorflow"  # This must be invoked before importing keras
from keras.models import Sequential
from keras.layers import Dense, Reshape, Flatten, MaxPooling2D
from keras.layers import Dropout, Convolution2D
from keras import callbacks
from keras.utils import np_utils


class KerasExecutor:
    # The number of neurons in the first and last layer included in network-structure is ommited.
    def __init__(self, dataset, test_size, metrics, early_stopping_patience, loss, first_data_column=1):

        self.dataset = dataset
        self.first_data_column = first_data_column
        self.test_size = test_size
        self.metrics = metrics
        self.early_stopping_patience = early_stopping_patience
        self.loss = loss

        # Carga del dataset
        # data = pd.read_csv(self.dataset, sep=',')
        data = dataset["data"]
        # self.x = data

        self.x = true_divide(data, 255)

        y = dataset["target"]
        # self.x = data.iloc[1:, self.first_data_column:-1].as_matrix()
        # y = data.iloc[1:, -1].as_matrix()

        le = preprocessing.LabelEncoder()
        le.fit(y)
        y_numeric = le.transform(y)
        self.y_hot_encoding = np_utils.to_categorical(y_numeric).astype(float)

        self.n_in = len(self.x[1])
        self.n_out = len(self.y_hot_encoding[1])

    def execute(self, individual):

        train, test, target_train, target_test = train_test_split(self.x, self.y_hot_encoding, test_size=self.test_size,
                                                                  random_state=int(time.time()))

        model = Sequential()

        # last_num_neurons = self.n_in

        # print(individual)
        list_layers_names = [l.type for l in individual.net_struct]
        print ",".join(list_layers_names)









        for index, layer in enumerate(individual.net_struct):

            # print "Adding layer " + layer.type
            # pprint.pprint(layer.parameters)

            if layer.type == "Dense":
                model.add(Dense(**layer.parameters))
                # last_num_neurons = layer.parameters["output_dim"]

            elif layer.type == "Dropout":
                model.add(Dropout(**layer.parameters))

            elif layer.type == "Convolution2D":
                layer.parameters['nb_row'] = min(layer.parameters['nb_row'], model.output_shape[1])
                layer.parameters['nb_col'] = min(layer.parameters['nb_col'], model.output_shape[2])
                model.add(Convolution2D(**layer.parameters))

            elif layer.type == "MaxPooling2D":
                # Pool size checking...
                pool_size = (min(layer.parameters['pool_size'][0], model.output_shape[1]),
                             min(layer.parameters['pool_size'][1], model.output_shape[2]))

                model.add(MaxPooling2D(pool_size=pool_size, strides=layer.parameters['strides']))

            elif layer.type == "Reshape":

                aspect_ratio = layer.parameters["target_shape"]

                if index == 0:
                    last_num_rows, last_num_cols = None, self.n_in
                else:
                    last_num_rows, last_num_cols = model.output_shape

                dividers = [k for k in range(2, int(sqrt(last_num_cols))) if last_num_cols % k == 0]
                num_columns = max(dividers)
                num_rows = last_num_cols / num_columns

                if 'input_shape' in layer.parameters:
                    model.add(Reshape(target_shape=(num_rows, num_columns, 1),
                                      input_shape=layer.parameters["input_shape"]))
                else:
                    model.add(Reshape(target_shape=(num_rows, num_columns, 1)))

            elif layer.type == "Flatten":
                model.add(Flatten(**layer.parameters))
                # last_num_neurons = layer.parameters["output_dim"]

            # print "Added layer " + layer.type + " with input shape: " + str(model.input_shape)+ " -- output shape: " +\
            #       str(model.output_shape)


            """


            if layer.type_layer == 'Dense' or index == len(individual.net_struct) - 1:

                init = layer.arguments['init']
                activation = layer.arguments['activation']

                if index == 0:

                    model.add(Dense(layer.arguments['num_outputs'], input_dim=len(train[1]), init=init,
                                    activation=activation))

                elif index == len(individual.net_struct) - 1:

                    model.add(Dense(self.n_out, init=init, activation=activation))

                else:

                    model.add(Dense(layer.arguments['num_outputs'], init=init, activation=activation))

            if layer.type_layer == 'Dropout' and len(individual.net_struct) - 1 > index:

                p = layer.arguments['p']

                if index == 0:

                    model.add(Dropout(p, input_shape=(len(train[1]),)))
                else:
                    model.add(Dropout(p))
            """

        train, validation, target_train, target_validation = train_test_split(train, target_train, test_size=0.2,
                                                                              random_state=int(time.time()))

        model.compile(loss=self.loss, optimizer=individual.global_attributes.optimizer, metrics=self.metrics)

        callbacks_array = [
            callbacks.EarlyStopping(monitor='val_acc', min_delta=0.00001, patience=self.early_stopping_patience, verbose=0,
                                    mode='max')]

        hist = model.fit(train, target_train, nb_epoch=individual.global_attributes.nb_epoch,
                         batch_size=individual.global_attributes.batch_size,
                         verbose=0, callbacks=callbacks_array, validation_data=(validation, target_validation)).__dict__


        scores_training = model.evaluate(train, target_train, verbose=0)
        scores_validation = model.evaluate(validation, target_validation, verbose=0)
        scores_test = model.evaluate(test, target_test, verbose=0)

        return model.metrics_names, scores_training, scores_validation, scores_test, model
