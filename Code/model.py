from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras import optimizers
from keras import backend as K
from constants import number_of_classes, img_dim, generated_files_path, default_params
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import confusion_matrix
import numpy as np
from keras.models import load_model

class Model:
    def __init__(self, X_train, y_train, X_test, y_test, learning_rate,
                optimizer, train_valid_split, activation_function, epochs,
                batch_size, loss_function, use_drop_out, file_save_name,
                drop_outs_value = None):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.train_valid_split = train_valid_split
        self.activation_function = activation_function
        self.epochs = epochs
        self.batch_size = batch_size
        self.loss_function = loss_function
        self.use_drop_out = use_drop_out
        self.drop_outs_value = drop_outs_value or default_params.drop_outs_value
        self.file_save_name = generated_files_path+file_save_name

        self.optimizer_instance = self.create_optimizer_instance()

        self.model = self.create_cnn_model()

    def create_optimizer_instance(self):
        opt_instance = None
        if self.optimizer == 'SGD':
            opt_instance = optimizers.SGD(lr=self.learning_rate, decay=1e-6, momentum=0.9, nesterov=True)
        elif self.optimizer == 'Adam':
            opt_instance = optimizers.Adam(lr=self.learning_rate, beta_1=0.9, beta_2=0.999, amsgrad=False)
        elif self.optimizer == 'RMSprop':
            opt_instance = optimizers.RMSprop(lr=self.learning_rate, rho=0.9)
        return opt_instance

    def create_cnn_model(self):
        model = Sequential()
        model.add(Conv2D(32, (3, 3), padding='same',
                        input_shape=(img_dim, img_dim, 3),
                        activation=self.activation_function))
        model.add(Conv2D(32, (3, 3), activation=self.activation_function))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        if self.use_drop_out:
            model.add(Dropout(self.drop_outs_value[0]))
        model.add(Conv2D(64, (3, 3), padding='same',
                        activation=self.activation_function))
        model.add(Conv2D(64, (3, 3), activation=self.activation_function))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        if self.use_drop_out:
            model.add(Dropout(self.drop_outs_value[1]))
        model.add(Conv2D(128, (3, 3), padding='same',
                        activation=self.activation_function))
        model.add(Conv2D(128, (3, 3), activation=self.activation_function))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        if self.use_drop_out:
            model.add(Dropout(self.drop_outs_value[2]))
        model.add(Flatten())
        model.add(Dense(512, activation=self.activation_function))
        model.add(Dense(number_of_classes, activation='softmax'))
        # print(model.summary())
        return model

    def train(self):
        X_train, X_val, Y_train, Y_val = train_test_split(self.X_train, self.y_train,
                            test_size=self.train_valid_split, random_state=42)
        self.model.compile(loss=self.loss_function,
                            optimizer=self.optimizer_instance,
                            metrics=['accuracy'])

        history = self.model.fit(X_train, Y_train,
                                 batch_size=self.batch_size,
                                 epochs=self.epochs,
                                 validation_data=(X_val, Y_val),
                                 callbacks=[ModelCheckpoint(self.file_save_name+'_model.h5', save_best_only=True)])

        self.evaluate_on_test_data()
        self.print_confusion_matrix()
        self.plot_accuracy_and_loss(history)
        return history

    def load_trained_model(self):
        self.model = load_model(self.file_save_name+'_model.h5')
        self.evaluate_on_test_data()

    def print_confusion_matrix(self):
        y_pred = self.model.predict(self.X_test)
        y_pred_not_cat = np.argmax(y_pred, axis=-1)
        y_true_not_cat = np.argmax(self.y_test, axis=-1)
        conf_matrix = confusion_matrix(y_true_not_cat, y_pred_not_cat)
        print(conf_matrix)
        with open(self.file_save_name+'_confusion_matrix.txt', 'w') as fp:
            fp.write(str(conf_matrix.tolist()))

    def evaluate_on_test_data(self):
        scores = self.model.evaluate(self.X_test, self.y_test)
        print('Accuracy: (%)', scores[1] * 100)

    def plot_accuracy_and_loss(self, history):
        plt.plot(history.history['acc'])
        plt.plot(history.history['val_acc'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper left')
        plt.savefig(self.file_save_name+'_accuracy.png')
        plt.cla()
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper left')
        plt.savefig(self.file_save_name+'_loss.png')
        plt.cla()
