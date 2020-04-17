from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras import optimizers
from keras import backend as K
from constants import number_of_classes, img_dim
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras.callbacks import ModelCheckpoint

class Model:
    def __init__(self, X_train, y_train, X_test, y_test, learning_rate, optimizer, train_valid_split, activation_function, epochs, batch_size, loss_function):
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

        self.optimizer_instance = self.create_optimizer_instance()

        self.model = self.create_cnn_model()

    def create_optimizer_instance(self):
        opt_instance = None
        if self.optimizer == 'SGD':
            opt_instance = optimizers.SGD(lr=self.learning_rate, decay=1e-6, momentum=0.9, nesterov=True)
        elif self.optimizer == 'Adam':
            opt_instance = optimizers.Adam(learning_rate=self.learning_rate, beta_1=0.9, beta_2=0.999, amsgrad=False)
        elif self.optimizer == 'RMSprop':
            opt_instance = optimizers.RMSprop(learning_rate=self.learning_rate, rho=0.9)
        return opt_instance

    def create_cnn_model(self):
        model = Sequential()
        model.add(Conv2D(32, (3, 3), padding='same',
                        input_shape=(img_dim, img_dim, 3),
                        activation=self.activation_function))
        model.add(Conv2D(32, (3, 3), activation=self.activation_function))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(64, (3, 3), padding='same',
                        activation=self.activation_function))
        model.add(Conv2D(64, (3, 3), activation=self.activation_function))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(128, (3, 3), padding='same',
                        activation=self.activation_function))
        model.add(Conv2D(128, (3, 3), activation=self.activation_function))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Flatten())
        model.add(Dense(512, activation=self.activation_function))
        model.add(Dense(number_of_classes, activation='softmax'))
        print(model.summary())
        return model

    def train(self):
        self.model.compile(loss=self.loss_function,
                            optimizer=self.optimizer_instance,
                            metrics=['accuracy'])
        
        history = self.model.fit(self.X_train, self.y_train,
                                 batch_size=self.batch_size,
                                 epochs=self.epochs,
                                 validation_split=self.train_valid_split,
                                 callbacks=[ModelCheckpoint('basic_model.h5', save_best_only=True)])

        self.evaluate_on_test_data()
        self.plot_accuracy_and_loss(history)

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
        plt.show()
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper left')
        plt.show()
