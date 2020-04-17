from data_reader import read_image_files
from constants import data_folder_path, train_csv_file_name, test_csv_file_name
from model import Model
from visualizer import draw_multiple_line_plots

def read_data():
    X_train, y_train = read_image_files(csv_file_name=data_folder_path+train_csv_file_name)
    X_test, y_test = read_image_files(csv_file_name=data_folder_path+test_csv_file_name)
    return X_train, y_train, X_test, y_test

def create_and_test_basic_model(X_train, y_train, X_test, y_test):
    print('> Basic model')
    learning_rate = 0.01
    optimizer = 'SGD'
    epochs = 2
    batch_size = 32
    file_save_name = 'basic_model_lr_{}_optimizer_{}_epochs_{}_batch_{}'.format(learning_rate, optimizer, epochs, batch_size)
    this_model = Model(X_train, y_train, X_test, y_test, learning_rate=learning_rate, optimizer=optimizer, train_valid_split=0.2, activation_function='relu', epochs=epochs, batch_size=batch_size, loss_function='categorical_crossentropy', use_drop_out=False, file_save_name=file_save_name)
    this_model.train()

def test_different_activation_functions(X_train, y_train, X_test, y_test):
    print('> Testing activation functions')
    learning_rate = 0.01
    optimizer = 'SGD'
    epochs = 15
    batch_size = 32
    history_of_models = {}
    for activation_function in ['relu', 'sigmoid', 'tanh']:
        file_save_name = 'testing_activation_{}_epochs_{}'.format(activation_function, epochs)
        this_model = Model(X_train, y_train, X_test, y_test, learning_rate=learning_rate, optimizer=optimizer, train_valid_split=0.2, activation_function=activation_function, epochs=epochs, batch_size=batch_size, loss_function='categorical_crossentropy', use_drop_out=False, file_save_name=file_save_name)
        hist = this_model.train()
        history_of_models[activation_function] = hist
    draw_multiple_line_plots(history_of_models, 
        field_to_draw='val_acc', 
        title = 'accuracy on validation data', 
        y_label = 'accuracy', 
        x_label = 'epoch',
        file_save_name = 'acc_of_different_activation_functions_{}_epochs.png'.format(epochs))

def test_different_optimizers(X_train, y_train, X_test, y_test):
    print('> Testing optimizers')
    learning_rate = 0.01
    activation_function = 'relu'
    epochs = 15
    batch_size = 32
    history_of_models = {}
    for optimizer in ['SGD', 'Adam']:
        file_save_name = 'testing_optimizer_{}_epochs_{}'.format(optimizer, epochs)
        this_model = Model(X_train, y_train, X_test, y_test, learning_rate=learning_rate, optimizer=optimizer, train_valid_split=0.2, activation_function=activation_function, epochs=epochs, batch_size=batch_size, loss_function='categorical_crossentropy', use_drop_out=False, file_save_name=file_save_name)
        hist = this_model.train()
        history_of_models[optimizer] = hist

    draw_multiple_line_plots(history_of_models, 
        field_to_draw='val_acc', 
        title = 'accuracy on validation data', 
        y_label = 'accuracy', 
        x_label = 'epoch',
        file_save_name = 'acc_of_different_optimizers_{}_epochs.png'.format(epochs))

def test_having_dropout(X_train, y_train, X_test, y_test):
    print('> Testing dropout')
    learning_rate = 0.01
    activation_function = 'relu'
    epochs = 100
    batch_size = 32
    optimizer = 'SGD'
    history_of_models = {}
    for use_drop_out in [True, False]:
        file_save_name = 'testing_dropout_{}_epochs_{}'.format(str(use_drop_out), epochs)
        this_model = Model(X_train, y_train, X_test, y_test, learning_rate=learning_rate, optimizer=optimizer, train_valid_split=0.2, activation_function=activation_function, epochs=epochs, batch_size=batch_size, loss_function='categorical_crossentropy', use_drop_out=use_drop_out, file_save_name=file_save_name)
        hist = this_model.train()
        history_of_models['Dropout = '+str(use_drop_out)] = hist

    draw_multiple_line_plots(history_of_models, 
        field_to_draw='val_acc', 
        title = 'accuracy on validation data', 
        y_label = 'accuracy', 
        x_label = 'epoch',
        file_save_name = 'acc_of_validation_having_and_not_having_dropout_{}_epochs.png'.format(epochs))

    draw_multiple_line_plots(history_of_models, 
        field_to_draw='acc', 
        title = 'accuracy on train data', 
        y_label = 'accuracy', 
        x_label = 'epoch',
        file_save_name = 'acc_of_train_having_and_not_having_dropout_{}_epochs.png'.format(epochs))

def train_model(X_train, y_train, X_test, y_test):
    create_and_test_basic_model(X_train, y_train, X_test, y_test)
    test_different_activation_functions(X_train, y_train, X_test, y_test)
    test_different_optimizers(X_train, y_train, X_test, y_test)
    test_having_dropout(X_train, y_train, X_test, y_test)

def main():
    print('Read and Resize Data')
    X_train, y_train, X_test, y_test = read_data()
    print('Create and Train Model')
    train_model(X_train, y_train, X_test, y_test)

main()
