from data_reader import read_image_files
from constants import data_folder_path, train_csv_file_name, test_csv_file_name
from model import Model 

def read_data():
    X_train, y_train = read_image_files(csv_file_name=data_folder_path+train_csv_file_name)
    X_test, y_test = read_image_files(csv_file_name=data_folder_path+test_csv_file_name)
    return X_train, y_train, X_test, y_test

def train_model(X_train, y_train, X_test, y_test):
    this_model = Model(X_train, y_train, X_test, y_test, learning_rate=0.1, optimizer='SGD', train_valid_split=0.2, activation_function='relu', epochs=2, batch_size=32, loss_function='categorical_crossentropy')
    this_model.train()

def main():
    print('Read and Resize Data')
    X_train, y_train, X_test, y_test = read_data()
    print('Create and Train Model')
    train_model(X_train, y_train, X_test, y_test)

main()