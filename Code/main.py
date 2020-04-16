from data_reader import read_image_files
from constants import data_folder_path, train_csv_file_name, test_csv_file_name

def read_data():
    X_train, y_train = read_image_files(csv_file_name=data_folder_path+train_csv_file_name)
    X_test, y_test = read_image_files(csv_file_name=data_folder_path+test_csv_file_name)
    return X_train, y_train, X_test, y_test

def main():
    print('Read and Resize Data')
    X_train, y_train, X_test, y_test = read_data()

main()