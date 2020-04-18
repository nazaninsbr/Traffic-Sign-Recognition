data_folder_path = '../gtsrb-german-traffic-sign/'

meta_csv_file_name = 'Meta.csv'
train_csv_file_name = 'Train.csv'
test_csv_file_name = 'Test.csv'

img_dim = 30
number_of_classes = 43

generated_files_path = '../Generated_Files/'

class DefaultObj: pass
default_params = DefaultObj()

default_params.batch_size = 128
default_params.learning_rate = 0.01
default_params.optimizer = "SGD"
default_params.activation_function = "relu"
default_params.train_valid_split = 0.2
default_params.loss_function = "categorical_crossentropy"
