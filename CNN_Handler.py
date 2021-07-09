### Imports from CNN notebook
from __future__ import print_function
# What version of Python do you have?
import sys

import tensorflow.keras
import pandas as pd
#import sklearn as sk
import tensorflow as tf
import os

import random

import numpy as np
from tensorflow.keras.models import load_model

import Individual_v1 as individual
import Genetic_Algorithm as ga


random.seed(865)




class CNN_handler:
    def __init__(self):
        #os.chdir("D:\Spring_Tasks_21\FNS_CNN_v1")
        print("CNN handler")
        #self.g_a = 'N/A'


    def keras_debug(self):
        print(f"Tensor Flow Version: {tf.__version__}")
        print(f"Keras Version: {tensorflow.keras.__version__}")
        print()
        print(f"Python {sys.version}")
        print(f"Pandas {pd.__version__}")
        #print(f"Scikit-Learn {sk.__version__}")
        gpu = len(tf.config.list_physical_devices('GPU')) > 0
        print("GPU is", "available" if gpu else "NOT AVAILABLE")

    def load_cnn_model(self, model_string, name = 'model'):
        setattr(self, name, load_model(model_string))
        #self.model = load_model(model_string)

    def cnn_predict(self, child_ind, variable_cassette_preprocess = True, material_matrix_val = ""):
        ### Putting material matrix into form expected by CNN
        if variable_cassette_preprocess:
            material_cnn_form_preproccessed = self.preprocess_variable_cassette_child_object(child_ind)
        else:
            material_cnn_form_preproccessed = getattr(child_ind, material_matrix_val)

        material_cnn_form, _ = self.build_3x3_fns_data_variable_cassette_A(single_case=True,
                                                                           single_case_material_matrix = material_cnn_form_preproccessed)

        return self.model.predict(material_cnn_form)[0][0]

    def return_pattern(self, pat_a, pat_b, pat_c, plate_number):
        row_1 = np.array([pat_c[plate_number], pat_b[plate_number],
                          pat_b[plate_number], pat_a[plate_number]])
        return np.reshape(row_1, (2, 2))

    def build_3x3_fns_data_v1(self,
                              data_file_string,
                              zones=[1, 2, 3],
                              number_of_plates=60,
                              number_of_materials=4,
                              void_value = 1):
        data_ = np.loadtxt(data_file_string,
                           delimiter=",")
        X = data_[:,0:150]
        Y = data_[:, 150:151]

        pattern_shapes_list = []
        for fns_pattern_count, pattern in enumerate(X):

            ### Setting patterns for this dataset
            pattern_1a = pattern[0:10]
            pattern_1b = pattern[10:30]
            pattern_1c = pattern[30:50]
            pattern_2a = pattern[50:70]
            pattern_2b = pattern[70:90]
            pattern_2c = pattern[90:110]
            pattern_3a = np.array([void_value, void_value, void_value, void_value, void_value,
                                   void_value, void_value, void_value, void_value, void_value,
                                   void_value, void_value, void_value, void_value, void_value,
                                   void_value, void_value, void_value, void_value, void_value])
            pattern_3b = pattern[110:130]
            pattern_3c = pattern[130:150]

            np_four_10_array = np.array([void_value, void_value, void_value, void_value, void_value,
                                         void_value, void_value, void_value, void_value, void_value])
            pattern_1a = np.concatenate((np_four_10_array, pattern_1a))

            for plate in range(number_of_plates):
                if plate == 0:
                    pattern_array = self.return_pattern(pattern_1a, pattern_1b, pattern_1c, plate)
                    continue

                if plate < 20:
                    pattern_ = self.return_pattern(pattern_1a, pattern_1b, pattern_1c, plate)
                    pattern_array = np.concatenate((pattern_array, pattern_))
                if plate < 40 and plate >= 20:
                    pattern_ = self.return_pattern(pattern_2a, pattern_2b, pattern_2c, plate - 20)
                    pattern_array = np.concatenate((pattern_array, pattern_))
                if plate >= 40:
                    pattern_ = self.return_pattern(pattern_3a, pattern_3b, pattern_3c, plate - 40)
                    pattern_array = np.concatenate((pattern_array, pattern_))

            pattern_array_reshaped = np.reshape(pattern_array, (60, 2, 2))
            # print(pattern_array_reshaped.shape)
            ### One hot encoding the 3x3x60 array into a 3x3x60x5 array. Stackoverflow magic:
            ### https://stackoverflow.com/questions/36960320/convert-a-2d-matrix-to-a-3d-one-hot-matrix-numpy
            ### 4 types of materials is hard coded inside np.arange(#)
            pattern_array_reshaped_1he = (np.arange(number_of_materials) == pattern_array_reshaped[..., None] - 1).astype(int)

            pattern_shapes_list.append(np.array(pattern_array_reshaped_1he))
            #print(fns_pattern_count, pattern_array_reshaped_1he.shape)
        pattern_shapes_array = np.array(pattern_shapes_list)
        return pattern_shapes_array, Y




    def train_network(self):
        pass
    #preprocess_variable_cassette_raw_ga_data(self, "test_variable_cassette_2a.csv", output_file_name="test_variable_data.csv")
    def preprocess_variable_cassette_child_object(self, child_object,
                                                  fixed_plate_count=120,
                                                  material_matricss = ['material_matrix', 'material_matrix_cassette_2A']):
        material_matrix_master_list = []
        for material_matrix in material_matricss:
            material_matrix_master_list.extend(getattr(child_object, material_matrix))

        size_of_all_patterns = len(material_matrix_master_list)
        #print('len(material_matrix)',len(material_matrix_master_list))
        size_of_variable_cassette = size_of_all_patterns - fixed_plate_count
        new_list_with_variable_cassette = []
        for count, value in enumerate(material_matrix_master_list):
            new_list_with_variable_cassette.append(int(value[0]))
            if count == 119:
                for _ in range(30 - size_of_variable_cassette):
                    new_list_with_variable_cassette.append(1)
        #print("final list", len(new_list_with_variable_cassette),
        #      np.array(new_list_with_variable_cassette))


        return np.array(new_list_with_variable_cassette)

    def preprocess_variable_cassette_raw_ga_data(self, data_file, output_file_name="", fixed_plate_count=120):
        with open(data_file, 'r') as data_file_obj:
            data_master_list = []
            for line in data_file_obj:
                list_ = line.split(",")

                # print(len(list_))
                # print(list_)
                new_list = []
                for value in list_:
                    if value.strip() == "":
                        continue
                    new_list.append(value.strip())
                # print(len(new_list), len(new_list) - fixed_plate_count - 1)
                # print(new_list)
                size_of_all_patterns = len(new_list) - 1

                size_of_variable_cassette = size_of_all_patterns - fixed_plate_count

                new_list_with_variable_cassette = []
                for count, value in enumerate(new_list):
                    new_list_with_variable_cassette.append(value)
                    if count == 120:
                        for _ in range(30 - size_of_variable_cassette):
                            new_list_with_variable_cassette.append('1')
                # print("final list", len(new_list_with_variable_cassette), new_list_with_variable_cassette)
                data_master_list.append(new_list_with_variable_cassette)
            # print(len(data_master_list))
            with open(output_file_name, 'w') as output_file_obj:
                for _ in data_master_list:
                    write_string = ""
                    for __ in _:
                        write_string +=  str(__) + ","
                    output_file_obj.write(write_string[:-1] + "\n")

    def build_CNN_ga_object(self, options_dict):
        self.g_a = ga.genetic_algorithm(options_dict)

    def build_3x3_fns_data_variable_cassette_A(self,
                              data_file_string = '',
                              number_of_plates=60,
                              number_of_materials=4,
                              void_value = 1,
                              number_of_plates_in_fixed_section = 120,
                              single_case = False,
                              single_case_material_matrix = []):
        if single_case == False:
            data_ = np.loadtxt(data_file_string,
                               delimiter=",")
            X = data_[:,1:151]
            Y = data_[:, 0:1]
            #print("X matrix", X)
            #print(type(X))

        else:
            X = single_case_material_matrix
            Y = "n/a"
            #print("X matrix", X)
            X = np.array([single_case_material_matrix])

        pattern_shapes_list = []
        #print(X)
        for fns_pattern_count, pattern in enumerate(X):

            ### Setting patterns for this dataset

            pattern_1b = pattern[0:20]
            pattern_1c = pattern[20:40]

            pattern_2b = pattern[40:60]
            pattern_2c = pattern[60:80]
            pattern_3a = np.array([void_value, void_value, void_value, void_value, void_value,
                                   void_value, void_value, void_value, void_value, void_value,
                                   void_value, void_value, void_value, void_value, void_value,
                                   void_value, void_value, void_value, void_value, void_value])
            pattern_3b = pattern[80:100]
            pattern_3c = pattern[100:120]

            pattern_1a = pattern[120:130]
            pattern_2a = pattern[130:150]

            np_four_10_array = np.array([void_value, void_value, void_value, void_value, void_value,
                                         void_value, void_value, void_value, void_value, void_value])
            pattern_1a = np.concatenate((np_four_10_array, pattern_1a))

            for plate in range(number_of_plates):
                if plate == 0:
                    pattern_array = self.return_pattern(pattern_1a, pattern_1b, pattern_1c, plate)
                    continue

                if plate < 20:
                    pattern_ = self.return_pattern(pattern_1a, pattern_1b, pattern_1c, plate)
                    pattern_array = np.concatenate((pattern_array, pattern_))
                if plate < 40 and plate >= 20:
                    pattern_ = self.return_pattern(pattern_2a, pattern_2b, pattern_2c, plate - 20)
                    pattern_array = np.concatenate((pattern_array, pattern_))
                if plate >= 40:
                    pattern_ = self.return_pattern(pattern_3a, pattern_3b, pattern_3c, plate - 40)
                    pattern_array = np.concatenate((pattern_array, pattern_))

            pattern_array_reshaped = np.reshape(pattern_array, (60, 2, 2))
            # print(pattern_array_reshaped.shape)
            ### One hot encoding the 3x3x60 array into a 3x3x60x(n) array. Stackoverflow magic:
            ### https://stackoverflow.com/questions/36960320/convert-a-2d-matrix-to-a-3d-one-hot-matrix-numpy
            pattern_array_reshaped_1he = (np.arange(number_of_materials) == pattern_array_reshaped[..., None] - 1).astype(
                int)

            pattern_shapes_list.append(np.array(pattern_array_reshaped_1he))
            #print(fns_pattern_count, pattern_array_reshaped_1he.shape)
        pattern_shapes_array = np.array(pattern_shapes_list)
        return pattern_shapes_array, Y
def main():
    print("HEEERRE!!")
    os.chdir("D:\Spring_Tasks_21\FNS_CNN_v1")
    cnn_ = CNN_handler()
    cnn_.preprocess_variable_cassette_raw_ga_data("test_variable_cassette_2a.csv",
                                                  output_file_name="test_variable_data.csv")
    x, y = cnn_.build_3x3_fns_data_variable_cassette_A("test_variable_data.csv")
    print("HEEERRE!! 2")
    np.save("test_variable_data_X", x)
    np.save("test_variable_data_Y", y)

    cnn_.load_cnn_model("keff_model_7_chkpnt_.2993-0.005611.hdf5")

    x_test = np.load("X_keff_validation_data.csv.npy")
    y_test = np.load("Y_keff_validation_data.csv.npy")
    pred_vals = cnn_.model.predict(x_test)
    # print(pred_vals)
    for pred_val, test_val in zip(pred_vals, y_test):
        print(pred_val[0], test_val[0])


if __name__ == '__main__':
    main()


"""os.chdir("D:\Spring_Tasks_21\FNS_CNN_v1")
cnn_ = CNN_handler()
cnn_.preprocess_variable_cassette_raw_ga_data("test_variable_rep_thru_gen_45.csv", output_file_name="test_variable_rep_data.csv")
x, y = cnn_.build_3x3_fns_data_variable_cassette_A("test_variable_rep_data.csv")
np.save("test_variable_rep_data_X", x)
np.save("test_variable_rep_data_Y", y)"""

#cnn_.load_cnn_model("keff_model_6_model_0_chkpnt_.338-0.007420.hdf5")

"""x_test = np.load("test_variable_rep_data_X.npy")
y_test = np.load("test_variable_rep_data_Y.npy")
pred_vals = cnn_.model.predict(x_test)
print('pred, real')
for pred_val, test_val in zip(pred_vals, y_test):
    print(pred_val[0], test_val[0])

model_checkpoint =\
    tensorflow.keras.callbacks.ModelCheckpoint(filepath="best_model",
                                               monitor='val_loss',
                                               save_best_only=True)

cnn_.model.fit(x = x_test,
               y = y_test,
               batch_size= 1000,
               shuffle = True,
               validation_split = 0.25,
               epochs = 25,
               callbacks = [model_checkpoint])

cnn_.model = load_model("best_model")
pred_vals = cnn_.model.predict(x_test)
print('pred, real')
for pred_val, test_val in zip(pred_vals, y_test):
    print(pred_val[0], test_val[0])"""