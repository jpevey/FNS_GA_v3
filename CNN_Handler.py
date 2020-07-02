### Imports from CNN notebook
from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.layers import LeakyReLU
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, BatchNormalization
from keras.layers import Conv3D, MaxPooling3D
from keras import backend as K
from keras import optimizers
from matplotlib import pyplot as plt
from IPython.display import clear_output
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from keras_radam import RAdam
import pandas as pd
import numpy as np
import random
import os
import collections
import math
import random
import copy
import time
random.seed(865)




class CNN_handler:
    def __init__(self, model_string):
        self.model = load_model(model_string, custom_objects={'RAdam': RAdam})

    ### Creating data
    def build_input_data_reshaped(self, data_array, X_val, Y_val, Z_val, channels):
        data_array_reshaped = self.build_input_data(data_array)
        # print("data_Array_reshaped", data_array_reshaped.shape)
        data_array_reshaped_final = np.reshape(data_array_reshaped, (len(data_array), X_val, Y_val, Z_val, channels))
        # print("reshaped the data")
        return data_array_reshaped_final

    def build_input_data(self, data_array):
        data_ = []
        for item in data_array:
            # print("here_1", len(item))
            data_.append(self.break_materials_into_rows(item,
                                                        number_of_materials=121,
                                                        materials_per_row=11,
                                                        number_of_materials_1he=2,
                                                        one_hot_encode=True))
        return np.array(data_)

    ### This function takes a list of individuals and creates an array that can be fed into build_input_data
    def build_individuals_array(self, individuals, generation):
        ind_list = []
        for individual in individuals:

            ind_pattern = []
            for pattern in individual.material_matrix:
                ind_pattern.extend(pattern)
            # print("here_2", len(ind_pattern))
            ind_list.append(np.array(ind_pattern))

        return np.array(ind_list)

    # for fns_pattern_count, pattern in enumerate(X):
    def break_materials_into_rows(self,
                                  pattern,
                                  number_of_materials_1he,
                                  number_of_materials,
                                  materials_per_row, one_hot_encode=False):

        data_ = []
        for _ in range(int(number_of_materials / materials_per_row)):

            pat = pattern[_ * materials_per_row:materials_per_row + _ * materials_per_row]

            if one_hot_encode == True:
                pat = self.one_hot_encode_array_of_numbers(pat, number_of_materials=number_of_materials_1he)

            data_.append(np.array(pat))
        return np.array(data_)

    def one_hot_encode_array_of_numbers(self, array, number_of_materials):
        array_ = []
        for item in array:
            # print("item", item)
            array_.append(self.one_hot_encode(item, number_of_materials))
        return array_

    def one_hot_encode(self, material_value, number_of_materials):
        zeros_ = np.zeros(number_of_materials)
        material_array = zeros_
        material_array[int(material_value) - 1] = 1
        return material_array