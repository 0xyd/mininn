import os
import sys
import pprint
import pathlib
import unittest

import numpy as np
import tensorflow as tf
import tensorflow.keras as keras

sys.path.append('../')
sys.path.append('../../')

from test_helpers import convert_keras_to_tflite 
from test_helpers import build_ikann_graph
from test_helpers import invoke_tensorflow_lite
from test_helpers import build_and_execute_c_code
from test_helpers import generate_c_code
from test_helpers import compare_results_between_tflite_and_ikann

pp = pprint.PrettyPrinter(indent=4)


class FcnnSigmTestCase(unittest.TestCase):


    def test_codegen(self):

        model = keras.Sequential([
            keras.layers.InputLayer(input_shape=(5)),
            keras.layers.Dense(5, activation='sigmoid'),
            keras.layers.Dense(10, activation='sigmoid'),
            keras.layers.Dense(5, activation='sigmoid'),
            keras.layers.Dense(10, activation='sigmoid'),
            keras.layers.Dense(5, activation='sigmoid'),
            keras.layers.Softmax()
        ])

        tfliteModel = convert_keras_to_tflite(model)
        ikannGraph = build_ikann_graph(tfliteModel)

        # Template path looks different when caller of test is local or global.
        # First condition is when we test the current test only
        # Second condition is when we test all cases
        if os.path.exists('../../../templates'):
            generate_c_code(
                ikannGraph, 
                templatePath='../../../templates',)
        else:
            generate_c_code(
                ikannGraph, 
                templatePath='../../templates', 
                outputPath='fcnn_sigmoid')

        inputData = np.array([[1., 2., 3., 4., 5.]], dtype=np.float32)

        outputResult = invoke_tensorflow_lite(
            tfliteModel, inputData).flatten().round(decimals=8)

        build_and_execute_c_code(codePath='fcnn_sigmoid')

        # Compare output results
        compare_results_between_tflite_and_ikann(outputResult)

        os.remove('model_output.txt')

if __name__ == '__main__':
    unittest.main()

