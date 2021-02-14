import os
import sys
import pprint
import inspect
import unittest

import numpy as np
import tensorflow as tf
import tensorflow.keras as keras

sys.path.append('../')
sys.path.append('../../')

from test_helpers import convert_keras_to_tflite 
from test_helpers import build_ikann_graph
from test_helpers import build_ikann_graph
from test_helpers import invoke_tensorflow_lite
from test_helpers import build_and_execute_c_code
from test_helpers import generate_c_code
from test_helpers import compare_results_between_tflite_and_ikann

pp = pprint.PrettyPrinter(indent=4)

class Conv2dTestCase(unittest.TestCase):

	def test_single_conv2d(self):
		
		model = keras.Sequential([
			keras.layers.Conv2D(
			5, 
			kernel_size=(3, 3), 
			strides=(1, 1),
			input_shape=(3, 3, 2),
			activation='relu'),
			keras.layers.Flatten(),
			keras.layers.Dense(10, activation='relu'),
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
				ikannGraph, templatePath='../../templates', outputPath='conv2d')

		## Comparing results between original tflite model and generated ikann model
		inputData = np.array([
			[
				[1., 2., 3.],
				[4., 5., 6.],
				[7., 8., 9.],
			],
			[
				[4., 5., 6.],
				[1., 2., 3.],
				[7., 8., 9.],
			],
		], dtype=np.float32)
		inputData = np.reshape(inputData, (1, 3, 3, 2))

		outputResult = invoke_tensorflow_lite(
			tfliteModel, inputData).flatten().round(decimals=8)

		# Compile generated model and executed it
		build_and_execute_c_code(codePath='conv2d')

		# Compare output results
		compare_results_between_tflite_and_ikann(outputResult)

		os.remove('model_output.txt')

	# def test_multiple_conv2d(self):
	#     pass
if __name__ == '__main__':
	unittest.main()
















