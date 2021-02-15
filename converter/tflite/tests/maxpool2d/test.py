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

class MaxPool2DTestCase(unittest.TestCase):

	def test_codegen(self):

		model = keras.Sequential([
			keras.layers.InputLayer(input_shape=(5, 5, 3)),
			keras.layers.MaxPool2D(pool_size=(3, 3), strides=(1, 1)),
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
				outputPath='maxpool2d')

		inputData = np.zeros(
			(1, 5, 5, 3), dtype=np.float32)

		inputData[0, 0, :, 0] = np.array([1., 2., 3., 4., 5.])
		inputData[0, 1, :, 0] = np.array([6., 7., 8., 9., 10.])
		inputData[0, 2, :, 0] = np.array([11., 12., 13., 14., 15.])
		inputData[0, 3, :, 0] = np.array([16., 17., 18., 19., 20.])
		inputData[0, 4, :, 0] = np.array([21., 22., 23., 24., 25.])

		inputData[0, 0, :, 1] = np.array([6., 7., 8., 9., 10.])
		inputData[0, 1, :, 1] = np.array([1., 2., 3., 4., 5.])
		inputData[0, 2, :, 1] = np.array([11., 12., 13., 14., 15.])
		inputData[0, 3, :, 1] = np.array([21., 22., 23., 24., 25.])
		inputData[0, 4, :, 1] = np.array([16., 17., 18., 19., 20.])

		inputData[0, 0, :, 2] = np.array([11., 12., 13., 14., 15.])
		inputData[0, 1, :, 2] = np.array([6., 7., 8., 9., 10.])
		inputData[0, 2, :, 2] = np.array([1., 2., 3., 4., 5.])
		inputData[0, 3, :, 2] = np.array([16., 17., 18., 19., 20.])
		inputData[0, 4, :, 2] = np.array([21., 22., 23., 24., 25.])

		outputResult = invoke_tensorflow_lite(
			tfliteModel, inputData)

		_outputResult = []
		for c in range(3):
			for r in range(3):
				d = outputResult[0, r, :, c].tolist()
				_outputResult.extend(d)

		build_and_execute_c_code(codePath='maxpool2d')

		# Compare output results
		compare_results_between_tflite_and_ikann(_outputResult)

		os.remove('model_output.txt')

if __name__ == '__main__':
	unittest.main()
