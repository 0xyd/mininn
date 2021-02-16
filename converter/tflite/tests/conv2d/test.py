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
from test_helpers import invoke_tensorflow_lite
from test_helpers import build_and_execute_c_code
from test_helpers import generate_c_code
from test_helpers import compare_results_between_tflite_and_ikann

pp = pprint.PrettyPrinter(indent=4)

inputData1Channel = np.zeros(
	(1, 5, 5, 1), dtype=np.float32)

inputData1Channel[0, 0, :, 0] = np.array([1., 2., 3., 4., 5.])
inputData1Channel[0, 1, :, 0] = np.array([6., 7., 8., 9., 10.])
inputData1Channel[0, 2, :, 0] = np.array([11., 12., 13., 14., 15.])
inputData1Channel[0, 3, :, 0] = np.array([16., 17., 18., 19., 20.])
inputData1Channel[0, 4, :, 0] = np.array([21., 22., 23., 24., 25.])

inputData3Channel = np.zeros(
	(1, 5, 5, 3), dtype=np.float32)

inputData3Channel[0, 0, :, 0] = np.array([1., 2., 3., 4., 5.])
inputData3Channel[0, 1, :, 0] = np.array([6., 7., 8., 9., 10.])
inputData3Channel[0, 2, :, 0] = np.array([11., 12., 13., 14., 15.])
inputData3Channel[0, 3, :, 0] = np.array([16., 17., 18., 19., 20.])
inputData3Channel[0, 4, :, 0] = np.array([21., 22., 23., 24., 25.])

inputData3Channel[0, 0, :, 1] = np.array([26., 27., 28., 29., 30.])
inputData3Channel[0, 1, :, 1] = np.array([31., 32., 33., 34., 35.])
inputData3Channel[0, 2, :, 1] = np.array([36., 37., 38., 39., 40.])
inputData3Channel[0, 3, :, 1] = np.array([41., 42., 43., 44., 45.])
inputData3Channel[0, 4, :, 1] = np.array([46., 47., 48., 49., 50.])

inputData3Channel[0, 0, :, 2] = np.array([51., 52., 53., 54., 55.])
inputData3Channel[0, 1, :, 2] = np.array([56., 57., 58., 59., 60.])
inputData3Channel[0, 2, :, 2] = np.array([61., 62., 63., 64., 65.])
inputData3Channel[0, 3, :, 2] = np.array([66., 67., 68., 69., 70.])
inputData3Channel[0, 4, :, 2] = np.array([71., 72., 73., 74., 75.])

class Conv2dTestCase(unittest.TestCase):

	def generate_code(self, ikannGraph):

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
				outputPath='conv2d')


	def test_codegen_with_1_channel(self):

		print("input for per channel:")
		for c in range(1):
			pp.pprint(inputData1Channel[0, :, :, c])
		print(f"inputData.shape: {inputData1Channel.shape}")

		model = keras.Sequential([
			keras.layers.Conv2D(
				1, 
				kernel_size=(5, 5), 
				strides=(1, 1),
				input_shape=(5, 5, 1),
				),
		])

		tfliteModel = convert_keras_to_tflite(model)
		ikannGraph = build_ikann_graph(tfliteModel)

		self.generate_code(ikannGraph)

		outputResult = invoke_tensorflow_lite(
			tfliteModel, inputData1Channel)
		print(f"outputResult.shape: {outputResult.shape}")
		print("output result:")
		outputResult = outputResult.flatten().tolist()

		# Compile generated model and executed it
		build_and_execute_c_code(
			codePath='conv2d', mainFileName='main_2.c')

		compare_results_between_tflite_and_ikann(
			outputResult, threshold=1e-3)

		os.remove('model_output.txt')


	def test_codegen_with_3_channels(self):

		print("input for per channel:")
		for c in range(3):
			pp.pprint(inputData3Channel[0, :, :, c])
		print(f"inputData.shape: {inputData3Channel.shape}")

		numFilters = 10
		
		model = keras.Sequential([
			keras.layers.Conv2D(
				numFilters, 
				kernel_size=(5, 5), 
				strides=(1, 1),
				input_shape=(5, 5, 3),
				),
		])

		# filterWeights, bias = model.get_layer(index=0).get_weights()
		# print(filterWeights.shape)

		# filterWeights[0, 0, 0, 0] = 1
		# filterWeights[0, 1, 0, 0] = 2
		# filterWeights[0, 2, 0, 0] = 3
		# filterWeights[0, 3, 0, 0] = 4
		# filterWeights[0, 4, 0, 0] = 5
		# filterWeights[1, 0, 0, 0] = 6
		# filterWeights[1, 1, 0, 0] = 7
		# filterWeights[1, 2, 0, 0] = 8
		# filterWeights[1, 3, 0, 0] = 9
		# filterWeights[1, 4, 0, 0] = 10
		# filterWeights[2, 0, 0, 0] = 11
		# filterWeights[2, 1, 0, 0] = 12
		# filterWeights[2, 2, 0, 0] = 13
		# filterWeights[2, 3, 0, 0] = 14
		# filterWeights[2, 4, 0, 0] = 15
		# filterWeights[3, 0, 0, 0] = 16
		# filterWeights[3, 1, 0, 0] = 17
		# filterWeights[3, 2, 0, 0] = 18
		# filterWeights[3, 3, 0, 0] = 19
		# filterWeights[3, 4, 0, 0] = 20
		# filterWeights[4, 0, 0, 0] = 21
		# filterWeights[4, 1, 0, 0] = 22
		# filterWeights[4, 2, 0, 0] = 23
		# filterWeights[4, 3, 0, 0] = 24
		# filterWeights[4, 4, 0, 0] = 25
		# filterWeights[0, 0, 1, 0] = 26
		# filterWeights[0, 1, 1, 0] = 27
		# filterWeights[0, 2, 1, 0] = 28
		# filterWeights[0, 3, 1, 0] = 29
		# filterWeights[0, 4, 1, 0] = 30
		# filterWeights[1, 0, 1, 0] = 31
		# filterWeights[1, 1, 1, 0] = 32
		# filterWeights[1, 2, 1, 0] = 33
		# filterWeights[1, 3, 1, 0] = 34
		# filterWeights[1, 4, 1, 0] = 35
		# filterWeights[2, 0, 1, 0] = 36
		# filterWeights[2, 1, 1, 0] = 37
		# filterWeights[2, 2, 1, 0] = 38
		# filterWeights[2, 3, 1, 0] = 39
		# filterWeights[2, 4, 1, 0] = 40
		# filterWeights[3, 0, 1, 0] = 41
		# filterWeights[3, 1, 1, 0] = 42
		# filterWeights[3, 2, 1, 0] = 43
		# filterWeights[3, 3, 1, 0] = 44
		# filterWeights[3, 4, 1, 0] = 45
		# filterWeights[4, 0, 1, 0] = 46
		# filterWeights[4, 1, 1, 0] = 47
		# filterWeights[4, 2, 1, 0] = 48
		# filterWeights[4, 3, 1, 0] = 49
		# filterWeights[4, 4, 1, 0] = 50

		# model.get_layer(index=0).set_weights([filterWeights, bias])

		tfliteModel = convert_keras_to_tflite(model)
		ikannGraph = build_ikann_graph(tfliteModel)

		self.generate_code(ikannGraph)
		
		outputResult = invoke_tensorflow_lite(
			tfliteModel, inputData3Channel)
		print(f"outputResult.shape: {outputResult.shape}")
		outputResult = outputResult.flatten().tolist()
		print('output result:')
		print(outputResult)

		# Compile generated model and executed it
		build_and_execute_c_code(codePath='conv2d')

		# # Compare output results
		compare_results_between_tflite_and_ikann(
			outputResult, threshold=1e-3)

		os.remove('model_output.txt')

	# def test_multiple_conv2d(self):
	#     pass
if __name__ == '__main__':
	unittest.main()

