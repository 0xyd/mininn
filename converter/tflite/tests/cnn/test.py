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

class CNNTestCase(unittest.TestCase):

	def test_codegen(self):


		inputData = np.zeros((1, 3, 3, 3), dtype=np.float32)


		inputData[0, 0, :, 0] = np.array([1., 2., 3.])
		inputData[0, 1, :, 0] = np.array([4., 5., 6.])
		inputData[0, 2, :, 0] = np.array([7., 8., 9.])
		inputData[0, 0, :, 1] = np.array([10., 11., 12.])
		inputData[0, 1, :, 1] = np.array([13., 14., 15.])
		inputData[0, 2, :, 1] = np.array([16., 17., 18.])
		inputData[0, 0, :, 2] = np.array([19., 20., 21.])
		inputData[0, 1, :, 2] = np.array([22., 23., 24.])
		inputData[0, 2, :, 2] = np.array([25., 26., 27.])

		model = keras.Sequential([
			keras.layers.Conv2D(
				5, 
				kernel_size=(2, 2), 
				strides=(1, 1),
				input_shape=(3, 3, 3),
				# kernel_initializer='zeros',
				),
			keras.layers.Flatten(),
			keras.layers.Dense(5),
				# kernel_initializer='zeros'),
		])

		# filterWeights, bias = model.get_layer(index=0).get_weights()
		# print('weights.shape:')
		# print(filterWeights.shape)

		# filterWeights[:, :, 0, 0] = np.ones((2, 2), dtype=np.float32)

		# filterWeights[0, :, 1, 0] = np.array([1, 2])
		# filterWeights[1, :, 1, 0] = np.array([3, 4])
		# # filterWeights[:, :, 1, 0] = np.ones((2, 2), dtype=np.float32)

		# filterWeights[:, :, 0, 1] = np.full((2,2), 2)
		# filterWeights[:, :, 1, 1] = np.full((2,2), 2)
		# filterWeights[:, :, 0, 2] = np.full((2,2), 3)
		# filterWeights[:, :, 1, 2] = np.full((2,2), 3)
		# filterWeights[:, :, 0, 3] = np.full((2,2), 4)
		# filterWeights[:, :, 1, 3] = np.full((2,2), 4)

		# model.get_layer(index=0).set_weights([filterWeights, bias])

		# denseWeights, bias = model.get_layer(index=2).get_weights()
		# print('dense weights.shape:')
		# print(denseWeights.shape)

		# denseWeights[:, 0] = np.array(
		# 	[
		# 		1, 2, 3, 4, 5, 6, 7, 8, 
		# 		9, 10, 11, 12, 13, 14, 15, 16
		# 	 ],
		# 	 dtype=np.float32)
		# # denseWeights[:, 0] = np.ones((16,), dtype=np.float32)
		# denseWeights[:, 1] = np.full((16,), 2)
		# denseWeights[:, 2] = np.full((16,), 3)
		# denseWeights[:, 3] = np.full((16,), 4)
		# denseWeights[:, 4] = np.full((16,), 5)

		# model.get_layer(index=2).set_weights([denseWeights, bias])
		
		tfliteModel = convert_keras_to_tflite(model)
		outputResult = invoke_tensorflow_lite(
			tfliteModel, inputData)
		print(f"outputResult.shape: {outputResult.shape}")
		outputs = outputResult[0].tolist()
		# pp.pprint(outputResult)

		# for i in range(5):
		# 	pp.pprint(outputResult[:, :, :, i])

		# print('flatten outputResult!!')
		# pp.pprint(outputResult.flatten())

		# inputData[0, 0, :, 0] = np.array([1., 2., 3., 4., 5.])
		# inputData[0, 1, :, 0] = np.array([6., 7., 8., 9., 10.])
		# inputData[0, 2, :, 0] = np.array([11., 12., 13., 14., 15.])
		# inputData[0, 3, :, 0] = np.array([16., 17., 18., 19., 20.])
		# inputData[0, 4, :, 0] = np.array([21., 22., 23., 24., 25.])

		# inputData[0, 0, :, 1] = np.array([26., 27., 28., 29., 30.])
		# inputData[0, 1, :, 1] = np.array([31., 32., 33., 34., 35.])
		# inputData[0, 2, :, 1] = np.array([36., 37., 38., 39., 40.])
		# inputData[0, 3, :, 1] = np.array([41., 42., 43., 44., 45.])
		# inputData[0, 4, :, 1] = np.array([46., 47., 48., 49., 50.])

		# inputData[0, 0, :, 2] = np.array([51., 52., 53., 54., 55.])
		# inputData[0, 1, :, 2] = np.array([56., 57., 58., 59., 60.])
		# inputData[0, 2, :, 2] = np.array([61., 62., 63., 64., 65.])
		# inputData[0, 3, :, 2] = np.array([66., 67., 68., 69., 70.])
		# inputData[0, 4, :, 2] = np.array([71., 72., 73., 74., 75.])
		# print(f"inputData.shape: {inputData.shape}")
		
		# numFilters = 10

		# model = keras.Sequential([
		# 	keras.layers.Conv2D(
		# 		numFilters, 
		# 		kernel_size=(4, 4), 
		# 		strides=(1, 1),
		# 		input_shape=(5, 5, 3)),
		# 	keras.layers.MaxPool2D(
		# 		pool_size=(1, 1), 
		# 		strides=(1, 1)),
		# 	# keras.layers.Flatten(),
		# 	# keras.layers.Dense(10),
		# 	# keras.layers.Softmax()
		# ])

		# tfliteModel = convert_keras_to_tflite(model)
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
				outputPath='cnn')


		# print("flatten outputResult:")
		# pp.pprint(outputResult.flatten())
		# outputs = []
		# for i in range(1):
		# 	outputs.extend(outputResult[i].tolist())
		# # # 			outputResult[i, :, :, j].flatten().tolist())
		# # # pp.pprint(outputs)

		# Compile generated model and executed it
		build_and_execute_c_code(codePath='cnn')

		# Compare output results
		print('outputs from tflite:')
		pp.pprint(outputs)
		compare_results_between_tflite_and_ikann(outputs)
		os.remove('model_output.txt')

	# def test_multiple_conv2d(self):
	#     pass
if __name__ == '__main__':
	unittest.main()

