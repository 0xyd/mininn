import sys
import pprint
import pathlib
import subprocess
import unittest

import numpy as np
import tensorflow as tf
import tensorflow.keras as keras

sys.path.append('../../')

from cgen.graph import iKannGraph, FlatbufferToDict
from cgen.snippet import CSnippetGenerator

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

		converter = tf.lite.TFLiteConverter.from_keras_model(model)
		# tfModelsDir = pathlib.Path('.')
		tfliteModel = converter.convert()
		# tfliteFile = tfModelsDir/"model.tflite"
		# tfliteFile.write_bytes(tfliteModel)

		ikannGraph = iKannGraph()
		graph = ikannGraph.parse(tfliteModel)

		cGenerator = CSnippetGenerator(templatePath='../../../templates')
		cGenerator.build_code(ikannGraph)

		## Comparing results between original tflite model and generated ikann model
		interpreter = tf.lite.Interpreter(model_content=tfliteModel)
		interpreter.allocate_tensors()
		inputDetails = interpreter.get_input_details()
		outputDetails = interpreter.get_output_details()

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
		interpreter.set_tensor(inputDetails[0]['index'], inputData)
		interpreter.invoke()
		outputResult = interpreter.get_tensor(
		outputDetails[0]['index']).flatten(
			).round(decimals=8)

		pp.pprint(outputResult)

		# Compile generated model and executed it
		subprocess.run(['make'], check=True)
		subprocess.run(['./hello'], check=True)

		# Compare output results
		with open('model_output.txt') as f:
			cModelOutput = [float(v) for v in f.readline().split(',') if len(v) > 0]

		for ov, cv in zip(outputResult, cModelOutput):
			if abs(ov-cv) > 1e-5:
				raise ValueError(f"output results between tflite and ikann model is different! tflite: {ov} ; ikann: {cv}")

	# def test_multiple_conv2d(self):
	#     pass
if __name__ == '__main__':
	unittest.main()
















