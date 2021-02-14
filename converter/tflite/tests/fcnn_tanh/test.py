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


class FcnnTanhTestCase(unittest.TestCase):


	def test_codegen(self):

		model = keras.Sequential([
			keras.layers.InputLayer(input_shape=(5)),
			keras.layers.Dense(5, activation='tanh'),
			keras.layers.Dense(10, activation='tanh'),
			keras.layers.Dense(5, activation='tanh'),
			keras.layers.Dense(10, activation='tanh'),
			keras.layers.Dense(5, activation='tanh'),
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
				outputPath='fcnn_tanh')

		inputData = np.array([[1., 2., 3., 4., 5.]], dtype=np.float32)

		outputResult = invoke_tensorflow_lite(
			tfliteModel, inputData).flatten().round(decimals=8)

		build_and_execute_c_code(codePath='fcnn_tanh')

		# Compare output results
		compare_results_between_tflite_and_ikann(outputResult)

		os.remove('model_output.txt')

if __name__ == '__main__':
	unittest.main()



# import sys
# import pprint
# import pathlib
# import subprocess

# import numpy as np
# import tensorflow as tf
# import tensorflow.keras as keras

# sys.path.append('../../')

# from cgen.graph import iKannGraph
# from cgen.snippet import CSnippetGenerator

# pp = pprint.PrettyPrinter(indent=4)

# model = keras.Sequential([
#     keras.layers.InputLayer(input_shape=(5)),
#     keras.layers.Dense(5, activation='tanh'),
#     keras.layers.Dense(10, activation='tanh'),
#     keras.layers.Dense(5, activation='tanh'),
#     keras.layers.Dense(10, activation='tanh'),
#     keras.layers.Dense(5, activation='tanh'),
#     keras.layers.Softmax()
# ])

# converter = tf.lite.TFLiteConverter.from_keras_model(model)
# tfModelsDir = pathlib.Path('.')
# tfliteModel = converter.convert()
# tfliteFile = tfModelsDir/"model.tflite"
# tfliteFile.write_bytes(tfliteModel)

# ikannGraph = iKannGraph()
# graph = ikannGraph.parse(tfliteModel)

# # pp.pprint(graph)

# cGenerator = CSnippetGenerator(templatePath='../../../templates')
# cGenerator.build_code(ikannGraph)

# ## Comparing results between original tflite model and generated ikann model
# interpreter = tf.lite.Interpreter(model_content=tfliteModel)
# interpreter.allocate_tensors()
# inputDetails = interpreter.get_input_details()
# outputDetails = interpreter.get_output_details()

# inputData = np.array([[1., 2., 3., 4., 5.]], dtype=np.float32)
# interpreter.set_tensor(inputDetails[0]['index'], inputData)
# interpreter.invoke()
# outputResult = interpreter.get_tensor(
# 	outputDetails[0]['index']).flatten(
# 		).round(decimals=8)

# # Compile generated model 
# subprocess.run(['make'])
# subprocess.run(['./hello'])

# # Compare output results
# with open('model_output.txt') as f:
# 	cModelOutput = [float(v) for v in f.readline().split(',') if len(v) > 0]

# for ov, cv in zip(outputResult, cModelOutput):
# 	if abs(ov-cv) > 1e-5:
# 		raise ValueError(f"output results between tflite and ikann model is different! tflite: {ov} ; ikann: {cv}")

