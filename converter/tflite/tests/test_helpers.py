import os
import sys
import tensorflow as tf

import pprint
import pathlib
import subprocess

import numpy as np
import tensorflow as tf
import tensorflow.keras as keras

sys.path.append('../')

from cgen.graph import iKannGraph, FlatbufferToDict
from cgen.snippet import CSnippetGenerator

def convert_keras_to_tflite(kerasModel):
	'''
	Convert a keras model to tflite model

	kerasModel:
	A keras model is being converted.
	'''

	converter = tf.lite.TFLiteConverter.from_keras_model(kerasModel)
	tfliteModel = converter.convert()
	return tfliteModel

def build_ikann_graph(tfliteModel):
	'''
	Build a ikann graph from tflite model.

	tfliteModel:
	The binary of tfliteModel
	'''
	ikannGraph = iKannGraph()
	ikannGraph.parse(tfliteModel)
	return ikannGraph

def generate_c_code(ikannGraph, templatePath, outputPath='.'):
	'''
	Generate c code from ikannGraph

	ikannGraph: <dict>
	ikannGraph contains layers of different operations

	templatePath: <str>
	templatePath defines path where template locate

	outputPath: <str>
	outputPath is the path where c code is generated
	'''
	cGenerator = CSnippetGenerator(templatePath=templatePath)
	cGenerator.build_code(ikannGraph, outputPath=outputPath)

def invoke_tensorflow_lite(tfliteModel, inputData):
	'''
	Invoke tflite model to see output result

	tfliteModel:
	The binary of tfliteModel

	inputData: <np.ndarray>
	Input data
	'''
	interpreter = tf.lite.Interpreter(model_content=tfliteModel)
	interpreter.allocate_tensors()
	inputDetails = interpreter.get_input_details()
	outputDetails = interpreter.get_output_details()

	interpreter.set_tensor(inputDetails[0]['index'], inputData)
	interpreter.invoke()
	outputResult = interpreter.get_tensor(outputDetails[0]['index'])
	return outputResult

def build_and_execute_c_code(codePath, mainFileName='main.c'):
	'''
	Build and execute the generated c code.

	codePath: <str>
	Path where make and binary are.

	mainFileName: <str>
	main to compile
	'''

	if os.path.exists('makefile'):
		subprocess.run([
			'make', '-f', 'makefile',
			f'MAIN_FILE={mainFileName}'], 
			check=True)
		subprocess.run(['./hello'], check=True)
	else:
		subprocess.run([
			'make', '-f', 
			f'{codePath}/makefile', 
			f'BASEDIR={codePath}',
			'IKANN_PATH=../../..'], check=True)
		subprocess.run([f'./{codePath}/hello'], check=True)


def compare_results_between_tflite_and_ikann(tfliteOutput, threshold=1e-5):
	'''
	Compare inferenced results between tflite model and ikann model

	tfliteOutput: <np.ndarray>
	Output result of tflite model
	'''
	with open('model_output.txt') as f:
		ikannOutput = [float(v) for v in f.readline().split(',') if len(v) > 0]
	for ov, cv in zip(tfliteOutput, ikannOutput):
		if abs(ov-cv) > threshold:
			raise ValueError(f"output results between tflite and ikann model is different! tflite: {ov} ; ikann: {cv}")

