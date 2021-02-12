import pprint
import numpy as np
import tensorflow as tf
from tensorflow.lite.python import schema_py_generated as schema_fb

pp = pprint.PrettyPrinter(indent=4)


def BuiltinCodeToName(code):
	"""Converts a builtin op code enum to a readable name."""
	for name, value in schema_fb.BuiltinOperator.__dict__.items():
		if value == code:
			return name
	return None

def FlatbufferToDict(fb, preserve_as_numpy):
	"""
	Converts a hierarchy of FB objects into a nested dict.
	We avoid transforming big parts of the flat buffer into python arrays. This
	speeds conversion from ten minutes to a few seconds on big graphs.
	Args:
	fb: a flat buffer structure. (i.e. ModelT)
		preserve_as_numpy: true if all downstream np.arrays should be preserved.
		false if all downstream np.array should become python arrays
	Returns:
		A dictionary representing the flatbuffer rather than a flatbuffer object.
	"""
	if isinstance(fb, int) or isinstance(fb, float) or isinstance(fb, str):
		return fb
	elif hasattr(fb, "__dict__"):
		result = {}
		for attribute_name in dir(fb):
			attribute = fb.__getattribute__(attribute_name)
			if not callable(attribute) and attribute_name[0] != "_":
				# snake_name = CamelCaseToSnakeCase(attribute_name)
				preserve = True if attribute_name == "buffers" else preserve_as_numpy
				result[attribute_name] = FlatbufferToDict(attribute, preserve)
				# result[snake_name] = FlatbufferToDict(attribute, preserve)
		return result
	elif isinstance(fb, np.ndarray):
		return fb if preserve_as_numpy else fb.tolist()
	elif hasattr(fb, "__len__"):
		return [FlatbufferToDict(entry, preserve_as_numpy) for entry in fb]
	else:
		return fb


class iKannGraph():

	def __init__(self):
		'''

		'''
		
		self.g = {
			'subgraphs': []
		}
		self.tfliteData = None
		self.visitedTensor = set()
		

	def parse(self, tfliteModel, dtype=np.float32):
		'''
		Op (operations) are composed of tensors.
	

		tfliteModel: <byte>
		tflite model in bytes. 

		dtype: <np.dtype>
		data type of weights
		'''

		data = schema_fb.Model.GetRootAsModel(tfliteModel, 0)
		data = FlatbufferToDict(
			schema_fb.ModelT.InitFromObj(data), 
			preserve_as_numpy=True)

		# Buffers are place where weights and bias values are allocated
		buffers = data['buffers']
		operatorCodes = data['operatorCodes']

		for subgraphIdx, g in enumerate(data["subgraphs"]):

			inputs = g['inputs']
			outputs = g['outputs']
			tensors = g['tensors']
			operators = g['operators']
			layers = {}

			# Indexing operator
			for idx, op in enumerate(g['operators']):
				op['operatorIdx'] = f'op{idx}'

			# Move weight values from buffer to tensor
			for idx, t in enumerate(tensors):
				d = buffers[t['buffer']]['data']
				if d is not None:
					t['data'] = d.view(dtype)

			# Build input layers
			for i in inputs:

				inputName = self.namelist_to_string(
					tensors[i]['name'])
				layers[str(i)] = {
					'name': inputName,
					'shape': tensors[i]['shape'],
					'nextLayer': []
				}

				for op in g['operators']:
					if i in op['inputs']:
						layers[str(i)]['nextLayer'].append(
							op['operatorIdx'])
				
				self.visitedTensor.add(str(i))

			# Build output layers
			for i, o in enumerate(outputs):
				# outputName = self.namelist_to_string(tensors[o]['name'])
				layers[str(o)] = {
					'name': f'output_{i}',
					'shape': tensors[o]['shape']
				}
				self.visitedTensor.add(str(o))

			# Build layers for operators
			for op in g['operators']:

				opName = BuiltinCodeToName(
					operatorCodes[
						op['opcodeIndex']]['builtinCode'])

				if opName == 'FULLY_CONNECTED':
					denseLayers = self._build_dense_layer(op, tensors)
					for l in denseLayers:
						layers.update(l)

				elif opName == 'LOGISTIC':
					logitLayer = self._build_logit_layer(op, tensors)
					layers.update(logitLayer)

				elif opName == 'RELU':
					reluLayer = self._build_relu_layer(op, tensors)
					layers.update(reluLayer)

				elif opName == 'TANH':
					tanhLayer = self._build_tanh_layer(op, tensors)
					layers.update(tanhLayer)

				elif opName == 'SOFTMAX':
					softmaxLayer = self._build_softmax_layer(op, tensors)
					layers.update(softmaxLayer)

				else:
					raise NotImplementedError

			# Link to next layers
			for layerId, layerData in layers.items():

				if 'outputs' in layerData:

					if len(layerData['outputs']) == len(layerData['nextLayer']):
						continue

					for o in layerData['outputs']:
						for _layerId, _layerData in layers.items():

							# Handle the input layer case
							if 'inputs' in _layerData:
								if o in _layerData['inputs']:
									layerData['nextLayer'].append(_layerId)
							elif str(o) == _layerId:
								layerData['nextLayer'].append(_layerId)


			self.g['subgraphs'].append(layers)
		
		return self.g

	def namelist_to_string(self, name):
		'''
		name: list<int>
		'''
		if isinstance(name, str):
			return name_list
		else:
			return ''.join([chr(c) for c in name])

	def _build_dense_layer(self, op, tensors):
		'''
		Parse the tensors and

		op <dict>:
		operators in tflite graph
		'''

		# Choose index that is not been used as index of dense layer
		layerId = op['operatorIdx']
		denseLayer = {
			layerId: {
				'name': 'dense',
				'inputs': op['inputs'],
				'outputs': op['outputs']
			}
		}

		for idx in op['inputs']:

			name = self.namelist_to_string(
				tensors[idx]['name'])

			if 'MatMul' in name and 'BiasAdd' not in name:
				denseLayer[layerId]['weights'] = {
					'value': tensors[idx]['data'],
					'shape': tensors[idx]['shape']
				}
				self.visitedTensor.add(str(idx))

			elif 'BiasAdd' in name and \
				'MatMul' not in name:
				denseLayer[layerId]['bias'] = {
					'value': tensors[idx]['data'],
					'shape': tensors[idx]['shape']
				}
				self.visitedTensor.add(str(idx))

			else:
				continue

		# Relu and dense layer will be fused together in tensorflow lite!
		# Hence we need to build an additional way to add relu
		if op['builtinOptions']['fusedActivationFunction']:

			fusedLayerId = 'r' + layerId
			reluLayer = {
				fusedLayerId: {
					'name': 'relu',
					'outputs': op['outputs'],
					'nextLayer': []
				},
			}
			denseLayer[layerId]['nextLayer'] = [fusedLayerId]

			return [denseLayer, reluLayer]

		else:
			denseLayer[layerId]['outputs'] = op['outputs']
			denseLayer[layerId]['nextLayer'] = []
			return [denseLayer]

	def _build_relu_layer(self, op, tensors):
		'''
		'''
		idx = op['inputs'][0]
		layer = {
			op['operatorIdx']: {
				'name': 'relu',
				'shape': tensors[idx]['shape'],
				'inputs':  op['inputs'],
				'outputs': op['outputs'],
				'nextLayer': []
			}
		}
		self.visitedTensor.add(idx)
		
		return layer

	def _build_tanh_layer(self, op, tensors):
		'''
		'''
		idx = op['inputs'][0]
		layer = {
			op['operatorIdx']: {
				'name': 'tanh',
				'shape': tensors[idx]['shape'],
				'inputs':  op['inputs'],
				'outputs': op['outputs'],
				'nextLayer': []
			}
		}
		self.visitedTensor.add(idx)
		
		return layer

	def _build_logit_layer(self, op, tensors):
		'''
		'''
		idx = op['inputs'][0]
		layer = {
			op['operatorIdx']: {
				'name': 'logit',
				'shape': tensors[idx]['shape'],
				'inputs':  op['inputs'],
				'outputs': op['outputs'],
				'nextLayer': []
			}
		}
		self.visitedTensor.add(str(idx))
		return layer

	def _build_softmax_layer(self, op, tensors):
		'''
		'''
		idx = op['inputs'][0]
		layer = {
			op['operatorIdx']: {
				'name': 'softmax',
				'shape': tensors[idx]['shape'],
				'inputs':  op['inputs'],
				'outputs': op['outputs'],
				'nextLayer': []
			}
		}
		self.visitedTensor.add(str(idx))
		return layer
