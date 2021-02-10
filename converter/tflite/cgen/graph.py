import pprint
import numpy as np
import tensorflow as tf
from tensorflow.lite.python import schema_py_generated as schema_fb
pp = pprint.PrettyPrinter(indent=4)


def TensorTypeToName(tensor_type):
	"""Converts a numerical enum to a readable tensor type."""
	for name, value in schema_fb.TensorType.__dict__.items():
		if value == tensor_type:
			return name
	return None

def BuiltinCodeToName(code):
	"""Converts a builtin op code enum to a readable name."""
	for name, value in schema_fb.BuiltinOperator.__dict__.items():
		if value == code:
			return name
	return None

class OpCodeMapper(object):
	"""Maps an opcode index to an op name."""

	def __init__(self, data):
		self.code2Name = {}
		for idx, d in enumerate(data["operatorCodes"]):
			self.code2Name[idx] = BuiltinCodeToName(d["builtinCode"])

	def __call__(self, x):
		if x not in self.code2Name:
			s = "<UNKNOWN>"
		else:
			s = self.code2Name[x]
		return "%s (%d)" % (s, x)

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

def NameListToString(name_list):
	"""Converts a list of integers to the equivalent ASCII string."""
	if isinstance(name_list, str):
		return name_list
	else:
		result = ""
		if name_list is not None:
			for val in name_list:
				result = result + chr(int(val))
	return result


class TensorMapper(object):
	"""Maps a list of tensor indices to a tooltip hoverable indicator of more."""

	def __init__(self, subgraph_data):
		self.data = subgraph_data

	def __call__(self, x):
		# html = ""
		# html += "<span class='tooltip'><span class='tooltipcontent'>"
		for i in x:
			tensor = self.data["tensors"][i]
			print(NameListToString(tensor["name"]))
			print(TensorTypeToName(tensor["type"]))
			print(repr(tensor["shape"]) if "shape" in tensor else [])
			print(repr(tensor["shape_signature"]) 
				if "shape_signature" in tensor else [])
			# html += str(i) + " "

			# html += NameListToString(tensor["name"]) + " "
			# html += TensorTypeToName(tensor["type"]) + " "
			# html += (repr(tensor["shape"]) if "shape" in tensor else "[]")
			# html += (repr(tensor["shape_signature"])
			# 	if "shape_signature" in tensor else "[]") + "<br>"

		print(x)
		# html += "</span>"
		# html += repr(x)
		# html += "</span>"
		# return html


def GenerateGraph(subgraphIdx, g, opcode_mapper):
	"""Produces the HTML required to have a d3 visualization of the dag."""

	def TensorName(idx):
		return "t%d" % idx

	def OpName(idx):
		return "o%d" % idx

	edges = []
	nodes = []
	first = {}
	second = {}
	# pixel_mult = 200  # TODO(aselle): multiplier for initial placement
	# width_mult = 170  # TODO(aselle): multiplier for initial placement
	for op_index, op in enumerate(g["operators"]):

		for tensor_input_position, tensor_index in enumerate(op["inputs"]):
			# if tensor_index not in first:
			# 	first[tensor_index] = ((op_index - 0.5 + 1) * pixel_mult,
   #  	                           (tensor_input_position + 1) * width_mult)
			edges.append({
				"source": TensorName(tensor_index),
				"target": OpName(op_index)
			})
		for tensor_output_position, tensor_index in enumerate(op["outputs"]):
			# if tensor_index not in second:
			# 	second[tensor_index] = ((op_index + 0.5 + 1) * pixel_mult,
   #                              	(tensor_output_position + 1) * width_mult)
			edges.append({
				"target": TensorName(tensor_index),
				"source": OpName(op_index)
			})

		print(f"op['opcodeIndex']: {op['opcodeIndex']}")

		nodes.append({
			"id": OpName(op_index),
			"name": opcode_mapper(op["opcodeIndex"]),
			"group": 2,
			# "x": pixel_mult,
			# "y": (op_index + 1) * pixel_mult
		})

	for tensor_index, tensor in enumerate(g["tensors"]):
		# initial_y = (
		# 	first[tensor_index] if tensor_index in first else
		# 	second[tensor_index] if tensor_index in second else (0, 0))

		nodes.append({
			"id": TensorName(tensor_index),
			"name": "%r (%d)" % (getattr(tensor, "shape", []), tensor_index),
			"group": 1,
			# "x": initial_y[1],
			# "y": initial_y[0]
		})

	return nodes, edges

	# graph_str = json.dumps({"nodes": nodes, "edges": edges})

	# html = _D3_HTML_TEMPLATE % (graph_str, subgraphIdx)
	# return html

class TfliteGraph():

	def __init__(self):
		'''

		'''
		# modelObj = schema_fb.Model.GetRootAsModel(tfModel, 0)
		# data = FlatbufferToDict(
		# 	schema_fb.ModelT.InitFromObj(modelObj), 
		# 	preserve_as_numpy=False)
		# pp.pprint(data)
		# print('='*5)
		# print(f'len(data["buffers"]): {len(data["buffers"])}')
		# print(f'len(data["subgraphs"][0]["operators"]): {len(data["subgraphs"][0]["operators"])}')
		# print(f'len(data["subgraphs"][0]["tensors"]): {len(data["subgraphs"][0]["tensors"])}')
		
		# for i in range(20):
		# 	print(f'BuiltinCodeToName({i}):{BuiltinCodeToName(i)}')
		# pp.pprint(len(data["subgraphs"]))
		# pp.pprint(data['operatorCodes'])

		 # Update builtin code fields.
		# for idx, d in enumerate(data["operatorCodes"]):
		# 	print(f"idx: {idx}")
		# 	d["builtinCode"] = max(
		# 		d["builtinCode"], d["deprecatedBuiltinCode"])

		# opcode_mapper = OpCodeMapper(data)
		# pp.pprint(opcode_mapper.code2Name)		

		# for subgraphIdx, g in enumerate(data["subgraphs"]):

		# 	print("inputs:")
		# 	pp.pprint(g["inputs"])
		# 	print("outputs:")
		# 	pp.pprint(g["outputs"])
		# 	print("tensors:")
		# 	pp.pprint(g["tensors"])
		# 	# print("op code mapper")
		# 	tensor_mapper = TensorMapper(g)
		# 	opcode_mapper = OpCodeMapper(data)
			
		# 	nodes, edges = GenerateGraph(subgraphIdx, g, opcode_mapper)
		# 	print("nodes:")
		# 	print(nodes)
		# 	print("edges:")
		# 	print(edges)
		# 	print('-'*5)
		# pp.pprint(model)
		self.g = {
			'subgraphs': []
		}
		self.tfliteData = None
		self.visitedTensor = set()
		# interpreter = tf.lite.Interpreter(
		# 	model_content=tfModel)
		# interpreter.allocate_tensors()
		# # layers = interpreter.get_tensor_details()
		# pp.pprint(interpreter._get_ops_details())
		# print('='*5)
		# pp.pprint(interpreter.get_tensor_details())
		# print('='*5)
		# # pp.pprint(interpreter.get_input_details())
		# # print('='*5)
		# for op in interpreter._get_ops_details():
		# 	print("tensors in op: ")
		# 	for i in op['inputs']:
		# 		pp.pprint(interpreter._get_tensor_details(i))
		# 		pp.pprint(interpreter.get_tensor(i))
		# 	print("*"*10)

		# for i in range(100):
		# 	print(BuiltinCodeToName(i))

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

		# pp.pprint(operatorCodes)

		for subgraphIdx, g in enumerate(data["subgraphs"]):

			inputs = g['inputs']
			outputs = g['outputs']
			tensors = g['tensors']
			operators = g['operators']
			layers = {}
			layerIdx = 0

			# pp.pprint(g)

			# Move weight values from buffer to tensor
			for idx, t in enumerate(tensors):
				d = buffers[t['buffer']]['data']
				if d is not None:
					t['data'] = d.view(dtype)

			# Build input layers
			for i in inputs:
				inputName = self._to_string(tensors[i]['name'])
				layers[tuple(str(i))] = {
					'name': inputName,
					'size': tensors[i]['shape'],
				}
				self.visitedTensor.add(str(i))

			# Build output layers
			for i, o in enumerate(outputs):
				# outputName = self._to_string(tensors[o]['name'])
				layers[tuple(str(o))] = {
					'name': f'output_{i}',
					'size': tensors[o]['shape']
				}
				self.visitedTensor.add(str(o))

			for op in g['operators']:

				opName = BuiltinCodeToName(
					operatorCodes[
						op['opcodeIndex']]['builtinCode'])

				if opName == 'FULLY_CONNECTED':
					denseLayers = self._parse_dense_op(op, tensors)
					for l in denseLayers:
						layers.update(l)

				elif opName == 'LOGISTIC':
					logitLayer = self._parse_logit_op(op, tensors)
					layers.update(logitLayer)

				elif opName == 'RELU':
					reluLayer = self._parse_relu_op(op, tensors)
					layers.update(reluLayer)

			pp.pprint(layers)
			# print('='*10)

	def _to_string(self, name):
		'''
		name: list<int>
		'''
		return ''.join([chr(c) for c in name])


	def _parse_op_inputs(self, op):
		'''
		Parse input of an operation

		op <dict>:

		'''
		return [str(i) for i in op['inputs'].tolist()]

	def _parse_op_outputs(self, op):
		'''
		Parse output of an operation

		op <dict>:

		'''
		return [str(i) for i in op['outputs'].tolist()]

	# def _parse_dense_op(self, layerIdx, layers, op, tfliteData):
	def _parse_dense_op(self, op, tensors):
		'''
		'''

		# Choose index that is not been used as index of dense layer
		layerIdx = tuple(s for s in self._parse_op_inputs(op))

		denseLayer = {
			layerIdx: {
				'name': 'dense',
				'inputs': self._parse_op_inputs(op)
			}
		}

		for idx in op['inputs']:

			name = self._to_string(
				tensors[idx]['name'])

			if 'MatMul' in name and 'BiasAdd' not in name:
				denseLayer[layerIdx]['weights'] = {
					'value': tensors[idx]['data'],
					'size': tensors[idx]['shape']
				}
				self.visitedTensor.add(str(idx))

			elif 'BiasAdd' in name and \
				'MatMul' not in name:
				denseLayer[layerIdx]['bias'] = {
					'value': tensors[idx]['data'],
					'size': tensors[idx]['shape']
				}
				self.visitedTensor.add(str(idx))

			else:
				continue

		# Relu and dense layer will be fused together in tensorflow lite!
		# Hence we need to build an additional way to add relu
		if op['builtinOptions']['fusedActivationFunction']:

			fusedLayerIdx = tuple('f') + layerIdx
			reluLayer = {
				fusedLayerIdx: {
					'name': 'relu',
					'inputs': [layerIdx],
					'outputs': self._parse_op_outputs(op) 
				},
			}
			denseLayer[layerIdx]['outputs'] = [fusedLayerIdx]

			return [denseLayer, reluLayer]

		else:
			denseLayer[layerIdx]['outputs'] = \
				self._parse_op_outputs(op)
			return [denseLayer]

	def _parse_relu_op(self, op, tensors):
		'''
		'''
		layerIdx = tuple(
			s for s in self._parse_op_inputs(op))
		layer = {
			layerIdx: {
				'name': 'relu',
				'shape': tensors[op['inputs'][0]]['shape'],
				'outputs': self._parse_op_outputs(op)
			}
		}
		
		return layer

	def _parse_logit_op(self, op, tensors):
		'''
		'''
		layerIdx = tuple(s for s in self._parse_op_inputs(op))
		# layerIdx = str(op['inputs'][0])
		layer = {
			layerIdx: {
				'name': 'logit',
				'shape': tensors[op['inputs'][0]]['shape'],
				'outputs': self._parse_op_outputs(op),
			}
		}
		return layer

# class iKannForwardGraph():

# 	def __init__(self):
# 		'''
# 		iKannForwardGraph is a graph based on ikann format

# 		'''
# 		interpreter = tf.lite.Interpreter(
# 			model_content=tfModel)
# 		interpreter.allocate_tensors()
# 		layers = interpreter.get_tensor_details()
# 		pp.pprint(layers)

# 	def parse(self, backpropGraph):
# 		'''
# 		parse backpropgation graph of pytorch model. 
# 		The parse algorithm takes idea from torchviz package:
# 		https://github.com/szagoruyko/pytorchviz/blob/master/torchviz/dot.py

# 		backpropGraph:
# 		backpropGraph of pytorch model
# 		'''
# 		startNodes = []
# 		for nid in backpropGraph.keys():
# 			if 'output' in backpropGraph[nid]:
# 				startNodes.append(nid)

# 		for nid in startNodes:
# 			self._parse(nid, backpropGraph)

# 	def _parse(self, nodeId, graph):
# 		'''
# 		The implementation details of recursive parsing.

# 		nodeId:
# 		Id of operation node.

# 		graph:
# 		backprop graph of pytorch model
# 		'''
# 		if nodeId not in self.visitedOperation:

# 			if 'ReluBackward' in graph[nodeId]['name']:
# 				self._build_relu_block(nodeId, graph)

# 			elif 'AddmmBackward' in graph[nodeId]['name']:
# 				self._build_dense_block(nodeId, graph)

# 			elif 'SigmoidBackward' in graph[nodeId]['name']:
# 				self._build_sigm_block(nodeId, graph)

# 			elif 'TanhBackward' in graph[nodeId]['name']:
# 				self._build_tanh_block(nodeId, graph)

# 			if self.layerIdx > 0:
# 				self.g[self.layerIdx]['next'] = self.layerIdx-1

# 			self.layerIdx += 1
# 			self.visitedOperation.add(nodeId)

# 			for c in graph[nodeId]['children']:
# 				self._parse(c, graph)


# 	def _build_dense_block(self, nodeId, graph):
# 		'''
# 		Build dense block. 

# 		nodeId:
# 		Id of operation node.

# 		graph:
# 		backprop graph of pytorch model
# 		'''
# 		self.g[self.layerIdx] = {'name': 'dense'}

# 		children = graph[nodeId]['children']

# 		for c in children:

# 			if graph[c]['name'] == 'AccumulateGrad':
# 				self.g[self.layerIdx]['bias'] = {
# 					'value': graph[c]['weights'],
# 					'size': graph[c]['size']
# 				}
# 				self.visitedOperation.add(c)

# 			# The children of 'TBackward' is weight
# 			elif 'TBackward' in graph[c]['name']:

# 				self.visitedOperation.add(c)
# 				c = graph[c]['children'][0]

# 				for c in graph[c]['children']:
# 					print(graph[c])
# 					print('='*10)
# 				self.g[self.layerIdx]['weights'] = {
# 					'value': graph[c]['weights'],
# 					'size': graph[c]['size']
# 				}
# 			else:
# 				continue

# 	def _build_relu_block(self, nodeId, graph):
# 		'''
# 		Build relu block in graph.

# 		nodeId:
# 		Id of operation node.

# 		graph:
# 		backprop graph of pytorch model
# 		'''

# 		self.g[self.layerIdx] = {'name': 'relu'}
# 		if 'output' in graph[nodeId]:
# 			self.g[self.layerIdx]['output'] = True

# 	def _build_sigm_block(self, nodeId, graph):
# 		'''
# 		Build sigm block in graph

# 		nodeId:
# 		Id of operation node.

# 		graph:
# 		backprop graph of pytorch model
# 		'''
# 		self.g[self.layerIdx] = {'name': 'sigm'}
# 		if 'output' in graph[nodeId]:
# 			self.g[self.layerIdx]['output'] = True

# 	def _build_tanh_block(self, nodeId, graph):
# 		'''
# 		Build sigm block in graph

# 		nodeId:
# 		Id of operation node.

# 		graph:
# 		backprop graph of pytorch model
# 		'''
# 		self.g[self.layerIdx] = {'name': 'tanh'}
# 		if 'output' in graph[nodeId]:
# 			self.g[self.layerIdx]['output'] = True

# 	def _build_softmax_block(self):
# 		pass




