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


def GenerateGraph(subgraph_idx, g, opcode_mapper):
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

	# html = _D3_HTML_TEMPLATE % (graph_str, subgraph_idx)
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

		# for subgraph_idx, g in enumerate(data["subgraphs"]):

		# 	print("inputs:")
		# 	pp.pprint(g["inputs"])
		# 	print("outputs:")
		# 	pp.pprint(g["outputs"])
		# 	print("tensors:")
		# 	pp.pprint(g["tensors"])
		# 	# print("op code mapper")
		# 	tensor_mapper = TensorMapper(g)
		# 	opcode_mapper = OpCodeMapper(data)
			
		# 	nodes, edges = GenerateGraph(subgraph_idx, g, opcode_mapper)
		# 	print("nodes:")
		# 	print(nodes)
		# 	print("edges:")
		# 	print(edges)
		# 	print('-'*5)
		# pp.pprint(model)
		self.g = {}
		self.visitedTensor = set()
		self.interpreter = None
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

	def parse(self, tfModel):
		'''
		Op (operations) are composed of tensors.
	

		tfModel: <byte>
		tflite model in bytes. 

		'''
		self.interpreter = tf.lite.Interpreter(
			model_content=tfModel)
		self.interpreter.allocate_tensors()
		
		# Parse operations
		for op in self.interpreter._get_ops_details():
			
			if op['op_name'] == 'FULLY_CONNECTED':
				self._parse_dense_op(op)
			# self.g[opIdx] = 
			# for 

		# Parse tensors:
		# Single tensors are regarded as operation

	def _parse_op_inputs(self, op):
		'''
		Parse input of an operation

		op <dict>:

		'''
		return op['inputs'].tolist()

	def _parse_op_outputs(self, op):
		'''
		Parse output of an operation

		op <dict>:

		'''
		return op['outputs'].tolist()		

	def _parse_dense_op(self, op):
		'''
		'''
		opIndex = op['index']
		self.g[opIndex] = {'name': 'dense'}
		self.g[opIndex]['inputs'] = \
			self._parse_op_inputs(op)
		self.g[opIndex]['outputs'] = \
			self._parse_op_outputs(op)

		# parse input tensors of operator
		for idx in self.g[opIndex]['inputs']:

			tensorMeta = self.interpreter._get_tensor_details(idx)
			tensorName = tensorMeta['name'] 
			# print("check TensorMeta")
			# pp.pprint(tensorMeta)
			# Access the weights from Matrix Multiplication
			if 'MatMul' in tensorName and \
				'BiasAdd' not in tensorName:
				self.g[opIndex]['weights'] = {
					'value': self.interpreter.get_tensor(idx),
					'size': tensorMeta['shape']
				}

				self.visitedTensor.add(id(tensorMeta))

			# Access the values from Bias
			elif 'BiasAdd' in tensorName and \
				'MatMul' not in tensorName:

				self.g[opIndex]['bias'] = {
					'value': self.interpreter.get_tensor(idx),
					'size': tensorMeta['shape']
				}

				self.visitedTensor.add(id(tensorMeta))

			else:
				continue


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




