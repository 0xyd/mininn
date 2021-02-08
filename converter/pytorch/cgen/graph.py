import torch

class BackpropGraph():

	def __init__(self, model):
		'''
		BackpropGraph is a graph generated after model is call

		model:
		Pytorch model
		'''
		self.g = {}
		self.params_map = {
			id(v): k for k, v in 
				dict(
					model.named_parameters()).items()}
		self.seen = set()

	def parse(self, var):
		'''
		Parse the backpropagation graph of model

		var:
		output of a pytorch model
		'''

		self.outputNodes = (var.grad_fn,) \
			if not isinstance(var, tuple) \
				else tuple(v.grad_fn for v in var)
		self._parse(var.grad_fn)

	def _parse(self, var):
		'''
		Parse and generate the graph
		
		var:
		output of a pytorch model
		'''

		vId = id(var)

		# if vId in self.g:
		if var in self.seen:
			return
		else:

			self.g[vId] = { 'name': str(type(var).__name__) }
			print(f'-'*10)
			print(f"id: {vId}; name: {self.g[vId]['name']}")

			if hasattr(var, 'variable'):
				vv = var.variable
				self.g[vId]['weights'] = vv.data
				self.g[vId]['size'] = vv.size()

			elif var in self.outputNodes:
				self.g[vId]['output'] = True
			else:
				pass

			self.seen.add(var)

			print('children:')
			if hasattr(var, 'next_functions'):
				self.g[vId]['children'] = []
				tmp = []
				for u in var.next_functions:
					if u[0] is not None:
						tmp.append(u[0])
						print(f'id: {id(u[0])} name : {str(type(u[0]).__name__)}')

				for t in tmp:
					self._parse(t)
					self.g[vId]['children'].append(id(t))

				
class iKannForwardGraph():

	def __init__(self):
		'''
		iKannForwardGraph is a graph based on ikann format

		'''
		self.g = {}
		self.layerIdx = 0
		self.visitedOperation = set()

	def parse(self, backpropGraph):
		'''
		parse backpropgation graph of pytorch model. 
		The parse algorithm takes idea from torchviz package:
		https://github.com/szagoruyko/pytorchviz/blob/master/torchviz/dot.py

		backpropGraph:
		backpropGraph of pytorch model
		'''
		startNodes = []
		for nid in backpropGraph.keys():
			if 'output' in backpropGraph[nid]:
				startNodes.append(nid)

		for nid in startNodes:
			self._parse(nid, backpropGraph)

	def _parse(self, nodeId, graph):
		'''
		The implementation details of recursive parsing.

		nodeId:
		Id of operation node.

		graph:
		backprop graph of pytorch model
		'''
		if nodeId not in self.visitedOperation:

			if 'ReluBackward' in graph[nodeId]['name']:
				self._build_relu_block(nodeId, graph)

			elif 'AddmmBackward' in graph[nodeId]['name']:
				self._build_dense_block(nodeId, graph)

			elif 'SigmoidBackward' in graph[nodeId]['name']:
				self._build_sigm_block(nodeId, graph)

			if self.layerIdx > 0:
				self.g[self.layerIdx]['next'] = self.layerIdx-1

			self.layerIdx += 1
			self.visitedOperation.add(nodeId)

			for c in graph[nodeId]['children']:
				self._parse(c, graph)


	def _build_dense_block(self, nodeId, graph):
		'''
		Build dense block. 

		nodeId:
		Id of operation node.

		graph:
		backprop graph of pytorch model
		'''
		self.g[self.layerIdx] = {'name': 'dense'}

		children = graph[nodeId]['children']

		for c in children:

			if graph[c]['name'] == 'AccumulateGrad':
				self.g[self.layerIdx]['bias'] = {
					'value': graph[c]['weights'],
					'size': graph[c]['size']
				}
				self.visitedOperation.add(c)

			# The children of 'TBackward' is weight
			elif 'TBackward' in graph[c]['name']:

				self.visitedOperation.add(c)
				c = graph[c]['children'][0]

				for c in graph[c]['children']:
					print(graph[c])
					print('='*10)
				self.g[self.layerIdx]['weights'] = {
					'value': graph[c]['weights'],
					'size': graph[c]['size']
				}
			else:
				continue

	def _build_relu_block(self, nodeId, graph):
		'''
		Build relu block in graph.

		nodeId:
		Id of operation node.

		graph:
		backprop graph of pytorch model
		'''

		self.g[self.layerIdx] = {'name': 'relu'}
		if 'output' in graph[nodeId]:
			self.g[self.layerIdx]['output'] = True

	def _build_sigm_block(self, nodeId, graph):
		'''
		Build sigm block in graph

		nodeId:
		Id of operation node.

		graph:
		backprop graph of pytorch model
		'''
		self.g[self.layerIdx] = {'name': 'sigm'}
		if 'output' in graph[nodeId]:
			self.g[self.layerIdx]['output'] = True

	def _build_softmax_block(self):
		pass




