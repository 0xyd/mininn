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

		if vId not in self.g:

			self.g[vId] = {
				'name': str(type(var).__name__)}

			if hasattr(var, 'variable'):
				vv = var.variable
				self.g[vId]['weights'] = vv.data
				self.g[vId]['size'] = vv.size()
			elif var in self.outputNodes:
				self.g[vId]['output'] = True

			if hasattr(var, 'next_functions'):
				self.g[vId]['children'] = []
				for u in var.next_functions:
					if u[0] is not None:
						self._parse(u[0])
						self.g[vId]['children'].append(id(u[0]))


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
		'''
		if nodeId not in self.visitedOperation:

			if 'ReluBackward' in graph[nodeId]['name']:
				self._build_relu_block(nodeId, graph)

			elif 'AddmmBackward' in graph[nodeId]['name']:
				self._build_dense_block(nodeId, graph)

			if self.layerIdx > 0:
				self.g[self.layerIdx]['next'] = self.layerIdx-1

			self.layerIdx += 1
			self.visitedOperation.add(nodeId)

			for c in graph[nodeId]['children']:
				self._parse(c, graph)


	def _build_dense_block(self, nodeId, graph):
		'''
		Build dense block. 


		'''
		self.g[self.layerIdx] = {'name': 'dense'}

		children = graph[nodeId]['children']
		for c in children:
			if graph[c]['name'] == 'AccumulateGrad':
				self.g[self.layerIdx]['bias'] = graph[c]
				self.visitedOperation.add(c)
			# The parent of 'TBackward' is weight
			elif 'TBackward' in graph[c]['name']:
				self.visitedOperation.add(c)
				c = graph[c]['children'][0]
				self.g[self.layerIdx]['weights'] = graph[c]
			else:
				continue

	def _build_relu_block(self, nodeId, graph):
		'''
		Build relu block in graph

		node:
		An operation in a backpropagation graph
		'''

		self.g[self.layerIdx] = {'name': 'relu'}
		if 'output' in graph[nodeId]:
			self.g[self.layerIdx]['output'] = True

	def _build_sigm_block(self):
		pass

	def _build_softmax_block(self):
		pass




