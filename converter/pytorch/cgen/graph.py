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
		pass

	def parse(self, backpropGraph):
		pass