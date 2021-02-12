import numpy as np
from jinja2 import Environment, FileSystemLoader

class CSnippetGenerator():

	def __init__(self, templatePath):
		'''
		CSnippetGenerator is used for generating ikann code.

		templatePath: <str>
		Path where templates are located.
		'''
		self.fileLoader = FileSystemLoader(templatePath)
		self.env = Environment(loader=self.fileLoader)
		self.base = self.env.get_template('base.c')
		self.header = self.env.get_template('model.h')

	def build_code(self, iKannGraph, outputPath='.'):
		'''
		iKannGraph: <iKannGraph>
		iKannGraph converted from tensorflow lite

		outputPath: <str>
		Path for output model's code
		'''
		codeBlocks = [] 

		for subgraph in iKannGraph.g['subgraphs']:

			# Find inputs
			inputLayers = []
			for k, layer in subgraph.items():
				if 'input' in layer['name']:
					inputLayers.append(layer)

			# Travese from the input layers
			for layer in inputLayers:

				while True:

					if 'nextLayer' in layer:
						# print(f"layer['nextLayer']: {layer['nextLayer']}")
						# print(f"layer['name']: {layer['name']}")
						# print('='*5)
						layerId = layer['nextLayer'][0]
						layer = subgraph[layerId]

						if layer['name'] == 'dense':
							codeBlocks.append(self._build_dense(layer))

						elif layer['name'] == 'relu':
							codeBlocks.append(self._build_relu())

						elif layer['name'] == 'logit':
							codeBlocks.append(self._build_sigm())

						elif layer['name'] == 'tanh':
							codeBlocks.append(self._build_tanh())

						elif layer['name'] == 'softmax':
							codeBlocks.append(self._build_softmax())

						else:
							if 'output' in layer['name']:
								pass
							else:
								raise NotImplementedError

					else: break
					
		with open(f'{outputPath}/model.c', 'w') as f:
			f.write(self.base.render(codeBlocks=codeBlocks))

		# Output header file of model
		with open(f'{outputPath}/model.h', 'w') as f:
			f.write(self.header.render())

	
	def _build_input(self, layer):
		'''
		'''
		template = self.env.get_template('input.c')
		_inputShape = []

		dims = 0
		for d in layer['shape']:
			if d > 0: 
				dims += 1
				_inputShape.append(d)

		_inputShapeLen = len(_inputShape)
		if _inputShapeLen < 4:
			for i in range(4-_inputShapeLen):
				_inputShape.append(0)

		return template.render(
			dims=dims, 
			batchNum=_inputShape[0], 
			channel=_inputShape[1], 
			height=_inputShape[2], 
			width=_inputShape[3])

	def _build_dense(self, layer):
		'''
		'''
		template = self.env.get_template('dense.c')
		weights = layer['weights']['value'].flatten()
		bias = layer['bias']['value']
		output = layer['bias']['size'][0]

		return template.render(
			weights=weights, bias=bias, output=output)

	def _build_relu(self):
		'''
		'''
		template = self.env.get_template('relu.c')
		return template.render()

	def _build_sigm(self):
		'''
		'''
		template = self.env.get_template('sigm.c')
		return template.render()

	def _build_tanh(self):
		'''
		'''
		template = self.env.get_template('tanh.c')
		return template.render()

	def _build_softmax(self):
		'''
		'''
		template = self.env.get_template('softmax.c')
		return template.render()

