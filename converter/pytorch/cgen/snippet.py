import torch
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

	def build_code(self, kannGraph, inputShape, outputPath):
		'''
		kannGraph: <iKannForwardGraph>


		'''
		sequentialBlocks = [(i, data) for i, data in kannGraph.g.items()]
		sequentialBlocks.sort(key=lambda x: x[0], reverse=True) # Sort reversely because the output layer has the lowest index
		
		inputBlock = self._build_input(inputShape)
		codeBlocks = [inputBlock]

		for idx, block in enumerate(sequentialBlocks):

			block = block[1]
			name  = block['name']

			if name == 'dense':
				codeBlock = self._build_dense(block)

			elif name == 'relu':
				codeBlock = self._build_relu(block)
				
			else:
				raise NotImplementedError

			codeBlocks.append(codeBlock)

		with open(f'{outputPath}/model.c', 'w') as f:
			f.write(self.base.render(codeBlocks=codeBlocks))

	def _build_input(self, inputShape):
		'''
		'''
		template = self.env.get_template('input.c')
		_inputShape = []

		dims = 0
		for d in inputShape:
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

	def _build_dense(self, block):
		'''
		'''
		template = self.env.get_template('dense.c')
		weights = torch.flatten(block['weights']['value'])
		bias = block['bias']['value']
		output = block['bias']['size'][0]

		return template.render(
			weights=weights, bias=bias, output=output)

	def _build_relu(self, block):
		'''
		'''
		template = self.env.get_template('relu.c')
		return template.render()

	def _build_sigm(self):
		'''
		'''
		pass

	def _build_tanh(self):
		'''
		'''
		pass

	def _build_softmax(self):
		'''
		'''
		pass

