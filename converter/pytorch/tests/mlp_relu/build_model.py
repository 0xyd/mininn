import sys
import torch
import torch.nn as nn

sys.path.append('../../')

from cgen.graph import BackpropGraph, iKannForwardGraph
from cgen.snippet import CSnippetGenerator

class Net(nn.Module):

	def __init__(self):
		super(Net, self).__init__()
		self.relu = nn.ReLU()
		self.relu2 = nn.ReLU()
		self.relu3 = nn.ReLU()
		self.fc1 = nn.Linear(3, 5)
		self.fc2 = nn.Linear(5, 3)
		self.fc3 = nn.Linear(3, 3)

	def forward(self, x):
		x = self.fc1(x)
		x = self.relu(x)
		x = self.fc2(x)
		x = self.relu2(x)
		x = self.fc3(x)
		x = self.relu3(x)
		return x

x = torch.tensor([[1.,2.,3.]])

net = Net()
y = net(x)


b = BackpropGraph(net)
b.parse(y)

k = iKannForwardGraph()
k.parse(b.g)

generator = CSnippetGenerator('../../cgen/templates')
generator.build_code(k, (1, 3), '.')

print(y)
