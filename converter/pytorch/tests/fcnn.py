import torch
import torch.nn as nn

class Net(nn.Module):

	def __init__(self):
		super(self, Net).__init__()
		self.relu = nn.ReLU()
		self.fc1 = nn.Linear(3, 3)
		self.fc2 = nn.Linear(3, 3)
		self.fc3 = nn.Linear(3, 3)

	def forward(self, x):
		x = self.fc1(x)
		x = self.relu(x)
		x = self.fc2(x)
		x = self.relu(x)
		x = self.fc3(x)
		x = self.relu(x)
		return x

net = Net()		
x = torch.rand(1, 3)
y = net(x)
