from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from xception import Xception
from skimage import io, color


def train(model, device, train_loader, optimizer, epoch):
	model.train()
	for batch_idx, (data, target) in enumerate(train_loader):
		data, target = data.to(device), target.to(device)
		# print(data.shape, target.shape)
		optimizer.zero_grad()
		output = model(data)
		loss = F.nll_loss(output, target)
		loss.backward()
		optimizer.step()

		if batch_idx % 10 ==0:
			print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
				epoch, batch_idx * len(data), len(train_loader.dataset),
				100. * batch_idx / len(train_loader), loss.item()))



def test(model, device, test_loader):
	model.eval()
	test_loss = 0
	correct = 0
	# no_grad() prevents codes from tracking records or using memories 
	with torch.no_grad():
		for data, target in test_loader:
			data, target = data.to(device), target.to(device)
			output = model(data)
			test_loss += F.nll_loss(output, target, reduction = 'sum').item()
			pred = output.argmax(dim = 1, keepdim = True) # get the index of the max log-probability
			correct += pred.eq(target.view_as(pred)).sum().item()
	test_loss/=len(test_loader.dataset)
	print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
		test_loss, correct, len(test_loader.dataset),
		100. * correct / len(test_loader.dataset)))


def save_models(model):
    print()
    torch.save(model.state_dict(), "models/trained.model")
    print("****----Checkpoint Saved----****")
    print()



def main():
	# parser = argparse.ArgumentParser(description = 'Deep Learning Framework 1 Argument')
	# parser.add_argument('--epochs', type = int, default = 100, metavar = 'N', help = 'number of epochs to train and test the model (default=100)')
	# args = parser.parse_args()
	transform = transforms.Compose([
			transforms.Grayscale(num_output_channels=1),
			transforms.ToTensor(),
			transforms.Normalize([0.4161,], [0.1688,]),
	])

	train_dataset = datasets.ImageFolder('cropped_trainset',transform)
	test_dataset = datasets.ImageFolder('cropped_testset',transform)

	train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=3,shuffle=True,num_workers=2)
	
	test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle= False, num_workers=2)


	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	model = Xception().to(device)

	optimizer = optim.Adam(model.parameters(), lr = 0.001)
	scheduler = StepLR(optimizer, step_size = 1, gamma = 0.8)

	# set you own epoch
	for epoch in range(100):
		train(model, device, train_loader, optimizer, epoch)
		test(model, device, test_loader)

		"""
		use train and test function to train and test your model

		"""
	save_models(model)

if __name__ == "__main__":
	main()