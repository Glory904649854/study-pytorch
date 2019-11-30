#coding:utf8
import torch;
import torch.nn as nn;
import torchvision;
import torchvision.transforms as transforms;
def loadDatas():
	tfs = transforms.Compose([
		transforms.ToTensor(),
		transforms.Normalize([.5,.5,.5],[.5,.5,.5]),
		
	]);
	train_datas = torchvision.datasets.CIFAR10(download=True,train=True,root="./data",transform=tfs);
	test_datas = torchvision.datasets.CIFAR10(download=True,train=False,root="./data",transform=tfs);
	train_loader = torch.utils.data.DataLoader(train_datas,shuffle=True,batch_size=64);
	test_loader = torch.utils.data.DataLoader(test_datas,shuffle=False,batch_size=64);
	return train_loader,test_loader;

	pass
class CIFAR10Net(nn.Module):
	def __init__(self):
		super(CIFAR10Net,self).__init__();
		self.layer = nn.Sequential(

			nn.Conv2d(3,8,3,padding=1),
			nn.MaxPool2d(2,2),
			nn.BatchNorm2d(8),
			nn.Conv2d(8,16,3,padding=1),
			nn.MaxPool2d(2,2),
			nn.BatchNorm2d(16),
			nn.Conv2d(16,32,3,padding=1), 
			nn.MaxPool2d(2,2),
			nn.BatchNorm2d(32),
			nn.Conv2d(32,64,3,padding=1),
			nn.MaxPool2d(2,2),
			nn.BatchNorm2d(64),

		);
		self.fc = nn.Sequential(

			nn.Linear(64*2*2,32),
			nn.Linear(32,10),
			nn.ReLU(inplace=True),

		);
	def forward(self,x):
		x = self.layer(x);
		x = x.view(x.size(0),-1);
		x = self.fc(x);
		return x;

def CIFAR10NetAction():
	_EPOCH = 10;
	_LR = 1e-3;
	train_datas,test_datas = loadDatas();
	model = CIFAR10Net();
	criterion = torch.nn.CrossEntropyLoss();
	optimizer = torch.optim.Adam(model.parameters(),lr=_LR);
	for epoch in range(_EPOCH):
		for i,(image,labels) in enumerate(train_datas):
			model.train();
			image = image;
			labels = labels;
			out = model(image);
			loss = criterion(out,labels);
			optimizer.zero_grad();
			loss.backward();
			optimizer.step();
			if (i % 100) == 0:
				print("NETWORD WAS RUNNING , EPOCH: {} , LOSS: {} .".format(
					epoch,loss
				));
		model.eval();
		with torch.no_grad():
			total = 0;
			corrent = 0;
			for image,labels in test_datas:
				image = image;
				labels = labels;
				out_t = model(image);
				_,predicted = torch.max(out_t,1);
				total += labels.size(0);
				corrent += (predicted == labels).sum().item();
			pass
		print("EPOCH {} TRAIN FINISH , TSET_DATAS RATE IS {}".format(epoch,(100 * corrent / total)));
	pass
if __name__ == '__main__':
	CIFAR10NetAction();
	pass