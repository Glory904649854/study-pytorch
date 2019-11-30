#coding:utf8;
import torch;
import torch.nn as nn;
import torch.nn.functional as F;
import torchvision;
import torchvision.transforms as transforms;
def loadDatas():
	tfs = transforms.Compose([
		transforms.ToTensor(),
		# transforms.Normalize([.5],[.5])
	]);
	trans_datas = torchvision.datasets.MNIST(root="./data",train=True,transform=tfs,download=True);
	test_datas = torchvision.datasets.MNIST(root="./data",train=False,transform=torchvision.transforms.ToTensor(),download=True);
	trans_loader = torch.utils.data.DataLoader(trans_datas,batch_size=64,shuffle=True);
	test_loader = torch.utils.data.DataLoader(test_datas,batch_size=64,shuffle=False);
	return trans_loader,test_loader;
	pass
class Net(nn.Module):
	def __init__(self):
		super(Net,self).__init__();
		# self.hiddenLayer = nn.Linear(28*28,100);
		# self.hiddenLayer2 = nn.Linear(20,50);
		
		self.outputLayer = nn.Linear(28*28,10);
	def forward(self,x):
		# x = F.relu(self.hiddenLayer(x));
		# x = F.relu(self.hiddenLayer2(x));
		x = self.outputLayer(x);
		return x;
if __name__ == '__main__':
	trans_datas,test_datas = loadDatas();
	model = Net();
	criterion = nn.CrossEntropyLoss();
	optimizer = torch.optim.SGD(model.parameters(),lr=1e-3);
	for epoch in range(5):
		for i,(img,labels) in enumerate(trans_datas):
			img = img.view(-1,28*28);
			labels = labels;
			out = model(img);
			loss = criterion(out,labels);
			optimizer.zero_grad();
			loss.backward();
			optimizer.step(); 
			if (i % 100) == 0: 
				print("EPOCH {},CURRENT LOSS : {}".format(epoch,loss));
		with torch.no_grad():
			correct = 0;
			total = 0;
			for img,labels in test_datas:
				img = img.view(-1,28*28);
				labels = labels;
				out_t_ = model(img);
				_,predicted = torch.max(out_t_,1);
				total += labels.size(0);
				correct += (predicted == labels).sum().item();
			pass;
		print("EPOCH {},NETWORD TEST IMAGES {}".format(epoch,(100 * correct / total)));
	pass