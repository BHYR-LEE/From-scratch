# imports
import torch # entire library
import torch.nn as nn # import all the nn modules (Linear, Conv, Maxpool, Loss fn, Activations, etc)
import torch.optim as optim ## SGD etc
import torch.nn.functional as F # functions without params
from torch.utils.data import DataLoader ## easier dataset management
import torchvision.datasets as datasets ## standart datsets like mnist,imagenet
import torchvision.transforms as transforms # in : dataset out : transformed dataset

# create ffn
class NN(nn.Module):
    def __init__(self, input_size, num_classes): # (28 * 28) 
        super(NN, self).__init__() ## calls parent initialization method of parent
        self.fc1 = nn.Linear(input_size,50)
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
input_size = 784
num_classes = 10
learning_rate = 0.1
batch_size = 64
num_epoch = 1

# Load Data
train_dataset = datasets.MNIST(root='dataset/', train = True, transform=transforms.ToTensor(), download=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = datasets.MNIST(root='dataset/', train = False, transform=transforms.ToTensor(), download=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

# Initialize Network

model = NN(input_size=input_size, num_classes=num_classes).to(device)

# Loss and Optimizer

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training

for epoch in range(num_epoch):
    for batch_idx ,(data, target) in enumerate(train_loader):
        data = data.to(device=device)
        target = target.to(device=device)

        # flatten for Linear layer
        data = data.view(data.shape[0],-1)  ## flatten

        # forward
        scores = model(data)
        loss = criterion(scores, target)


        #backward
        optimizer.zero_grad() ## 초기화 시킴
        loss.backward()

        # gradient descent 
        optimizer.step()






# Check our accuracy & test to see how good our model

def check_accuracy(loader, model):
    if loader.dataset.train:
        print('train set accuracy')
    
    num_correct = 0
    num_samples = 0

    model.eval() # 

    with torch.no_grad(): ## we don't need to calculate any gredient during this
        for x,y in loader:
            x = x.to(device=device)
            y = y.to(device=device)
            x = x.reshape(x[0, -1])

            scores = model(x)
            _ , predictions = torch.max(scores,dim=1)
            result = (score == indices)
            num_correct += result.sum()
            num_samples += x.shape[0]

    model.train()

    return (num_correct/num_samples)        