from __future__ import print_function
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import argparse
import itertools
import os
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import timm


parser = argparse.ArgumentParser(description='PyTorch KMUFED shuffvit-DFER Training')
parser.add_argument('--dataset', type=str, default='/kaggle/working/shufflnetw1', help='CNN architecture')
parser.add_argument('--model', type=str, default='/kaggle/working/efficientvitwcc_Ourmodel', help='CNN architecture')


opt, unknown = parser.parse_known_args()

device = "cuda:0" if torch.cuda.is_available() else "cpu"


transforms_vaild = torchvision.transforms.Compose([
                                     torchvision.transforms.ToPILImage(),
                                     torchvision.transforms.Resize((224,)),
                                     torchvision.transforms.Grayscale(num_output_channels=3),
                                     torchvision.transforms.ToTensor(),
                                     torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225,))
                                     ])


def plot_confusion_matrix(cm, classes,
                          normalize=True,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion 
    Normalization can be applied by setting `normalize=True
    """

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=16)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")


    plt.ylabel('True label', fontsize=18)
    plt.xlabel('Predicted label', fontsize=18)
    plt.tight_layout()

class_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sadness', 'Surprise']


if opt.model == 'efficientViT':
    #pretrained model
    net = EfficientViT_M2(pretrained='efficientvit_m2') 
    num_features = net.head.l.in_features
    net.head.l = nn.Linear(num_features, 6)
if opt.model == 'Ourmodel':
   num_classes = 6  # Adjust based on your task
  
   net =  shufflenetw.CombinedModel(num_classes)
  

correct = 0
total = 0
all_target = []

for i in range(1):
    print("%d fold" % (i+1))
    path = os.path.join(opt.dataset + '_' + opt.model,  '%d' %(i+1))
    checkpoint = torch.load(os.path.join(path, '/kaggle/working/Test_model.t7'))

    net.load_state_dict(checkpoint['net'])
    #net.cuda()
    net.to(device)
    net.eval()
    testset = KMU(split = 'Testing', fold = i+1, transform=transforms_vaild)
    testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False)

    for batch_idx, (inputs, targets) in enumerate(testloader):
       
        inputs, targets = inputs.to(device), targets.to(device)
        inputs, targets = Variable(inputs, volatile=True), Variable(targets)
        
        outputs = net(inputs)
        #outputs = net(inputs.to('cuda:1'))
         # avg over crops
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        if batch_idx == 0 and i == 0:
            all_predicted = predicted
            all_targets = targets
        else:
            all_predicted = torch.cat((all_predicted, predicted), 0)
            all_targets = torch.cat((all_targets, targets), 0)

acc = 100. * correct / total
print("accuracy: %0.3f" % acc)

# Compute confusion matrix
matrix = confusion_matrix(all_targets.data.cpu().numpy(), all_predicted.cpu().numpy())
np.set_printoptions(precision=2)


print('Classification Report:\n', classification_report(all_targets.data.cpu().numpy(), all_predicted.cpu().numpy(), target_names=class_names))

 
plt.figure(figsize=(10, 8))
plot_confusion_matrix(matrix, classes=class_names, normalize=True,
                      title= 'Confusion Matrix (Accuracy: %0.3f%%)' %acc)
#plt.show()
plt.savefig(os.path.join(opt.dataset + '_' + opt.model, '/kaggle/working/lightmtrcmobilevit.png'))
plt.close()
