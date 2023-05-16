import json
import matplotlib.pyplot as plt#Import here
import seaborn as sb
import torch
from torch import nn
from torch import optim
import argparse as ag
#import torch.nn.functional as F
from torchvision import datasets, transforms, models
#from collections import OrderedDict as OD
parser=ag.ArgumentParser()
parser.add_argument('--dir', type=str, default='checkpoint.pth', help='Path to checkpoint')
parser.add_argument('--arch', type=str, default='vgg16', help='architectures available are vgg16 and alexnet')
parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')
parser.add_argument('--hidden_units', type=int, default=4096, help='hidden units')
parser.add_argument('--epochs', type=int, default=10, help='epochs')
parser.add_argument('--gpu', type= str, default='cuda', help='choose either cpu or gpu')
args=parser.parse_args()

IMG='ImageClassifier/'
data_dir = IMG + 'flowers'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

trainings_transforms=transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

transforms = transforms.Compose([ transforms.Resize(226),
                                transforms.CenterCrop(224),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406],
                                                     [0.229, 0.224, 0.225])])

# TODO: Load the datasets with ImageFolder
#image_datasets = datasets.ImageFolder(train_dir)
train_data=datasets.ImageFolder(train_dir, trainings_transforms)
validation_data=datasets.ImageFolder(valid_dir, transforms)
test_data=datasets.ImageFolder(test_dir, transforms)
# TODO: Using the image datasets and the trainforms, define the dataloaders
train_loaders = torch.utils.data.DataLoader(train_data,32,shuffle=True)
test_loaders = torch.utils.data.DataLoader(test_data,32,shuffle=True)
val_loaders=torch.utils.data.DataLoader(validation_data,32,shuffle=True)

with open('ImageClassifier/cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)

if args.arch=='vgg16':
    cls_input=25088
    model=models.vgg16(True)
elif args.arch=='alexnet':
    cls_input=9216
    model=model.alexnet(True)
    

if args.gpu=='cuda':
    device='cuda'
else:
    device='cpu'
#device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
hid_units=args.hidden_units
#freezing the feature parameters.
for param in model.parameters():
    param.requires_grad=False
    
classifier=nn.Sequential(nn.Linear(cls_input,hid_units),nn.ReLU(),nn.Dropout(0.5),nn.Linear(hid_units,102),nn.LogSoftmax(dim=1))
model.classifier=classifier

#validation
def validation(model,val_loaders,criterion):
    #calculating the validation loss
    All_ones=[]
    Total_equals=[]
    val_loss=0
    accuracy=0
    
    for images,labels in val_loaders:
        
        images,labels=images.to(device),labels.to(device)
        log_ps=model(images)
        val_loss+=criterion(log_ps,labels).item()
        #actual probabilities
        ps=torch.exp(log_ps)
        #checking for equality betwen predicted values and actual labels.
        #top_p,top_class=ps.topk(1,1)
        #equals=top_class==labels.view(*top_class.shape)
        #print(type(equals))
        #for i in list(equals):
            #Total_equals.append('*')#List containing all the equlities(both 0 and 1 inclusive)
            #if i == 1:
                #All_ones.append('*')#list containing * for all the ones or the correst equalities.
        #accuracy=(len(All_ones)/len(Total_equals))*100
        equals=(labels.data == ps.max(dim=1)[1])
        accuracy+=equals.type_as(torch.FloatTensor()).mean().item()
    return val_loss,accuracy


#Training
criterion=nn.NLLLoss()
optimizer=optim.Adam(model.classifier.parameters(),0.01)
def training():
    epochs=5
    steps=0
    #fTloss=0
    #fValLoss=0
    #fAccuracy=0
    model.to(device)
    running_loss=0
    print_every=32
    for i in range(epochs):
        model.train()
        #running_loss=0
        for images,labels in train_loaders:
            images,labels = images.to(device),labels.to(device)
            steps+=1
            optimizer.zero_grad()#gradients zeroed out such that they don't accumulate; since doin' the latter prevents us from getting the desired results.
            log_ps=model.forward(images)
            loss=criterion(log_ps,labels)
            loss.backward()
            optimizer.step()
            running_loss+=loss.item()
            
            if steps % print_every==0:#every single step we're gonna perform and evaluation.
                model.eval()
                
                #to save memory we can turn off the gradients
                with torch.no_grad():
                    validation_loss, accuracy= validation(model,val_loaders,criterion)
                    
                print("Epochs: ", i+1, 'outta', epochs, end=' ')#printing the epoch we're in.
                fTloss=running_loss/print_every
                print('Training_loss: ', fTloss, end=' ')#the training loss
                fValLoss=validation_loss/len(val_loaders)
                print('Validation loss: ',fValLoss , end=' ')
                fAccuracy=accuracy/len(val_loaders)
                print('validation accuracy: ', fAccuracy)
                running_loss=0
                model.train()
                
training()

#validation on test set
def test(model, test_loaders):
    model.eval()
    model.to(device)
    with torch.no_grad():
            accuracy=0
            for images,labels in test_loaders:
                images,labels=images.to(device),labels.to(device)
                log_ps=model(images)
                ps=torch.exp(log_ps)
                
                equals=(labels.data==ps.max(dim=1)[1])
                accuracy+=equals.type_as(torch.FloatTensor()).mean().item()
    print('Accuracy: ',(accuracy/len(test_loaders)))
    
test(model, test_loaders)

#saving a checkpoint
def save_checkPoint(model):
    model.class_to_idx=train_data.class_to_idx
    checkpoint=['vgg16', model.class_to_idx, model.state_dict()]
    torch.save(checkpoint,'checkpoint.pth')
    
save_checkPoint(model)
