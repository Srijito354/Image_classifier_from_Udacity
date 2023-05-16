from PIL import Image
import json
import matplotlib.pyplot as plt#Import here
import seaborn as sb
import torch
from torch import nn
from torch import optim
import argparse as ag
from torchvision import datasets, transforms, models

parser=ag.ArgumentParser()
parser.add_argument('--image_dir', type=str, default='ImageClassifier/flowers/test/15/image_06369.jpg', help='Path to image')
parser.add_argument('--checkpoint', type=str, default='checkpoint.pth', help='path_to_checkpoint')
parser.add_argument('--topk', type=int, default=5, help='top k classes and probabilities')
parser.add_argument('--category names', type=str, default='cat_to_name.json', help='class_to_name json file.')
parser.add_argument('--gpu', type=str, default='cuda', help='you may choose either cpu or gpu.')
args=parser.parse_args()

#loading the checkpoint.
def load_checkPoint(filepath):
    checkpoint = torch.load(filepath)
    if checkpoint[0]=='vgg16':
        model=models.vgg16(True)
        for param in model.parameters():
            param.requires_grad=False
            
    else:
        print('Architecture not found for already trained model...')
        
    model.class_to_idx=checkpoint[1]
    
    classifier=nn.Sequential(nn.Linear(25088,4096),nn.ReLU(),nn.Dropout(0.5),nn.Linear(4096,102),nn.LogSoftmax(dim=1))
    model.classifier=classifier
    model.load_state_dict(checkpoint[2])
    return model
model=load_checkPoint(args.checkpoint)
if args.gpu=='cuda':
    device='cuda'
else:
    device='cpu'

def process_image(image_path):
    '''Scales, crops, and normalizes a PIL image for a PyTorch model,
       return a Numpy array
    '''
    # Process a PIL image for use in pytorch model
    
    pil_image = Image.open(image_path)
    
    #resizing the PIL image
    if pil_image.size[0]> pil_image.size[1]:
        pil_image.thumbnail((5000,256))
    else:
        pil_image.thumbnail((256,5000))
        
    #crop
    l_margin=(pil_image.width-224)/2   #left_margin
    b_margin=(pil_image.height-224)/2   #bottom_margin
    r_margin=l_margin+224   #right_margin
    t_margin=b_margin+224   #top_margin
    pil_image=pil_image.crop((l_margin,b_margin,r_margin,t_margin))
    
    #normalize
    np_image=np.array(pil_image)/225
    m=np.array([0.485,0.456,0.406]) #means
    std=np.array([0.229,0.224,0.225]) #standard deviations
    np_image=(np_image-m)/std
    
    np_image=np_image.transpose((2,0,1))
    return np_image

def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax

import numpy as np


def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    # TODO: Implement the code to predict the class from an image file
    image=process_image(args.image_dir)
    image=torch.from_numpy(image).type(torch.cuda.FloatTensor)
    image=image.unsqueeze(0)
    model.to(device)
    log_ps=model(image)
    ps=torch.exp(log_ps)
    top_p, top_class=ps.topk(topk)
    #type(torch.FloatTensor)
    top_p=top_p.cpu()
    top_class=top_class.cpu()
    top_p=list(top_p.detach().type(torch.FloatTensor).numpy())[0]
    top_class=list(top_class.detach().type(torch.FloatTensor).numpy())[0]
    #idx_to_class = {value: key for key, value in model.class_to_idx.items()}
    #top_classes = [idx_to_class[index] for index in top_indices]
    
    return list(top_p), list(top_class)
#path=input('f')
#'ImageClassifier/flowers/test/15/image_06369.jpg'
#IMG='ImageClassifier/'+str(args.image_dir)

probs, classes=predict(args.image_dir, model, args.topk)
print(probs)
#print(type(probs))
print(classes)
#print(type(classes))

#plotting the predictions and displaying the top predicted image.
def Sanity_check(model):
    plt.figure(figsize = (7,9))
    plot=plt.subplot(2,1,1)
    image=process_image(args.image_dir)
    flower_tag = cat_to_name['15']
    imshow(image, plot, title=flower_tag)
    #model=model.to(device)
    #dictionary for passing from idx to class
    model.class_to_idx=train_data.class_to_idx
    idx_to_class = {idx:cl for cl, idx in model.class_to_idx.items()}
    #list of the 5 names, the previous code had a for loop but didn't create a list
    flower_names= [cat_to_name[idx_to_class[c]] for c in classes]
    #flower_names = cat_to_name[str(i)] for i in [c.item() for c in classes]]
    plt.subplot(2,1,2)
    sb.barplot(x=probs, y=flower_names, color=sb.color_palette()[0])
    plt.show()
Sanity_check(model)