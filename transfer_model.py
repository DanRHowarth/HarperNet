## Create a class that contains our model architecture 
import torch
from torch import nn
from torchvision import models
import torch.nn.functional as F

class ResNetCNN(nn.Module):
    
    # class_size relates to the final layer 
    def __init__(self, class_size):
        
        # this is necessary - but need to explain this more
        # when we initialise, then we combine the variable name with EncoderCNN module
        super(ResNetCNN, self).__init__()
        
        # here we instantiate the restnet50 module
        # presumably pretrained gives us our weights?
        resnet = models.resnet50(pretrained=True)
        
        # interesting that this needs to be a loop rather than resnet.parameters().required_grad_(False)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        # does order matter? here is what we have done:
        # 1. instantiated a model as resnet, with the pretrained weights
        # 2. determined that this model does not need training 
        # 3. created a list of the model layers (or params, or children!) except the last FC layer
        # 4. stored this list as the layers in the sequential element of the model 
        # 5. added a new layer, nn.Linear, which takes in the size of the final fc layer we want in resnet
        # and returns the class_size number as output
        modules = list(resnet.children())[:-1]
        
        # note the difference between resent and self.resnet
        self.resnet = nn.Sequential(*modules)
        self.category = nn.Linear(resnet.fc.in_features, class_size)

    def forward(self, images):#, class_size):
        
        # our features is a variable - presumably the output of the original image * weights etc
        # - of whatever goes through the resnet layers
        features = self.resnet(images)
        
        # we then need to reshape the output for the linear layers
        features = features.view(features.size(0), -1)
        
        ## we need to perform something here like a softmax? 
        # if class_size == 2:
            # features = self.category(features)
        
        # else:
            # F.softmax(self.category(features), dim=0)
            
            # features = self.category(F.softmax(features, dim=0))

        features = F.softmax(self.category(features), dim=0)

        return features