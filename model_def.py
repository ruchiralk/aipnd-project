import torch
import torch.nn.functional as F
from torch import nn, optim
from torchvision import models
from collections import OrderedDict
from img_utils import process_image

# create vgg16 network
def make_vgg16(hidden_layer, output_size):
    model = models.vgg16(pretrained=True)
    
    # freeze model parameters so we won't do back prop
    for param in model.parameters():
        param.require_grad = False
    
    model.classifier = nn.Sequential(OrderedDict([
                                 ('fc1', nn.Linear(25088, hidden_layer)),
                                 ('relu', nn.ReLU()),
                                 ('dropout', nn.Dropout(0.2)),
                                 ('fc3', nn.Linear(hidden_layer, output_size)),
                                 ('out', nn.LogSoftmax(dim=1))
                                 ]))
    model.model_name = "vgg16"
    model.criterion = nn.NLLLoss()
    return model

# create densenet121 network
def make_densenet121(hidden_layer, output_size):
    model = models.densenet121(pretrained=True)

    # freeze model parameters so we won't do back prop
    for param in model.parameters():
        param.require_grad = False
    
    model.classifier = nn.Sequential(OrderedDict([
                                 ('fc1', nn.Linear(1024, hidden_layer)),
                                 ('relu', nn.ReLU()),
                                 ('dropout', nn.Dropout(0.2)),
                                 ('fc3', nn.Linear(hidden_layer, output_size)),
                                 ('out', nn.LogSoftmax(dim=1))
                                 ]))
    model.model_name = "densenet121"
    model.criterion = nn.NLLLoss()
    return model

# keep supported models in dictionary to simiplify loading model from check point
supported_models = {'vgg16': make_vgg16, 'densenet121': make_densenet121}

def convert_idx_to_class(class_to_idx):
    return {v:k for k,v in class_to_idx.items()}

 # The code to predict the class from an image file
def predict(image_path, model, device, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    img = process_image(image_path)
    img = img.to(device)
    
    model = load_model(model, device)
    model = model.to(device)
    model.eval()

    with torch.no_grad():
        img = img.unsqueeze(0) # convert to 4D tensor
        
        logps = model.forward(img)
        ps = torch.exp(logps)
        
        top_p, top_class = ps.topk(topk, dim=1)
        
        # convert idx to class
        idx_to_class = convert_idx_to_class(model.class_to_idx)
        
        # resize top_class tensor, next we can iterate over it
        top_class = [class_idx.item() for class_idx in top_class.reshape(-1, 1)]
        # convert class indexes to actual classes
        top_class = [idx_to_class[class_idx] for class_idx in top_class]
        
        # convert probability tensor to array
        top_p = [p.item() for p in top_p.reshape(-1,1)]
        
        model.train()
        return top_p, top_class

# define model saving function
def save_checkpoint(checkpoint_file, epochs, hidden_layer, output_size, model):
    checkpoint = {'output_size': output_size,
                  'hidden_layer': hidden_layer,
                  'model_state': model.state_dict(),
                  'epoches': epochs,
                  'class_to_idx': model.class_to_idx,
                  'model_name': model.model_name}
    torch.save(checkpoint, checkpoint_file)

# function that loads a checkpoint and rebuilds the model
def load_model(checkpoint_file, device):
    
    device_str = device.type
    if device.index != None:
        # if device is gpu, current device index needs to be appended to device_str
        device_str += f":{device.index}"
        
    checkpoint = torch.load(checkpoint_file, map_location= device_str)
    hidden_layer = checkpoint['hidden_layer']
    output_size = checkpoint['output_size']
    model = supported_models[checkpoint['model_name']](hidden_layer, output_size) # initialize model
    model.class_to_idx = checkpoint['class_to_idx']
    model.load_state_dict(checkpoint['model_state'])
    return model