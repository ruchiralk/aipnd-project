import argparse
import torch
from torch import nn, optim
from torchvision import datasets, transforms
from device_utils import current_device
from model_def import predict, supported_models, save_checkpoint

parser = argparse.ArgumentParser(description="Trainning network on a data set. Printing trainning loss and validation accuracy as the network trains.")
parser.add_argument("data_dir", help="path to directory containning trainning image set.")
parser.add_argument("--save_dir", default=".", help="Set directory to save checkpoints")
parser.add_argument("--arch", default="vgg16", choices=["vgg16", "densenet121"], help="Choose architecture")
parser.add_argument("--gpu", action="store_true", default=True ,help="Use GPU for trainning")
parser.add_argument("--learning_rate", default=0.0001, type=float, help="learning rate")
parser.add_argument("--hidden_units", default=512, type=int, help="hidden units in network")
parser.add_argument("--epochs", default=10, type=int, help="number of epoches")
args = parser.parse_args()

print('\n')
print(' Input Args '.center(40, '#'))
print(f"data_dir: {args.data_dir}")
print(f"checkpoint_dir: {args.save_dir}")
print(f"arch: {args.arch}")
print(f"gpu: {args.gpu}")
print(f"learning_rate: {args.learning_rate}")
print(f"hidden_units: {args.hidden_units}")
print(f"epochs: {args.epochs}")
print('\n')

device = current_device(use_gpu = args.gpu)

# define the directory structure
train_dir = args.data_dir + '/train'
valid_dir = args.data_dir + '/valid'
test_dir = args.data_dir + '/test'

# transforms for the training, validation, and testing sets
normalize_transform = transforms.Normalize([0.485, 0.456, 0.406],
                                           [0.229, 0.224, 0.225])
data_transforms = {
    'train': transforms.Compose([transforms.RandomRotation(30),
                                      transforms.RandomResizedCrop(224),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(),
                                      normalize_transform]) ,
    
   'test': transforms.Compose([transforms.Resize(225),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     normalize_transform])
}

# Load the datasets with ImageFolder
image_datasets = {
    'train': datasets.ImageFolder(train_dir, transform = data_transforms['train']),
    'test': datasets.ImageFolder(test_dir, transform = data_transforms['test']),
    'validation': datasets.ImageFolder(valid_dir, transform = data_transforms['test'])
}

# Using the image datasets and the trainforms, define the dataloaders
dataloaders = {
    'train': torch.utils.data.DataLoader(image_datasets['train'], batch_size=32, shuffle=True),
    'test': torch.utils.data.DataLoader(image_datasets['test'], batch_size=32),
    'validation': torch.utils.data.DataLoader(image_datasets['validation'], batch_size=32)
}

# define network parameters and initialize the network
output_size = 102
hidden_layer = args.hidden_units
learning_rate = args.learning_rate

model = supported_models[args.arch](hidden_layer, output_size)
model = model.to(device)

# configure the optimizer
# only train classifier parameters
optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)

# Build and train network
epochs = args.epochs
steps = 0
running_loss = 0
print_every = 20

print("starting to train ...")
for epoch in range(epochs):
    for inputs, labels in dataloaders['train']:
        steps += 1
        
        # move input and labels to default device
        inputs, labels = inputs.to(device), labels.to(device)
        logps = model.forward(inputs)
        loss = model.criterion(logps, labels)
        
        # zero the optimizer
        optimizer.zero_grad()
        
        # do the gradient decent
        loss.backward()
        optimizer.step()
        
        # accumilate running loss
        running_loss += loss.item()
        
        # validate model accuracy
        if steps % print_every == 0:
            test_loss = 0
            accuracy = 0
            
            model.eval() # disable dropouts
            
            with torch.no_grad():
                for inputs, labels in dataloaders['validation']:
                    inputs, labels = inputs.to(device), labels.to(device)
                    
                    logps = model.forward(inputs)
                    
                    batch_loss = model.criterion(logps, labels)
                    test_loss += batch_loss.item()
                    
                    # calculate accuracy
                    ps = torch.exp(logps)
                    
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                    
            # print statistic after calculating accuracy based on validation set
            print(f"Epoch {epoch+1}/{epochs}.. "
                  f"Train loss: {running_loss/print_every:.3f}.. "
                  f"Test loss: {test_loss/len(dataloaders['validation']):.3f}.. "
                  f"Test accuracy: {accuracy/len(dataloaders['validation']):.3f}")
            running_loss = 0
            model.train() # enable dropouts
print("training completed\n")

# save checkpoint after trainning
model.class_to_idx = image_datasets['train'].class_to_idx

# define checkpint file name
if args.save_dir[-1] != "/":
    args.save_dir = args.save_dir + "/"
checkpoint_file = args.save_dir + 'checkpoint.pth'
print(f"saving checkpoint at: {checkpoint_file}")

save_checkpoint(
    checkpoint_file=checkpoint_file, 
    epochs=args.epochs, 
    hidden_layer=args.hidden_units, 
    output_size=output_size, 
    model=model
)
