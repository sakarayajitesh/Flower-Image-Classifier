import argparse
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import torch
from torch import nn
from torch import optim
from torch.optim import lr_scheduler
import time
import json
import copy
import json
import matplotlib.pyplot as plt
import torch.optim.lr_scheduler 


def main():
    print("started now")
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir")
    parser.add_argument("--save_dir", dest="save_dir", type=str,default="")
    parser.add_argument("--arch", dest="arch", type=str,default="vgg")
    parser.add_argument("--learning_rate", dest="learning_rate", type=float,default=0.001)
    parser.add_argument("--hidden_units", dest="hidden_units", type=str,default='512')
    parser.add_argument("--epochs", dest="epochs", type=int,default=10)
    parser.add_argument("-gpu", dest="gpu", action="store_true",default=False)
    results = parser.parse_args()
    print("Taking arguments over")
    data_dir = results.data_dir
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'valid': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    image_datasets = {x: datasets.ImageFolder(data_dir+'/'+x,
                                          data_transforms[x])
                  for x in ['train', 'valid','test']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=32,
                                             shuffle=True)
              for x in ['train', 'valid','test']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'valid','test']}
    class_names = image_datasets['train'].classes
    print("directory and loaders has been made")
    arch = results.arch
    print(arch)
    model = ''
    input_size=''
    if arch=='vgg':
      model = models.vgg16(pretrained=True)
      input_size = 25088
      print("Model selected is {}".format(arch)) 
    elif arch=='densenet':
      model = models.densenet121(pretrained=True)
      input_size = 1024
      print("Model selected is {}".format(arch)) 
    else:
      print("This model is not supported Enter either vgg or densenet")

    from collections import OrderedDict
    classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(input_size, 4096)),
                          ('relu', nn.ReLU()),('drop1',nn.Dropout(0.4)),
                          ('fc2', nn.Linear(4096, 102)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
    
    model.classifier = classifier
    
    hidden_units = results.hidden_units
    hidden_units = hidden_units.split(',')
    hidden_units = [int(x) for x in hidden_units]
    hidden_units.append(102)
    
    layers = nn.ModuleList([nn.Linear(input_size,hidden_units[0])])
    layer_sizes = zip(hidden_units[:-1], hidden_units[1:])
    layers.extend([nn.Linear(h1, h2) for h1, h2 in layer_sizes])   
    
    net_layers = OrderedDict()
    print("training starts now")
    for x in range(len(layers)):
        layerid = x + 1
        if x == 0:
            net_layers.update({'drop{}'.format(layerid):nn.Dropout(p=0.5)})
            net_layers.update({'fc{}'.format(layerid):layers[x]})
        else:
            net_layers.update({'relu{}'.format(layerid):nn.ReLU()})
            net_layers.update({'drop{}'.format(layerid):nn.Dropout(p=0.5)})
            net_layers.update({'fc{}'.format(layerid):layers[x]})
        
    net_layers.update({'output':nn.LogSoftmax(dim=1)})
    
    classifier = nn.Sequential(net_layers)
    print("Model created")
    model.classifier = classifier
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=results.learning_rate)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    
    steps = 0
    print_every = 40
    run_loss = 0
    num_epochs = results.epochs
    
    gpu = results.gpu
    
    device = "cpu"

    if gpu and torch.cuda.is_available():
        model.cuda()
        device ="cuda"
    elif gpu and torch.cuda.is_available() == False:
        print('Not possible')
        device = "cpu"
    
        
    since = time.time()
    print("device = {}".format(device))
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        for phase in ['train', 'valid']:
            if phase == 'train':
                exp_lr_scheduler.step()
                model.train() 
            else:
                model.eval()   

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            if phase == 'valid' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    model.load_state_dict(best_model_wts)
    print("testing starts now")
    correct = 0
    total = 0
    with torch.no_grad():
        for data in dataloaders['test']:
            images, labels = data
            images,labels = images.to(device),labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the test images: %d %%' % (
        100 * correct / total))
    
    model.class_to_idx = image_datasets['train'].class_to_idx
    model.cpu()
    save_dir = results.save_dir
    save_path = ""
    if len(save_dir) == 0:
        save_path = save_dir + 'checkpoint.pth'
    else:
        save_path = save_dir + '/checkpoint.pth'
        
    save_model = {
        'arch':arch,
        'state_dict':model.state_dict(),
        'class_to_idx':model.class_to_idx,
        'input_size':input_size,
        'epochs':num_epochs,
        'classifier':classifier,
        'optim_dict':optimizer,
        'learning_rate':results.learning_rate,
        'hidden_units':[each.out_features for each in model.classifier if hasattr(each, 'out_features') == True]
    }
    torch.save(save_model,save_path)
    
if __name__ == '__main__':
    main()
      
      
      

