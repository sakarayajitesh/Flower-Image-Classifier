import argparse
import torch
from PIL import Image
import numpy as np
import json
from torch.autograd import Variable
import torchvision.models as models

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image_path")
    parser.add_argument("checkpoint")
    parser.add_argument("--top_k", dest="top_k", type=int,default=3)
    parser.add_argument("--category_names", dest="category_names", type=str,default="cat_to_name.json")
    parser.add_argument("--gpu", dest="gpu",action="store_true",default=False)
    results = parser.parse_args()
    
    checkpoint = torch.load(results.checkpoint)
    checkpoint_arch = checkpoint['arch']
    if checkpoint_arch=='vgg':
        pre_model = models.vgg16(pretrained=True)
    elif arch == 'densenet':
        pre_model = models.densenet121(pretrained=True)
    else:
        print("unable to find architecture")
    
    for param in pre_model.parameters():
        param.requires_grad = False

    pre_model.classifier = checkpoint['classifier']
    pre_model.load_state_dict(checkpoint['state_dict'])
    
    image = Image.open(results.image_path)
    image.thumbnail((256,256), Image.ANTIALIAS)
    h,w = image.size
    image  = image.crop((w//2 - 224//2, h//2 - 224//2, w//2 + 224//2, h//2 + 224//2))
    np_image = np.array(image, dtype=np.float64)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = np_image / 255.0
    np_image = np_image - mean / std
    processed_image = np_image.transpose((2,0,1))
    image = torch.from_numpy(processed_image)
    
    image = image.unsqueeze(0).float()
    image = Variable(image)
    gpu = results.gpu
    
    if gpu and torch.cuda.is_available():
        pre_model.cuda()
        image = image.cuda()
        print("cuda")
    else:
        print("Cpu")
    
    with torch.no_grad():
        out = pre_model.forward(image)
        res = torch.exp(out).data.topk(results.top_k)
        
    classes = np.array(res[1][0], dtype=np.int)
    probs = Variable(res[0][0]).data
    category_names = results.category_names
    class_idx = checkpoint['class_to_idx']
       
    if len(category_names) > 0:
        with open(category_names, 'r') as f:
            cat_to_name = json.load(f)
        mapped_names = {}
        for k in class_idx:
            mapped_names[cat_to_name[k]] = class_idx[k]
        mapped_names = {v:k for k,v in mapped_names.items()}
        
        classes = [mapped_names[x] for x in classes]
        probs = list(probs)
    else:
        class_idx = {v:k for k,v in class_idx.items()}
        classes = [class_idx[x] for x in classes]
        probs = list(probs)
        
    predictions = list(zip(classes, probs))
    for i in range(len(predictions)):
        print('{} : {:.3%}'.format(predictions[i][0], predictions[i][1]))
     
    
    

if __name__ == '__main__':
    main()