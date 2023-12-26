import torch
from PIL import Image
from torchvision import ds, trans

def loading_data(path):
    print("Now the data is loading {} ...".format(path))
    
    train_dir = path + '/train'
    valid_dir = path + '/valid'
    test_dir = path + '/test'
    
    tr_transform = trans.Compose([trans.RandomRotation(50),trans.RandomResizedCrop(224), trans.RandomHorizontalFlip(), trans.ToTensor(), trans.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])



    
##### NOW valid trans
    val_transform = trans.Compose([trans.Resize(255),trans.CenterCrop(224), trans.ToTensor(),trans.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])





    
##### NOW test trans
    test_transform = trans.Compose([trans.Resize(255),trans.CenterCrop(224), trans.ToTensor(), trans.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])


''''define each data with loader loader'''

    
    trdata = ds.ImageFolder(train_dir, transform = tr_transform)
    trloader = torch.utils.data.DataLoader(trdata, batch_size = 65, shuffle = True)

    vldata = ds.ImageFolder(valid_dir, transform = val_transform)
    valloader = torch.utils.data.DataLoader(vldata, batch_size = 65)



    tstData = ds.ImageFolder(test_dir, transform = test_transform)
    tstloader = torch.utils.data.DataLoader(tstData, batch_size = 65)
    
    


    
    print("****Finished loading and preprocessing data****")
    
    return trdata, trloader, valloader, tstloader




##############
'''process image'''


def processing_image(img):
    img = Image.open(img)
    image_transform = trans.Compose([trans.Resize(255), trans.CenterCrop(224), trans.ToTensor(),  trans.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])
    return image_transform(img)



''''' THIS CLASS IS DONE'''' 
