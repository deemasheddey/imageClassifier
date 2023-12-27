import torch
from torch import nn as nanan,optim
from torchvision import models

def building_network(arc, hideUnits):

    
    print("==========================================")    

    
    print("*** currently we're building the network,"+
          "architecture: {}, hidden units: {} ***".format(arc, hideUnits))

    if arc == 'vgg16':
        model = models.vgg16(pretrained=True)
        input_units = 25088
    elif arc == 'vgg13':
        model = models.vgg13(pretrained=True)
        input_units = 25088
    elif arc == 'alexnet':
        model = models.alexnet(pretrained=True)
        input_units = 9216
    else:
        raise ValueError(f"Sorry this architecture: {arc} is unsppurted")




   iterator = iter(model.parameters())
    while True:
        try:
            i = next(iterator)
            i.requires_grad = False
        except StopIteration:
            break
    
    classfier = nanan.Sequential( nanan.Linear(input_units, hideUnits), nanan.ReLU(), nanan.Dropout(p=0.2),
        nanan.Linear(hideUnits, 256),nanan.ReLU(),nanan.Dropout(p=0.2),nanan.Linear(256, 102),nanan.LogSoftmax(dim=1))
    
    



    model.classifier = classfier
    print("==========================================")    
    ##Fun. finshed print and return
    print("******* Finished from building network *******")
    
    return model


def training_network(model, ep, learnrate, trloader, valloader, g_p_u):

    print("==========================================")  
    print(f"Currently we're training the network ... epochs: {ep}, learning_rate: {learnrate}, gpu used for training: {g_p_u}")
    if g_p_u and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")    

    Improver = optim.Adam(model.classifier.parameters(), lr=learnrate)

    Standard = nanan.NLLLoss()
    steps = 0
    print_each = 10
    trainLoss = 0
    
    model.to(device)
    


    for epch in range(ep):
        
        model.train()
        
        for inputs, labels in trloader:


            steps =steps+ 1
            inputs = inputs.to(device)
            labels= labels.to(device)
            Improver.zero_grad()
            
            pslog= model.forward(inputs)
            loss = Standard(pslog, labels)

            loss.backward()
            Improver.step()

            trainLoss = trainLoss+loss.item()
            
            if steps % print_each == 0:
                model.eval()
                
                validloss = 0
                validAccuracy = 0
                
                with torch.no_grad():
                    for inputs, labels in valloader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        
                        pslog = model.forward(inputs)
                        batch_loss = Standard(pslog, labels)
                        validloss =validloss+ batch_loss.item()
                        ps = torch.exp(pslog)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        validAccuracy =validAccuracy+ torch.mean(equals.type(torch.FloatTensor)).item()
                print("==========================================")    

                print("Epoch {}/{}, Train loss: {:.4f}, Valid loss: {:.4f}, Valid accuracy: {:.3f}".format(epch+1, ep, trainLoss/print_each, validloss/len(valloader), validAccuracy/len(valloader)))
                
                trainLoss = 0
                
                model.train()
    print("==========================================")    

    print("***Training network is Finished.***")   
    print("==========================================")    
       
    return model, Standard
    
def evaluating_model(model, tstloader, st, g_p_u):
    print("==========================================")    

    print("*** we're currently testing the network -- gpu -- used to test: {} ***".format(g_p_u))
    
    if g_p_u and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")    



    tstloss = 0
    tstacu = 0

    
    model.eval()  
    with torch.no_grad():
        for inputs, labels in tstloader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            logps = model.forward(inputs)

            batch_loss = st(logps, labels)

            tstloss =tstloss+ batch_loss.item()

            ps = torch.exp(logps)
            #### 

            
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            
            tstacu = tstacu+torch.mean(equals.type(torch.FloatTensor)).item()


    resultestloss=tstloss/len(tstloader)
    resultstacc=tstacu/len(tstloader)

    print("*** CHECK  >> test loss: {:.4f}, test accuracy: {:.4f}".format(resultestloss,resultstacc ))
    print("==========================================")    
   
    print("***Testing network is Finished.***")


    
def saving_model(model, arc, hideunits, epocs, learnrate, savedircto):

    print("==========================================")    

    print(f"-- Currently we're saving model -- epochs: {epocs}, learning_rate: {learnrate}, save_dir: {savedircto}")    
    cp = {
        'architecture': arc,
        'hidden_units': hideunits,
        'epochs': epocs,
        'learning_rate': learnrate,
        'model_state_dict': model.state_dict(),
        'class_to_idx': model.class_to_idx
    }
    
    cp_path = savedircto + "checkpoint.pth"


    #Saving
    torch.save(cp, cp_path)
    print("==========================================")    

    print(f"Model is saved to {cp_path}")
    print("==========================================")  
    
def loading_model(fileP):
    print(f"Loading and building model from {fileP}")
    
    #Currently loading using torch 
    cp = torch.load(fileP)

    model = building_network(cp['architecture'], cp['hidden_units'])


    model.load_state_dict(cp['model_state_dict'])


    model.class_to_idx = cp['class_to_idx']
    ##DONE LOADING
    return model
    
 
def predict(processimg, model, tk):
    # Set the model to evaluation mode
    model.eval()
    
    # Disable gradient calculation for inference
    with torch.no_grad():
        # Forward pass through the model
        sp_log = model.forward(processimg.unsqueeze(0))

        # Compute the probabilities by applying softmax to the output
        ps__ = torch.exp(sp_log)

        # Get the top-k probabilities and labels
        probs = ps__.topk(tk, dim=1) 
        labels = ps__.topk(tk, dim=1)
        
        # Create a dictionary to map class indices to class labels
        classInto_idx_inv = {}
        idx = 0
        while idx < len(model.class_to_idx):
            key = list(model.class_to_idx.keys())[idx]
            value = model.class_to_idx[key]
            classInto_idx_inv[value] = key
            idx += 1
            
        # Create a list to store the predicted class labels
        class__ = []
        
        # Convert labels to class labels using the dictionary
        for label in labels.numpy()[0]:
            class__.append(classInto_idx_inv[label])
        
        # Return the probabilities and predicted class labels
        return probs.numpy()[0], class__

