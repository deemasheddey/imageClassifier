import torch
from torch import nn as nanan,optim
from torchvision import models

def building_network(arc, hideUnits):
    print("*** currently we're building the network, architecture: {}, hidden units: {} ***".format(arc, hideUnits))
    
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




    
    for i in model.parameters():
        i.requires_grad = False
    
    classfier = nanan.Sequential( nanan.Linear(input_units, hideUnits), nanan.ReLU(), nanan.Dropout(p=0.2),
        nanan.Linear(hideUnits, 256),nanan.ReLU(),nanan.Dropout(p=0.2),nanan.Linear(256, 102),nanan.LogSoftmax(dim=1))
    
    model.classifier = classfier




    ##Fun. finshed print and return
    print("******* Finished from building network *******")
    
    return model


def training_network(model, epochs, learning_rate, trainloader, validloader, gpu):
    print("Training network ... epochs: {}, learning_rate: {}, gpu used for training: {}".format(
        epochs, learning_rate, gpu))
    
    device = torch.device("cuda" if gpu and torch.cuda.is_available() else "cpu")
    
    criterion = nanan.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
    
    model.to(device)
    
    # Training the network
    steps = 0
    print_every = 10
    train_loss = 0
    
    # Note: I looked at the notebooks from the last module and decided to do it in a similar way
    for epoch in range(epochs):
        model.train()
        
        for inputs, labels in trainloader:
            steps += 1
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            if steps % print_every == 0:
                model.eval()
                
                valid_loss = 0
                valid_accuracy = 0
                
                with torch.no_grad():
                    for inputs, labels in validloader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        
                        logps = model.forward(inputs)
                        batch_loss = criterion(logps, labels)
                        valid_loss += batch_loss.item()
                        
                        # Calculate validation accuracy
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        valid_accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                
                print(f"Epoch {epoch+1}/{epochs}, Train loss: {train_loss/print_every:.3f}, "
                      f"Valid loss: {valid_loss/len(validloader):.3f}, "
                      f"Valid accuracy: {valid_accuracy/len(validloader):.3f}")
                
                train_loss = 0
                
                model.train()
    
    print("***Training network is Finished.***")          
    return model, criterion
def evaluating_model(model, testloader, criterion, gpu):
    print("Testing network ... gpu used for testing: {}".format(gpu))
    
    device = torch.device("cuda" if gpu and torch.cuda.is_available() else "cpu")
    
    # Validation on the test set
    test_loss = 0
    test_accuracy = 0
    model.eval()  # We just want to evaluate and not train the model
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            logps = model.forward(inputs)
            batch_loss = criterion(logps, labels)
            test_loss += batch_loss.item()
            
            # Calculate accuracy of test set
            ps = torch.exp(logps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            test_accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
    
    print(f"Test loss: {test_loss/len(testloader):.3f}, "
          f"Test accuracy: {test_accuracy/len(testloader):.3f}")
    
    print("***Testing network is Finished.***")
def saving_model(model, architecture, hidden_units, epochs, learning_rate, save_dir):
    print("Saving model ... epochs: {}, learning_rate: {}, save_dir: {}".format(epochs, learning_rate, save_dir))
    
    checkpoint = {
        'architecture': architecture,
        'hidden_units': hidden_units,
        'epochs': epochs,
        'learning_rate': learning_rate,
        'model_state_dict': model.state_dict(),
        'class_to_idx': model.class_to_idx
    }
    
    checkpoint_path = save_dir + "checkpoint.pth"
    
    torch.save(checkpoint, checkpoint_path)
    
    print("Model is saved to {}".format(checkpoint_path))
def loading_model(filepath):
    print("Loading and building model from {}".format(filepath))
    
    checkpoint = torch.load(filepath)
    model = building_network(checkpoint['architecture'], checkpoint['hidden_units'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    
    return model
def predict(processed_image, model, topk): 
    model.eval()
    with torch.no_grad():
        logps = model.forward(processed_image.unsqueeze(0))
        ps = torch.exp(logps)
        probs, labels = ps.topk(topk, dim=1)
        
        class_to_idx_inv = {model.class_to_idx[i]: i for i in model.class_to_idx}
        classes = list()
    
        for label in labels.numpy()[0]:
            classes.append(class_to_idx_inv[label])
        
        return probs.numpy()[0], classes
