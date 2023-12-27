import argparse
from data_mang import loading_data
import model_mang

# Define the argument dictionary with argument names and their attributes
args_dict = {
    'description': 'Training a neural network on a given dataset',
    'data_directory': {'help': 'Path to dataset on which the neural network should be trained on'},
    '--save_dir': {'help': 'Path to directory where the checkpoint should be saved'},
    '--arch': {'help': 'Network architecture (default \'vgg16\')'},
    '--learning_rate': {'help': 'Learning rate'},
    '--hidden_units': {'help': 'Number of hidden units'},
    '--epochs': {'help': 'Number of epochs'},
    '--gpu': {'help': 'Use GPU for training', 'action': 'store_true'}
}

# Create the argument parser using the argument dictionary
parser = argparse.ArgumentParser(**args_dict)

# Parse the command-line arguments
args = parser.parse_args()

# Set the default values for optional arguments
# Assign save_dir based on the value of args.save_dir
if args.save_dir is None:
    saveDir = ''
else:
    saveDir = args.save_dir

# Assign network_architecture based on the value of args.arch
if args.arch is None:
    nw_arch = 'vgg16'
else:
    nw_arch = args.arch

# Assign learning_rate based on the value of args.learning_rate
if args.learning_rate is None:
    learRate = 0.0025
else:
    learRate = float(args.learning_rate)

# Assign hidden_units based on the value of args.hidden_units
if args.hidden_units is None:
    hideunit = 512
else:
    hideunit = int(args.hidden_units)

# Assign epochs based on the value of args.epochs
if args.epochs is None:
    epochs = 5
else:
    epochs = int(args.epochs)

# Assign gpu based on the value of args.gpu
if args.gpu is None:
    gpu = False
else:
    gpu = args.gpu

# Load the data and create data loaders
tr_data, trload, valload, tstload = loading_data(args.data_directory)

# Build the model
model = model_mang.building_network(nw_arch, hideunit)
model.class_to_idx = tr_data.class_to_idx

# Train the model
model =model_mang.training_network(model, epochs, learRate, trload, valload, gpu)

#ctritera
st = model_mang.training_network(model, epochs, learRate, trload, valload, gpu)

# Evaluate the model
model_mang.evaluating_model(model, tstload, st, gpu)

# Save the model checkpoint
model_mang.saving_model(model, nw_arch, hideunit, epochs, learRate, saveDir)
