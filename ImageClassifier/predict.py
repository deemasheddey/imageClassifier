import argparse
import torch
import json
import model_mang
import data_mang

# Define the argument dictionary with argument names and their attributes
args_dict = {
    'description': 'Predicting flower name from an image along with the probability of that name.',
    'image_path': {'help': 'Path to image'},
    'checkpoint': {'help': 'Given checkpoint of a network'},
    '--top_k': {'help': 'Return top k most likely classes', 'type': int, 'default': 1},
    '--category_names': {'help': 'Use a mapping of categories to real names', 'default': 'cat_to_name.json'},
    '--gpu': {'help': 'Use GPU for inference', 'action': 'store_true'}
}

# Create the argument parser using the argument dictionary
parser = argparse.ArgumentParser(**args_dict)

# Parse the command-line arguments
args = parser.parse_args()

# Check if GPU should be used for inference
gpu = args.gpu and torch.cuda.is_available()

# Load the model using the provided checkpoint
model = model_mang.loading_model(args.checkpoint)
print("====================================")
print("Loaded model:", model)
print("====================================")


# Process the image and make predictions using the model
probs, predict_classes = model_mang.predict(data_mang.process_image(args.image_path), model, args.top_k)

# Load the category names mapping from the provided JSON file
with open(args.category_names, 'r') as f:
    cat_to_name = json.load(f)

# Map the predicted class indices to their corresponding names
classes = [cat_to_name[predict_class] for predict_class in predict_classes]

# Print the predicted probabilities and class names
print("Top probabilities:", probs)

# Print the predicted class names
print("Predicted classes:", classes)
