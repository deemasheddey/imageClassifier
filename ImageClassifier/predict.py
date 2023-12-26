import argparse
import json
import model_mang
import data_mang


parser = argparse.ArgumentParser(description='Predicting flower name from an image along with the probability of that name.')
parser.add_argument('image_path', help='Path to image')
parser.add_argument('checkpoint', help='Given checkpoint of a network')
parser.add_argument('--top_k', help='Return top k most likely classes', type=int, default=1)
parser.add_argument('--category_names', help='Use a mapping of categories to real names', default='cat_to_name.json')
parser.add_argument('--gpu', help='Use GPU for inference', action='store_true')

args = parser.parse_args()

gpu = args.gpu and torch.cuda.is_available()

model = model_mang.loading_model(args.checkpoint)
print(model)

probs, predict_classes = model_mang.predict(data_mang.process_image(args.image_path), model, args.top_k)

with open(args.category_names, 'r') as f:
    cat_to_name = json.load(f)

classes = [cat_to_name[predict_class] for predict_class in predict_classes]

print(probs)
print(classes)