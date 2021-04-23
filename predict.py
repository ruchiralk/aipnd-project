import argparse
from device_utils import current_device
from model_def import predict
import json

parser = argparse.ArgumentParser(description="Predict flower name from an image, along with the probability of that name. Pass in a single image path and return the flower name and class probability")
parser.add_argument("image_path", help="file path to image")
parser.add_argument("checkpoint", help="path to model checkpoint")
parser.add_argument("--top_k", type=int, default=1, help="Return top K most likely classes")
parser.add_argument("--category_names", default="cat_to_name.json", help="mapping file to map category to real names")
parser.add_argument("--gpu", action="store_true", help="use gpu")
args = parser.parse_args()

print('\n')
print(' Input Args '.center(40, '#'))
print(f"imge-path: {args.image_path}")
print(f"checkpoint: {args.checkpoint}")
print(f"top_k: {args.top_k}")
print(f"gpu: {args.gpu}")
print(f"category_names: {args.category_names}")
print('\n')

device = current_device(use_gpu = args.gpu)

top_p, top_class = predict(image_path = args.image_path, model=args.checkpoint, device=device, topk=args.top_k)

# convert top classes to class names
with open(args.category_names, 'r') as f:
    cat_to_name = json.load(f)
top_names = [cat_to_name[cat] for cat in top_class]

print(' Predictions '.center(40, '#'))
for top_name, top_p, top_class in zip(top_names, top_p, top_class):
    print(f"image class: {top_class}, probability: {top_p:.3f}, name: {top_name}")
print('\n')