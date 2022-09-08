import io
import json
from optparse import Values

from torch import nn
import torchvision.transforms as transforms
from PIL import Image
from flask import Flask, jsonify, request
import torch

app = Flask(__name__)

imagenet_class_index = json.load(open('labels.json'))
path = 'model/model.pt'
AlexNet_model = torch.hub.load('pytorch/vision:v0.6.0', 'alexnet', pretrained=True)
AlexNet_model.classifier[4] = nn.Linear(4096,1024)
AlexNet_model.classifier[6] = nn.Linear(1024,len(imagenet_class_index))
device = torch.device("cpu")
AlexNet_model.to(device)
model = AlexNet_model
CheckPoint = torch.load(path, map_location=torch.device('cpu'))
model.load_state_dict(CheckPoint['model_state_dict'])
model.eval()

def transform_image(image_bytes):                                            
    my_transforms = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor()])
        
    image = Image.open(io.BytesIO(image_bytes))
    return my_transforms(image).unsqueeze(0)

# get predictions with levels
def get_prediction(image_bytes):
    tensor = transform_image(image_bytes=image_bytes)
    outputs = model.forward(tensor)
    # If we want one prediction
    #_, y_hat = torch.max(outputs, 1)
    # Get top 3 predictions
    values, top = torch.topk(outputs, 3)
    names = []
    for i in top[0].numpy():
        names.append(imagenet_class_index[str(i)])
    return names, values.detach()[0].numpy().tolist()

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        img_bytes = file.read()
        displayNames, confidences = get_prediction(image_bytes=img_bytes)
        return jsonify({'displayNames': displayNames, 'confidences': confidences})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)