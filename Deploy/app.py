from flask import Flask
from flask import request
from model import HAM10000_Model
import torch
from torchvision.transforms import Compose, Resize, ToTensor, Normalize, RandomAffine, RandomVerticalFlip, RandomHorizontalFlip
from PIL import Image

import urllib.request

model = HAM10000_Model()
model.load_state_dict(torch.load("resnet50_SVD.pth"), strict=True)
model.freeze()

transform = Compose([
    RandomAffine(45, (0.1, 0.1), (0.75, 1.25)),
    RandomVerticalFlip(),
    RandomHorizontalFlip(),
    Resize((299, 299)),
    ToTensor(),
    #Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

app = Flask(__name__)

number_label_to_label = ['dermatofibroma', 'vascular lesions', "actinic keratoses and intraepithelial carcinoma / bowen's disease", 'basal cell carcinoma', 'benign keratosis-like lesions', 'melanoma', 'melanocytic nevi']
number_label_to_label_vietnamese = ['u xơ da', 'tổn thương mạch máu', "dày sừng ánh sáng và ung thư biểu mô nội mô / bệnh Bowen", 'ung thư biểu mô tế bào đáy', 'các tổn thương tương tự sừng hóa lành tính', 'u hắc tố', 'nốt ruồi hắc tố']

@app.route('/', methods=['POST', 'GET'])
def hello_world():
    if request.method == 'POST':
        
        
        f = request.get_json()
        image_links = f["image_links"]
        sum = torch.zeros((7, ))
        for image_link in image_links:
            urllib.request.urlretrieve(image_link, "input_iamge.jpg")
            image = Image.open("./input_iamge.jpg")
            image = transform(image)
            print(image.unsqueeze(0).shape)
            print(sum)
            sum += model(image.unsqueeze(0)).squeeze()
            print(sum)
            print(torch.argmax(sum).item())
        result = number_label_to_label_vietnamese[torch.argmax(sum).item()]
        sum = torch.zeros((7, ))
        return [result];
    else:
        return "<p>Welcome to the Skin lesion detection AI, feel free to explore</p>"

if __name__ == '__main__':
    app.run()