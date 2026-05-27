from PIL import Image
import os
import joblib
import torch
import torchvision.transforms as transforms
import torchvision.models as models

svm = joblib.load("svm_model.pkl")
# Load class map (if you used label encoding)
class_map = joblib.load("class_map.pkl")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet50(pretrained=True)
model.fc = torch.nn.Identity()
model = model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],
                         [0.229,0.224,0.225])
])

def extract_feature(img_path):
    img = Image.open(img_path).convert("RGB")
    #print(img.size)
    
    img = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        feature = model(img)
    return feature.cpu().numpy().flatten()

lis=os.listdir("test")
for li in lis:
    feature = extract_feature("test/"+li)
    prediction = svm.predict([feature])
    # print("Predicted class:", list(class_map.keys())[prediction[0]])

    if list(class_map.keys())[prediction[0]]:
        print(li, list(class_map.keys())[prediction[0]])