# FastAPI backend
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import io

CLASSES = [
    'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
    'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
]

class ConvNeuralNetwork(nn.Module):
    def __init__(self):
        super(ConvNeuralNetwork, self).__init__()
        self.classifier = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.25),
            nn.Conv2d(32, 64, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.25),
        )
        self.flatten = nn.Flatten()
        self.Linear = nn.Linear(64 * 7 * 7, 10)

    def forward(self, x):
        x = self.classifier(x)
        x = self.flatten(x)
        output = self.Linear(x)
        return output

model = ConvNeuralNetwork()
state_dict = torch.load('./model_weights.pth', map_location=torch.device('cpu'))
model.load_state_dict(state_dict)
model.eval()

def preprocess_image(image_bytes):
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.Grayscale(1),
        transforms.ToTensor(),
        transforms.Normalize((0.5), (0.5))
    ])
    image = Image.open(io.BytesIO(image_bytes)).convert('L')
    return transform(image).unsqueeze(0)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

@app.post("/classify")
async def classify_image(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        input_tensor = preprocess_image(image_bytes)

        with torch.no_grad():
            outputs = model(input_tensor)
            _, predicted = torch.max(outputs, 1)
            label = CLASSES[predicted.item()]

        return JSONResponse(content={"label": label})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
