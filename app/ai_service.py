import torch
from torch.nn import functional as F
from PIL import Image
from torchvision import transforms
from pathlib import Path
import json
from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights

# 모델 로드 및 설정
model_path = Path(__file__).parent.parent / "models" / "efficientnet_b3_food_classification.pth"
class_names_path = Path(__file__).parent / "foodname.json"

# 모델 로드
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = efficientnet_b3(weights=EfficientNet_B3_Weights.IMAGENET1K_V1)
num_ftrs = model.classifier[1].in_features
model.classifier[1] = torch.nn.Linear(num_ftrs, len(json.load(open(class_names_path, "r", encoding="utf-8"))))
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# 이미지 전처리 함수
transform = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 음식 이름 로드
with open(class_names_path, "r", encoding="utf-8") as f:
    class_names = json.load(f)

# 예측 함수
def predict_image(image: Image.Image):
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image)
        probabilities = F.softmax(outputs, dim=1)

    # 상위 3개 클래스
    top_prob, top_idx = torch.max(probabilities, dim=1)
    top_class = class_names[top_idx.item()]


    return {
        "prediction": {
            "class": top_class
        }
    }
