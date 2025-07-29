# 👕 FashionMNIST 이미지 분류기

👗 웹앱 실행 화면입니다.
사용자가 이미지를 업로드하면, CNN 모델이 어떤 패션 아이템인지 예측해줍니다.

<img src="images/Ankle_boot.png" width="900"/>

---

<img src="images/Shirt.png" width="900"/>

---

<img src="images/Coat.png" width="900"/>

---

PyTorch 기반 CNN 모델을 이용해 FashionMNIST 옷 이미지를 분류하는 프로젝트입니다.  
모델 학습 → FastAPI 백엔드 → Gradio 프론트엔드까지 전 과정을 포함합니다.

---

## 📁 폴더 구조

```
FASHION_MNIST/
├── __pycache__/                   # Python 캐시 파일
├── data/                          # FashionMNIST 데이터 저장 위치
├── saved_test_images/             # 저장된 테스트 이미지들 (예: 10장 샘플)
├── images/                        # README에서 사용하는 예측 결과 이미지들
│   ├── Ankle_boot.png
│   ├── Shirt.png
│   └── Coat.png
├── venv/                          # Python 가상환경 폴더
├── .gradio/                       # Gradio의 자동 로그 저장 폴더 (flagged 이미지 등)
├── .gitignore                     # Git에서 추적하지 않을 파일/폴더 목록
├── fashion_classifier.py          # 모델 학습 및 평가 코드 (model_weights.pth, model.pt 저장)
├── fashion_client.py              # Gradio 프론트엔드 (이미지 업로드 UI)
├── fashion_server.py              # FastAPI 백엔드 (이미지 분류 API)
├── model_weights.pth              # 학습된 모델 파라미터 (state_dict만 저장)
├── model.pt                       # 전체 모델 저장본 (모델 + 구조)
├── requirements.txt               # 프로젝트 의존 패키지 목록
├── save_fashionmnist_images.py    # FashionMNIST 테스트 이미지 10장 저장용 스크립트
├── README.md                      # 프로젝트 설명 문서

```

---

## 🚀 실행 방법

### 1. 가상환경 생성

```
python -m venv venv
source venv/bin/activate       # macOS/Linux
venv\Scripts\activate          # Windows
```

### 2. 패키지 설치

```
pip install -r requirements.txt
```

### 3. 모델 학습 및 저장

```
python fashion_classifier.py
```

- `model_weights.pth` 와 `model.pt` 파일이 저장됩니다.

### 4. FastAPI 서버 실행

```
uvicorn fashion_server:app --reload
```

### 5. Gradio 프론트 실행

```
python fashion_client.py
```

- 웹 브라우저에서 이미지 업로드 시, 분류 결과가 출력됩니다.

---

## 👗 FashionMNIST 테스트 이미지 10장 저장

```
python save_fashionmnist_images.py
```

- `./saved_test_images/` 폴더에 10장의 예시 이미지가 저장됩니다.

---

## 📚 분류 클래스

```
['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
```

---

## 📌 사용 기술

- PyTorch
- FastAPI
- Gradio
- torchvision.datasets (FashionMNIST)
