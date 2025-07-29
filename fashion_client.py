# Gradio frontend
import gradio as gr
import requests
import io

def classify_with_backend(image):
    url = "http://127.0.0.1:8000/classify"
    image_bytes = io.BytesIO()
    image.save(image_bytes, format="PNG")
    image_bytes = image_bytes.getvalue()
    response = requests.post(url, files={"file": ("image.png", image_bytes, "image/png")})
    if response.status_code == 200:
        return response.json().get("label", "Error")
    else:
        return "Error"

iface = gr.Interface(
    fn=classify_with_backend,
    inputs=gr.Image(type="pil"),
    outputs="text",
    title="👕 FashionMNIST 이미지 분류기",
    description="28x28 흑백 옷 이미지를 넣으면 어떤 옷인지 예측합니다 (예: 셔츠, 신발, 가방 등)"
)

if __name__ == "__main__":
    iface.launch()