from torchvision import datasets
import os

testset = datasets.FashionMNIST(
    root="./data",
    train=False,
    download=True
)

save_dir = './saved_test_images'
os.makedirs(save_dir, exist_ok=True)

for i in range(10):
    image, label = testset[i]
    filename = f"{i:04d}_label_{label}.png"
    image.save(os.path.join(save_dir, filename))

print("저장 완료!")
