"""Cách đọc ảnh đầu vào chung của thực nghiệm RGB224."""
import cv2


def read_rgb224(image_path):
    image = cv2.imread(str(image_path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return cv2.resize(image, (224, 224), interpolation=cv2.INTER_LINEAR)
