import cv2

def evaluate_contrast(img):
    return cv2.Laplacian(img, cv2.CV_64F).var()

