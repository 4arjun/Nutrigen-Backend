import cv2
from PIL import Image

def rotate_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        rect = cv2.minAreaRect(max(contours, key=cv2.contourArea))
        angle = rect[-1]
        if angle < -45:
            angle += 90

        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(image, M, (w, h))
    
    return image

def crop_image(file_path):
    with Image.open(file_path) as img:
        width, height = img.size
        box_size = min(width, height)
        
        left = (width - box_size) / 2
        top = (height - box_size) / 2
        right = left + box_size
        bottom = top + box_size
        
        cropped_img = img.crop((left, top, right, bottom))
        cropped_file_path = file_path.replace("uploaded_images", "uploads")  # Ensure different dir
        cropped_img.save(cropped_file_path)
        
    return cropped_file_path
