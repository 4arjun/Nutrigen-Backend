import re
import cv2
from pyzbar.pyzbar import decode

def is_url(data):
    return re.match(r'^https?://', data) is not None

def BarcodeReader(image_path):
    img = cv2.imread(image_path)

    detectedBarcodes = decode(img)

    if not detectedBarcodes:
        img = rotate_image(img)
        detectedBarcodes = decode(img)

    if not detectedBarcodes:
        return "error: barcode not detected"
    
    barcode_data = []
    for barcode in detectedBarcodes:
        barcode_data.append({
            "data": barcode.data.decode("utf-8"),
            "type": barcode.type
        })

    non_url_data = [item['data'] for item in barcode_data 
                    if 'data' in item and not is_url(item['data'])]
    print("barcode")
    
    return non_url_data[0] if non_url_data else None
