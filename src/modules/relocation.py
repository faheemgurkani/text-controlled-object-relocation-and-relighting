import cv2
import numpy as np



# def relocate_object(image_path, mask_path, bbox, location_shift):
#     image = cv2.imread(image_path)
#     mask = cv2.imread(mask_path, 0)

#     object_pixels = cv2.bitwise_and(image, image, mask=mask)
    
#     # Extract object
#     x, y, w, h = bbox
#     object_crop = object_pixels[y:y+h, x:x+w]
    
#     # Move to new location
#     new_image = image.copy()
#     new_x, new_y = x + location_shift[0], y + location_shift[1]
#     new_image[new_y:new_y+h, new_x:new_x+w] = object_crop
    
#     return new_image

def relocate_object(image_path, mask_path, bbox, location_shift):
    image = cv2.imread(image_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    # Ensure proper dtype
    if mask.dtype != np.uint8:
        mask = (mask * 255).astype(np.uint8)

    # Ensure mask shape matches image
    if mask.shape != image.shape[:2]:
        raise ValueError(f"Shape mismatch: image shape {image.shape}, mask shape {mask.shape}")

    object_pixels = cv2.bitwise_and(image, image, mask=mask)
    
    # Extract object crop
    x, y, x2, y2 = bbox
    w, h = x2 - x, y2 - y
    object_crop = object_pixels[y:y+h, x:x+w]
    
    # Move to new location
    new_image = image.copy()
    new_x, new_y = x + location_shift[0], y + location_shift[1]
    new_image[new_y:new_y+h, new_x:new_x+w] = object_crop
    
    return new_image