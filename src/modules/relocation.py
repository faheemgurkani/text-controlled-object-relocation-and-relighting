import cv2
import numpy as np



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

# def relocate_object(image_path, mask_path, bbox, location_shift, scale=1.0, add_shadow=True):
#     # Load image and mask
#     image = cv2.imread(image_path)
#     mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

#     if mask.dtype != np.uint8:
#         mask = (mask * 255).astype(np.uint8)

#     if mask.shape != image.shape[:2]:
#         raise ValueError("Image and mask dimensions must match.")

#     # Masked object
#     object_pixels = cv2.bitwise_and(image, image, mask=mask)

#     # Bounding box
#     x0, y0, x1, y1 = bbox
#     w, h = x1 - x0, y1 - y0

#     # Crop object and mask
#     object_crop = object_pixels[y0:y1, x0:x1]
#     mask_crop = mask[y0:y1, x0:x1]

#     # Resize for scale consistency
#     new_w = int(w * scale)
#     new_h = int(h * scale)
#     object_crop = cv2.resize(object_crop, (new_w, new_h), interpolation=cv2.INTER_AREA)
#     mask_crop = cv2.resize(mask_crop, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
#     mask_crop_3ch = cv2.merge([mask_crop] * 3) / 255.0  # Normalize for blending

#     # New location
#     new_x = x0 + location_shift[0]
#     new_y = y0 + location_shift[1]

#     # Bounds check
#     if new_x < 0 or new_y < 0 or new_x + new_w > image.shape[1] or new_y + new_h > image.shape[0]:
#         raise ValueError("Relocated object goes out of image bounds.")

#     result = image.copy()

#     # Optional shadow
#     if add_shadow:
#         shadow = cv2.GaussianBlur(mask_crop, (15, 15), 10)
#         shadow_bgr = cv2.merge([shadow] * 3)
#         shadow_bgr = (shadow_bgr * 0.3).astype(np.uint8)

#         shadow_x = new_x + 5
#         shadow_y = new_y + 5

#         # Bounds check for shadow
#         if (0 <= shadow_x < image.shape[1] - new_w) and (0 <= shadow_y < image.shape[0] - new_h):
#             shadow_region = result[shadow_y:shadow_y+new_h, shadow_x:shadow_x+new_w]
#             result[shadow_y:shadow_y+new_h, shadow_x:shadow_x+new_w] = cv2.add(
#                 shadow_region, shadow_bgr
#             )

#     # Alpha blend with target region
#     target_crop = result[new_y:new_y+new_h, new_x:new_x+new_w]
#     blended = (object_crop * mask_crop_3ch + target_crop * (1 - mask_crop_3ch)).astype(np.uint8)

#     result[new_y:new_y+new_h, new_x:new_x+new_w] = blended

#     return result