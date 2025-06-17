import replicate
from transformers import SamProcessor, SamModel, DetrImageProcessor, DetrForObjectDetection
from PIL import Image
import torch
import numpy as np



def normalize_box(bbox, width, height):
    x0, y0, x1, y1 = bbox
    
    normalized_bbox = np.array([
        x0 / width,
        y0 / height,
        x1 / width,
        y1 / height
    ], dtype=np.float32)

    normalized_bbox = np.clip(normalized_bbox, 0.0, 1.0).tolist()

    return normalized_bbox

# DETR via Hugging Face Transformers
def detect_object(image_path, object_name, threshold=0.5):
    processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
    model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")

    img = Image.open(image_path).convert("RGB")
    inputs = processor(images=img, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model(**inputs)

    target_sizes = torch.tensor([img.size[::-1]])  # (H, W)
    results = processor.post_process_object_detection(
        outputs, target_sizes=target_sizes, threshold=threshold
    )[0]

    # Collect candidate boxes where label text matches object_name (case-insensitive)
    bboxes = []
    for score, label_idx, box in zip(results["scores"], results["labels"], results["boxes"]):
        label = model.config.id2label[label_idx.item()]
    
        if label.lower() == object_name.lower():
            bboxes.append((score.item(), [round(v) for v in box.tolist()]))

    if not bboxes:
        raise ValueError(f"No '{object_name}' found at threshold {threshold}")

    # Return bbox with highest confidence
    best = max(bboxes, key=lambda x: x[0])
    
    return best[1]

# # Grounding DINO via Replicate API
# def detect_object(image_path, object_name, replicate_api_key):
#     client = replicate.Client(api_token=replicate_api_key)
#     # model = client.models.get("adirik/grounding-dino")

#     # output = model.run(
#     #     image=open(image_path, "rb"),
#     #     query=object_name,
#     #     box_threshold=0.5,
#     #     text_threshold=0.25
#     # )

#     # output = client.run(
#     #     "adirik/grounding-dino",
#     #     input={
#     #         "image": open(image_path, "rb"),
#     #         "query": object_name,
#     #         "box_threshold": 0.5,
#     #         "text_threshold": 0.25
#     #     }
#     # )

#     # # Replicate returns list of boxes, take first
#     # bbox = output["boxes"][0]  # format: [x_min, y_min, x_max, y_max]

#     uploaded_file = client.files.create(open(image_path, "rb"))

#     # print(uploaded_file)    # For, testing

#     uploaded_file_url = uploaded_file.urls['get']

#     outputs = client.run(
#         "adirik/grounding-dino:efd10a8ddc57ea28773327e881ce95e20cc1d734c589f7dd01d2036921ed78aa",  # model identifier
#         use_file_output=False,
#         input={
#             "image": uploaded_file_url,
#             "query": object_name,
#             "box_threshold": 0.5,
#             "text_threshold": 0.25
#         },
#         api_token=replicate_api_key
#     )
    
#     # outputs is a list; extract from the first element
#     # print(outputs)  # For, testing

#     bbox = outputs['detections'][0]['bbox']
    
#     return bbox

def segment_object(image_path, bbox):
    processor = SamProcessor.from_pretrained("facebook/sam-vit-base")
    model = SamModel.from_pretrained("facebook/sam-vit-base")

    image = Image.open(image_path).convert("RGB")

    # # For, testing
    # print("Image size:", image.size)   # (width, height)
    # print("Bounding box:", bbox)       # (x0, y0, x1, y1)

    # Sanity checks
    width, height = image.size
    x0, y0, x1, y1 = bbox
    assert 0 <= x0 < x1 <= width
    assert 0 <= y0 < y1 <= height

    # NOTE: SAM expects input_boxes as [ [ [x0, y0, x1, y1] ] ]
    # normalized_bbox = normalize_box(bbox, width, height)

    # # For, testing
    # print("Image size:", width, height)
    # print("Normalized bbox:", normalized_bbox)

    inputs = processor(images=image, input_boxes=[[list(bbox)]], return_tensors="pt")
    # inputs = processor(images=image, input_boxes=[[list(normalized_bbox)]], return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)

        # masks = processor.post_process_masks(
        #     outputs.pred_masks, 
        #     inputs["original_sizes"], 
        #     inputs["reshaped_input_sizes"]
        # )

        # original_sizes = [(int(w), int(h)) for (w, h) in inputs["original_sizes"].tolist()]
        # reshaped_sizes = [(int(w), int(h)) for (w, h) in inputs["reshaped_input_sizes"].tolist()]

        original_sizes = [(int(h), int(w)) for (h, w) in inputs["original_sizes"].tolist()]
        reshaped_sizes = [(int(h), int(w)) for (h, w) in inputs["reshaped_input_sizes"].tolist()]

        masks = processor.post_process_masks(
            outputs.pred_masks,
            original_sizes,
            reshaped_sizes
        )

    # print("Masks:", masks)  # For, testing
    # print("Masks shape:", np.array(masks).shape)  # For, testing

    # binary_mask = masks[0][0].numpy()
    # binary_mask = masks[0][0].squeeze().numpy()
    binary_mask = np.squeeze(masks[0][0][0].numpy())

    # # For, testing
    # print("Unique mask values:", np.unique(binary_mask))
    
    return binary_mask
