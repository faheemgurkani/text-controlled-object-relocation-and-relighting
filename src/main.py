from modules import parser, detection, inpainting, relocation, relighting, utils
import cv2
from dotenv import load_dotenv
import os



def draw_bbox_on_image(image_path, bbox, output_path="../results/example_1/bbox_test.png"):
    image = cv2.imread(image_path)
    
    # Ensuring bbox is a list of ints
    x, y, w, h = map(int, bbox)

    cv2.rectangle(image, (x, y), (x + w, y + h), color=(0, 0, 255), thickness=2)

    cv2.imwrite(output_path, image)

    # print(f"Bounding box drawn and saved to {output_path}")   # For, testing

if __name__ == "__main__":
    load_dotenv()

    api_keys = {
        "openai_api_key": os.getenv("OPENAI_API_KEY"),
        "replicate_api_key": os.getenv("REPLICATE_API_KEY")
    }

    user_instruction = "Move the car to the left and add sunset light."
    system_instruction = """
    You are an intelligent instruction parser.
    Extract the object, action, location, and lighting from the user input.
    Always return a valid JSON object with the following format:
    {"object": object (string), "action": action (string), "location": location (string), "lighting": lighting (string)}.
    Do not include any additional text or explanation.
    """

    # # Parsing instruction
    # parsed = parser.parse_instruction(user_instruction, system_instruction, api_keys['openai_api_key'])

    # # For, testing
    # print(parsed)

    parsed = {'object': 'car', 'action': 'move', 'location': 'left', 'lighting': 'night'}  # For, testing

    # Resizing: To match the inpaiting model's requirements
    image = cv2.imread("./inputs/scene.png")

    image_resized = cv2.resize(image, (512, 512), interpolation=cv2.INTER_LANCZOS4)
    cv2.imwrite("./inputs/scene_resized.png", image_resized)

    image_path = "./inputs/scene_resized.png"

    # Detection
    # bbox = detection.detect_object(image_path, parsed['object'], api_keys['replicate_api_key'])
    bbox = detection.detect_object(image_path, parsed['object'])

    # # For, testing
    # draw_bbox_on_image(image_path, bbox)

    # print("Detected BBox:", bbox) # For, testing

    # bbox = [248, 232, 472, 417] # For, testing

    # Segmentation (SAM via Hugging Face Transformers)
    mask = detection.segment_object(image_path, bbox)

    binary_mask = (mask > 0.5).astype("uint8")*255

    # Dilating the mask to cover the shadow
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))  # You can tune the size
    dilated_mask = cv2.dilate(binary_mask, kernel, iterations=1)

    cv2.imwrite("../results/example_1/mask.png", dilated_mask)

    # Inpainting (To remove the object)
    inpainted_url = inpainting.inpaint_image(image_path, "../results/example_1/mask_dilated.png", f"Remove the object {parsed['object']} and its shadow, and restore the scene")
    
    utils.download_image(inpainted_url[0], "../results/example_1/inpainted.png")
    
    # # For, testing
    # inpainting.inpaint_image(image_path, "../results/example_1/mask.png")

    # # Pre-mature code
    # # Relocation (toy example shift)
    # bbox_int = list(map(int, bbox))
    # shift = (-200, 0)
    # relocated = relocation.relocate_object(
    #     "../results/example_1/inpainted.png",
    #     "../results/example_1/mask.png",
    #     bbox_int,
    #     shift
    #     # scale=0.9,
    #     # add_shadow=True
    # )

    # cv2.imwrite("../results/example_1/relocated.png", relocated)

    # # Relighting
    # relighted_url = relighting.relight_image("../results/example_1/relocated.png", "../results/example_1/mask.png",  "relight the scene to match" + parsed['lighting'], api_keys['replicate_api_key'])

    # utils.download_image(relighted_url[0], "../results/example_1/final_output.png")

    # print("Pipeline Complete")
