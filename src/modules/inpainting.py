# import replicate



# # Does not work
# def inpaint_image(image_path, mask_path, prompt, replicate_api_key):
#     client = replicate.Client(api_token=replicate_api_key)

#     uploaded_file = client.files.create(open(image_path, "rb"))
#     uploaded_file_url = uploaded_file.urls['get']

#     mask_file = client.files.create(open(mask_path, "rb"))
#     mask_file_url = mask_file.urls['get']

#     # model = client.models.get("stability-ai/stable-diffusion-inpainting")

#     # output = model.predict(
#     #     image=open(image_path, "rb"),
#     #     mask=open(mask_path, "rb"),
#     #     prompt=prompt
#     # )

#     outputs = client.run(
#         "stability-ai/stable-diffusion-inpainting:95b7223104132402a9ae91cc677285bc5eb997834bd2349fa486f53910fd68b3",  # model identifier
#         use_file_output=False,
#         input={
#             "image": uploaded_file_url,
#             "mask": mask_file_url,
#             "prompt": prompt
#         },
#         api_token=replicate_api_key
#     )

#     return outputs  # url returned by replicate

# # statbilityai method (Primarily used due to better expected results | Does not work)
# from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler, StableDiffusionInpaintPipeline
# from PIL import Image
# import cv2



# def inpaint_image(image_path, mask_path, instruction, negative_instrction, device="cpu"):
#     image = Image.open(image_path).convert("RGB")
#     mask = Image.open(mask_path).convert("L")

#     pipe = StableDiffusionInpaintPipeline.from_pretrained(
#         "stabilityai/stable-diffusion-2-inpainting"
#         # revision="fp16",
#         # torch_dtype=torch.float16
#     ).to(device)

#     out = pipe(
#         prompt=instruction,
#         negative_prompt=negative_instrction,
#         image=image,
#         mask_image=mask,
#         guidance_scale=7.5,
#         num_inference_steps=50
#     )

#     return out.images[0]

# # Suggested paper method (Works)
# from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler, StableDiffusionInpaintPipeline
# from PIL import Image
# import cv2



# def inpaint_image(image_path, mask_path, instruction, device="cpu", model_name="paint-by-inpaint/general-finetuned-mb"):
    
#     image = Image.open(image_path).convert("RGB")
#     mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    
#     pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
#         model_name
#     ).to(device)
#     pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)

#     # Applying edit
#     output = pipe(
#         prompt=instruction,
#         image=image,
#         guidance_scale=7.5,
#         image_guidance_scale=1.5,
#         num_inference_steps=50,
#         num_images_per_prompt=1
#     )

#     return output.images[0]  # PIL.Image

# Another googled approach (Under testing)
import cv2
from PIL import Image, ImageChops

from controlnet_union import ControlNetModel_Union
from diffusers import AutoencoderKL, StableDiffusionXLControlNetPipeline, TCDScheduler, ControlNetUnionModel



def inpaint_image(image_path, mask_path):
    image = Image.open(image_path).convert("RGB")
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    
    width, height = image.size
    min_dimension = min(width, height)

    left = (width - min_dimension) / 2
    top = (height - min_dimension) / 2
    right = (width + min_dimension) / 2
    bottom = (height + min_dimension) / 2

    final_source = image.crop((left, top, right, bottom))
    final_source = final_source.resize((512, 512), Image.LANCZOS).convert("RGBA")

    mask = Image.fromarray(mask).convert("L")  # Converting to grayscale PIL image

    binary_mask = mask.point(lambda p: 255 if p > 0 else 0)
    inverted_mask = ImageChops.invert(binary_mask)

    alpha_image = Image.new("RGBA", final_source.size, (0, 0, 0, 0))
    cnet_image = Image.composite(final_source, alpha_image, inverted_mask)

    vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix").to("cpu")

    # controlnet_model = ControlNetUnionModel.from_pretrained(
    #     "xinsir/controlnet-union-sdxl-1.0"
    # )

    controlnet_model = ControlNetModel_Union.from_pretrained(
        "xinsir/controlnet-union-sdxl-1.0"
    )

    pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
        "SG161222/RealVisXL_V5.0_Lightning",
        vae=vae,
        custom_pipeline="OzzyGT/pipeline_sdxl_fill",
        controlnet=controlnet_model,
        low_cpu_mem_usage=True
    ).to("cpu")
    pipe.scheduler = TCDScheduler.from_config(pipe.scheduler.config)

    prompt = "high quality"
    (
        prompt_embeds,
        negative_prompt_embeds,
        pooled_prompt_embeds,
        negative_pooled_prompt_embeds,
    ) = pipe.encode_prompt(prompt, "cpu", True)

    image = pipe(
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
        image=cnet_image,
    )

    image = image.convert("RGBA")
    cnet_image.paste(image, (0, 0), binary_mask)

    cnet_image.save("../results/example_1/inpainted.png")
