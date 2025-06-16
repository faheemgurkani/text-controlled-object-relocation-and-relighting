# import replicate



# def relight_image(image_path, relight_prompt, replicate_api_key):
#     replicate.Client(api_token=replicate_api_key)

#     output = replicate.run(
#         "stability-ai/stable-diffusion-xl-inpainting",
#         input={
#             "image": open(image_path, "rb"),
#             "prompt": relight_prompt,
#         }
#     )
    
#     return output['output_url']

import requests



def download_image(url, save_path):
    response = requests.get(url)
    
    with open(save_path, 'wb') as f:
        f.write(response.content)
