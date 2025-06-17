import replicate



def inpaint_image(image_path, mask_path, prompt, replicate_api_key):
    client = replicate.Client(api_token=replicate_api_key)

    uploaded_file = client.files.create(open(image_path, "rb"))
    uploaded_file_url = uploaded_file.urls['get']

    mask_file = client.files.create(open(mask_path, "rb"))
    mask_file_url = mask_file.urls['get']

    # model = client.models.get("stability-ai/stable-diffusion-inpainting")

    # output = model.predict(
    #     image=open(image_path, "rb"),
    #     mask=open(mask_path, "rb"),
    #     prompt=prompt
    # )

    outputs = client.run(
        "stability-ai/stable-diffusion-inpainting:95b7223104132402a9ae91cc677285bc5eb997834bd2349fa486f53910fd68b3",  # model identifier
        use_file_output=False,
        input={
            "image": uploaded_file_url,
            "mask": mask_file_url,
            "prompt": prompt
        },
        api_token=replicate_api_key
    )

    return outputs  # url returned by replicate
