import requests



def download_image(url, save_path):
    response = requests.get(url)
    
    with open(save_path, 'wb') as f:
        f.write(response.content)
