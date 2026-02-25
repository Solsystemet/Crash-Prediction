import requests

def data_fetcher(url: str, filename: str):
    response = requests.get(url, stream=True)
    with open(f"data/{filename}", "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)