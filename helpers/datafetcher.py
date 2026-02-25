import requests

def data_fetcher(url: str, filename: str, headers: dict | None = None, payload: dict | None = None):
    """
    Fetch data from API using POST request.
    
    Args:
        url: The API endpoint URL
        filename: Output filename to save the response
        headers: HTTP headers dictionary (optional)
        payload: JSON payload for POST request (optional)
    """
    if headers is None:
        headers = {}
    
    if payload is None:
        response = requests.post(url, headers=headers, stream=True)
    else:
        response = requests.post(url, headers=headers, json=payload, stream=True)
    
    response.raise_for_status()
    
    with open(f"data/{filename}", "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)