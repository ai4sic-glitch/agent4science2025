import requests
import os

# Optional: Set your default API key here (fallback if not provided explicitly)
DEFAULT_API_KEY = ""
DEFAULT_MODEL = "deepseek-chat"

def query_deepseek(messages: list, api_key: str = None, model: str = None) -> str:
    """
    Query DeepSeek Chat API with given messages and return the response text.

    Args:
        messages (list): A list of message dicts, e.g., [{"role": "user", "content": "Hello"}].
        api_key (str, optional): The API key for authorization. Defaults to DEFAULT_API_KEY.
        model (str): The model to use (default: "deepseek-chat").

    Returns:
        str: The text response from DeepSeek Chat.
    """
    
    api_key = api_key or DEFAULT_API_KEY
    if not api_key or api_key == "your-default-api-key-here":
        raise ValueError("API key must be provided either as argument or via DEFAULT_API_KEY.")

    model = model or DEFAULT_MODEL
    url = "https://api.deepseek.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model,
        "messages": messages
    }

    response = requests.post(url, json=payload, headers=headers)
    response.raise_for_status()
    data = response.json()
    return data["choices"][0]["message"]["content"]


if __name__ == "__main__":
    test_messages = [
        {"role": "system", "content": "You are a helpful plant biology assistant."},
        {"role": "user", "content": "What are key factors affecting grape yield?"}
    ]
    try:
        response_text = query_deepseek(test_messages)
        print("Response from DeepSeek API:")
        print(response_text)
    except Exception as e:
        print(f"Error: {e}")
