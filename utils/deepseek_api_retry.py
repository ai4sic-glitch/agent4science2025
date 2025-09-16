import requests
import os
import time

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

def query_deepseek_with_retries(
    messages: list,
    api_key: str = None,
    model: str = None,
    max_attempts: int = 3,
    initial_delay: int = 2
) -> str:
    """
    Query DeepSeek API with retry logic on failure (e.g., connection error, truncated response).

    Args:
        messages (list): LLM chat messages.
        api_key (str): DeepSeek API key.
        model (str): DeepSeek model name.
        max_attempts (int): Max number of attempts before giving up.
        initial_delay (int): Initial wait time (in seconds) before first retry.

    Returns:
        str: The LLM-generated response.
    """
    for attempt in range(max_attempts):
        try:
            return query_deepseek(messages, api_key=api_key, model=model)
        except Exception as e:
            print(f"[DeepSeek Retry] Attempt {attempt+1} failed: {e}")
            if attempt < max_attempts - 1:
                wait_time = initial_delay * (2 ** attempt)
                print(f"Waiting {wait_time} seconds before retrying...")
                time.sleep(wait_time)
            else:
                raise

if __name__ == "__main__":
    test_messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Summarize the importance of GPS in agriculture."}
    ]
    try:
        print("Testing DeepSeek API with retry logic...")
        response_text = query_deepseek_with_retries(test_messages, model="deepseek-reasoner")
        print("\nResponse from DeepSeek API:")
        print(response_text)
    except Exception as e:
        print(f"Final error after retries: {e}")
