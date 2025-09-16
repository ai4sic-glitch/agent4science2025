import os
import time
from openai import OpenAI

DEFAULT_API_KEY = os.getenv("OPENAI_API_KEY", "")
DEFAULT_MODEL = "gpt-5-mini"

client = OpenAI(api_key=DEFAULT_API_KEY)

def query_gpt(messages: list, api_key: str = None, model: str = None) -> str:
    """
    Query GPT API with given messages and return the response text.
    """
    response = client.chat.completions.create(
        model=model or DEFAULT_MODEL,
        messages=messages
    )
    return response.choices[0].message.content

def query_gpt_with_retries(
    messages: list,
    api_key: str = None,
    model: str = None,
    max_attempts: int = 3,
    initial_delay: int = 2
) -> str:
    """
    Query GPT API with retry logic on failure.
    """
    for attempt in range(max_attempts):
        try:
            return query_gpt(messages, api_key=api_key, model=model)
        except Exception as e:
            print(f"[GPT Retry] Attempt {attempt+1} failed: {e}")
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
        print("Testing GPT API with retry logic...")
        response_text = query_gpt(test_messages, model="gpt-5-nano")
        print("\nResponse from GPT API:")
        print(response_text)
    except Exception as e:
        print(f"Final error after retries: {e}")
