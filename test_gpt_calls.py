import os
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
import base64
import requests

from dotenv import load_dotenv

load_dotenv()

# Initialize models
model_anthropic = ChatAnthropic(
    model="claude-3-7-sonnet-20250219",
    temperature=0,
    max_tokens=8000,
    api_key=os.getenv("ANTHROPIC_API_KEY"),
)
model_claude_3_5_sonnet = ChatAnthropic(
    model="claude-3-5-sonnet-20240620",
    temperature=0,
    max_tokens=8000,
    api_key=os.getenv("ANTHROPIC_API_KEY"),
)
model_openai = ChatOpenAI(
    model="gpt-4o",
    temperature=0,
    max_tokens=8000,
    api_key=os.getenv("OPENAI_API_KEY"),
)
model_o3mini = ChatOpenAI(
    model="o3-mini",
    max_tokens=8000,
    api_key=os.getenv("OPENAI_API_KEY"),
)

# Test function to check if APIs are working
def test_api_connectivity():
    try:
        # Get a sample image
        image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/7/7e/Circle-icons-profile.svg/2048px-Circle-icons-profile.svg.png"
        response = requests.get(image_url)
        base64_image = base64.b64encode(response.content).decode("utf-8")

        # Test Anthropic API with image
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "What do you see in this image?"
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{base64_image}"}
                    }
                ]
            }
        ]
        response = model_anthropic.invoke(messages)
        print("Anthropic API Response:", response.content)
        print("Anthropic API Test: Success")
        
        # Test OpenAI API with image
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "What do you see in this image?"
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": image_url}
                    }
                ]
            }
        ]
        response = model_openai.chat(messages)
        print("OpenAI API Response:", response.content)
        print("OpenAI API Test: Success")
        
    except Exception as e:
        print(f"API Test Failed with error: {str(e)}")

if __name__ == "__main__":
    test_api_connectivity()