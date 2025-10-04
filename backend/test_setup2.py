import requests
import os
from dotenv import load_dotenv

load_dotenv()

HF_API_KEY = os.getenv("HUGGINGFACE_API_KEY")

def test_api_key():
    print("🔑 Testing Hugging Face API key...")
    
    # Test with a simple text model to verify API key
    url = "https://api-inference.huggingface.co/models/gpt2"
    headers = {"Authorization": f"Bearer {HF_API_KEY}"}
    payload = {"inputs": "Hello, how are you?"}
    
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            print("✅ API Key is valid!")
            return True
        elif response.status_code == 401:
            print("❌ Invalid API Key")
            return False
        elif response.status_code == 403:
            print("❌ API Key doesn't have access")
            return False
        else:
            print(f"❌ Other error: {response.status_code}")
            print(f"Response: {response.text[:200]}")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    test_api_key()