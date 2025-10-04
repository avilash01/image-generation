import replicate
import os
from dotenv import load_dotenv

load_dotenv()

def test_flux():
    print("🧪 Testing FREE Flux model on Replicate...")
    
    token = os.getenv("REPLICATE_API_TOKEN")
    if not token:
        print("❌ No token found")
        return False
    
    try:
        client = replicate.Client(api_token=token)
        
        print("Testing Flux model (should be free)...")
        output = client.run(
            "black-forest-labs/flux-1.1-pro",
            input={
                "prompt": "a cute cat wearing a hat",
                "width": 512,
                "height": 512,
                "num_outputs": 1
            }
        )
        
        print("✅ SUCCESS! Flux model works!")
        print(f"Image URL: {output[0]}")
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    test_flux()