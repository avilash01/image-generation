from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import replicate
import os
from dotenv import load_dotenv
import logging
import requests
import base64
from io import BytesIO
import time
import asyncio
import aiohttp
from typing import List, Optional
import json
import sys

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

app = FastAPI(title="AI Image Generator", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN")
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")

if not REPLICATE_API_TOKEN:
    logger.error("REPLICATE_API_TOKEN not found in .env file!")
else:
    logger.info(f"Replicate API token loaded: {REPLICATE_API_TOKEN[:10]}...")

if not HUGGINGFACE_API_KEY:
    logger.warning("HUGGINGFACE_API_KEY not found - HuggingFace fallback disabled")
else:
    logger.info(f"HuggingFace API token loaded: {HUGGINGFACE_API_KEY[:10]}...")

class PromptRequest(BaseModel):
    prompt: str

class ImageResponse(BaseModel):
    success: bool
    image_data: str = None
    image_url: str = None
    error: str = None
    model_used: str = None
    provider: str = None

REPLICATE_MODELS = [
    {
        "name": "flux-schnell",
        "model": "black-forest-labs/flux-schnell",
        "params": {
            "prompt": "",
            "width": 512,
            "height": 512,
            "num_outputs": 1,
            "steps": 4,
            "guidance_scale": 3.5
        },
        "timeout": 120
    },
    {
        "name": "flux-dev", 
        "model": "black-forest-labs/flux-dev",
        "params": {
            "prompt": "",
            "num_outputs": 1,
            "steps": 4,
            "guidance_scale": 3.5
        },
        "timeout": 120
    },
    {
        "name": "sdxl-lightning",
        "model": "bytedance/sdxl-lightning-4step",
        "params": {
            "prompt": "",
            "width": 512,
            "height": 512,
            "num_outputs": 1
        },
        "timeout": 120
    }
]

HUGGINGFACE_MODELS = [
    {
        "name": "stable-diffusion-2.1",
        "url": "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-2-1",
        "params": {
            "inputs": "",
            "options": {
                "wait_for_model": True,
                "use_cache": True
            }
        }
    },
    {
        "name": "sd-xl",
        "url": "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-xl-base-1.0",
        "params": {
            "inputs": "",
            "options": {
                "wait_for_model": True,
                "use_cache": True
            }
        }
    }
]

def sanitize_prompt(prompt: str) -> str:
    """Clean and sanitize the prompt"""
    prompt = ' '.join(prompt.split())
    if len(prompt) > 900:
        prompt = prompt[:900] + "..."
    return prompt

async def download_image_async(url: str) -> Optional[bytes]:
    """Download image with async requests and better error handling"""
    try:
        timeout = aiohttp.ClientTimeout(total=30)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(url) as response:
                if response.status == 200:
                    content = await response.read()
                    if len(content) > 1000:
                        return content
                    else:
                        logger.error(f"Downloaded content too small: {len(content)} bytes")
                else:
                    logger.error(f"HTTP {response.status} for URL: {url}")
    except Exception as e:
        logger.error(f"Async download failed: {e}")
    return None

def download_image_sync(url: str) -> Optional[bytes]:
    """Synchronous download fallback"""
    try:
        response = requests.get(url, timeout=30, stream=True)
        response.raise_for_status()
        content = response.content
        if len(content) > 1000:
            return content
    except Exception as e:
        logger.error(f"Sync download failed: {e}")
    return None

def image_to_base64(image_bytes: bytes) -> Optional[str]:
    """Convert image bytes to base64 data URL"""
    try:
        if image_bytes.startswith(b'\xff\xd8'):
            mime_type = "image/jpeg"
        elif image_bytes.startswith(b'\x89PNG'):
            mime_type = "image/png"
        elif image_bytes.startswith(b'GIF'):
            mime_type = "image/gif"
        elif image_bytes.startswith(b'RIFF') and image_bytes[8:12] == b'WEBP':
            mime_type = "image/webp"
        else:
            mime_type = "image/png"
        
        image_data = base64.b64encode(image_bytes).decode('utf-8')
        return f"data:{mime_type};base64,{image_data}"
    except Exception as e:
        logger.error(f"Base64 conversion failed: {e}")
        return None

async def try_replicate_model(client, model_config, prompt: str) -> Optional[dict]:
    """Try a specific Replicate model"""
    try:
        logger.info(f"Trying Replicate model: {model_config['name']}")
        
        params = model_config['params'].copy()
        params['prompt'] = prompt
        
        # Remove timeout from client.run() as it's not supported
        output = await asyncio.wait_for(
            asyncio.get_event_loop().run_in_executor(
                None, 
                lambda: client.run(
                    model_config['model'],
                    input=params
                )
            ),
            timeout=model_config['timeout']
        )
        
        if output and isinstance(output, list) and len(output) > 0:
            image_url = output[0]
            if isinstance(image_url, str) and image_url.startswith(('http://', 'https://')):
                logger.info(f"Replicate {model_config['name']} succeeded: {image_url[:50]}...")
                return {
                    "image_url": image_url,
                    "model_used": model_config['name'],
                    "provider": "replicate"
                }
            else:
                logger.warning(f"Invalid URL from {model_config['name']}: {image_url}")
                
    except asyncio.TimeoutError:
        logger.warning(f"Replicate {model_config['name']} timeout")
    except Exception as e:
        logger.warning(f"Replicate {model_config['name']} failed: {str(e)}")
    
    return None

@app.get("/test-all")
async def test_all_apis():
    """Test all available APIs with detailed diagnostics"""
    test_prompt = "a beautiful sunset over mountains, digital art"
    results = []
    
    logger.info("=== Starting comprehensive API test ===")
    
    if REPLICATE_API_TOKEN:
        logger.info("Testing Replicate API connectivity...")
        try:
            client = replicate.Client(api_token=REPLICATE_API_TOKEN)
            models_list = list(client.models.list())
            replicate_accessible = True
            logger.info(f"Replicate API: OK - Found {len(models_list)} models")
        except Exception as e:
            replicate_accessible = False
            logger.error(f"Replicate API connection failed: {e}")
    else:
        replicate_accessible = False
        logger.warning("Replicate API token not configured")
    
    if HUGGINGFACE_API_KEY:
        logger.info("Testing HuggingFace API connectivity...")
        try:
            headers = {"Authorization": f"Bearer {HUGGINGFACE_API_KEY}"}
            response = requests.get(
                "https://huggingface.co/api/whoami", 
                headers=headers, 
                timeout=10
            )
            if response.status_code == 200:
                huggingface_accessible = True
                user_info = response.json()
                logger.info(f"HuggingFace API: OK - User: {user_info.get('name', 'Unknown')}")
            else:
                huggingface_accessible = False
                logger.error(f"HuggingFace API failed: HTTP {response.status_code}")
        except Exception as e:
            huggingface_accessible = False
            logger.error(f"HuggingFace API connection failed: {e}")
    else:
        huggingface_accessible = False
        logger.warning("HuggingFace API token not configured")
    
    # Test individual models only if APIs are accessible
    if REPLICATE_API_TOKEN and replicate_accessible:
        for model in REPLICATE_MODELS[:2]:  # Test first two models to save time
            try:
                logger.info(f"Testing Replicate model: {model['name']}")
                client = replicate.Client(api_token=REPLICATE_API_TOKEN)
                
                test_params = model['params'].copy()
                test_params['prompt'] = test_prompt
                test_params['steps'] = 2  
                output = await asyncio.wait_for(
                    asyncio.get_event_loop().run_in_executor(
                        None, 
                        lambda: client.run(
                            model['model'],
                            input=test_params
                        )
                    ),
                    timeout=30 
                )
                
                if output and isinstance(output, list) and len(output) > 0:
                    image_url = output[0]
                    if isinstance(image_url, str) and image_url.startswith(('http://', 'https://')):
                        results.append({
                            "provider": "replicate",
                            "model": model["name"],
                            "status": "success",
                            "details": f"Generated URL: {image_url[:50]}..."
                        })
                        logger.info(f"Replicate {model['name']}: SUCCESS")
                    else:
                        results.append({
                            "provider": "replicate",
                            "model": model["name"],
                            "status": "invalid_output",
                            "details": f"Invalid URL format: {str(image_url)[:100]}"
                        })
                        logger.warning(f"Replicate {model['name']}: INVALID OUTPUT")
                else:
                    results.append({
                        "provider": "replicate",
                        "model": model["name"],
                        "status": "no_output",
                        "details": "No output received from model"
                    })
                    logger.warning(f"Replicate {model['name']}: NO OUTPUT")
                    
            except asyncio.TimeoutError:
                results.append({
                    "provider": "replicate",
                    "model": model["name"],
                    "status": "timeout",
                    "details": "Model timed out during test (30s)"
                })
                logger.warning(f"Replicate {model['name']}: TIMEOUT")
            except Exception as e:
                results.append({
                    "provider": "replicate",
                    "model": model["name"],
                    "status": "error",
                    "details": str(e)[:200]
                })
                logger.error(f"Replicate {model['name']}: ERROR - {e}")
    
    if HUGGINGFACE_API_KEY and huggingface_accessible:
        for model in HUGGINGFACE_MODELS[:1]:  
            try:
                logger.info(f"Testing HuggingFace model: {model['name']}")
                
                headers = {
                    "Authorization": f"Bearer {HUGGINGFACE_API_KEY}",
                    "Content-Type": "application/json"
                }
                
                data = model['params'].copy()
                data['inputs'] = test_prompt
                
                timeout = aiohttp.ClientTimeout(total=60)
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    async with session.post(
                        model['url'], 
                        headers=headers, 
                        json=data
                    ) as response:
                        
                        if response.status == 200:
                            image_bytes = await response.read()
                            if len(image_bytes) > 1000:
                                results.append({
                                    "provider": "huggingface",
                                    "model": model["name"],
                                    "status": "success",
                                    "details": f"Generated image: {len(image_bytes)} bytes"
                                })
                                logger.info(f"HuggingFace {model['name']}: SUCCESS")
                            else:
                                results.append({
                                    "provider": "huggingface",
                                    "model": model["name"], 
                                    "status": "small_output",
                                    "details": f"Image too small: {len(image_bytes)} bytes"
                                })
                                logger.warning(f"HuggingFace {model['name']}: SMALL OUTPUT")
                        else:
                            error_text = await response.text()
                            results.append({
                                "provider": "huggingface",
                                "model": model["name"],
                                "status": f"http_{response.status}",
                                "details": error_text[:200]
                            })
                            logger.warning(f"HuggingFace {model['name']}: HTTP {response.status}")
                            
            except asyncio.TimeoutError:
                results.append({
                    "provider": "huggingface",
                    "model": model["name"],
                    "status": "timeout",
                    "details": "Model timed out during test (60s)"
                })
                logger.warning(f"HuggingFace {model['name']}: TIMEOUT")
            except Exception as e:
                results.append({
                    "provider": "huggingface",
                    "model": model["name"],
                    "status": "error",
                    "details": str(e)[:200]
                })
                logger.error(f"HuggingFace {model['name']}: ERROR - {e}")
    
    if not results:
        if REPLICATE_API_TOKEN and not replicate_accessible:
            results.append({
                "provider": "replicate",
                "model": "api_connectivity",
                "status": "api_connection_failed",
                "details": "Could not connect to Replicate API"
            })
        if HUGGINGFACE_API_KEY and not huggingface_accessible:
            results.append({
                "provider": "huggingface", 
                "model": "api_connectivity",
                "status": "api_connection_failed",
                "details": "Could not connect to HuggingFace API"
            })
    
    logger.info("=== API test completed ===")
    
    return {
        "test_results": results,
        "api_status": {
            "replicate_configured": bool(REPLICATE_API_TOKEN),
            "huggingface_configured": bool(HUGGINGFACE_API_KEY),
            "replicate_accessible": replicate_accessible if REPLICATE_API_TOKEN else False,
            "huggingface_accessible": huggingface_accessible if HUGGINGFACE_API_KEY else False
        },
        "prompt_used": test_prompt,
        "timestamp": time.time()
    }

async def try_huggingface_model(model_config, prompt: str) -> Optional[dict]:
    """Try a specific HuggingFace model"""
    if not HUGGINGFACE_API_KEY:
        return None
        
    try:
        logger.info(f"Trying HuggingFace model: {model_config['name']}")
        
        headers = {
            "Authorization": f"Bearer {HUGGINGFACE_API_KEY}",
            "Content-Type": "application/json"
        }
        
        data = model_config['params'].copy()
        data['inputs'] = prompt
        
        timeout = aiohttp.ClientTimeout(total=120)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(
                model_config['url'], 
                headers=headers, 
                json=data
            ) as response:
                
                if response.status == 200:
                    image_bytes = await response.read()
                    if len(image_bytes) > 1000:
                        image_data = image_to_base64(image_bytes)
                        if image_data:
                            logger.info(f"HuggingFace {model_config['name']} succeeded")
                            return {
                                "image_data": image_data,
                                "model_used": model_config['name'],
                                "provider": "huggingface"
                            }
                else:
                    error_text = await response.text()
                    logger.warning(f"HuggingFace {model_config['name']} HTTP {response.status}: {error_text}")
                    
    except asyncio.TimeoutError:
        logger.warning(f"HuggingFace {model_config['name']} timeout")
    except Exception as e:
        logger.warning(f"HuggingFace {model_config['name']} failed: {str(e)}")
    
    return None

async def generate_with_local_fallback(prompt: str) -> Optional[dict]:
    """Generate a simple image locally as final fallback"""
    try:
        from PIL import Image, ImageDraw, ImageFont
        import textwrap
        
        width, height = 512, 512
        image = Image.new('RGB', (width, height), color=(73, 109, 137))
        draw = ImageDraw.Draw(image)
        
        try:
            font = ImageFont.truetype("Arial", 20)
        except:
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 20)
            except:
                font = ImageFont.load_default()
        
        wrapped_text = textwrap.fill(prompt, width=30)
        
        bbox = draw.textbbox((0, 0), wrapped_text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        x = (width - text_width) / 2
        y = (height - text_height) / 2
        
        draw.text((x, y), wrapped_text, font=font, fill=(255, 255, 255))
        
        img_byte_arr = BytesIO()
        image.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()
        
        image_data = image_to_base64(img_byte_arr)
        
        if image_data:
            logger.info("Local fallback image created")
            return {
                "image_data": image_data,
                "model_used": "local-fallback",
                "provider": "local"
            }
            
    except Exception as e:
        logger.error(f"Local fallback failed: {e}")
    
    return None

@app.get("/")
async def root():
    return {
        "message": "AI Image Generator API v2.0", 
        "status": "healthy",
        "replicate_configured": bool(REPLICATE_API_TOKEN),
        "huggingface_configured": bool(HUGGINGFACE_API_KEY),
        "features": ["Multiple APIs", "Async Processing", "Smart Fallbacks", "Local Backup"]
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy", 
        "apis_configured": {
            "replicate": bool(REPLICATE_API_TOKEN),
            "huggingface": bool(HUGGINGFACE_API_KEY)
        },
        "models_available": len(REPLICATE_MODELS) + len(HUGGINGFACE_MODELS) + 1,
        "version": "2.0.0"
    }

@app.get("/status")
async def system_status():
    """Quick system status check"""
    return {
        "status": "operational",
        "backend": "running",
        "apis_configured": {
            "replicate": bool(REPLICATE_API_TOKEN),
            "huggingface": bool(HUGGINGFACE_API_KEY)
        },
        "version": "2.0.0",
        "timestamp": time.time()
    }

@app.post("/generate-image", response_model=ImageResponse)
async def generate_image(request: PromptRequest):
    start_time = time.time()
    logger.info(f"Generation request: {request.prompt[:50]}...")
    
    if not request.prompt.strip():
        return JSONResponse(
            status_code=400,
            content={"success": False, "error": "Prompt cannot be empty"}
        )
    
    prompt = sanitize_prompt(request.prompt)
    
    replicate_results = []
    if REPLICATE_API_TOKEN:
        try:
            client = replicate.Client(api_token=REPLICATE_API_TOKEN)
            
            replicate_tasks = [
                try_replicate_model(client, model, prompt) 
                for model in REPLICATE_MODELS
            ]
            replicate_results = await asyncio.gather(*replicate_tasks, return_exceptions=True)
            
        except Exception as e:
            logger.error(f"Replicate client failed: {e}")
            replicate_results = []
    
    hf_tasks = [
        try_huggingface_model(model, prompt) 
        for model in HUGGINGFACE_MODELS
    ]
    hf_results = await asyncio.gather(*hf_tasks, return_exceptions=True)
    
    all_results = replicate_results + hf_results
    
    for result in all_results:
        if isinstance(result, dict) and result:
            if result.get('image_url'):
                image_bytes = await download_image_async(result['image_url'])
                if not image_bytes:
                    image_bytes = download_image_sync(result['image_url'])
                
                if image_bytes:
                    image_data = image_to_base64(image_bytes)
                    if image_data:
                        result['image_data'] = image_data
            
            if result.get('image_data') or result.get('image_url'):
                generation_time = time.time() - start_time
                logger.info(f"Generation succeeded in {generation_time:.1f}s with {result['provider']}/{result['model_used']}")
                
                response_data = {
                    "success": True,
                    "model_used": result['model_used'],
                    "provider": result['provider'],
                    "generation_time": round(generation_time, 1)
                }
                
                if result.get('image_data'):
                    response_data["image_data"] = result['image_data']
                if result.get('image_url'):
                    response_data["image_url"] = result['image_url']
                
                return response_data
    
    logger.warning("All APIs failed, trying local fallback...")
    local_result = await generate_with_local_fallback(prompt)
    
    if local_result:
        generation_time = time.time() - start_time
        logger.info(f"Local fallback succeeded in {generation_time:.1f}s")
        
        return {
            "success": True,
            "image_data": local_result['image_data'],
            "model_used": local_result['model_used'],
            "provider": local_result['provider'],
            "generation_time": round(generation_time, 1),
            "error": "AI services unavailable - using text visualization"
        }
    
    error_msg = "All image generation services are currently unavailable. Please try again later."
    logger.error(error_msg)
    return JSONResponse(
        status_code=503,
        content={"success": False, "error": error_msg}
    )

@app.get("/models")
async def list_models():
    return {
        "replicate_models": [m["name"] for m in REPLICATE_MODELS],
        "huggingface_models": [m["name"] for m in HUGGINGFACE_MODELS],
        "fallback": "local-text-renderer",
        "status": "Multiple providers with fallback"
    }

@app.get("/test-all")
async def test_all_apis():
    """Test all available APIs with detailed diagnostics"""
    test_prompt = "a beautiful sunset over mountains, digital art"
    results = []
    
    logger.info("=== Starting comprehensive API test ===")
    if REPLICATE_API_TOKEN:
        logger.info("Testing Replicate API connectivity...")
        try:
            client = replicate.Client(api_token=REPLICATE_API_TOKEN)
            models_list = list(client.models.list())
            replicate_accessible = True
            logger.info(f"Replicate API: OK - Found {len(models_list)} models")
        except Exception as e:
            replicate_accessible = False
            logger.error(f"Replicate API connection failed: {e}")
    else:
        replicate_accessible = False
        logger.warning("Replicate API token not configured")
    
    if HUGGINGFACE_API_KEY:
        logger.info("Testing HuggingFace API connectivity...")
        try:
            headers = {"Authorization": f"Bearer {HUGGINGFACE_API_KEY}"}
            response = requests.get(
                "https://huggingface.co/api/whoami", 
                headers=headers, 
                timeout=10
            )
            if response.status_code == 200:
                huggingface_accessible = True
                user_info = response.json()
                logger.info(f"HuggingFace API: OK - User: {user_info.get('name', 'Unknown')}")
            else:
                huggingface_accessible = False
                logger.error(f"HuggingFace API failed: HTTP {response.status_code}")
        except Exception as e:
            huggingface_accessible = False
            logger.error(f"HuggingFace API connection failed: {e}")
    else:
        huggingface_accessible = False
        logger.warning("HuggingFace API token not configured")
    if REPLICATE_API_TOKEN and replicate_accessible:
        for model in REPLICATE_MODELS[:2]:  
            try:
                logger.info(f"Testing Replicate model: {model['name']}")
                client = replicate.Client(api_token=REPLICATE_API_TOKEN)
                
                test_params = model['params'].copy()
                test_params['prompt'] = test_prompt
                test_params['steps'] = 2  
                output = await asyncio.wait_for(
                    asyncio.get_event_loop().run_in_executor(
                        None, 
                        lambda: client.run(
                            model['model'],
                            input=test_params,
                            timeout=30  
                        )
                    ),
                    timeout=40
                )
                
                if output and isinstance(output, list) and len(output) > 0:
                    image_url = output[0]
                    if isinstance(image_url, str) and image_url.startswith(('http://', 'https://')):
                        results.append({
                            "provider": "replicate",
                            "model": model["name"],
                            "status": "success",
                            "details": f"Generated URL: {image_url[:50]}..."
                        })
                        logger.info(f"Replicate {model['name']}: SUCCESS")
                    else:
                        results.append({
                            "provider": "replicate",
                            "model": model["name"],
                            "status": "invalid_output",
                            "details": f"Invalid URL format: {str(image_url)[:100]}"
                        })
                        logger.warning(f"Replicate {model['name']}: INVALID OUTPUT")
                else:
                    results.append({
                        "provider": "replicate",
                        "model": model["name"],
                        "status": "no_output",
                        "details": "No output received from model"
                    })
                    logger.warning(f"Replicate {model['name']}: NO OUTPUT")
                    
            except asyncio.TimeoutError:
                results.append({
                    "provider": "replicate",
                    "model": model["name"],
                    "status": "timeout",
                    "details": "Model timed out during test (30s)"
                })
                logger.warning(f"Replicate {model['name']}: TIMEOUT")
            except Exception as e:
                results.append({
                    "provider": "replicate",
                    "model": model["name"],
                    "status": "error",
                    "details": str(e)[:200]
                })
                logger.error(f"Replicate {model['name']}: ERROR - {e}")
    
    if HUGGINGFACE_API_KEY and huggingface_accessible:
        for model in HUGGINGFACE_MODELS[:1]:  
            try:
                logger.info(f"Testing HuggingFace model: {model['name']}")
                
                headers = {
                    "Authorization": f"Bearer {HUGGINGFACE_API_KEY}",
                    "Content-Type": "application/json"
                }
                
                data = model['params'].copy()
                data['inputs'] = test_prompt
                
                model_status_url = model['url'].replace('/api/', '/') + '/'
                status_response = requests.get(model_status_url, timeout=10)
                
                timeout = aiohttp.ClientTimeout(total=60)
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    async with session.post(
                        model['url'], 
                        headers=headers, 
                        json=data
                    ) as response:
                        
                        if response.status == 200:
                            image_bytes = await response.read()
                            if len(image_bytes) > 1000:
                                results.append({
                                    "provider": "huggingface",
                                    "model": model["name"],
                                    "status": "success",
                                    "details": f"Generated image: {len(image_bytes)} bytes"
                                })
                                logger.info(f"HuggingFace {model['name']}: SUCCESS")
                            else:
                                results.append({
                                    "provider": "huggingface",
                                    "model": model["name"], 
                                    "status": "small_output",
                                    "details": f"Image too small: {len(image_bytes)} bytes"
                                })
                                logger.warning(f"HuggingFace {model['name']}: SMALL OUTPUT")
                        else:
                            error_text = await response.text()
                            results.append({
                                "provider": "huggingface",
                                "model": model["name"],
                                "status": f"http_{response.status}",
                                "details": error_text[:200]
                            })
                            logger.warning(f"HuggingFace {model['name']}: HTTP {response.status}")
                            
            except asyncio.TimeoutError:
                results.append({
                    "provider": "huggingface",
                    "model": model["name"],
                    "status": "timeout",
                    "details": "Model timed out during test (60s)"
                })
                logger.warning(f"HuggingFace {model['name']}: TIMEOUT")
            except Exception as e:
                results.append({
                    "provider": "huggingface",
                    "model": model["name"],
                    "status": "error",
                    "details": str(e)[:200]
                })
                logger.error(f"HuggingFace {model['name']}: ERROR - {e}")
    
    if not results:
        if REPLICATE_API_TOKEN and not replicate_accessible:
            results.append({
                "provider": "replicate",
                "model": "api_connectivity",
                "status": "api_connection_failed",
                "details": "Could not connect to Replicate API"
            })
        if HUGGINGFACE_API_KEY and not huggingface_accessible:
            results.append({
                "provider": "huggingface", 
                "model": "api_connectivity",
                "status": "api_connection_failed",
                "details": "Could not connect to HuggingFace API"
            })
    
    logger.info("=== API test completed ===")
    
    return {
        "test_results": results,
        "api_status": {
            "replicate_configured": bool(REPLICATE_API_TOKEN),
            "huggingface_configured": bool(HUGGINGFACE_API_KEY),
            "replicate_accessible": replicate_accessible if REPLICATE_API_TOKEN else False,
            "huggingface_accessible": huggingface_accessible if HUGGINGFACE_API_KEY else False
        },
        "prompt_used": test_prompt,
        "timestamp": time.time()
    }

if __name__ == "__main__":
    import uvicorn
    
    logger.info("AI Image Generator v2.0 Starting...")
    logger.info("Multiple APIs with smart fallback configured")
    logger.info(f"Replicate: {'OK' if REPLICATE_API_TOKEN else 'NOT CONFIGURED'}")
    logger.info(f"HuggingFace: {'OK' if HUGGINGFACE_API_KEY else 'NOT CONFIGURED'}")
    logger.info("Local fallback: OK")
    logger.info("Server: http://0.0.0.0:8000")
    logger.info("Test endpoint: http://localhost:8000/test-all")
    
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
