import axios from 'axios';
import sharp from 'sharp';
import { CONFIG } from '../config.js';

export class ImageService {
    static async generateImage(prompt, modelName = null, width = 512, height = 512, retryCount = 0) {
        const model = modelName || this.getRandomModel();
        
        const validWidth = Math.min(Math.max(width, 64), 1024);
        const validHeight = Math.min(Math.max(height, 64), 1024);

        try {
            console.log(`Generating image with model: ${model}`);
            console.log(`Dimensions: ${validWidth}x${validHeight}`);
            console.log(`Prompt: ${prompt}`);
            console.log(`Attempt: ${retryCount + 1}/${CONFIG.HF_RETRY_ATTEMPTS}`);

            const response = await axios.post(
                `https://api-inference.huggingface.co/models/${model}`,
                {
                    inputs: prompt,
                    parameters: {
                        num_inference_steps: 25,
                        guidance_scale: 7.5,
                        width: validWidth,
                        height: validHeight
                    },
                    options: {
                        wait_for_model: true,
                        use_cache: true
                    }
                },
                {
                    headers: {
                        'Authorization': `Bearer ${CONFIG.HF_TOKEN}`,
                        'Content-Type': 'application/json'
                    },
                    responseType: 'arraybuffer',
                    timeout: CONFIG.HF_TIMEOUT
                }
            );

            console.log('Image generated successfully');
            
            const optimizedBuffer = await sharp(response.data)
                .jpeg({ 
                    quality: 90,
                    chromaSubsampling: '4:4:4'
                })
                .toBuffer();

            return {
                success: true,
                imageBuffer: optimizedBuffer,
                model: model,
                dimensions: { width: validWidth, height: validHeight }
            };

        } catch (error) {
            console.error('Hugging Face API error:', error.response?.status, error.response?.statusText);
            
            if (error.response?.status === 503 && retryCount < CONFIG.HF_RETRY_ATTEMPTS - 1) {
                const waitTime = Math.pow(2, retryCount) * 5000; 
                console.log(`Model is loading, waiting ${waitTime/1000} seconds before retry...`);
                
                await new Promise(resolve => setTimeout(resolve, waitTime));
                return this.generateImage(prompt, modelName, width, height, retryCount + 1);
            }
            
            let errorMessage = 'Image generation failed';
            
            if (error.response?.status === 503) {
                errorMessage = 'The AI model is currently loading. This can take 30-60 seconds. Please try again in a moment.';
            } else if (error.response?.status === 429) {
                errorMessage = 'Rate limit exceeded. Please wait a moment before generating another image.';
            } else if (error.code === 'ECONNABORTED') {
                errorMessage = 'Image generation timed out. Please try again with a simpler prompt.';
            } else if (error.response?.status === 401) {
                errorMessage = 'Invalid Hugging Face API token. Please check your configuration.';
            } else if (error.response?.status === 404) {
                errorMessage = 'The selected AI model is not available. Please try a different model.';
            } else if (error.message?.includes('Network Error') || error.message?.includes('Failed to fetch')) {
                errorMessage = 'Cannot connect to the AI service. Please check your internet connection and try again.';
            } else if (error.response?.data) {
                try {
                    const errorData = JSON.parse(Buffer.from(error.response.data).toString());
                    errorMessage = errorData.error || errorMessage;
                } catch {
                    errorMessage = error.response.data.toString() || errorMessage;
                }
            } else {
                errorMessage = error.message || errorMessage;
            }

            return {
                success: false,
                error: errorMessage,
                retryable: this.isRetryableError(error)
            };
        }
    }

    static isRetryableError(error) {
        return error.response?.status === 503 || 
               error.code === 'ECONNABORTED' ||
               error.message?.includes('Network Error');
    }

    static getRandomModel() {
        const models = CONFIG.AVAILABLE_MODELS.filter(model => model.trim());
        return models[Math.floor(Math.random() * models.length)];
    }

    static getAvailableModels() {
        return CONFIG.AVAILABLE_MODELS.map(model => ({
            id: model,
            name: model.split('/').pop().replace(/-/g, ' ').replace(/_/g, ' '),
            description: this.getModelDescription(model)
        }));
    }

    static getModelDescription(modelId) {
        const descriptions = {
            'runwayml/stable-diffusion-v1-5': 'Standard Stable Diffusion model, good for general purpose',
            'stabilityai/stable-diffusion-2-1': 'Improved version with better details and coherence',
            'stabilityai/stable-diffusion-xl-base-1.0': 'High-quality model with enhanced resolution'
        };
        return descriptions[modelId] || 'AI Image Generation Model';
    }

    static validateDimensions(width, height) {
        const validWidth = Math.min(Math.max(parseInt(width), 64), 1024);
        const validHeight = Math.min(Math.max(parseInt(height), 64), 1024);
        return { width: validWidth, height: validHeight };
    }

    static async testConnection() {
        try {
            const response = await axios.get('https://huggingface.co/api/whoami-v2', {
                headers: {
                    'Authorization': `Bearer ${CONFIG.HF_TOKEN}`
                },
                timeout: 10000
            });
            return { success: true, user: response.data };
        } catch (error) {
            return { 
                success: false, 
                error: 'Cannot connect to Hugging Face' 
            };
        }
    }
}
