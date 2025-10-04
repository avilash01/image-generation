import { config } from 'dotenv';

config();

export const CONFIG = {
    // Server
    PORT: process.env.PORT || 3000,
    NODE_ENV: process.env.NODE_ENV || 'development',
    
    // AI APIs
    GEMINI_API_KEY: process.env.GEMINI_API_KEY,
    HF_TOKEN: process.env.HF_TOKEN,
    
    // Models
    AVAILABLE_MODELS: (process.env.AVAILABLE_MODELS || 'runwayml/stable-diffusion-v1-5').split(','),
    
    // Image Settings
    DEFAULT_WIDTH: parseInt(process.env.DEFAULT_WIDTH) || 512,
    DEFAULT_HEIGHT: parseInt(process.env.DEFAULT_HEIGHT) || 512,
    MAX_PROMPT_LENGTH: parseInt(process.env.MAX_PROMPT_LENGTH) || 1000,
    RATE_LIMIT_PER_MINUTE: parseInt(process.env.RATE_LIMIT_PER_MINUTE) || 10,
    
    // API Settings
    HF_TIMEOUT: 120000, // 2 minutes
    HF_RETRY_ATTEMPTS: 3,
    
    // Art Styles
    ART_STYLES: {
        realistic: {
            name: "Photorealistic",
            description: "Professional photography style with high detail",
            enhancement: "professional photography, 4K, highly detailed, sharp focus, studio lighting, photorealistic"
        },
        artistic: {
            name: "Digital Art",
            description: "Trending digital art style",
            enhancement: "digital art, trending on artstation, dramatic lighting, intricate details, masterpiece"
        },
        cinematic: {
            name: "Cinematic",
            description: "Movie-like atmospheric scenes",
            enhancement: "cinematic, atmospheric, moody lighting, detailed, film noir, dramatic"
        },
        fantasy: {
            name: "Fantasy",
            description: "Magical and imaginative scenes",
            enhancement: "concept art, character design, detailed, fantasy, dramatic, magical"
        },
        anime: {
            name: "Anime",
            description: "Japanese animation style",
            enhancement: "anime style, vibrant colors, clean lines, detailed, professional illustration"
        },
        oil_painting: {
            name: "Oil Painting",
            description: "Classical painting style",
            enhancement: "oil painting, textured, brush strokes, classical art, masterpiece"
        },
        watercolor: {
            name: "Watercolor",
            description: "Soft watercolor painting style",
            enhancement: "watercolor painting, soft colors, fluid, transparent, artistic"
        },
        cyberpunk: {
            name: "Cyberpunk",
            description: "Futuristic neon-lit scenes",
            enhancement: "cyberpunk, neon lights, futuristic, detailed, sci-fi, atmospheric"
        },
        minimalist: {
            name: "Minimalist",
            description: "Simple and clean design",
            enhancement: "minimalist, clean lines, simple composition, modern, elegant"
        },
        surreal: {
            name: "Surreal",
            description: "Dreamlike and imaginative",
            enhancement: "surreal, dreamlike, imaginative, bizarre, psychological, symbolic"
        }
    }
};

export default CONFIG;