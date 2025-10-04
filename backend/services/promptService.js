import { GoogleGenerativeAI } from '@google/generative-ai';
import { CONFIG } from '../config.js';

const genAI = new GoogleGenerativeAI(CONFIG.GEMINI_API_KEY);
const model = genAI.getGenerativeModel({ model: 'gemini-1.5-flash' });

export class PromptService {
    static async enhancePrompt(originalPrompt, style = 'realistic') {
        try {
            const styleConfig = CONFIG.ART_STYLES[style] || CONFIG.ART_STYLES.realistic;
            
            const enhancementPrompt = `
                Enhance this image generation prompt for Stable Diffusion. Make it more descriptive and detailed.
                Focus on: composition, lighting, colors, details, and mood.
                Incorporate this style: ${styleConfig.enhancement}
                Keep it under 400 characters.
                Return ONLY the enhanced prompt, no explanations or additional text.
                
                Original prompt: ${originalPrompt}
            `;

            const result = await model.generateContent(enhancementPrompt);
            const response = await result.response;
            
            let enhanced = response.text().trim().replace(/["']/g, '');
            
            // Fallback if Gemini returns unexpected format
            if (enhanced.length < originalPrompt.length + 10) {
                enhanced = `${originalPrompt}, ${styleConfig.enhancement}`;
            }
            
            return enhanced;
        } catch (error) {
            console.log('Failed to enhance prompt, using fallback:', error.message);
            const styleConfig = CONFIG.ART_STYLES[style] || CONFIG.ART_STYLES.realistic;
            return `${originalPrompt}, ${styleConfig.enhancement}`;
        }
    }

    static async generateImageDescription(originalPrompt, enhancedPrompt) {
        try {
            const descriptionPrompt = `
                Based on this image generation process:
                Original idea: "${originalPrompt}"
                Enhanced prompt: "${enhancedPrompt}"
                
                Write a brief, creative description (1-2 sentences) of what the generated image looks like.
                Make it engaging and descriptive, but concise.
            `;

            const result = await model.generateContent(descriptionPrompt);
            const response = await result.response;
            return response.text().trim();
        } catch (error) {
            return `An AI-generated image based on: "${originalPrompt}"`;
        }
    }

    static validatePrompt(prompt) {
        if (!prompt || prompt.trim().length === 0) {
            return { valid: false, error: 'Please enter a prompt to generate an image' };
        }

        if (prompt.length > CONFIG.MAX_PROMPT_LENGTH) {
            return { 
                valid: false, 
                error: `Prompt too long. Please keep it under ${CONFIG.MAX_PROMPT_LENGTH} characters.` 
            };
        }

        if (prompt.length < 5) {
            return { 
                valid: false, 
                error: 'Please enter a more detailed prompt (at least 5 characters)' 
            };
        }

        return { valid: true };
    }

    static getAvailableStyles() {
        return Object.entries(CONFIG.ART_STYLES).map(([id, style]) => ({
            id,
            name: style.name,
            description: style.description,
            enhancement: style.enhancement
        }));
    }
}