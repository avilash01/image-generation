require('dotenv').config(); // Load environment variables from .env file

const express = require('express');
const cors = require('cors'); // For handling CORS if frontend and backend are on different ports/domains
const OpenAI = require('openai'); // OpenAI's official Node.js library

const app = express();
const PORT = process.env.PORT || 5000; // Use port from environment or default to 5000

// --- Middleware ---
// Enable CORS for all routes. In production, you might restrict this to your frontend's domain.
app.use(cors());
// Enable parsing of JSON request bodies (e.g., the 'prompt' sent from frontend)
app.use(express.json());

// --- OpenAI API Configuration ---
// Get the API key from environment variables.
// The 'openai' library will automatically pick it up if process.env.OPENAI_API_KEY is set.
const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
});

// --- API Route for Image Generation ---
app.post('/api/generate-image', async (req, res) => {
  const { prompt } = req.body; // Extract the prompt from the request body

  // Basic validation: ensure prompt is provided
  if (!prompt || prompt.trim() === '') {
    return res.status(400).json({ success: false, message: 'Prompt is required to generate an image.' });
  }

  try {
    console.log(`Backend: Attempting to generate image for prompt: "${prompt}"`);

    // Call the OpenAI DALL-E API
    const response = await openai.images.generate({
      model: "dall-e-3", // Using DALL-E 3 for best quality. Can change to "dall-e-2" if preferred.
      prompt: prompt,
      n: 1, // Generate 1 image
      size: "1024x1024", // Choose desired resolution. DALL-E 3 supports 1024x1024, 1792x1024, 1024x1792
      response_format: "url", // Request the image URL
    });

    const imageUrl = response.data[0].url; // Extract the URL of the generated image
    console.log("Backend: Image URL generated:", imageUrl);

    // Send the image URL back to the frontend
    res.status(200).json({ success: true, imageUrl });

  } catch (error) {
    console.error("Backend: Error generating image with OpenAI:", error);

    // Handle different types of errors from OpenAI API
    if (error.response) {
      // OpenAI API returned an error response (e.g., invalid key, policy violation)
      console.error("OpenAI API Error Data:", error.response.data);
      res.status(error.response.status).json({
        success: false,
        message: error.response.data.error.message || 'Error from OpenAI API.',
        code: error.response.status
      });
    } else if (error.request) {
      // The request was made but no response was received (e.g., network issue)
      res.status(500).json({ success: false, message: 'No response from OpenAI API. Please check network connection.' });
    } else {
      // Something else happened in setting up the request that triggered an Error
      res.status(500).json({ success: false, message: error.message || 'Internal server error during image generation.' });
    }
  }
});

// Start the Express server
app.listen(PORT, () => {
  console.log(`Backend server running on http://localhost:${PORT}`);
  console.log('Ensure OPENAI_API_KEY environment variable is set in backend/.env file.');
});