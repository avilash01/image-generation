require('dotenv').config(); 

const express = require('express');
const cors = require('cors'); 
const OpenAI = require('openai'); 

const app = express();
const PORT = process.env.PORT || 5000; 

app.use(cors());
app.use(express.json());
const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
});

app.post('/api/generate-image', async (req, res) => {
  const { prompt } = req.body; 

  if (!prompt || prompt.trim() === '') {
    return res.status(400).json({ success: false, message: 'Prompt is required to generate an image.' });
  }

  try {
    console.log(`Backend: Attempting to generate image for prompt: "${prompt}"`);

    const response = await openai.images.generate({
      model: "dall-e-3", 
      prompt: prompt,
      n: 1, 
      size: "1024x1024", 
      response_format: "url", 
    });

    res.status(200).json({ success: true, imageUrl });

  } catch (error) {
    console.error("Backend: Error generating image with OpenAI:", error);

    if (error.response) {
      console.error("OpenAI API Error Data:", error.response.data);
      res.status(error.response.status).json({
        success: false,
        message: error.response.data.error.message || 'Error from OpenAI API.',
        code: error.response.status
      });
    } else if (error.request) {
      res.status(500).json({ success: false, message: 'No response from OpenAI API.' });
    } 
    else {
      res.status(500).json({ success: false, message: error.message || 'Internal server error during image generation.' });
    }
  }
});

app.listen(PORT, () => {
  console.log(`Backend server running on http://localhost:${PORT}`);
  console.log('environment variable is set in backend/.env file.');
});
