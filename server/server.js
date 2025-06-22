const express = require('express');
const axios = require('axios');

const app = express();
const port = 8000;

app.use(express.json());

app.post('/api/generate-image', async (req, res) => {
  try {
    const { prompt } = req.body;
    if (!prompt) {
      return res.status(400).json({ error: 'Prompt is required' });
    }

    const response = await axios.post(
      'https://generativelanguage.googleapis.com/v1beta/models/gemini-pro-vision:generateContent',
      {
        contents: [{ parts: [{ text: prompt }] }],
        generationConfig: { temperature: 0.7, topP: 0.8, topK: 40, candidateCount: 1 }
      },
      {
        headers: { 
          'Authorization': `Bearer ${process.env.GEMINI_API_KEY}`,
          'Content-Type': 'application/json'
        }
      }
    );

    const imageUrl = response.data.candidates[0].content.parts[0].inlineData.uri;
    res.json({ success: true, imageUrl });
  } catch (error) {
    console.error('Error:', error);
    res.status(500).json({ 
      error: 'Failed to generate image',
      details: error.message 
    });
  }
});

app.listen(port, () => {
  console.log(`Server running on port ${port}`);
});
