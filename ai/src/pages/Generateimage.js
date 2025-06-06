
import OpenAI from 'openai';
export default async function handler(req, res) {
  if (req.method !== 'POST') {
    return res.status(405).json({ message: 'Method Not Allowed' });
  }
  const { prompt } = req.body;
  if (!prompt || prompt.trim() === '') {
    return res.status(400).json({ message: 'Prompt is required.' });
  }

  const openai = new OpenAI({
    apiKey: process.env.OPENAI_API_KEY,
  });

  try {
    const response = await openai.images.generate({
      model: "dall-e-2", 
      prompt: prompt,
      n: 1,
      size: "512x512", 
      response_format: "url", 
    });

    const imageUrl = response.data[0].url;

    res.status(200).json({ imageUrl });

  } catch (error) {
    console.error('Error generating image from OpenAI:', error.response ? error.response.data : error.message);

    let errorMessage = 'Failed to generate image.';
    if (error.response && error.response.status) {
        errorMessage += ` Status: ${error.response.status}.`;
        if (error.response.data && error.response.data.error && error.response.data.error.message) {
            errorMessage += ` ${error.response.data.error.message}`;
        }
    } else {
        errorMessage += ` ${error.message}`;
    }

    res.status(error.response?.status || 500).json({ message: errorMessage });
  }
}