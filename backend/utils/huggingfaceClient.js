const axios = require('axios');

const huggingfaceClient = axios.create({
  baseURL: 'https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-2',
  headers: {
    Authorization: `Bearer ${process.env.HUGGINGFACE_API_KEY}`
  },
  responseType: 'arraybuffer',
  timeout: 30000,
});

module.exports = huggingfaceClient;
