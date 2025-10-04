const huggingfaceClient = require('../utils/huggingfaceClient');

exports.generateImage = async (req, res, next) => {
  const { prompt } = req.body;
  if (!prompt) return res.status(400).json({ error: 'Prompt is required' });

  try {
    const response = await huggingfaceClient.post('', { inputs: prompt });
    res.set('Content-Type', 'image/png');
    res.send(response.data);
  } catch (error) {
    next(error);
  }
};
