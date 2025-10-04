module.exports = (err, req, res, next) => {
  console.error(err);
  res.status(err.response?.status || 500).json({
    error: err.response?.data?.error || 'Internal Server Error',
  });
};
