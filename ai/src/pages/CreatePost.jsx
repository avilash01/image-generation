import React, { useState } from 'react';
import styled from 'styled-components';
import axios from 'axios';
import { Button as MuiButton, CircularProgress, TextField } from '@mui/material';
import { ThemeProvider } from '@mui/material/styles';
import { muiTheme } from '../styles/Theme';

const CreatePost = () => {
  const [prompt, setPrompt] = useState(''); 
  const [generatedImageUrl, setGeneratedImageUrl] = useState(null); 
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null); 

  const generateImage=async()=>{
    if (!prompt || prompt.trim() === '') {
      setError("Please enter a prompt to generate an image.");
      return; 
    }
    setLoading(true); 
    setGeneratedImageUrl(null); 
    setError(null); 
    try {
      const response = await axios.post('/api/generate-image', { prompt });
      if (response.data.success) {
        setGeneratedImageUrl(response.data.imageUrl);
      } else {
        setError(response.data.message || 'Unknown error from backend.');
      }
    }
    catch(err){
      console.error("Frontend: Image generation failed:", err); 
      let errorMessage = 'Failed to generate image. Please try again.'; 
      if (err.response) {
        if (err.response.status === 404) {
          errorMessage = 'API endpoint not found. Please ensure your backend server is running and configured correctly.';
        } else if (err.response.data && err.response.data.message) {
          errorMessage = `Error: ${err.response.data.message}`;
        } else {
          errorMessage = `Server responded with status: ${err.response.status}`;
        }
      } else if (err.request) {
        errorMessage = 'No response from backend server. Please ensure the backend is running.';
      } else {
        errorMessage = `Request setup error: ${err.message}`;
      }
      setError(errorMessage); 
    } finally {
      setLoading(false); 
    }
  };
  return (
    <ThemeProvider theme={muiTheme}>
      <Wrapper>
        <PageTitle>Generate a new image with DALL-E 2</PageTitle>

        <InputGroup>
          <TextField
            id="prompt-input"
            label="Image Prompt"
            variant="outlined"
            fullWidth
            multiline
            rows={4} 
            placeholder="e.g., A majestic cat wearing a crown in a futuristic city"
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            disabled={loading} 
            margin="normal" 
          />
        </InputGroup>
        {error && <ErrorMessage>{error}</ErrorMessage>}

        <StyledButtonContainer>
          <MuiButton
            variant="contained" 
            color="primary" 
            fullWidth 
            onClick={generateImage} 
            disabled={loading} 
            sx={{ height: '50px', fontSize: '16px', fontWeight: 'bold' }} 
          >
            {loading ? <CircularProgress size={24} sx={{ color: 'white' }} /> : "Generate Image"}
          </MuiButton>
        </StyledButtonContainer>
        {generatedImageUrl && (
          <GeneratedImageContainer>
            <ImageTitle>Generated Image:</ImageTitle>
            <ImagePreview src={generatedImageUrl} alt="Generated AI Image" />
            <DownloadLink href={generatedImageUrl} target="_blank" download="dalle-2-image.png">
              Download Image
            </DownloadLink>
          </GeneratedImageContainer>
        )}
      </Wrapper>
    </ThemeProvider>
  );
};

export default CreatePost;
const Wrapper = styled.div`
  padding: var(--spacing-lg, 24px); /* Fallback value 24px */
  background-color: var(--background-dark, #282c34); /* Dark background */
  min-height: calc(100vh - 64px); /* Adjust for Navbar height if you have one */
  display: flex;
  flex-direction: column;
  gap: var(--spacing-lg, 24px); /* Space between elements */
  align-items: center; /* Center items horizontally */
  color: var(--text-light, #f0f0f0); /* Light text color */
`;

const PageTitle = styled.h2`
  color: var(--text-light, #f0f0f0);
  font-size: 28px;
  font-weight: bold;
  margin-bottom: var(--spacing-sm, 8px);
  text-align: center;

  @media (max-width: 600px) {
    font-size: 22px;
  }
`;

const InputGroup = styled.div`
  width: 100%;
  max-width: 600px; /* Constrain width for larger screens */
`;

const StyledButtonContainer = styled.div`
  width: 100%;
  max-width: 600px; /* Constrain width for larger screens */
`;

const GeneratedImageContainer = styled.div`
  margin-top: var(--spacing-xl, 48px);
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: var(--spacing-md, 16px);
  width: 100%;
  max-width: 600px;
  padding: var(--spacing-lg, 24px);
  background-color: var(--card-background, #3a3f47);
  border-radius: 12px;
  box-shadow: 0 4px 10px rgba(0, 0, 0, 0.2);
  border: 1px solid var(--border-color, #4f555e); /* Subtle border */
`;

const ImageTitle = styled.h3`
  color: var(--primary-color, #61dafb);
  font-size: 20px;
  font-weight: bold;
`;

const ImagePreview = styled.img`
  max-width: 100%;
  height: auto;
  border-radius: 8px;
  box-shadow: 0 2px 5px rgba(0, 0, 0, 0.15);
  object-fit: contain; /* Ensures image fits within its container without cropping */
`;

const DownloadLink = styled.a`
  color: var(--accent-color, #a8d5e2);
  text-decoration: none;
  font-weight: bold;
  &:hover {
    text-decoration: underline;
  }
`;

const ErrorMessage = styled.p`
  color: var(--error-color, #ff6b6b);
  font-size: 14px;
  margin-top: var(--spacing-sm, 8px); /* Adjusted for consistency */
  margin-bottom: var(--spacing-md, 16px);
  text-align: center;
  width: 100%;
  max-width: 600px;
  padding: 8px 12px;
  background-color: rgba(255, 107, 107, 0.1);
  border-radius: 4px;
  border: 1px solid var(--error-color, #ff6b6b);
`;