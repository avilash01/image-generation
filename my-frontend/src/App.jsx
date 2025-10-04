import React, { useState } from "react";
import axios from "axios";
import "./App.css";

const API_BASE_URL = "http://localhost:8000";

function App() {
  const [prompt, setPrompt] = useState("");
  const [generatedImage, setGeneratedImage] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState("");
  const [generationTime, setGenerationTime] = useState(null);
  const [modelUsed, setModelUsed] = useState("");
  const [providerUsed, setProviderUsed] = useState("");

  const testBackendConnection = async () => {
    try {
      const response = await axios.get(`${API_BASE_URL}/health`);
      console.log("Backend health check:", response.data);
      return true;
    } catch (error) {
      console.error("Backend connection failed:", error);
      return false;
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();

    if (!prompt.trim()) {
      setError("Please enter a prompt");
      return;
    }

    setIsLoading(true);
    setError("");
    setGeneratedImage("");
    setModelUsed("");
    setProviderUsed("");
    setGenerationTime(null);
    const startTime = Date.now();

    try {
      // First test backend connection
      const isBackendConnected = await testBackendConnection();
      if (!isBackendConnected) {
        setError(
          "Cannot connect to backend server. Make sure it's running on port 8000."
        );
        setIsLoading(false);
        return;
      }

      // Test all APIs (optional - remove if not needed)
      console.log("Testing all APIs...");
      const apiTest = await axios.get(`${API_BASE_URL}/test-all`);
      console.log("API Test results:", apiTest.data);

      console.log("Sending request to backend...");
      const response = await axios.post(
        `${API_BASE_URL}/generate-image`,
        {
          prompt: prompt.trim(),
        },
        {
          timeout: 180000, // 3 minute timeout
        }
      );

      const endTime = Date.now();
      setGenerationTime(Math.round((endTime - startTime) / 1000));

      console.log("Backend response:", response.data);

      if (response.data.success) {
        if (response.data.image_data) {
          setGeneratedImage(response.data.image_data);
          console.log("Using base64 image data");
        } else if (response.data.image_url) {
          setGeneratedImage(response.data.image_url);
          console.log("Using image URL");
        } else {
          setError("No image data received from server");
        }

        // Store model and provider info if available
        if (response.data.model_used) {
          setModelUsed(response.data.model_used);
          console.log(`Image generated using: ${response.data.model_used}`);
        }
        if (response.data.provider) {
          setProviderUsed(response.data.provider);
          console.log(`Provider: ${response.data.provider}`);
        }

        // Show warning if there's a secondary error message
        if (response.data.error) {
          console.warn("Backend warning:", response.data.error);
        }
      } else {
        setError(response.data.error || "Failed to generate image");
      }
    } catch (err) {
      const endTime = Date.now();
      setGenerationTime(Math.round((endTime - startTime) / 1000));

      console.error("Full error details:", err);

      if (
        err.code === "NETWORK_ERROR" ||
        err.message.includes("Network Error")
      ) {
        setError(
          "Network error: Cannot connect to backend. Make sure the Python server is running on port 8000."
        );
      } else if (err.code === "ECONNREFUSED") {
        setError(
          "Connection refused: Backend server is not running. Start it with: python main.py"
        );
      } else if (err.response?.data?.error) {
        setError(`Generation error: ${err.response.data.error}`);
      } else if (err.message.includes("timeout")) {
        setError(
          "Request timeout: Image generation took too long. The free models might be busy. Try again in a moment."
        );
      } else {
        setError(`Unexpected error: ${err.message}`);
      }
    } finally {
      setIsLoading(false);
    }
  };

  const handleClear = () => {
    setPrompt("");
    setGeneratedImage("");
    setError("");
    setModelUsed("");
    setProviderUsed("");
    setGenerationTime(null);
  };

  const handleExamplePrompt = (examplePrompt) => {
    setPrompt(examplePrompt);
  };

  const copyToClipboard = async () => {
    try {
      await navigator.clipboard.writeText(prompt);
      // Temporary visual feedback
      const copyBtn = document.querySelector(".copy-btn");
      if (copyBtn) {
        const originalText = copyBtn.textContent;
        copyBtn.textContent = "✅ Copied!";
        setTimeout(() => {
          copyBtn.textContent = originalText;
        }, 2000);
      }
    } catch (err) {
      console.error("Failed to copy prompt: ", err);
    }
  };

  const handleImageError = (e) => {
    console.error("Image failed to load");
    setError("Failed to display generated image. Please try generating again.");
    e.target.style.display = "none";
  };

  const handleImageLoad = () => {
    console.log("Image loaded successfully");
    setError(""); // Clear any previous errors
  };

  const examplePrompts = [
    "A serene landscape with mountains and a lake at sunset",
    "A cute cat wearing a wizard hat, digital art",
    "A cyberpunk cityscape at night with neon lights",
    "A beautiful flower garden with butterflies, watercolor style",
    "An astronaut riding a horse on Mars, photorealistic",
    "A majestic dragon soaring over a medieval castle",
    "A cozy cabin in a snowy forest, warm lights in the windows",
    "A futuristic spaceship landing on an alien planet",
    "A bowl of fruit on a wooden table, still life painting",
    "A samurai warrior in a cherry blossom forest",
  ];

  const getModelDisplayName = (model) => {
    if (!model) return "AI Model";
    const modelMap = {
      "flux-schnell": "FLUX Schnell (Fast)",
      "flux-dev": "FLUX Dev",
      "flux-1.1-pro": "FLUX 1.1 Pro",
      "sdxl-lightning": "SDXL Lightning",
      "stable-diffusion-2.1": "Stable Diffusion 2.1",
      "sd-xl": "Stable Diffusion XL",
      "local-fallback": "Local Text Renderer",
    };
    return modelMap[model] || model;
  };

  const getProviderDisplayName = (provider) => {
    if (!provider) return "";
    const providerMap = {
      replicate: "Replicate API",
      huggingface: "HuggingFace API",
      local: "Local Fallback",
    };
    return providerMap[provider] || provider;
  };

  return (
    <div className="app">
      <div className="container">
        <header className="header">
          <h1>AI Image Generator</h1>
          <p>Transform your text into stunning images - Completely Free!</p>
          <div className="model-info">
            <span className="model-tag">Multiple AI Providers</span>
            <span className="free-tag">Smart Fallback System</span>
          </div>
        </header>

        {/* Example Prompts */}
        <div className="examples-section">
          <h3>Try these examples:</h3>
          <div className="example-prompts">
            {examplePrompts.map((example, index) => (
              <button
                key={index}
                className="example-btn"
                onClick={() => handleExamplePrompt(example)}
                disabled={isLoading}
              >
                {example}
              </button>
            ))}
          </div>
        </div>

        <form onSubmit={handleSubmit} className="prompt-form">
          <div className="input-group">
            <label htmlFor="prompt">
              Describe your image:
              <button
                type="button"
                className="copy-btn"
                onClick={copyToClipboard}
                title="Copy prompt to clipboard"
                disabled={!prompt.trim()}
              >
                📋 Copy
              </button>
            </label>
            <textarea
              id="prompt"
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              placeholder="Describe the image you want to generate... (e.g., 'A serene landscape with mountains and a lake at sunset')"
              disabled={isLoading}
              rows={4}
            />
            <div className="prompt-stats">
              <span>{prompt.length}/1000 characters</span>
              {prompt.length > 800 && (
                <span className="character-warning"> (Getting long)</span>
              )}
            </div>
          </div>

          <div className="button-group">
            <button
              type="submit"
              disabled={isLoading || !prompt.trim()}
              className="generate-btn"
            >
              {isLoading ? (
                <>
                  <span className="spinner-small"></span>
                  Generating...
                </>
              ) : (
                " Generate Image"
              )}
            </button>

            <button
              type="button"
              onClick={handleClear}
              disabled={isLoading}
              className="clear-btn"
            >
              🗑️ Clear All
            </button>
          </div>
        </form>

        {error && (
          <div className="error-message">
            <div className="error-header">
              <span className="error-icon">⚠️</span>
              <strong>Error</strong>
            </div>
            <div className="error-content">{error}</div>
            <div className="troubleshooting">
              <details>
                <summary>Troubleshooting Tips</summary>
                <ul>
                  <li>
                    Make sure the Python backend is running:{" "}
                    <code>python main.py</code>
                  </li>
                  <li>
                    Check that port 8000 is not being used by another
                    application
                  </li>
                  <li>Try a simpler or more descriptive prompt</li>
                  <li>Check your internet connection</li>
                  <li>
                    Free models may be busy - wait 1-2 minutes and try again
                  </li>
                  <li>The system will automatically try different providers</li>
                  <li>Check your API tokens in the .env file</li>
                </ul>
              </details>
            </div>
          </div>
        )}

        <div className="result-section">
          {isLoading && (
            <div className="loading">
              <div className="spinner"></div>
              <p>Testing available AI services...</p>
              <p className="loading-details">
                Trying multiple providers with automatic fallback
              </p>
              <div className="loading-steps">
                <div className="step">🔍 Testing Replicate API...</div>
                <div className="step">🔍 Testing HuggingFace API...</div>
                <div className="step">🔄 Preparing fallbacks...</div>
              </div>
              <p className="loading-note">
                Free models may be loading or in queue. Please be patient.
              </p>
              {generationTime && generationTime > 60 && (
                <p className="loading-warning">
                  Still working... Trying different models. This can take up to
                  3 minutes.
                </p>
              )}
            </div>
          )}

          {generatedImage && !isLoading && (
            <div className="image-result">
              <div className="result-header">
                <h3>🎉 Your Generated Image</h3>
                <div className="result-meta">
                  {generationTime && (
                    <span className="generation-time">
                      Generated in {generationTime} seconds
                    </span>
                  )}
                  {modelUsed && (
                    <span className="model-used">
                      🤖 {getModelDisplayName(modelUsed)}
                    </span>
                  )}
                  {providerUsed && (
                    <span className="provider-used">
                      ⚡ {getProviderDisplayName(providerUsed)}
                    </span>
                  )}
                  <span className="base64-notice">📁 Embedded Image</span>
                </div>
              </div>

              <div className="image-container">
                <img
                  src={generatedImage}
                  alt={`Generated from prompt: ${prompt}`}
                  onError={handleImageError}
                  onLoad={handleImageLoad}
                />
              </div>

              <div className="image-actions">
                <a
                  href={generatedImage}
                  download={`ai-generated-image-${Date.now()}.png`}
                  className="download-btn"
                >
                  💾 Download Image
                </a>

                <button
                  onClick={() => {
                    navigator.clipboard
                      .writeText(prompt)
                      .then(() => {
                        const btn = document.querySelector(".copy-prompt-btn");
                        if (btn) {
                          const originalText = btn.textContent;
                          btn.textContent = "✅ Copied!";
                          setTimeout(() => {
                            btn.textContent = originalText;
                          }, 2000);
                        }
                      })
                      .catch((err) => console.error("Copy failed:", err));
                  }}
                  className="copy-prompt-btn"
                  title="Copy prompt to clipboard"
                >
                  📋 Copy Prompt
                </button>

                <button onClick={handleClear} className="new-image-btn">
                  🎨 Create New Image
                </button>
              </div>

              <div className="prompt-display">
                <strong>Prompt used:</strong>
                <p>"{prompt}"</p>
              </div>

              {modelUsed && (
                <div className="model-info-display">
                  <strong>AI Model Used:</strong>{" "}
                  {getModelDisplayName(modelUsed)}
                  {providerUsed && (
                    <span> via {getProviderDisplayName(providerUsed)}</span>
                  )}
                </div>
              )}
            </div>
          )}

          {!isLoading && !generatedImage && !error && (
            <div className="welcome-message">
              <div className="welcome-icon">🎨</div>
              <h3>Ready to Create!</h3>
              <p>
                Enter a prompt above or click an example to generate your first
                AI image.
              </p>
              <div className="features">
                <div className="feature">
                  <span>💵</span>
                  <span>Completely FREE - No payments ever</span>
                </div>
                <div className="feature">
                  <span>⚡</span>
                  <span>Multiple AI providers with smart fallback</span>
                </div>
                <div className="feature">
                  <span>🎯</span>
                  <span>High quality 512x512 images</span>
                </div>
                <div className="feature">
                  <span>🔄</span>
                  <span>Automatic fallback if services are busy</span>
                </div>
              </div>
              <div className="free-notice">
                <p>
                  <strong>Note:</strong> Using Replicate API, HuggingFace API,
                  and local fallback. Generation may take 30-120 seconds.
                </p>
              </div>
            </div>
          )}
        </div>

        <footer className="footer">
          <p>
            <strong>Powered by Multiple AI APIs + Smart Fallbacks</strong>
          </p>
          <div className="footer-links">
            <a
              href="https://replicate.com/"
              target="_blank"
              rel="noopener noreferrer"
            >
              About Replicate
            </a>
            <span>•</span>
            <a
              href="https://huggingface.co/"
              target="_blank"
              rel="noopener noreferrer"
            >
              About HuggingFace
            </a>
            <span>•</span>
            <a
              href="https://replicate.com/account"
              target="_blank"
              rel="noopener noreferrer"
            >
              Get Free API Keys
            </a>
          </div>
          <p className="footer-note">
            💵 Completely Free - Multiple providers with automatic retry!
          </p>
          <p className="footer-tech">
            Using: Replicate API • HuggingFace API • Local Fallback
          </p>
        </footer>
      </div>
    </div>
  );
}

export default App;
