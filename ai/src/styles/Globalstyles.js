import { createGlobalStyle } from 'styled-components';

const GlobalStyle = createGlobalStyle`
  :root {
    --primary-color: #007bff;
    --secondary-color: #6c757d;
    --accent-color: #28a745;
    --background-dark: #0d1117;
    --card-background: #161b22;
    --text-light: #f8f9fa;
    --text-secondary: #ccd6f6;
    --border-color: #30363d;
    --error-color: #dc3545;

    --spacing-xs: 4px;
    --spacing-sm: 8px;
    --spacing-md: 16px;
    --spacing-lg: 24px;
    --spacing-xl: 32px;
  }

  * {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
  }

  body {
    font-family: 'Inter', sans-serif; /* You might need to import 'Inter' font via Google Fonts or self-host */
    background-color: var(--background-dark);
    color: var(--text-light);
    line-height: 1.6;
    -webkit-font-smoothing: antialiased;
    -moz-osx-font-smoothing: grayscale;
  }

  a {
    text-decoration: none;
    color: var(--primary-color);
  }

  h1, h2, h3, h4, h5, h6 {
    color: var(--text-light);
    margin-bottom: var(--spacing-sm);
    line-height: 1.2;
  }

  p {
    margin-bottom: var(--spacing-md);
  }
`;

export default GlobalStyle;