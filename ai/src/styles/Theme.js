import { createTheme } from '@mui/material/styles';

export const muiTheme = createTheme({
  palette: {
    primary: {
      main: '#61dafb',
    },
    secondary: {
      main: '#a8d5e2', 
    },
    error: {
      main: '#ff6b6b', 
    },
    background: {
      default: '#282c34', 
      paper: '#3a3f47', 
    },
    text: {
      primary: '#f0f0f0', 
      secondary: '#bbbbbb', 
    },
  },
  typography: {
    fontFamily: 'Roboto, sans-serif', 
  },
  spacing: 8, 
  customVariables: {
    spacingLg: '24px',
    spacingMd: '16px',
    spacingSm: '8px',
    spacingXl: '48px',
    backgroundDark: '#282c34',
    cardBackground: '#3a3f47',
    textLight: '#f0f0f0',
    primaryColor: '#61dafb',
    accentColor: '#a8d5e2',
    errorColor: '#ff6b6b',
    borderColor: '#4f555e',
  },
});