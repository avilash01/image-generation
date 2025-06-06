import React from "react";
import styled from "styled-components";
import { CircularProgress } from "@mui/material";

const StyledButton = styled.button`
  border-radius: 10px;
  background-color: var(--primary-color);
  color: var(--text-light);
  font-size: 14px;
  font-weight: 600;
  cursor: pointer;
  transition: all 0.3s ease;
  display: flex;
  align-items: center;
  justify-content: center;
  gap: var(--spacing-sm);
  height: min-content;
  padding: var(--spacing-sm) var(--spacing-lg);
  border: none;

  opacity: ${({ disabled }) => (disabled ? 0.6 : 1)};
  pointer-events: ${({ disabled }) => (disabled ? "none" : "auto")};

  &:hover {
    background-color: #0056b3;
    box-shadow: 0 4px 8px rgba(0,0,0,0.2);
  }

  &:active {
    transform: translateY(1px);
  }

  @media (max-width: 600px) {
    padding: var(--spacing-xs) var(--spacing-md);
    font-size: 12px;
  }
`;

const Button = ({ text, isLoading = false, isDisabled = false, onClick, flex = 1, style }) => {
  return (
    <StyledButton
      onClick={() => {
        if (!isDisabled && !isLoading && onClick) onClick();
      }}
      disabled={isDisabled}
      style={{ flex, ...style }}
    >
      {isLoading && <CircularProgress size={18} sx={{ color: 'inherit' }} />}
      {!isLoading && text}
    </StyledButton>
  );
};

export default Button;