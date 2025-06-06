import React from 'react';
import styled from 'styled-components';
import { useNavigate, useLocation } from 'react-router-dom';
import Button from './button'; 

const Navbar = () => {
  const navigate = useNavigate();
  const location = useLocation();
  const path = location.pathname.split('/');

  return (
    <Nav>
      <Title>
        GenAI
        {path[1] === 'create' && (
          <SmallButton
            onClick={() => navigate('/')}
            text="Explore Posts"
          />
        )}
      </Title>
      <Button onClick={() => navigate('/create')} text="+ Create new post" />
    </Nav>
  );
};

export default Navbar;

const Nav = styled.div`
  display: flex;
  justify-content: space-between;
  align-items: center;
  background-color: var(--background-dark);
  padding: var(--spacing-md) var(--spacing-lg);
  box-shadow: 0 0 10px rgba(0, 0, 0, 0.15);
  height: 64px;
  position: sticky;
  top: 0;
  z-index: 1000;
  border-bottom: 1px solid var(--border-color); /* Subtle border */
`;

const Title = styled.h1`
  color: var(--text-light);
  font-size: 20px;
  display: flex;
  align-items: center;
  gap: var(--spacing-md);
  font-weight: bold;
`;

const SmallButton = styled(Button)`
  font-size: 14px;
  padding: var(--spacing-xs) var(--spacing-md);
  border-radius: 6px;
  background-color: var(--secondary-color);
  &:hover {
    background-color: #5a6268;
  }
`;