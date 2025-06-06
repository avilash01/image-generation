import React from 'react';
import styled from 'styled-components';

const Home = () => {
  return (
    <HomeWrapper>
      <HomeTitle>Welcome to the AI Image Generation Gallery!</HomeTitle>
      <HomeText>Go to "+ Create new post" to generate your own images.</HomeText>
      <HomeText>
      </HomeText>
    </HomeWrapper>
  );
};

export default Home;

const HomeWrapper = styled.div`
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  min-height: calc(100vh - 64px); /* Adjust for Navbar height */
  background-color: var(--background-dark);
  color: var(--text-light);
  padding: var(--spacing-lg);
  text-align: center;
`;

const HomeTitle = styled.h1`
  font-size: 36px;
  font-weight: bold;
  margin-bottom: var(--spacing-md);

  @media (max-width: 768px) {
    font-size: 28px;
  }
`;

const HomeText = styled.p`
  font-size: 18px;
  max-width: 700px;
  margin-bottom: var(--spacing-lg);

  @media (max-width: 768px) {
    font-size: 16px;
  }
`;