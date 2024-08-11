import React from 'react';
import { ChakraProvider } from '@chakra-ui/react';
import HomePage from './Home/home';

function App() {
  return (
    <ChakraProvider>
      <HomePage/>
    </ChakraProvider>
  );
}

export default App;
