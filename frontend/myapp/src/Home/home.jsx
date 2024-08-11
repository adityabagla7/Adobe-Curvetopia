import { 
    Box, 
    Button, 
    Checkbox, 
    Container, 
    Flex, 
    FormControl, 
    FormLabel, 
    Heading, 
    Image, 
    Input, 
    Modal, 
    ModalBody, 
    ModalCloseButton, 
    ModalContent, 
    ModalFooter, 
    ModalHeader, 
    ModalOverlay, 
    Text, 
    useBreakpointValue, 
    useDisclosure, 
    VStack 
} from '@chakra-ui/react';
import React, { useState, useEffect } from 'react';
import axios from 'axios';
const HomePage = () => {
    const [showOutput, setShowOutput] = useState(false);
    const [drawData, setDrawData] = useState([]);
    const [csvFile, setCsvFile] = useState(null);
    const [googleDoodles, setGoogleDoodles] = useState([]);
    const [selectedDoodles, setSelectedDoodles] = useState([]);
    const { isOpen: isDrawOpen, onOpen: onDrawOpen, onClose: onDrawClose } = useDisclosure();
    const { isOpen: isDoodlesOpen, onOpen: onDoodlesOpen, onClose: onDoodlesClose } = useDisclosure();

    const imageSize = useBreakpointValue({ base: 'full', md: '300px' });

    useEffect(() => {
        const fetchGoogleDoodles = async () => {
            try {
                const response = await fetch('https://www.google.com/doodles/json');
                if (!response.ok) {
                    throw new Error(`HTTP error! status: ${response.status}`);
                }
                const doodles = await response.json();
                setGoogleDoodles(doodles);
            } catch (error) {
                console.error('Error fetching Google Doodles:', error);
            }
        };
        fetchGoogleDoodles();
    }, []);
    const serializeDrawDataToXML = (data) => {
        const gestureName = "circle05";  // Example name
        const subject = "1";  // Example subject
        const speed = "medium";  // Example speed
        const number = "5";  // Example number
        const numPts = data.length.toString();  // Number of points
        const milliseconds = "453";  // Example milliseconds
        const appName = "Gestures";  // Example app name
        const appVer = "3.5.0.0";  // Example app version
        const date = "Monday, March 05, 2007";  // Example date
        const timeOfDay = "5:01:35 PM";  // Example time of day
    
        let xml = `<?xml version="1.0" encoding="utf-8" standalone="yes"?>\n`;
        xml += `<Gesture Name="${gestureName}" Subject="${subject}" Speed="${speed}" Number="${number}" NumPts="${numPts}" Millseconds="${milliseconds}" AppName="${appName}" AppVer="${appVer}" Date="${date}" TimeOfDay="${timeOfDay}">\n`;
        
        data.forEach((point, index) => {
            const time = 1157641 + index * 8;  // Example time calculation
            xml += `  <Point X="${point.x}" Y="${point.y}" T="${time}" />\n`;
        });
    
        xml += `</Gesture>`;
        return xml;
    };
    const handleSendClick = async () => {
        try {
            const xmlData = serializeDrawDataToXML(drawData);
            const blob = new Blob([xmlData], { type: 'text/xml' });
            const formData = new FormData();
            formData.append('file', blob, 'draw.xml');
            try {
              const response = await axios.post('http://127.0.0.1:5000/upload', formData, {
                headers: {
                  'Content-Type': 'multipart/form-data',
                },
              });
              alert(response.data);
            } catch (error) {
              console.error('Error uploading file:', error);
              alert('Error uploading file');
            }
        } catch (error) {
            console.error('Error sending data:', error);
        }
    };

    const handleDraw = (e) => {
        const canvas = e.target;
        const ctx = canvas.getContext('2d');
        let isDrawing = false;
        let lastX = 0;
        let lastY = 0;

        const handleMouseDown = (e) => {
            isDrawing = true;
            [lastX, lastY] = [e.offsetX, e.offsetY];
        };

        const handleMouseMove = (e) => {
            if (!isDrawing) return;
            ctx.beginPath();
            ctx.moveTo(lastX, lastY);
            ctx.lineTo(e.offsetX, e.offsetY);
            ctx.strokeStyle = 'teal';
            ctx.lineWidth = 3;
            ctx.stroke();
            [lastX, lastY] = [e.offsetX, e.offsetY];
            setDrawData((prevData) => [...prevData, [lastX, lastY]]);
        };

        const handleMouseUp = () => {
            isDrawing = false;
        };

        canvas.addEventListener('mousedown', handleMouseDown);
        canvas.addEventListener('mousemove', handleMouseMove);
        canvas.addEventListener('mouseup', handleMouseUp);
    };

    const handleCsvFileChange = (e) => {
        setCsvFile(e.target.files[0]);
    };
    const [file, setFile] = useState(null);
  
    const handleSubmitCSV = async (event) => {
      event.preventDefault();
      if (!csvFile) {
        alert('No file selected');
        return;
      }
  
      const formData = new FormData();
      formData.append('file', csvFile);
  
      try {
        const response = await axios.post('http://127.0.0.1:5000/upload', formData, {
          headers: {
            'Content-Type': 'multipart/form-data',
          },
        });
        alert(response.data);
      } catch (error) {
        console.error('Error uploading file:', error);
        alert('Error uploading file');
      }
    };
    const handleGoogleDoodleSelect = (doodle) => {
        setSelectedDoodles((prevSelected) => {
            if (prevSelected.includes(doodle)) {
                return prevSelected.filter((d) => d !== doodle);
            } else {
                return [...prevSelected, doodle];
            }
        });
    };

    return (
        <Container maxW="container.xl" centerContent p={0}>
            <Box p={8} shadow="md" borderWidth="1px" borderRadius="md" bg="gray.100" w="100%">
                <Flex direction="column" align="center" justify="center" h="100%">
                    <VStack spacing={8} align="stretch" w="100%">
                        <Flex justify="space-between" align="center" w="100%">
                            <Box p={5} shadow="md" borderWidth="1px" borderRadius="md" flex="1" bg="white">
                                <FormControl>
                                    <FormLabel htmlFor="uploadCsv" fontWeight="bold" mb={2}>
                                        <Flex direction="column" align="start" justify="center">
                                            <Heading size="md" mb={2}>Upload CSV</Heading>
                                            <Button
                                                as="label"
                                                htmlFor="uploadCsv"
                                                colorScheme="teal"
                                                size="lg"
                                                width="full"
                                                cursor="pointer"
                                            >
                                                {csvFile ? csvFile.name : 'Choose File'}
                                            </Button>
                                        </Flex>
                                    </FormLabel>
                                    <Input
                                        id="uploadCsv"
                                        type="file"
                                        accept=".csv"
                                        borderColor="teal.300"
                                        display="none"
                                        onChange={handleCsvFileChange}
                                    />
                                </FormControl>
                            </Box>
                            <Box p={5} shadow="md" borderWidth="1px" borderRadius="md" flex="1" bg="white" ml={8}>
                                <Flex direction="column" align="center" justify="center" h="100%">
                                    <Heading size="md" mb={4}>Draw Manually</Heading>
                                    <Button colorScheme="teal" size="lg" width="full" onClick={onDrawOpen}>
                                        Draw
                                    </Button>
                                </Flex>
                            </Box>
                            <Box p={5} shadow="md" borderWidth="1px" borderRadius="md" flex="1" bg="white" ml={8}>
                                <Flex direction="column" align="center" justify="center" h="100%">
                                    <Heading size="md" mb={4}>Use Google Doodles</Heading>
                                    <Button colorScheme="teal" size="lg" width="full" onClick={onDoodlesOpen}>
                                        Choose Doodles
                                    </Button>
                                </Flex>
                            </Box>
                        </Flex>
                        <Button colorScheme="blue" size="lg" onClick={handleSubmitCSV} width="full">
                            Send
                        </Button>
                        {showOutput && (
                            <VStack spacing={8} align="stretch" w="100%">
                                <Box p={5} shadow="md" borderWidth="1px" borderRadius="md" bg="white">
                                    <Heading size="md" mb={4}>Output Preview</Heading>
                                    <Image src="output-preview.jpg" alt="Output Preview" boxSize={imageSize} objectFit="cover" borderRadius="md" />
                                </Box>
                                <Box p={5} shadow="md" borderWidth="1px" borderRadius="md" bg="white">
                                    <Heading size="md" mb={4}>Output Description</Heading>
                                    <VStack spacing={4} align="start">
                                        <Checkbox colorScheme="teal">Check Point 1</Checkbox>
                                        <Checkbox colorScheme="teal">Check Point 2</Checkbox>
                                        <Checkbox colorScheme="teal">Check Point 3</Checkbox>
                                        <Checkbox colorScheme="teal">Check Point 4</Checkbox>
                                    </VStack>
                                </Box>
                            </VStack>
                        )}
                    </VStack>
                </Flex>
            </Box>

            <Modal isOpen={isDrawOpen} onClose={onDrawClose} size="xl">
                <ModalOverlay />
                <ModalContent>
                    <ModalHeader>Draw Manually</ModalHeader>
                    <ModalCloseButton />
                    <ModalBody>
                        <canvas id="drawCanvas" width="600" height="600" onMouseDown={handleDraw}></canvas>
                    </ModalBody>
                    <ModalFooter>
                        <Button colorScheme="gray" mr={3} onClick={onDrawClose}>
                            Close
                        </Button>
                        <Button colorScheme="teal" onClick={handleSendClick}>
                            Send
                        </Button>
                    </ModalFooter>
                </ModalContent>
            </Modal>

            <Modal isOpen={isDoodlesOpen} onClose={onDoodlesClose} size="xl">
                <ModalOverlay />
                <ModalContent>
                    <ModalHeader>Choose Google Doodles</ModalHeader>
                    <ModalCloseButton />
                    <ModalBody>
                        <VStack spacing={4} align="start">
                            {googleDoodles.map((doodle, index) => (
                                <Flex key={index} align="center" justify="space-between" w="full">
                                    <Image src={doodle.url} alt={doodle.title} boxSize="50px" mr={4} />
                                    <Text>{doodle.title}</Text>
                                    <Checkbox
                                        isChecked={selectedDoodles.includes(doodle)}
                                        onChange={() => handleGoogleDoodleSelect(doodle)}
                                        colorScheme="teal"
                                    />
                                </Flex>
                            ))}
                        </VStack>
                    </ModalBody>
                    <ModalFooter>
                        <Button colorScheme="gray" mr={3} onClick={onDoodlesClose}>
                            Close
                        </Button>
                        <Button colorScheme="teal" onClick={handleSendClick}>
                            Send
                        </Button>
                    </ModalFooter>
                </ModalContent>
            </Modal>
        </Container>
    );
};

export default HomePage;
