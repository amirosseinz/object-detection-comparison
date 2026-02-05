const sampleImagesContainer = document.getElementById('sample-images-container');
const lionBtn = document.getElementById('lion-btn');
const warthogBtn = document.getElementById('warthog-btn');
const ostrichBtn = document.getElementById('ostrich-btn');
const randomizeBtn = document.getElementById('randomize-btn');

// Initialize history from localStorage or create a new Set
let usedImageUrls = new Set(JSON.parse(localStorage.getItem('usedImageUrls')) || []);
const animals = ['lion', 'warthog', 'ostrich'];
const imagesPerFetch = 5; // Number of images to display per button press
const maxImagesToFetch = 50; // Larger pool to fetch from to ensure variety
let imageCache = { lion: [], warthog: [], ostrich: [] }; // Cache for fetched images
let currentQuery = 'lion'; // Default query

// Function to save used image URLs to localStorage
function saveUsedImageUrls() {
    localStorage.setItem('usedImageUrls', JSON.stringify([...usedImageUrls]));
}

// Function to fetch images from Pixabay
async function fetchImages(query) {
    try {
        const response = await fetch(`https://pixabay.com/api/?key=49941637-4734d2eff6646055d67a757ea&q=${encodeURIComponent(query)}&image_type=photo&per_page=${maxImagesToFetch}`);
        if (!response.ok) throw new Error('Failed to fetch images');
        const data = await response.json();
        return data.hits.map(hit => ({
            url: hit.webformatURL,
            alt: hit.tags || query
        }));
    } catch (error) {
        console.error('Error fetching images:', error);
        showToast('Failed to load sample images.', 'error');
        return [];
    }
}

// Function to get fresh images for an animal, avoiding repeats
async function getFreshImages(animal, count) {
    // If cache is empty or running low, fetch more images
    if (imageCache[animal].length < count) {
        const newImages = await fetchImages(animal);
        // Filter out images that have already been used
        const freshImages = newImages.filter(img => !usedImageUrls.has(img.url));
        imageCache[animal] = freshImages;
    }

    // Select 'count' number of images from the cache
    const selectedImages = imageCache[animal].slice(0, count);
    // Remove selected images from cache
    imageCache[animal] = imageCache[animal].slice(count);
    // Add selected images to usedImageUrls
    selectedImages.forEach(img => usedImageUrls.add(img.url));
    saveUsedImageUrls();

    // If we didn't get enough images, fetch more
    if (selectedImages.length < count) {
        const additionalImages = await getFreshImages(animal, count - selectedImages.length);
        return [...selectedImages, ...additionalImages];
    }
    return selectedImages;
}

// Function to display images
function displayImages(images) {
    sampleImagesContainer.innerHTML = '';
    if (images.length === 0) {
        sampleImagesContainer.innerHTML = '<p class="text-center">No new images available.</p>';
        return;
    }
    images.forEach(image => {
        const imgElement = document.createElement('img');
        imgElement.src = image.url;
        imgElement.alt = image.alt;
        imgElement.className = 'sample-image w-full h-32 object-cover rounded-lg';
        imgElement.draggable = true;
        imgElement.addEventListener('dragstart', (e) => {
            e.dataTransfer.setData('text/uri-list', image.url);
            e.dataTransfer.setData('text/plain', image.url);
        });
        sampleImagesContainer.appendChild(imgElement);
    });
}

// Load images for a specific query
async function loadImages(query) {
    sampleImagesContainer.innerHTML = '<p class="text-center">Loading...</p>';
    const images = await getFreshImages(query, imagesPerFetch);
    displayImages(images);
}

// Function to handle randomize button
async function randomizeImages() {
    sampleImagesContainer.innerHTML = '<p class="text-center">Loading...</p>';
    const sequenceLength = 5; // Number of animals to show in the sequence
    const sequence = [];

    // Generate a random sequence of animals
    for (let i = 0; i < sequenceLength; i++) {
        const randomAnimal = animals[Math.floor(Math.random() * animals.length)];
        sequence.push(randomAnimal);
    }

    // Fetch one image per animal in the sequence
    const images = [];
    for (const animal of sequence) {
        const image = (await getFreshImages(animal, 1))[0];
        if (image) images.push(image);
    }

    displayImages(images);
}

// Event listeners for buttons
lionBtn.addEventListener('click', () => {
    currentQuery = 'lion';
    loadImages(currentQuery);
});
warthogBtn.addEventListener('click', () => {
    currentQuery = 'warthog';
    loadImages(currentQuery);
});
ostrichBtn.addEventListener('click', () => {
    currentQuery = 'ostrich';
    loadImages(currentQuery);
});
randomizeBtn.addEventListener('click', randomizeImages);

// Initial load with a random animal
const randomAnimal = animals[Math.floor(Math.random() * animals.length)];
currentQuery = randomAnimal;
loadImages(currentQuery);