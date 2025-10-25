import { EntityDB } from "./entity-db.js";
import { pipeline, TextStreamer } from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.7.6/dist/transformers.min.js";

// DOM Elements
const chatContainer = document.getElementById('chatContainer');
const userInput = document.getElementById('userInput');
const sendButton = document.getElementById('sendButton');
const statusIndicator = document.getElementById('statusIndicator');
const statusText = document.getElementById('statusText');

// State
let generator = null;
let vectorDB = null;
let conversationHistory = [
    { role: "system", content: "You are a helpful technical support assistant." }
];
let isGenerating = false;

// Initialize the vector database
async function initializeVectorDB() {
    try {
        console.log('Initializing EntityDB for RAG...');
        
        vectorDB = new EntityDB({
            vectorPath: "askmiku-memory",
            model: "Xenova/all-MiniLM-L6-v2"
        });
        
        console.log('EntityDB initialized successfully');
    } catch (error) {
        console.error('Error initializing EntityDB:', error);
        // Non-fatal - RAG features will be unavailable but chatbot still works
    }
}

// Populate database with dummy technical support data
async function populateDummyData() {
    if (!vectorDB) return;
    
    try {
        console.log('Populating database with dummy technical support data...');
        
        const dummyData = [
            {
                text: "To reset your password, go to Settings > Security > Change Password. You'll need to verify your email address first.",
                category: "password",
                topic: "account_security"
            },
            {
                text: "If your application is running slowly, try clearing the cache by going to Settings > Advanced > Clear Cache. This often resolves performance issues.",
                category: "performance",
                topic: "troubleshooting"
            },
            {
                text: "To enable two-factor authentication, navigate to Settings > Security > Two-Factor Authentication and follow the setup wizard.",
                category: "security",
                topic: "account_security"
            },
            {
                text: "Error code 404 means the requested resource was not found. Check the URL for typos or contact support if the problem persists.",
                category: "errors",
                topic: "troubleshooting"
            },
            {
                text: "To export your data, go to Settings > Data & Privacy > Export Data. The export process may take several minutes depending on data size.",
                category: "data",
                topic: "data_management"
            },
            {
                text: "Browser compatibility issues can often be resolved by updating to the latest version or trying a different browser like Chrome or Firefox.",
                category: "compatibility",
                topic: "troubleshooting"
            },
            {
                text: "API rate limits are 1000 requests per hour for free tier and 10000 for premium. Upgrade your plan if you need higher limits.",
                category: "api",
                topic: "technical_limits"
            },
            {
                text: "To integrate with third-party services, use the API key found in Settings > Developer > API Keys. Keep this key secret.",
                category: "integration",
                topic: "api_integration"
            }
        ];

        for (const item of dummyData) {
            await vectorDB.insert(item);
        }
        
        console.log(`Successfully inserted ${dummyData.length} knowledge base entries`);
    } catch (error) {
        console.error('Error populating dummy data:', error);
    }
}

// Initialize the model
async function initializeModel() {
    try {
        updateStatus('loading', 'Loading Qwen3 model... This may take a minute on first load.');

        generator = await pipeline(
            "text-generation",
            "onnx-community/Qwen3-0.6B-ONNX",
            {
                dtype: "q4f16",
                device: "wasm",
                progress_callback: (progress) => {
                    if (progress.status === 'progress') {
                        const percent = Math.round((progress.loaded / progress.total) * 100);
                        updateStatus('loading', `Loading model... ${percent}%`);
                    }
                }
            }
        );

        updateStatus('ready', 'Model ready! Initializing RAG database...');
        
        // Initialize vector database for RAG
        await initializeVectorDB();
        
        // Populate database with dummy technical support data
        await populateDummyData();

        updateStatus('ready', 'Ready!');
        userInput.disabled = false;
        sendButton.disabled = false;
        userInput.focus();
        
    } catch (error) {
        console.error('Error initializing model:', error);
        updateStatus('error', 'Failed to load model. Please refresh the page.');
        addMessage('assistant', `Error: ${error.message}. Please refresh the page to try again.`);
    }
}

// Update status indicator
function updateStatus(state, text) {
    statusText.textContent = text;
    statusIndicator.className = `status-indicator ${state}`;
}

// Add message to chat
function addMessage(role, content) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${role}`;
    
    const avatar = document.createElement('div');
    avatar.className = 'message-avatar';
    avatar.textContent = role === 'user' ? '👤' : '🤖';
    
    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';
    contentDiv.textContent = content;
    
    messageDiv.appendChild(avatar);
    messageDiv.appendChild(contentDiv);
    chatContainer.appendChild(messageDiv);
    
    // Scroll to bottom
    chatContainer.scrollTop = chatContainer.scrollHeight;
    
    return contentDiv;
}

// Create streaming message container
function createStreamingMessage() {
    const messageDiv = document.createElement('div');
    messageDiv.className = 'message assistant';
    
    const avatar = document.createElement('div');
    avatar.className = 'message-avatar';
    avatar.textContent = '🤖';
    
    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';
    contentDiv.innerHTML = '<span class="loading-dots">Thinking</span>';
    
    messageDiv.appendChild(avatar);
    messageDiv.appendChild(contentDiv);
    chatContainer.appendChild(messageDiv);
    
    chatContainer.scrollTop = chatContainer.scrollHeight;
    
    return contentDiv;
}

// Generate response
async function generateResponse(userMessage) {
    if (isGenerating || !generator) return;
    
    isGenerating = true;
    sendButton.disabled = true;
    userInput.disabled = true;
    
    try {
        // Query the vector database for relevant context
        let contextInfo = "";
        if (vectorDB) {
            const relevantDocs = await vectorDB.query(userMessage, { limit: 3 });
            
            if (relevantDocs && relevantDocs.length > 0) {
                console.log('Found relevant knowledge base entries:', relevantDocs.length);
                contextInfo = "\n\nRelevant information from knowledge base:\n";
                relevantDocs.forEach((doc, index) => {
                    contextInfo += `${index + 1}. ${doc.text}\n`;
                });
            }
        }

        // Create enhanced user message with context
        const enhancedMessage = contextInfo
            ? userMessage + contextInfo
            : userMessage;

        // Add user message to history (with context if available)
        conversationHistory.push({ role: "user", content: enhancedMessage });
        
        // Create streaming message container
        const responseContainer = createStreamingMessage();

        // Generate response with streaming
        const streamer = new TextStreamer(generator.tokenizer, {
            skip_prompt: true,
            skip_special_tokens: true
        });

        const output = await generator(conversationHistory, {
            max_new_tokens: 512,
            do_sample: false,
            streamer: streamer
        });

        // Extract the generated text
        const generatedText = output[0].generated_text.at(-1).content;

        // Add assistant response to history
        conversationHistory.push({ role: "assistant", content: generatedText });

        // Update final response
        responseContainer.textContent = generatedText || "I apologize, but I couldn't generate a response. Please try again.";
        
    } catch (error) {
        console.error('Error generating response:', error);
        addMessage('assistant', `Error generating response: ${error.message}`);
    } finally {
        isGenerating = false;
        sendButton.disabled = false;
        userInput.disabled = false;
        userInput.focus();
    }
}

// Handle send message
async function handleSendMessage() {
    const message = userInput.value.trim();
    
    if (!message || isGenerating) return;
    
    // Add user message to UI
    addMessage('user', message);
    
    // Clear input
    userInput.value = '';
    
    // Generate response
    await generateResponse(message);
}

// Event listeners
sendButton.addEventListener('click', handleSendMessage);

userInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        handleSendMessage();
    }
});

// Initialize model on page load
initializeModel();
// ============================================================================
// RAG (Retrieval-Augmented Generation) Helper Functions
// ============================================================================
// These functions provide vector search capabilities using EntityDB
// Population of the database will be handled separately

/**
 * Insert text into the vector database with automatic embedding generation
 * @param {string} text - The text to embed and store
 * @param {object} metadata - Optional metadata to store with the vector
 * @returns {Promise<void>}
 */
async function insertMemory(text, metadata = {}) {
    if (!vectorDB) {
        console.warn('VectorDB not initialized. RAG features unavailable.');
        return;
    }
    
    try {
        await vectorDB.insert({
            text: text,
            ...metadata
        });
        console.log('Memory inserted:', text.substring(0, 50) + '...');
    } catch (error) {
        console.error('Error inserting memory:', error);
    }
}

/**
 * Query the vector database using semantic similarity
 * @param {string} queryText - The text to search for
 * @param {number} topK - Number of results to return (default: 5)
 * @returns {Promise<Array>} Array of similar results
 */
async function queryMemory(queryText, topK = 5) {
    if (!vectorDB) {
        console.warn('VectorDB not initialized. RAG features unavailable.');
        return [];
    }
    
    try {
        const results = await vectorDB.query(queryText, topK);
        console.log(`Found ${results.length} similar memories`);
        return results;
    } catch (error) {
        console.error('Error querying memory:', error);
        return [];
    }
}

/**
 * Insert text with binary vector encoding (faster, slightly less accurate)
 * @param {string} text - The text to embed and store
 * @param {object} metadata - Optional metadata to store with the vector
 * @returns {Promise<void>}
 */
async function insertBinaryMemory(text, metadata = {}) {
    if (!vectorDB) {
        console.warn('VectorDB not initialized. RAG features unavailable.');
        return;
    }
    
    try {
        await vectorDB.insertBinary({
            text: text,
            ...metadata
        });
        console.log('Binary memory inserted:', text.substring(0, 50) + '...');
    } catch (error) {
        console.error('Error inserting binary memory:', error);
    }
}

/**
 * Query using binary vectors (extremely fast, uses Hamming distance)
 * @param {string} queryText - The text to search for
 * @param {number} topK - Number of results to return (default: 5)
 * @returns {Promise<Array>} Array of similar results
 */
async function queryBinaryMemory(queryText, topK = 5) {
    if (!vectorDB) {
        console.warn('VectorDB not initialized. RAG features unavailable.');
        return [];
    }
    
    try {
        const results = await vectorDB.queryBinary(queryText, topK);
        console.log(`Found ${results.length} similar memories (binary)`);
        return results;
    } catch (error) {
        console.error('Error querying binary memory:', error);
        return [];
    }
}

/**
 * Query using binary vectors with SIMD acceleration (fastest option)
 * @param {string} queryText - The text to search for
 * @param {number} topK - Number of results to return (default: 5)
 * @returns {Promise<Array>} Array of similar results
 */
async function queryBinaryMemorySIMD(queryText, topK = 5) {
    if (!vectorDB) {
        console.warn('VectorDB not initialized. RAG features unavailable.');
        return [];
    }
    
    try {
        const results = await vectorDB.queryBinarySIMD(queryText, topK);
        console.log(`Found ${results.length} similar memories (binary SIMD)`);
        return results;
    } catch (error) {
        console.error('Error querying binary memory with SIMD:', error);
        return [];
    }
}

// Export RAG functions to global scope for console testing
window.ragHelpers = {
    insertMemory,
    queryMemory,
    insertBinaryMemory,
    queryBinaryMemory,
    queryBinaryMemorySIMD
};

