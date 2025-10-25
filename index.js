import { EntityDB } from "./entity-db.js";
import { pipeline, TextStreamer } from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.7.6/dist/transformers.min.js";
import * as IDBExportImport from "https://cdn.jsdelivr.net/npm/indexeddb-export-import@2.1.5/index.min.js";

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

async function initializeVectorDB() {
    try {
        console.log('Initializing EntityDB for RAG...');

        vectorDB = new EntityDB({
            vectorPath: "askmiku-memory",
            model: "Xenova/all-MiniLM-L6-v2"
        });

        console.log('EntityDB initialized successfully');

        // Only load from database.json if not already loaded.
        const dbLoadedFromJson = localStorage.getItem('dbLoadedFromJson');
        if (!dbLoadedFromJson) {
            try {
                console.log('Checking for database.json...');
                const response = await fetch('data/database.json');

                if (response.ok) {
                    console.log('Found database.json, importing...');
                    const jsonString = await response.text();
                    await importDatabaseFromString(jsonString);
                    localStorage.setItem('dbLoadedFromJson', 'true');
                    console.log('Database imported from database.json successfully (first time)');
                    return true; // Signal that we loaded from export
                }
            } catch (error) {
                console.log('No database.json found or failed to load, will use dummy data');
            }
        } else {
            console.log('Database already loaded from JSON, skipping import.');
        }

        return false; // Signal that we need to load dummy data
    } catch (error) {
        console.error('Error initializing EntityDB:', error);
        // Non-fatal - RAG features will be unavailable but chatbot still works
        return false;
    }
}

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
        await initializeVectorDB();

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

function updateStatus(state, text) {
    statusText.textContent = text;
    statusIndicator.className = `status-indicator ${state}`;
}

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

function createStreamingMessageContainer() {
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
                console.log(contextInfo);
            }
        }

        const enhancedMessage = contextInfo
            ? userMessage + contextInfo
            : userMessage;
        conversationHistory.push({ role: "user", content: enhancedMessage });
        const responseContainer = createStreamingMessageContainer();
        const streamer = new TextStreamer(generator.tokenizer, {
            skip_prompt: true,
            skip_special_tokens: true
        });
        const output = await generator(conversationHistory, {
            max_new_tokens: 512,
            do_sample: false,
            streamer: streamer
        });
        const generatedText = output[0].generated_text.at(-1).content;
        conversationHistory.push({ role: "assistant", content: generatedText });
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

async function handleSendMessage() {
    const message = userInput.value.trim();
    if (!message || isGenerating) return;
    addMessage('user', message);
    userInput.value = '';
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

initializeModel();
// ============================================================================
// Database Export/Import Functions
// ============================================================================

/**
 * Export the entire vector database to a JSON string
 * @returns {Promise<string>} JSON string containing all database data
 */
async function exportDatabaseToString() {
    if (!vectorDB) {
        console.warn('VectorDB not initialized. Cannot export.');
        return null;
    }

    try {
        const nativeDB = await vectorDB.getNativeDB();
        
        return new Promise((resolve, reject) => {
            IDBExportImport.exportToJsonString(nativeDB, (err, jsonString) => {
                if (err) {
                    console.error('Error exporting database:', err);
                    reject(err);
                } else {
                    console.log('Database exported successfully');
                    resolve(jsonString);
                }
            });
        });
    } catch (error) {
        console.error('Error exporting database:', error);
        return null;
    }
}

/**
 * Import database data from a JSON string
 * @param {string} jsonString - JSON string containing database data
 * @returns {Promise<boolean>} True if successful
 */
async function importDatabaseFromString(jsonString) {
    if (!vectorDB) {
        console.warn('VectorDB not initialized. Cannot import.');
        return false;
    }

    try {
        const nativeDB = await vectorDB.getNativeDB();
        
        return new Promise((resolve, reject) => {
            IDBExportImport.importFromJsonString(nativeDB, jsonString, (err) => {
                if (err) {
                    console.error('Error importing database:', err);
                    reject(err);
                } else {
                    console.log('Database imported successfully');
                    resolve(true);
                }
            });
        });
    } catch (error) {
        console.error('Error importing database:', error);
        return false;
    }
}

/**
 * Clear all data from the database
 * @returns {Promise<boolean>} True if successful
 */
async function clearDatabaseData() {
    if (!vectorDB) {
        console.warn('VectorDB not initialized. Cannot clear.');
        return false;
    }

    try {
        const nativeDB = await vectorDB.getNativeDB();
        
        return new Promise((resolve, reject) => {
            IDBExportImport.clearDatabase(nativeDB, (err) => {
                if (err) {
                    console.error('Error clearing database:', err);
                    reject(err);
                } else {
                    console.log('Database cleared successfully');
                    resolve(true);
                }
            });
        });
    } catch (error) {
        console.error('Error clearing database:', error);
        return false;
    }
}

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
