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