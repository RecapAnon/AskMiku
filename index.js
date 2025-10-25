import { pipeline, TextStreamer } from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.7.6/dist/transformers.min.js";

// DOM Elements
const chatContainer = document.getElementById('chatContainer');
const userInput = document.getElementById('userInput');
const sendButton = document.getElementById('sendButton');
const statusIndicator = document.getElementById('statusIndicator');
const statusText = document.getElementById('statusText');

// State
let generator = null;
let conversationHistory = [
    { role: "system", content: "You are a helpful technical support assistant." }
];
let isGenerating = false;

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
        
        updateStatus('ready', 'Model ready!');
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
        // Add user message to history
        conversationHistory.push({ role: "user", content: userMessage });
        
        // Create streaming message container
        const responseContainer = createStreamingMessage();
        let fullResponse = '';
        
        // Create custom streamer that updates the UI
        class UIStreamer extends TextStreamer {
            put(value) {
                super.put(value);
                const text = this.decode(value);
                fullResponse += text;
                responseContainer.textContent = fullResponse;
                chatContainer.scrollTop = chatContainer.scrollHeight;
            }
        }
        
        // Generate response with streaming
        const streamer = new UIStreamer(generator.tokenizer, {
            skip_prompt: true,
            skip_special_tokens: true
        });
        
        await generator(conversationHistory, {
            max_new_tokens: 512,
            do_sample: false,
            streamer: streamer
        });
        
        // Add assistant response to history
        conversationHistory.push({ role: "assistant", content: fullResponse.trim() });
        
        // Update final response
        responseContainer.textContent = fullResponse.trim() || "I apologize, but I couldn't generate a response. Please try again.";
        
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