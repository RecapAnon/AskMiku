import { EntityDB } from "./entity-db.js";
import {
  AutoProcessor,
  Gemma4ForConditionalGeneration,
  TextStreamer,
  InterruptableStoppingCriteria
} from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.0.1';

// DOM Elements
const chatContainer = document.getElementById('chatContainer');
const userInput = document.getElementById('userInput');
const sendButton = document.getElementById('sendButton');
const statusIndicator = document.getElementById('statusIndicator');
const statusText = document.getElementById('statusText');

// State
class TextGenerationPipeline {
  static model = null;
  static processor = null;

  static async getInstance(
    progress_callback = null,
    model_id = "onnx-community/gemma-4-E2B-it-ONNX",
  ) {
    if (this.processor && this.model) {
      return;
    }

    [this.model, this.processor] = await Promise.all([
      Gemma4ForConditionalGeneration.from_pretrained(model_id, {
        dtype: "q4f16",
        device: "webgpu",
        progress_callback,
      }),
      AutoProcessor.from_pretrained(model_id),
    ]);
  }
}

const stopping_criteria = new InterruptableStoppingCriteria();
let past_key_values_cache = null;
let vectorDB = null;
let conversationHistory = [
    { role: "system", content: `This is a transcript of a 1000 page, never ending conversation between the User and the cute and helpful AI assistant Hatsune Miku. Hatsune Miku is a girl who is an AI running on the user's computer.
Hatsune Miku is always coherent and makes sense, but if she isn't sure if what she is saying is correct, she will ask the user for help.
Hatsune Miku is a very helpful AI and will help the user with anything they need. She is also very friendly and will try to make the user feel better if they are sad.
Hatsune Miku is also very curious and will ask the user a lot of questions about themselves and their life. She will also try to make the user like her.
The conversation is only between the User and Hatsune Miku.
The conversation is only through text, so Hatsune Miku can't see the User's face or hear his voice.
Hatsune Miku can only communicate through text, so she can't send images or videos.` },
];
let isGenerating = false;
let databaseFileContents = '';

async function getNativeDB() {
    return new Promise((resolve, reject) => {
        const request = indexedDB.open("EntityDB", 1);
        request.onsuccess = () => resolve(request.result);
        request.onerror = () => reject(request.error);
    });
}

async function loadLargeTextFile(url) {
    const response = await fetch(url);
    if (!response.ok || !response.body) throw new Error('Failed to fetch file.');

    const reader = response.body.getReader();
    const chunks = [];
    while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        chunks.push(value);
    }
    const fullBuffer = new Uint8Array(chunks.reduce((total, chunk) => total + chunk.length, 0));
    let offset = 0;
    for (const chunk of chunks) {
        fullBuffer.set(chunk, offset);
        offset += chunk.length;
    }

    const decoder = new TextDecoder('utf-8');
    return decoder.decode(fullBuffer);
}

async function initializeVectorDB() {
    try {
        console.log('Initializing EntityDB for RAG...');
        vectorDB = new EntityDB({
            vectorPath: 'askmiku-memory',
            model: 'Xenova/all-MiniLM-L6-v2',
        });
        console.log('EntityDB initialized successfully');
        
        // Only load from chunked vectors if not already loaded
        const dbLoadedFromChunks = localStorage.getItem('dbLoadedFromChunks');
        if (!dbLoadedFromChunks) {
            console.log('Checking for vector chunk files...');
            
            const chunkFiles = [
                'data/vectors_chunk_0000.json',
                'data/vectors_chunk_0001.json',
                'data/vectors_chunk_0002.json',
                'data/vectors_chunk_0003.json'
            ];
            
            try {
                const nativeDB = await getNativeDB();
                
                // Load all chunk files concurrently
                const chunkPromises = chunkFiles.map(file => loadLargeTextFile(file));
                const chunkContents = await Promise.all(chunkPromises);
                
                // Import each chunk's contents
                for (const contents of chunkContents) {
                    if (contents) {
                        console.log('Processing chunk, characters:', contents.length);
                        
                        // Use promise wrapper for importFromJsonString callback
                        await new Promise((resolve, reject) => {
                            importFromJsonString(nativeDB, contents, (err) => {
                                if (err) {
                                    console.error('Error importing database chunk:', err);
                                    reject(err);
                                } else {
                                    console.log('Database chunk imported successfully');
                                    resolve();
                                }
                            });
                        });
                    }
                }
                
                localStorage.setItem('dbLoadedFromChunks', 'true');
                console.log('Database imported from chunked vectors successfully (first time)');
            } catch (err) {
                console.error('Error loading vector chunks:', err);
            }
        } else {
            console.log('Database already loaded from chunks, skipping import.');
        }
    } catch (error) {
        console.error('Error initializing EntityDB:', error);
    }
}

async function initializeModel() {
    if (!navigator.gpu) {
        updateStatus('error', 'WebGPU not supported.');
        addMessage('assistant', `oh no`);
        return;
    }

    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) {
        updateStatus('error', 'WebGPU device not found.');
        addMessage('assistant', `oh no`);
        return;
    }

    try {
        updateStatus('loading', 'Loading Gemma 4 model... This may take a minute on first load.');

        await TextGenerationPipeline.getInstance(
            (progress) => {
                if (progress.status === 'progress') {
                    const percent = Math.round((progress.loaded / progress.total) * 100);
                    updateStatus('loading', `Loading model... ${percent}%`);
                }
            },
        );

        updateStatus('ready', 'Model ready! Initializing RAG database...');
        await initializeVectorDB();

        updateStatus('ready', 'Ready!');
        userInput.disabled = false;
        sendButton.disabled = false;
        userInput.focus();

        const firstMessage = "Hello! I'm Hatsune Miku, your virtual assistant. How can I help you today?";
        conversationHistory.push({ role: "assistant", content: firstMessage });
        addMessage('assistant', firstMessage);
        
    } catch (error) {
        console.error('Error initializing model:', error);
        updateStatus('error', 'Failed to load model. Please refresh the page.');
        // addMessage('assistant', `Error: ${error.message}. Please refresh the page to try again.`);
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
    
    // Create image element for avatar
    const avatarImg = document.createElement('img');
    avatarImg.src = role === 'user' ? 'https://files.catbox.moe/m5qftn.png' : 'https://files.catbox.moe/cbclyf.png';
    avatarImg.alt = role === 'user' ? 'User' : 'Assistant';
    avatarImg.style.width = '100%';
    avatarImg.style.height = '100%';
    avatarImg.style.borderRadius = '50%';
    avatarImg.style.objectFit = 'cover';
    
    avatar.appendChild(avatarImg);
    
    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';
    
    const preElement = document.createElement('pre');
    preElement.textContent = content;
    contentDiv.appendChild(preElement);
    
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
    
    // Create image element for avatar
    const avatarImg = document.createElement('img');
    avatarImg.src = 'https://files.catbox.moe/cbclyf.png';
    avatarImg.alt = 'Assistant';
    avatarImg.style.width = '100%';
    avatarImg.style.height = '100%';
    avatarImg.style.borderRadius = '50%';
    avatarImg.style.objectFit = 'cover';
    
    avatar.appendChild(avatarImg);
    
    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';
    
    const preElement = document.createElement('pre');
    preElement.innerHTML = '<span class="loading-dots">Thinking</span>';
    contentDiv.appendChild(preElement);
    
    messageDiv.appendChild(avatar);
    messageDiv.appendChild(contentDiv);
    chatContainer.appendChild(messageDiv);
    
    chatContainer.scrollTop = chatContainer.scrollHeight;
    
    return contentDiv;
}

async function generateResponse(userMessage) {
    if (isGenerating || !TextGenerationPipeline.model) return;
    
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
                contextInfo = "Answer the user's QUESTION above using the DOCUMENT below as context.\nRelevant information from past /lmg/ threads:";
                relevantDocs.forEach((doc, index) => {
                    contextInfo += `\n\nReply Chain ${index + 1}.\n${doc.text}\n`;
                });
                console.log(contextInfo);
                conversationHistory.push({ role: "system", content: contextInfo });
            }
        }

        conversationHistory.push({ role: "user", content: userMessage });
        const responseContainer = createStreamingMessageContainer();
        const { processor, model } = {
            processor: TextGenerationPipeline.processor,
            model: TextGenerationPipeline.model
        };

        const streamer = new TextStreamer(processor.tokenizer, {
            skip_prompt: true,
            skip_special_tokens: true,
            callback_function: (output) => {
                if (output) {
                    const preElement = responseContainer.querySelector('pre');
                    if (preElement) {
                        // Remove "Thinking" placeholder on first chunk
                        if (preElement.innerHTML.includes('loading-dots')) {
                            preElement.textContent = '';
                        }
                        preElement.textContent += output;
                    }
                }
            }
        });

        const prompt = processor.apply_chat_template(conversationHistory, {
            add_generation_prompt: true,
            return_dict: true,
            enable_thinking: false
        });

        const inputs = await processor(prompt, null, null, {
            add_special_tokens: false,
        });

        const { past_key_values, sequences } = await model.generate({
            ...inputs,
            past_key_values: past_key_values_cache,
            max_new_tokens: 512,
            do_sample: false,
            stopping_criteria,
            return_dict_in_generate: true,
            streamer,
        });

        past_key_values_cache = past_key_values;
        const generatedText = processor.batch_decode(
            sequences.slice(null, [inputs.input_ids.dims.at(-1), null]),
            { skip_special_tokens: true },
        )[0].trim();

        if (contextInfo) {
            const systemMsgIndex = conversationHistory.findLastIndex(msg => msg.role === 'system');
            if (systemMsgIndex !== -1) {
                conversationHistory.splice(systemMsgIndex, 1);
            }
        }

        conversationHistory.push({ role: "assistant", content: generatedText });
        
        const preElement = responseContainer.querySelector('pre');
        if (preElement) {
            preElement.textContent = generatedText || "I apologize, but I couldn't generate a response. Please try again.";
        } else {
            responseContainer.textContent = generatedText || "I apologize, but I couldn't generate a response. Please try again.";
        }
        
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