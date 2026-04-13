# Ask Miku

Ask Miku is a browser-based technical support chatbot featuring Retrieval-Augmented Generation (RAG) capabilities. It runs entirely client-side using WebAssembly, ensuring privacy and eliminating the need for a backend server.

## Features

- **Fully Client-Side**: Powered by Hugging Face Transformers.js and WebAssembly. No data leaves your browser.
- **RAG Integration**: Uses a vector database to retrieve relevant technical context before generating responses, improving accuracy and reducing hallucinations.
- **Local Vector Storage**: Persistent storage using IndexedDB for fast retrieval of knowledge base entries.
- **Database Management**: Built-in web UI for importing, exporting, and managing the knowledge base.
- **WebGPU Acceleration**: Supports hardware acceleration for faster model inference.

## Technology Stack

- **LLM**: gemma-4-E4B-it-ONNX (quantized q4f16)
- **Embeddings**: `Xenova/all-MiniLM-L6-v2`
- **ML Framework**: [Transformers.js](https://huggingface.co/docs/transformers.js)
- **Vector Database**: EntityDB
- **Storage**: IndexedDB

## Getting Started

Since this is a static web application, you can run it using any local web server:

```bash
# Example using static-web-server
static-web-server
```

1. Open the application in a modern browser.
2. Click **"Load model"** to initialize the LLM and RAG system.
3. Start chatting with Miku!

## Knowledge Base

The chatbot is powered by a specialized dataset of technical threads.

- **Dataset**: [lmg-neo-lora-v0.3](https://huggingface.co/datasets/quasar-of-mikus/lmg-neo-lora-v0.3) by quasar-of-mikus
  - *Description*: An incomplete collection of threads from June 2023 - March 2025.

## Credits

- **Vector Database**: This project uses [`entity-db.js`](https://github.com/babycommando/entity-db), an in-browser vector database wrapping IndexedDB and Transformers.js over WebAssembly.
- **Dataset**: Special thanks to quasar-of-mikus for the [lmg-neo-lora-v0.3](https://huggingface.co/datasets/quasar-of-mikus/lmg-neo-lora-v0.3) dataset.
