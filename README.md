#  Signvrse: English to 3D Kenyan Sign Language Translator

**Live Web Demo:** [Signvrse on Vercel](https://sign-motion.vercel.app/)  
**Cloud Backend:** [Hugging Face Space](https://huggingface.co/spaces/madeeha01/signmotion)

---

##  Overview
**Signvrse** is an end-to-end Hybrid AI engine that translates natural English into real-time 3D Kenyan Sign Language (KSL) animations. It utilizes a LangGraph agentic workflow to convert English to KSL gloss and intelligently routes the request to either a FAISS-based retrieval database or a custom PyTorch generative pipeline. This custom deep learning architecture, trained on over 12,000 spatial-temporal sequences, uses a decoupled Autoencoder and Bi-LSTM to dynamically synthesize exact 3D skeletal coordinates for unseen phrases. The system completely bypasses heavy video rendering by streaming these raw mathematical joint rotations directly to a React Three Fiber frontend, instantly animating a 3D avatar at a seamless 60 frames per second.

##  Key Features
* **Real-Time 3D Rendering:** Bypasses heavy MP4 generation by rendering motion mathematically in the browser at 60 FPS.
* **Agentic NLP Translation:** Uses an LLM agent to accurately translate English grammar into proper KSL Gloss (dropping articles, restructuring verbs).
* **Hybrid RAG Pipeline:** Intelligently searches a vector database for authentic human motion sequences before attempting to generate new ones.
* **Dynamic Generative Fallback:** Uses a custom-trained Autoencoder and Bi-LSTM to invent mathematically accurate 3D skeletal movements for completely unseen vocabulary.
* **Device Agnostic:** Runs entirely in the cloud, meaning the 3D web interface works instantly on mobile phones, tablets, and low-end laptops.

## Tech Stack
* **Frontend:** React.js, Vite, React Three Fiber, Drei (WebGL).
* **Backend:** Python, FastAPI, LangGraph, OpenAI API.
* **Machine Learning:** PyTorch, FAISS, NumPy, Pandas.
* **Deployment:** Vercel (Frontend), Hugging Face Spaces & Docker (Backend).

---

##  How It Works (The Pipeline)

### 1. The Data & Training
The foundation of the project is built on 12,467 continuous spatial-temporal sequences. Instead of a direct text-to-3D map, the AI was trained in two stages:
* **The Body (Autoencoder):** Compresses 668 physical features into a stable 512-dimensional latent space, teaching the AI the physical limits of human joints.
* **The Brain (Bi-LSTM):** Reads KSL vocabulary and learns to project it into the safe 512-dimensional latent space.

### 2. The Backend Engine
When a user submits text, the backend acts as an autonomous router:
* It translates the English to KSL Gloss.
* It checks the FAISS index for high-fidelity, real human motion.
* If no video exists, it triggers the PyTorch generative models to synthesize the raw $X, Y, Z$ Euler rotations dynamically.

### 3. The Frontend UI
The frontend receives a lightweight JSON payload of raw math. Using a `useFrame` loop in React Three Fiber, it applies these exact Euler rotations to the skeleton of a `.glb` 3D model (or a custom node visualizer) in real-time.

---

##  Project Structure
```text
├── backend/                 # FastAPI server, LangGraph agents, and ML models[cite: 3]
│   └── data/                # Contains metadata and FAISS indices (faiss_index.bin)[cite: 3]
├── frontend/                # React Three Fiber UI and 3D Avatar components
├── Dockerfile               # Containerization instructions for Hugging Face deployment[cite: 3]
├── .gitignore               # Git ignore rules[cite: 3]
└── README.md                # Project documentation[cite: 3]