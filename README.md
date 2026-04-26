# Signvrse Translator

Signvrse is a  Sign Language (SL) translator prototype that converts English text into SL gloss and renders the result as 3D avatar motion in the browser.

The frontend is built with React, Vite, Three.js, React Three Fiber, and Drei. It connects to a FastAPI backend that handles English-to-KSL gloss translation, motion generation, and animation-frame creation.

## Project Overview

The application flow is:

1. A user types an English sentence in the frontend.
2. The React app sends the sentence to the backend endpoint `/translate-to-sign`.
3. The backend translates the sentence into SL gloss.
4. The backend generates avatar animation frames from retrieval data or trained PyTorch models.
5. The frontend receives the gloss and animation data.
6. The 3D avatar applies the returned body, hand, and finger rotations in real time.

## Frontend Stack

- React
- Vite
- Axios
- Three.js
- `@react-three/fiber`
- `@react-three/drei`

## Backend Stack

- Python
- FastAPI
- Uvicorn
- PyTorch
- LangGraph
- OpenAI API
- NumPy
- KSL vocabulary and trained model checkpoints

## Main Frontend Files

- `src/App.jsx` - Main UI, input handling, API call, and 3D canvas setup.
- `src/Avatar.jsx` - Loads the GLB avatar and applies generated animation frames to skeleton bones.
- `src/main.jsx` - React app entry point.
- `public/models/avatar.glb` - 3D avatar model.

## How It Works

The frontend keeps three important pieces of state:

- `inputText` - the English sentence entered by the user.
- `glossText` - the KSL gloss returned by the backend.
- `animationData` - frame-by-frame avatar rotation data returned by the backend.

When the user clicks **TRANSLATE**, the app sends:

```js
axios.post('http://localhost:8000/translate-to-sign', {
  sentence: inputText,
})
```

The backend responds with:

```json
{
  "status": "success",
  "gloss_used": "KSL GLOSS",
  "final_3d_data": []
}
```

`Avatar.jsx` receives `final_3d_data` and applies each frame to the avatar skeleton using React Three Fiber's `useFrame`. The animation updates around 30 times per second and controls the spine, head, shoulders, arms, hands, and simplified finger curls.

## Running The Project

Start the backend first from the project root:

```bash
cd backend
python main.py
```

The backend should run on:

```text
http://localhost:8000
```

Then start the frontend:

```bash
cd frontend
npm install
npm run dev
```

The frontend will usually run on:

```text
http://localhost:5173
```

## Required Backend Files

The backend depends on model and data files such as:

- `backend/ksl_vocab.json`
- `backend/ksl_text_brain.pth`
- `backend/ksl_movement_dictionary.pth`
- `backend/ml/rvq_vae_best.pth`
- `backend/data/index_meta.pkl`
- `backend/data/faiss_index.bin`

