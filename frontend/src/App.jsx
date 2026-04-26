import React, { useState, Suspense } from 'react';
import axios from 'axios';
import { Canvas } from '@react-three/fiber';
import { OrbitControls } from '@react-three/drei';
import Avatar from './Avatar';

export default function App() {
  const [inputText, setInputText] = useState("");
  const [glossText, setGlossText] = useState("");
  const [animationData, setAnimationData] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleTranslate = async () => {
    if (!inputText) return;
    setLoading(true);

    try {
      const response = await axios.post('http://localhost:8000/translate-to-sign', {
        sentence: inputText,
      });

      setGlossText(response.data.gloss_used);
      setAnimationData(response.data.final_3d_data);
    } catch (error) {
      console.error('Error talking to backend:', error);
      setGlossText('Error connecting to server.');
    }

    setLoading(false);
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100vh', fontFamily: 'sans-serif' }}>
      <div style={{ padding: '20px', background: '#2c3e50', color: 'white' }}>
        <h2>Signvrse Translator</h2>
      </div>

      <div style={{ flex: 1, background: '#ecf0f1', position: 'relative' }}>
        <Canvas camera={{ position: [0, 1, 4], fov: 50 }}>
          <ambientLight intensity={0.7} />
          <directionalLight position={[2, 5, 2]} intensity={1} />

          <Suspense fallback={null}>
            <Avatar animationData={animationData} />
          </Suspense>

          <OrbitControls enableZoom={true} />
        </Canvas>
      </div>

      <div style={{ padding: '20px', background: '#fff', borderTop: '2px solid #ccc', display: 'flex', gap: '20px' }}>
        <div style={{ flex: 1 }}>
          <label style={{ fontWeight: 'bold' }}>English Input</label>
          <input
            type="text"
            value={inputText}
            onChange={(e) => setInputText(e.target.value)}
            placeholder="Type a sentence here..."
            style={{ width: '100%', padding: '10px', marginTop: '5px', fontSize: '16px' }}
          />
        </div>

        <div style={{ flex: 1, padding: '10px', background: '#f8f9fa', border: '1px dashed #aaa' }}>
          <label style={{ fontWeight: 'bold', color: '#666' }}>KSL Gloss Translation</label>
          <p style={{ margin: '5px 0', fontSize: '18px', color: '#2c3e50' }}>
            {loading ? "Translating..." : (glossText || "...")}
          </p>
        </div>

        <button
          onClick={handleTranslate}
          disabled={loading}
          style={{ padding: '15px 30px', fontSize: '16px', background: '#3498db', color: 'white', border: 'none', cursor: 'pointer' }}
        >
          {loading ? "Processing..." : "TRANSLATE"}
        </button>

      </div>
    </div>
  );
}
