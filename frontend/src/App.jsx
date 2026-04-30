import React, { useState, Suspense } from 'react';
import axios from 'axios';
import { Canvas } from '@react-three/fiber';
import { OrbitControls } from '@react-three/drei';
import Avatar from './Avatar';

const API_BASE_URL = 'https://madeeha01-signvrse.hf.space';

export default function App() {
  const [inputText, setInputText] = useState("");
  const [glossText, setGlossText] = useState("");
  const [animationData, setAnimationData] = useState([]);
  const [loading, setLoading] = useState(false);

  const handleTranslate = async () => {
    if (!inputText.trim()) return;
    setLoading(true);

    try {
      const response = await axios.post(`${API_BASE_URL}/translate-to-sign`, {
        sentence: inputText.trim(),
      });

      setGlossText(response.data.gloss_used || '');
      setAnimationData(Array.isArray(response.data.final_3d_data) ? response.data.final_3d_data : []);
    } catch (error) {
      console.error('Error talking to backend:', error);
      setGlossText('Error connecting to server.');
      setAnimationData([]);
    }

    setLoading(false);
  };

  return (
    <div
      style={{
        display: 'flex',
        flexDirection: 'column',
        minHeight: '100vh',
        fontFamily: 'Segoe UI, sans-serif',
        background: 'radial-gradient(circle at top, #152238 0%, #070a12 52%, #030407 100%)',
        color: '#e8f7ff',
      }}
    >
      <div
        style={{
          padding: '22px 28px',
          background: 'rgba(7, 12, 22, 0.88)',
          borderBottom: '1px solid rgba(0, 255, 204, 0.25)',
          boxShadow: '0 12px 38px rgba(0, 0, 0, 0.35)',
        }}
      >
        <h2 style={{ margin: 0, letterSpacing: '0.08em', textTransform: 'uppercase', color: '#00ffcc' }}>
          Signvrse Translator
        </h2>
      </div>

      <div
        style={{
          flex: 1,
          background:
            'linear-gradient(135deg, rgba(0, 255, 204, 0.08), transparent 35%), #090d18',
          position: 'relative',
        }}
      >
        <Canvas camera={{ position: [0, 0, 4], fov: 50 }}>
          <color attach="background" args={['#090d18']} />
          <ambientLight intensity={0.7} />
          <directionalLight position={[2, 5, 2]} intensity={1} />

          <Suspense fallback={null}>
            <Avatar animationData={animationData} />
          </Suspense>

          <OrbitControls enableZoom={true} />
        </Canvas>
      </div>

      <div
        style={{
          padding: '20px',
          background: 'rgba(5, 9, 18, 0.96)',
          borderTop: '1px solid rgba(0, 255, 204, 0.22)',
          display: 'flex',
          gap: '20px',
          boxShadow: '0 -14px 38px rgba(0, 0, 0, 0.32)',
        }}
      >
        <div style={{ flex: 1 }}>
          <label style={{ fontWeight: 'bold', color: '#9df7e6' }}>English Input</label>
          <input
            type="text"
            value={inputText}
            onChange={(e) => setInputText(e.target.value)}
            placeholder="Type a sentence here..."
            style={{
              width: '100%',
              boxSizing: 'border-box',
              padding: '12px',
              marginTop: '6px',
              fontSize: '16px',
              color: '#e8f7ff',
              background: '#0f1728',
              border: '1px solid rgba(0, 255, 204, 0.35)',
              borderRadius: '10px',
              outline: 'none',
            }}
          />
        </div>

        <div
          style={{
            flex: 1,
            padding: '12px',
            background: '#0f1728',
            border: '1px dashed rgba(0, 255, 204, 0.45)',
            borderRadius: '10px',
          }}
        >
          <label style={{ fontWeight: 'bold', color: '#9df7e6' }}>KSL Gloss Translation</label>
          <p style={{ margin: '5px 0', fontSize: '18px', color: '#ffffff' }}>
            {loading ? "Translating..." : (glossText || "...")}
          </p>
        </div>

        <button
          onClick={handleTranslate}
          disabled={loading}
          style={{
            padding: '15px 30px',
            fontSize: '16px',
            fontWeight: 'bold',
            background: loading ? '#1b3340' : 'linear-gradient(135deg, #00ffcc, #20a4f3)',
            color: '#031018',
            border: 'none',
            borderRadius: '12px',
            cursor: loading ? 'not-allowed' : 'pointer',
            boxShadow: '0 0 24px rgba(0, 255, 204, 0.22)',
          }}
        >
          {loading ? "Processing..." : "TRANSLATE"}
        </button>

      </div>
    </div>
  );
}
