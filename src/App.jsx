import { useEffect, useRef, useState } from 'react';
import { HandLandmarker, FilesetResolver } from '@mediapipe/tasks-vision';
import { Trash2, PenTool, Eraser, Type, Loader2 } from 'lucide-react';
import './index.css';

const COLORS = [
  { id: 'white', value: '#ffffff' },
  { id: 'blue', value: '#3b82f6' },
  { id: 'green', value: '#10b981' },
  { id: 'pink', value: '#ec4899' },
  { id: 'yellow', value: '#f59e0b' },
];

export default function App() {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const contextRef = useRef(null);
  const [isModelLoaded, setIsModelLoaded] = useState(false);
  const [status, setStatus] = useState('Initializing...');
  
  const [currentColor, setCurrentColor] = useState(COLORS[0].value);
  const [isEraser, setIsEraser] = useState(false);
  const [isPinched, setIsPinched] = useState(false);
  
  const landmarkerRef = useRef(null);
  const lastVideoTimeRef = useRef(-1);
  const requestRef = useRef(null);
  const isDrawingRef = useRef(false);
  const lastPointRef = useRef(null);
  const lastGestureRef = useRef('NONE');
  const clearGestureStartRef = useRef(null);
  const colorIndexRef = useRef(0);
  const cursorRef = useRef(null);
  const [activeGestureInfo, setActiveGestureInfo] = useState('');
  
  const [extractedText, setExtractedText] = useState('');
  const [isExtracting, setIsExtracting] = useState(false);

  useEffect(() => {
    async function loadModel() {
      try {
        const vision = await FilesetResolver.forVisionTasks(
          "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3/wasm"
        );
        landmarkerRef.current = await HandLandmarker.createFromOptions(vision, {
          baseOptions: {
            modelAssetPath: "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
            delegate: "GPU"
          },
          runningMode: "VIDEO",
          numHands: 2
        });
        setIsModelLoaded(true);
        setStatus('Ready - Pinch to draw!');
      } catch (err) {
        console.error(err);
        setStatus('Failed to load model.');
      }
    }
    loadModel();
  }, []);

  useEffect(() => {
    if (!isModelLoaded) return;
    
    // Setup camera
    navigator.mediaDevices.getUserMedia({ video: { width: 1280, height: 720 }, audio: false })
      .then((stream) => {
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          videoRef.current.addEventListener('loadeddata', predictWebcam);
        }
      })
      .catch((err) => {
        console.error("Camera error:", err);
        setStatus("Camera access denied.");
      });
      
    // Setup Canvas
    if (canvasRef.current && videoRef.current) {
      canvasRef.current.widthRef = window.innerWidth;
      canvasRef.current.heightRef = window.innerHeight;
      
      const updateCanvasSize = () => {
        if (!canvasRef.current) return;
        canvasRef.current.width = window.innerWidth;
        canvasRef.current.height = window.innerHeight;
        
        const ctx = canvasRef.current.getContext('2d');
        ctx.lineCap = 'round';
        ctx.lineJoin = 'round';
        contextRef.current = ctx;
      };
      
      updateCanvasSize();
      window.addEventListener('resize', updateCanvasSize);
      return () => window.removeEventListener('resize', updateCanvasSize);
    }
  }, [isModelLoaded]);

  // Sync state to refs for use in animation frame callbacks
  const colorRef = useRef(currentColor);
  const eraserRef = useRef(isEraser);
  
  useEffect(() => { colorRef.current = currentColor; }, [currentColor]);
  useEffect(() => { eraserRef.current = isEraser; }, [isEraser]);

  const predictWebcam = () => {
    if (!videoRef.current || !landmarkerRef.current) return;
    
    const startTimeMs = performance.now();
    if (lastVideoTimeRef.current !== videoRef.current.currentTime) {
      lastVideoTimeRef.current = videoRef.current.currentTime;
      
      const results = landmarkerRef.current.detectForVideo(videoRef.current, startTimeMs);
      
      if (results.landmarks && results.landmarks.length > 0) {
        // Sort hands by wrist X coordinate.
        // In physical space, when facing a mirror (CSS scaleX(-1)),
        // the left hand has a LARGER raw X coordinate (closer to 1.0) 
        // than the right hand (closer to 0.0).
        const sortedHands = results.landmarks.slice().sort((a, b) => b[0].x - a[0].x);
        
        let controlHand = null;
        let penHand = null;
        
        if (sortedHands.length === 2) {
          controlHand = sortedHands[0];
          penHand = sortedHands[1];
        } else if (sortedHands.length === 1) {
          penHand = sortedHands[0];
          controlHand = null;
        }

        const getDist = (landmarks, i, j) => {
          const p1 = landmarks[i]; const p2 = landmarks[j];
          return Math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2 + (p1.z - p2.z)**2);
        };

        const getGesture = (landmarks) => {
          if (!landmarks) return 'NONE';
          const wrist = landmarks[0];
          
          const isIndexUp = getDist(landmarks, 0, 8) > getDist(landmarks, 0, 6);
          const isMiddleUp = getDist(landmarks, 0, 12) > getDist(landmarks, 0, 10);
          const isRingUp = getDist(landmarks, 0, 16) > getDist(landmarks, 0, 14);
          const isPinkyUp = getDist(landmarks, 0, 20) > getDist(landmarks, 0, 18);
          
          const isThumbExtended = getDist(landmarks, 0, 4) > getDist(landmarks, 0, 3);
          const thumbToPinkyBase = getDist(landmarks, 4, 17);
          const indexToPinkyBase = getDist(landmarks, 5, 17);
          const isThumbUp = isThumbExtended && (thumbToPinkyBase > indexToPinkyBase * 1.2);
          
          const isOpenPalm = isIndexUp && isMiddleUp && isRingUp && isPinkyUp;
          const isOneFinger = isIndexUp && !isMiddleUp && !isRingUp && !isPinkyUp;
          const isThreeFingers = isIndexUp && isMiddleUp && isRingUp && !isPinkyUp;
          const isThumbsUpGesture = isThumbUp && !isIndexUp && !isMiddleUp && !isRingUp && !isPinkyUp;
          
          if (isOneFinger) return 'DRAW';
          if (isOpenPalm) return 'ERASE';
          if (isTwoFingers) return 'COLOR';
          if (isThumbsUpGesture) return 'CLEAR_ALL';
          return 'NONE';
        };

        let currentGesture = 'NONE';
        
        if (controlHand) {
           currentGesture = getGesture(controlHand);
        }

        // Edge triggering and hold-to-clear logic
        if (currentGesture === 'CLEAR_ALL') {
          if (!clearGestureStartRef.current) {
            clearGestureStartRef.current = performance.now();
            setActiveGestureInfo('Hold to clear...');
          } else if (performance.now() - clearGestureStartRef.current > 1000) {
            clearCanvas();
            setActiveGestureInfo('Board Cleared');
            setTimeout(() => setActiveGestureInfo(''), 2000);
            clearGestureStartRef.current = null;
            currentGesture = 'NONE';
          }
        } else {
          if (clearGestureStartRef.current) {
            clearGestureStartRef.current = null;
            setActiveGestureInfo('');
          }
          
          if (currentGesture !== lastGestureRef.current) {
            if (currentGesture === 'COLOR') {
              colorIndexRef.current = (colorIndexRef.current + 1) % COLORS.length;
              const nextColor = COLORS[colorIndexRef.current].value;
              setCurrentColor(nextColor);
              setActiveGestureInfo('Color Changed: ' + COLORS[colorIndexRef.current].id);
              setTimeout(() => setActiveGestureInfo(''), 2000);
            }
          }
        }
        
        lastGestureRef.current = currentGesture;
        
        const isCurrentlyInteracting = currentGesture === 'DRAW' || currentGesture === 'ERASE';
        const forceEraser = currentGesture === 'ERASE';
        
        // Draw using the PEN hand's coordinates if it exists
        if (penHand) {
          const targetTip = penHand[8];
          
          // The visual position of the index tip
          const rawX = targetTip.x * (canvasRef.current?.width || window.innerWidth);
          const rawY = targetTip.y * (canvasRef.current?.height || window.innerHeight);
          
          let isHoveringBtn = false;
          const hoverEl = document.elementFromPoint(rawX, rawY);
          const hoveredBtn = hoverEl ? hoverEl.closest('button') : null;
          if (hoveredBtn) isHoveringBtn = true;
          
          if (isDrawingRef.current !== isCurrentlyInteracting) {
            isDrawingRef.current = isCurrentlyInteracting;
            setIsPinched(currentGesture === 'DRAW');
            
            // Air Click: engage tool buttons!
            if (isCurrentlyInteracting && currentGesture === 'DRAW' && hoveredBtn) {
              hoveredBtn.click();
            }
          }
          
          // Update visual floating cursor immediately
          if (cursorRef.current) {
            cursorRef.current.style.transform = `translate(${rawX}px, ${rawY}px)`;
            cursorRef.current.style.opacity = '1';
            
            if (isHoveringBtn && !isCurrentlyInteracting) {
               // Snap / highlight cursor to show clickable
               cursorRef.current.style.backgroundColor = 'rgba(255,255,255,0.8)';
               cursorRef.current.style.border = '2px solid var(--primary)';
               cursorRef.current.style.width = '30px';
               cursorRef.current.style.height = '30px';
               cursorRef.current.style.marginLeft = '-15px';
               cursorRef.current.style.marginTop = '-15px';
               cursorRef.current.style.boxShadow = '0 0 20px var(--primary)';
            } else if (forceEraser) {
              cursorRef.current.style.backgroundColor = 'transparent';
              cursorRef.current.style.border = '2px solid white';
              cursorRef.current.style.width = '60px';
              cursorRef.current.style.height = '60px';
              cursorRef.current.style.marginLeft = '-30px';
              cursorRef.current.style.marginTop = '-30px';
              cursorRef.current.style.boxShadow = 'none';
            } else if (isCurrentlyInteracting && !hoveredBtn) {
              cursorRef.current.style.backgroundColor = colorRef.current;
              cursorRef.current.style.border = 'none';
              cursorRef.current.style.width = '20px';
              cursorRef.current.style.height = '20px';
              cursorRef.current.style.marginLeft = '-10px';
              cursorRef.current.style.marginTop = '-10px';
              cursorRef.current.style.boxShadow = `0 0 15px ${colorRef.current}`;
            } else {
              // Standard hover over canvas
              cursorRef.current.style.backgroundColor = 'rgba(255,255,255,0.4)';
              cursorRef.current.style.border = '2px solid white';
              cursorRef.current.style.width = '15px';
              cursorRef.current.style.height = '15px';
              cursorRef.current.style.marginLeft = '-7.5px';
              cursorRef.current.style.marginTop = '-7.5px';
              cursorRef.current.style.boxShadow = 'none';
            }
          }
          
          if (isCurrentlyInteracting && contextRef.current && canvasRef.current && !hoveredBtn) {
            const ctx = contextRef.current;
            
            if (!lastPointRef.current) {
              lastPointRef.current = { x: rawX, y: rawY };
            }
            
            // Low-pass filter for massive stroke smoothing
            const smoothX = lastPointRef.current.x * 0.6 + rawX * 0.4;
            const smoothY = lastPointRef.current.y * 0.6 + rawY * 0.4;
            
            ctx.beginPath();
            ctx.moveTo(lastPointRef.current.x, lastPointRef.current.y);
            ctx.lineTo(smoothX, smoothY);
            
            if (eraserRef.current || forceEraser) {
              ctx.globalCompositeOperation = 'destination-out';
              ctx.lineWidth = 60;
              ctx.strokeStyle = 'rgba(0,0,0,1)';
            } else {
              ctx.globalCompositeOperation = 'source-over';
              ctx.lineWidth = 6;
              ctx.strokeStyle = colorRef.current;
            }
            
            ctx.stroke();
            lastPointRef.current = { x: smoothX, y: smoothY };
          } else {
            lastPointRef.current = null;
          }
        } else {
           lastPointRef.current = null;
           if (cursorRef.current) cursorRef.current.style.opacity = '0';
           
           if (isDrawingRef.current !== isCurrentlyInteracting) {
             isDrawingRef.current = isCurrentlyInteracting;
             setIsPinched(false);
           }
        }
      } else {
        if (isDrawingRef.current) {
          isDrawingRef.current = false;
          setIsPinched(false);
        }
        lastPointRef.current = null;
        if (cursorRef.current) cursorRef.current.style.opacity = '0';
      }
    }
    
    requestRef.current = window.requestAnimationFrame(predictWebcam);
  };

  const clearCanvas = () => {
    if (contextRef.current && canvasRef.current) {
      contextRef.current.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
    }
  };

  const selectColor = (color) => {
    setCurrentColor(color);
    setIsEraser(false);
    colorIndexRef.current = COLORS.findIndex((c) => c.value === color);
  };

  const convertToText = async () => {
    if (!canvasRef.current || isExtracting) return;
    setIsExtracting(true);
    setExtractedText('');
    setActiveGestureInfo('Extracting text...');
    
    try {
      const Tesseract = (await import('tesseract.js')).default;
      
      // 1: Flip the original drawing (so it matches the CSS mirror)
      const flippedCanvas = document.createElement('canvas');
      flippedCanvas.width = canvasRef.current.width;
      flippedCanvas.height = canvasRef.current.height;
      const fCtx = flippedCanvas.getContext('2d');
      
      fCtx.translate(flippedCanvas.width, 0);
      fCtx.scale(-1, 1);
      fCtx.drawImage(canvasRef.current, 0, 0);
      
      // 2: Force all strokes to be solid black, while preserving the anti-aliasing/alpha
      fCtx.globalCompositeOperation = 'source-in';
      fCtx.fillStyle = '#000000';
      fCtx.fillRect(0, 0, flippedCanvas.width, flippedCanvas.height);
      
      // 3: Draw a white background and place the black strokes on top
      const finalCanvas = document.createElement('canvas');
      finalCanvas.width = canvasRef.current.width;
      finalCanvas.height = canvasRef.current.height;
      const fnCtx = finalCanvas.getContext('2d');
      
      fnCtx.fillStyle = '#ffffff';
      fnCtx.fillRect(0, 0, finalCanvas.width, finalCanvas.height);
      
      // Increase visibility/thickness slightly for OCR reading by drawing it twice! (optional but helps thin air-written letters)
      fnCtx.drawImage(flippedCanvas, 0, 0);
      fnCtx.drawImage(flippedCanvas, 0, 0);
      
      const dataUrl = finalCanvas.toDataURL('image/png');
      
      const result = await Tesseract.recognize(dataUrl, 'eng');
      const text = result.data.text.trim();
      setExtractedText(text || "No readable text found.");
      setTimeout(() => setActiveGestureInfo(''), 500);
      
    } catch (err) {
      console.error(err);
      setExtractedText("Failed to extract text. See console.");
      setTimeout(() => setActiveGestureInfo(''), 500);
    } finally {
      setIsExtracting(false);
    }
  };

  return (
    <div className="container">
      <div ref={cursorRef} className="floating-cursor" />
      
      <header className="header">
        <h1 className="title">
          <PenTool size={28} color="var(--primary)" />
          AI Board <span className="badge">Vision</span>
        </h1>
        
        <div className="status-badge">
          <div className={`status-dot ${isModelLoaded ? 'active' : ''} ${isPinched ? 'pinched' : ''}`} />
          <span>{status}</span>
        </div>
      </header>

      <div className="tools-panel">
        {COLORS.map((c) => (
          <button
            key={c.id}
            className={`color-btn ${currentColor === c.value && !isEraser ? 'active' : ''}`}
            style={{ backgroundColor: c.value }}
            onClick={() => selectColor(c.value)}
            title={c.id}
          />
        ))}
        <div style={{ height: '1px', background: 'rgba(255,255,255,0.2)', margin: '0.5rem 0' }} />
        <button 
          className={`button-icon ${isEraser ? 'active' : ''}`} 
          style={isEraser ? { backgroundColor: 'rgba(255,255,255,0.25)', border: '2px solid white' } : {}}
          onClick={() => setIsEraser(true)} 
          title="Eraser"
        >
          <Eraser size={20} />
        </button>
        <button className="button-icon clear" onClick={clearCanvas} title="Clear Board">
          <Trash2 size={20} />
        </button>
        <button className="button-icon" onClick={convertToText} title="Convert to Text" disabled={isExtracting}>
          {isExtracting ? <Loader2 size={20} className="animate-spin" /> : <Type size={20} />}
        </button>
      </div>

      {extractedText && (
        <div className="text-extraction-panel">
          <div className="panel-header">
            <h3>Extracted Text</h3>
            <button onClick={() => setExtractedText('')}>×</button>
          </div>
          <p>{extractedText}</p>
        </div>
      )}

      <div className="video-container">
        {!isModelLoaded && (
          <div style={{ position: 'absolute', top: '50%', left: '50%', transform: 'translate(-50%, -50%)' }} className="loader">
            <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" style={{ animation: 'spin 1s linear infinite' }}>
              <path d="M21 12a9 9 0 1 1-6.219-8.56"></path>
            </svg>
            <p>Loading AI Vision Model...</p>
          </div>
        )}
        <video 
          ref={videoRef} 
          className="video-feed" 
          autoPlay 
          playsInline
        />
        <canvas 
          ref={canvasRef} 
          className="board-canvas"
        />
      </div>

      <div className="instructions">
        <span>✋ **Left Hand:** Command Center</span>
        <span>👉 **Right Hand:** Pen</span>
        <br />
        <span>☝️ <b>Draw:</b> Left hand 1 finger up</span>
        <span>🖐️ <b>Erase:</b> Left hand open palm</span>
        <span>✌️ <b>Next Color:</b> Left hand 2 fingers up</span>
        <span>👍 <b>Clear All:</b> Left hand Thumbs up (Hold)</span>
      </div>

      {activeGestureInfo && (
        <div className="toast-notification">
          <span>🎨 {activeGestureInfo}</span>
        </div>
      )}
      
      <style>{`
        @keyframes spin { 100% { transform: rotate(360deg); } }
        
        .toast-notification {
          position: absolute;
          top: 30%;
          left: 50%;
          transform: translate(-50%, -50%);
          background: var(--primary);
          color: white;
          padding: 1rem 2rem;
          border-radius: 50px;
          font-size: 1.5rem;
          font-weight: bold;
          z-index: 100;
          animation: popIn 0.3s cubic-bezier(0.175, 0.885, 0.32, 1.275);
          box-shadow: 0 10px 30px rgba(0,0,0,0.5);
          text-transform: capitalize;
        }

        @keyframes popIn {
          0% { transform: translate(-50%, -50%) scale(0.5); opacity: 0; }
          100% { transform: translate(-50%, -50%) scale(1); opacity: 1; }
        }

        .text-extraction-panel {
          position: absolute;
          right: 20px;
          top: 80px;
          bottom: 120px;
          width: 320px;
          background: rgba(30, 41, 59, 0.85);
          backdrop-filter: blur(16px);
          border: 1px solid rgba(255,255,255,0.1);
          border-radius: 1.5rem;
          padding: 1.5rem;
          display: flex;
          flex-direction: column;
          z-index: 30;
          box-shadow: 0 20px 40px rgba(0,0,0,0.4);
          animation: slideIn 0.3s ease;
          overflow-y: auto;
        }

        @keyframes slideIn {
          0% { transform: translateX(50px); opacity: 0; }
          100% { transform: translateX(0); opacity: 1; }
        }

        .panel-header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          margin-bottom: 1rem;
          border-bottom: 1px solid rgba(255,255,255,0.1);
          padding-bottom: 0.5rem;
        }

        .panel-header h3 {
          margin: 0;
          font-size: 1.1rem;
          color: white;
        }

        .panel-header button {
          background: none;
          border: none;
          color: rgba(255,255,255,0.6);
          font-size: 1.5rem;
          cursor: pointer;
        }

        .panel-header button:hover {
          color: white;
        }

        .text-extraction-panel p {
          color: #e2e8f0;
          line-height: 1.6;
          white-space: pre-wrap;
          font-family: inherit;
          margin: 0;
        }
      `}</style>
    </div>
  );
}
