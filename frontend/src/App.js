import React, { useState, useEffect, useRef, useCallback } from 'react';
import './App.css';
import AgentPanel from './AgentPanel';

const BACKEND_URL       = process.env.REACT_APP_BACKEND_URL || 'http://localhost:8000';
const SEQUENCE_LENGTH   = 30;
const FEATURE_SIZE      = 1662;
const CONFIDENCE_THRESH = 0.70;
const STABILITY_FRAMES  = 10;
const ADD_COOLDOWN_MS   = 1500;
const APP_NAME          = 'GestureMind';

const POSE_CONNECTIONS = [
  [11,12],[11,13],[13,15],[12,14],[14,16],[11,23],[12,24],[23,24],
  [23,25],[25,27],[24,26],[26,28]
];
const HAND_CONNECTIONS = [
  [0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],
  [0,9],[9,10],[10,11],[11,12],[0,13],[13,14],[14,15],[15,16],
  [0,17],[17,18],[18,19],[19,20],[5,9],[9,13],[13,17]
];

export default function App() {
  const [backendReady, setBackendReady] = useState(false);
  const [modelLoaded,  setModelLoaded]  = useState(false);
  const [emailReady,   setEmailReady]   = useState(false);
  const [backendError, setBackendError] = useState('');
  const [isRunning,    setIsRunning]    = useState(false);
  const [fps,          setFps]          = useState(0);
  const [now,          setNow]          = useState(new Date());

  const [gesture,      setGesture]      = useState('');
  const [confidence,   setConfidence]   = useState(0);
  const [isConfident,  setIsConfident]  = useState(false);
  const [landmarkCount,setLandmarkCount]= useState(0);
  const [confHistory,  setConfHistory]  = useState(Array(40).fill(0));

  const [words,          setWords]          = useState([]);
  const [gestureHistory, setGestureHistory] = useState([]);
  const [activeNav,      setActiveNav]      = useState('live');

  const videoRef=useRef(null), canvasRef=useRef(null), streamRef=useRef(null), animRef=useRef(null);
  const sequenceRef=useRef([]), predBufferRef=useRef([]), lastAddedRef=useRef(0), lastGestureRef=useRef('');
  const fpsClockRef=useRef(Date.now()), fpsCountRef=useRef(0);
  const mpHandsRef=useRef(null), mpPoseRef=useRef(null), handsResultRef=useRef(null), poseResultRef=useRef(null);
  const isRunningRef=useRef(false), predictingRef=useRef(false);

  useEffect(() => { const t=setInterval(()=>setNow(new Date()),1000); return()=>clearInterval(t); },[]);

  useEffect(()=>{
    const check=async()=>{
      try{
        const controller=new AbortController(); const timeout=setTimeout(()=>controller.abort(),3000);
        const r=await fetch(`${BACKEND_URL}/health`,{signal:controller.signal}); clearTimeout(timeout);
        const d=await r.json();
        setBackendReady(d.status==='ok'); setModelLoaded(d.model_loaded); setEmailReady(d.email_alerts_ready);
        setBackendError('');
      }catch(err){
        setBackendReady(false); setModelLoaded(false); setEmailReady(false);
        setBackendError(err.name==='AbortError' ? 'Backend timeout — is uvicorn running?' : 'Cannot reach backend at '+BACKEND_URL);
      }
    };
    check(); const t=setInterval(check,5000); return()=>clearInterval(t);
  },[]);

  const loadMediaPipe = useCallback(()=>{
    return new Promise(resolve=>{
      if(window.Hands&&window.Pose){resolve();return;}
      const scripts=[
        'https://cdn.jsdelivr.net/npm/@mediapipe/drawing_utils/drawing_utils.js',
        'https://cdn.jsdelivr.net/npm/@mediapipe/hands/hands.js',
        'https://cdn.jsdelivr.net/npm/@mediapipe/pose/pose.js',
      ];
      let loaded=0;
      scripts.forEach(src=>{
        if(document.querySelector(`script[src="${src}"]`)){loaded++;if(loaded===scripts.length)resolve();return;}
        const s=document.createElement('script'); s.src=src; s.crossOrigin='anonymous';
        s.onload=()=>{loaded++;if(loaded===scripts.length)resolve();};
        document.head.appendChild(s);
      });
    });
  },[]);

  const extractKeypoints = useCallback((handRes, poseRes)=>{
    const kp=new Float32Array(FEATURE_SIZE); let off=0;
    if(poseRes?.poseLandmarks){ poseRes.poseLandmarks.forEach(lm=>{kp[off++]=lm.x;kp[off++]=lm.y;kp[off++]=lm.z;kp[off++]=lm.visibility??0;}); }
    else{ off+=132; }
    off+=1404;
    const lh=new Float32Array(63); const rh=new Float32Array(63);
    if(handRes?.multiHandLandmarks&&handRes?.multiHandedness){
      handRes.multiHandLandmarks.forEach((lms,i)=>{
        const label=handRes.multiHandedness[i]?.label;
        const arr=label==='Left'?lh:rh;
        lms.forEach((lm,j)=>{arr[j*3]=lm.x;arr[j*3+1]=lm.y;arr[j*3+2]=lm.z;});
      });
    }
    kp.set(lh,off); kp.set(rh,off+63);
    return Array.from(kp);
  },[]);

  const drawLandmarks = useCallback((ctx,handRes,poseRes,w,h)=>{
    if(poseRes?.poseLandmarks){
      const lms=poseRes.poseLandmarks;
      ctx.strokeStyle='rgba(255,120,30,0.7)'; ctx.lineWidth=2;
      POSE_CONNECTIONS.forEach(([a,b])=>{ if(!lms[a]||!lms[b])return; ctx.beginPath();ctx.moveTo(lms[a].x*w,lms[a].y*h);ctx.lineTo(lms[b].x*w,lms[b].y*h);ctx.stroke(); });
      lms.forEach(lm=>{ctx.beginPath();ctx.arc(lm.x*w,lm.y*h,3,0,Math.PI*2);ctx.fillStyle='rgba(255,140,50,0.9)';ctx.fill();});
    }
    if(handRes?.multiHandLandmarks){
      handRes.multiHandLandmarks.forEach(landmarks=>{
        ctx.strokeStyle='rgba(0,200,255,0.85)'; ctx.lineWidth=2;
        HAND_CONNECTIONS.forEach(([a,b])=>{ctx.beginPath();ctx.moveTo(landmarks[a].x*w,landmarks[a].y*h);ctx.lineTo(landmarks[b].x*w,landmarks[b].y*h);ctx.stroke();});
        landmarks.forEach(lm=>{ctx.beginPath();ctx.arc(lm.x*w,lm.y*h,4,0,Math.PI*2);ctx.fillStyle='#00dfa2';ctx.fill();});
      });
    }
  },[]);

  const runPrediction = useCallback(async(sequence)=>{
    try{
      const controller=new AbortController(); const timeout=setTimeout(()=>controller.abort(),5000);
      const res=await fetch(`${BACKEND_URL}/predict`,{
        method:'POST', headers:{'Content-Type':'application/json'},
        body:JSON.stringify({frames:sequence.map(kp=>({keypoints:kp}))}), signal:controller.signal
      });
      clearTimeout(timeout);
      if(!res.ok)return null; return await res.json();
    }catch{return null;}
  },[]);

  const processFrame = useCallback(()=>{
    if(!isRunningRef.current)return;
    const video=videoRef.current; const canvas=canvasRef.current;
    if(!video||!canvas||video.readyState<2){ animRef.current=requestAnimationFrame(processFrame); return; }
    const ctx=canvas.getContext('2d');
    const w=canvas.width=video.videoWidth||640; const h=canvas.height=video.videoHeight||480;
    ctx.save(); ctx.scale(-1,1); ctx.drawImage(video,-w,0,w,h); ctx.restore();
    const handRes=handsResultRef.current; const poseRes=poseResultRef.current;
    let count=0;
    if(handRes?.multiHandLandmarks)handRes.multiHandLandmarks.forEach(h2=>count+=h2.length);
    setLandmarkCount(count);
    drawLandmarks(ctx,handRes,poseRes,w,h);
    const kp=extractKeypoints(handRes,poseRes);
    sequenceRef.current.push(kp);
    if(sequenceRef.current.length>SEQUENCE_LENGTH)sequenceRef.current.shift();
    if(sequenceRef.current.length===SEQUENCE_LENGTH&&!predictingRef.current){
      predictingRef.current=true;
      const seqCopy=[...sequenceRef.current];
      runPrediction(seqCopy).then(result=>{
        predictingRef.current=false;
        if(!result)return;
        const{gesture:g,confidence:c,is_confident:ic}=result;
        setGesture(g); setConfidence(c); setIsConfident(ic);
        setConfHistory(prev=>[...prev.slice(1),Math.round(c*100)]);
        const pred=ic?g:'';
        predBufferRef.current.push(pred);
        if(predBufferRef.current.length>STABILITY_FRAMES)predBufferRef.current.shift();
        const allSame=predBufferRef.current.length===STABILITY_FRAMES&&new Set(predBufferRef.current).size===1&&pred!=='';
        if(allSame){
          const t=Date.now();
          if(g!==lastGestureRef.current||t-lastAddedRef.current>ADD_COOLDOWN_MS*3){
            if(t-lastAddedRef.current>ADD_COOLDOWN_MS){
              setWords(prev=>[...prev,{text:g,id:Date.now()}]);
              setGestureHistory(prev=>[{
                text:g, confidence:Math.round(c*100),
                time:new Date().toLocaleTimeString('en-US',{hour:'2-digit',minute:'2-digit',second:'2-digit',hour12:false}),
                id:Date.now()
              },...prev].slice(0,50));
              lastGestureRef.current=g; lastAddedRef.current=t;
            }
          }
        }
      });
    }
    fpsCountRef.current++;
    const el=Date.now()-fpsClockRef.current;
    if(el>=1000){setFps(Math.round(fpsCountRef.current/(el/1000)));fpsCountRef.current=0;fpsClockRef.current=Date.now();}
    animRef.current=requestAnimationFrame(processFrame);
  },[drawLandmarks,extractKeypoints,runPrediction]);

  const startCamera=useCallback(async()=>{
    try{
      await loadMediaPipe();
      const stream=await navigator.mediaDevices.getUserMedia({video:{width:1280,height:720,facingMode:'user'}});
      streamRef.current=stream; videoRef.current.srcObject=stream; await videoRef.current.play();
      const hands=new window.Hands({locateFile:f=>`https://cdn.jsdelivr.net/npm/@mediapipe/hands/${f}`});
      hands.setOptions({maxNumHands:2,modelComplexity:1,minDetectionConfidence:0.7,minTrackingConfidence:0.5});
      hands.onResults(r=>{handsResultRef.current=r;});
      mpHandsRef.current=hands;
      const pose=new window.Pose({locateFile:f=>`https://cdn.jsdelivr.net/npm/@mediapipe/pose/${f}`});
      pose.setOptions({modelComplexity:1,smoothLandmarks:true,minDetectionConfidence:0.6,minTrackingConfidence:0.5});
      pose.onResults(r=>{poseResultRef.current=r;});
      mpPoseRef.current=pose;
      const sendFrames=async()=>{
        if(!isRunningRef.current)return;
        if(videoRef.current&&videoRef.current.readyState>=2){
          await hands.send({image:videoRef.current});
          await pose.send({image:videoRef.current});
        }
        setTimeout(sendFrames,50);
      };
      isRunningRef.current=true; setIsRunning(true);
      sendFrames(); animRef.current=requestAnimationFrame(processFrame);
    }catch(err){
      console.error(err);
      if(err.name==='NotAllowedError') alert('Camera access denied. Please allow camera permission and try again.');
      else if(err.name==='NotFoundError') alert('No camera found. Please connect a webcam and try again.');
      else alert('Could not open camera: '+err.message);
    }
  },[loadMediaPipe,processFrame]);

  const stopCamera=useCallback(()=>{
    isRunningRef.current=false; setIsRunning(false);
    if(animRef.current)cancelAnimationFrame(animRef.current);
    if(streamRef.current){streamRef.current.getTracks().forEach(t=>t.stop());streamRef.current=null;}
    if(mpHandsRef.current)mpHandsRef.current.close();
    if(mpPoseRef.current)mpPoseRef.current.close();
    mpHandsRef.current=null; mpPoseRef.current=null;
    handsResultRef.current=null; poseResultRef.current=null;
    sequenceRef.current=[]; predBufferRef.current=[];
    setGesture(''); setConfidence(0); setIsConfident(false); setLandmarkCount(0);
    const canvas=canvasRef.current;
    if(canvas){const ctx=canvas.getContext('2d');ctx.clearRect(0,0,canvas.width,canvas.height);}
  },[]);

  const clearText=()=>{setWords([]);predBufferRef.current=[];};
  const copyText=()=>navigator.clipboard.writeText(words.map(w=>w.text.replace('_',' ')).join(' ')).catch(()=>{});
  const saveText=()=>{
    const text=words.map(w=>w.text.replace('_',' ')).join(' ');
    const blob=new Blob([text],{type:'text/plain'});
    const url=URL.createObjectURL(blob);
    const a=document.createElement('a');a.href=url;a.download=`gesturemind_${Date.now()}.txt`;a.click();
    URL.revokeObjectURL(url);
  };

  const wordCount=words.length;
  const charCount=words.map(w=>w.text.replace('_',' ')).join(' ').length;
  const sentenceCount=Math.max(0, words.length ? Math.ceil(words.length/4) : 0);
  const timeStr=now.toLocaleTimeString('en-US',{hour:'2-digit',minute:'2-digit',second:'2-digit',hour12:true});
  const dateStr=now.toLocaleDateString('en-US',{month:'short',day:'numeric',year:'numeric'});

  const navItems=[
    {id:'live',label:'Live SLR'},{id:'text',label:'Text Output'},
    {id:'history',label:'Gesture History'},{id:'settings',label:'Settings'},
    {id:'help',label:'Help'},{id:'about',label:'About'},
  ];

  return(
    <div className="app">
      {backendError && (
        <div className="error-banner">
          <span className="error-banner-icon">⚠️</span>
          <span>{backendError}</span>
          <span className="error-banner-hint">Run: <code>uvicorn main:app --host 0.0.0.0 --port 8000 --reload</code></span>
        </div>
      )}

      <header className="hdr">
        <div className="hdr-logo">
          <svg width="36" height="36" viewBox="0 0 36 36" fill="none">
            <circle cx="18" cy="18" r="17" stroke="#00dfa2" strokeWidth="1.5"/>
            <path d="M12 24L12 14L15 14L15 18L18 18L18 14L21 14L21 24" stroke="#00dfa2" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" fill="none"/>
            <circle cx="24" cy="16" r="2" fill="#00dfa2"/>
          </svg>
          <div>
            <div className="hdr-name">{APP_NAME}</div>
            <div className="hdr-sub">Sign Language Recognition System</div>
          </div>
        </div>
        <div className="hdr-title-wrap"><div className="hdr-title">REAL-TIME SIGN LANGUAGE RECOGNITION</div></div>
        <div className="hdr-right">
          <div className="hdr-time">{timeStr}</div>
          <div className="hdr-date">{dateStr}</div>
          <div className={`hdr-status ${backendReady?'on':'off'}`}><span className="hdr-dot"/>{backendReady?'ONLINE':'OFFLINE'}</div>
        </div>
      </header>

      <div className="body">
        <aside className="sidebar">
          <nav>
            {navItems.map(item=>(
              <button key={item.id} className={`nav-btn ${activeNav===item.id?'active':''}`} onClick={()=>setActiveNav(item.id)}>
                <NavIcon id={item.id}/>{item.label}
              </button>
            ))}
          </nav>
          <div className="sys-box">
            <div className="sys-hdr">SYSTEM STATUS</div>
            <SysRow label="Camera"     val={isRunning?'Active':'Inactive'}   green={isRunning}/>
            <SysRow label="Model"      val={modelLoaded?'Loaded':'Not Loaded'} green={modelLoaded}/>
            <SysRow label="Connection" val={backendReady?'Stable':'Offline'}  green={backendReady}/>
            <SysRow label="Alerts"     val={emailReady?'Enabled':'Disabled'}  green={emailReady}/>
            <SysRow label="FPS"        val={`${fps} FPS`} plain/>
          </div>
          <div className="sb-brand">
            <svg width="54" height="54" viewBox="0 0 54 54" fill="none">
              <circle cx="27" cy="27" r="23" stroke="#00dfa2" strokeWidth="1" strokeDasharray="4 3" opacity="0.35"/>
              <path d="M21 36C21 36 23 29 27 26C31 23 33 18 33 14" stroke="#00dfa2" strokeWidth="1.5" strokeLinecap="round" fill="none" opacity="0.8"/>
              <circle cx="27" cy="12" r="3" fill="#00dfa2" opacity="0.9"/>
            </svg>
            <span className="sb-brand-txt">AI-Powered<br/>Accessibility for All</span>
          </div>
        </aside>

        <main className="center">
          {activeNav==='live' && (
            <>
              <div className="cam-card">
                <div className="cam-hdr">
                  <span className="cam-lbl">LIVE CAMERA FEED</span>
                  <span className={`live-badge ${isRunning?'on':''}`}><span className="live-dot"/>{isRunning?'LIVE':'IDLE'}</span>
                </div>
                <div className="cam-body">
                  <video ref={videoRef} style={{display:'none'}} playsInline muted/>
                  <canvas ref={canvasRef} className="cam-canvas"/>
                  {!isRunning&&(
                    <div className="cam-placeholder">
                      <svg width="72" height="72" viewBox="0 0 24 24" fill="none" stroke="#2a2f4a" strokeWidth="1">
                        <rect x="2" y="7" width="15" height="13" rx="2"/><path d="M17 12l5-3v10l-5-3"/>
                      </svg>
                      <div className="cam-ph-text">Click <b>START LIVE SLR</b> to begin recognition</div>
                      {!backendReady && <div className="cam-ph-warn">⚠️ Backend offline — start uvicorn first</div>}
                    </div>
                  )}
                  {isRunning&&(
                    <div className="cam-hud-overlay">
                      <div className="hud-line"><span className="hud-green-dot"/>Landmarks: {landmarkCount}</div>
                      <div className="hud-line"><span className="hud-blue-dot"/>Confidence: {(confidence*100).toFixed(1)}%</div>
                    </div>
                  )}
                </div>
                <div className="gesture-bar">
                  <span className="gest-label">Gesture:</span>
                  <span className={`gest-val ${isConfident&&gesture?'active':''}`}>
                    {isRunning?(isConfident&&gesture?gesture.replace(/_/g,' ').toUpperCase():'—'):'WAITING...'}
                  </span>
                  <div className="sig-bars">
                    {[0.25,0.50,0.75,1.0].map((t,i)=><div key={i} className={`sig-bar sig-bar-${i+1} ${confidence>=t?'lit':''}`}/>)}
                  </div>
                </div>
              </div>
              <div className="ctrl-row">
                <button className={`btn-start ${isRunning?'stop':''}`} onClick={isRunning?stopCamera:startCamera} disabled={!backendReady&&!isRunning}>
                  {isRunning?<><StopIco/> STOP LIVE SLR</>:<><PlayIco/> START LIVE SLR</>}
                </button>
                <button className="btn-clear" onClick={clearText}><TrashIco/> CLEAR TEXT</button>
              </div>
              <div className="qa-card">
                <div className="qa-hdr">QUICK ACTIONS</div>
                <div className="qa-row">
                  <QABtn icon={<CopyIco/>} label="COPY" sub="TEXT" onClick={copyText}/>
                  <QABtn icon={<SaveIco/>} label="SAVE" sub="TEXT" onClick={saveText}/>
                  <QABtn icon={<MsgIco/>}  label="AI"   sub="ASSISTANT" onClick={()=>{}}/>
                </div>
              </div>
            </>
          )}

          {activeNav==='text' && (
            <TextOutputView words={words} onClear={clearText} onCopy={copyText} onSave={saveText}
              wordCount={wordCount} charCount={charCount} sentenceCount={sentenceCount}/>
          )}

          {activeNav==='settings' && (
            <SettingsView backendUrl={BACKEND_URL} confThresh={CONFIDENCE_THRESH}
              modelLoaded={modelLoaded} emailReady={emailReady} appName={APP_NAME}/>
          )}

          {activeNav==='help' && <HelpView/>}
          {activeNav==='about' && <AboutView appName={APP_NAME}/>}
          {activeNav==='history' && (
            <div className="center-history-hint">
              <div className="chh-icon">⟳</div>
              <div>Gesture history is shown in the right panel →</div>
            </div>
          )}
        </main>

        <aside className="rp">
          {activeNav!=='history' ? (
            <>
              <div className="rp-card">
                <div className="rp-hdr">RECOGNIZED TEXT</div>
                <div className="rp-text">
                  {words.length===0
                    ?<span className="rp-empty">Recognized words will appear here as you sign...</span>
                    :words.map((w,i)=>(<span key={w.id} className={`rp-word ${i===words.length-1?'new':''}`}>{w.text.replace(/_/g,' ')}{' '}</span>))
                  }
                </div>
              </div>
              <div className="rp-card">
                <div className="rp-hdr">TEXT STATISTICS</div>
                <div className="stats-row">
                  <div className="stat-box"><span className="stat-ico">T</span><div><div className="stat-lbl">Words</div><div className="stat-val">{wordCount}</div></div></div>
                  <div className="stat-box"><span className="stat-ico">Aa</span><div><div className="stat-lbl">Characters</div><div className="stat-val">{charCount}</div></div></div>
                </div>
              </div>
              <div className="rp-card">
                <div className="rp-hdr">CONFIDENCE METER</div>
                <div className="conf-body">
                  <div className="conf-gauge">
                    <svg width="96" height="96" viewBox="0 0 96 96">
                      <circle cx="48" cy="48" r="40" fill="none" stroke="#1e2235" strokeWidth="9"/>
                      <circle cx="48" cy="48" r="40" fill="none" stroke={confidence>=CONFIDENCE_THRESH?'#00dfa2':'#0ea5e9'} strokeWidth="9" strokeLinecap="round"
                        strokeDasharray={`${2*Math.PI*40}`} strokeDashoffset={`${2*Math.PI*40*(1-confidence)}`} transform="rotate(-90 48 48)" style={{transition:'stroke-dashoffset 0.3s'}}/>
                    </svg>
                    <div className="gauge-val">{(confidence*100).toFixed(1)}%</div>
                  </div>
                  <div className="conf-chart">
                    <div className="cc-labels"><span>100%</span><span>50%</span><span>0%</span></div>
                    <div className="cc-bars">
                      {confHistory.map((v,i)=>(<div key={i} className="cc-col"><div className="cc-fill" style={{height:`${v}%`,background:v>=70?'#00dfa2':'#0ea5e9',opacity:0.4+(i/confHistory.length)*0.6}}/></div>))}
                    </div>
                    <div className="cc-label">Real-time Confidence</div>
                  </div>
                </div>
              </div>
              <AgentPanel recognizedWords={words} isRunning={isRunning}/>
            </>
          ) : (
            <div className="rp-card history-card">
              <div className="rp-hdr-row">
                <div className="rp-hdr">GESTURE HISTORY</div>
                <button className="history-clear-btn" onClick={()=>setGestureHistory([])}>Clear</button>
              </div>
              {gestureHistory.length===0 ? (
                <div className="history-empty">
                  <div className="history-empty-icon">⟳</div>
                  <div className="history-empty-text">No gestures recorded yet.<br/>Start Live SLR to begin.</div>
                </div>
              ) : (
                <div className="history-list">
                  {gestureHistory.map((item,i)=>(
                    <div key={item.id} className={`history-item ${i===0?'history-newest':''}`}>
                      <div className="history-sign">{item.text.replace(/_/g,' ').toUpperCase()}</div>
                      <div className="history-meta">
                        <span className={`history-conf ${item.confidence>=70?'high':'low'}`}>{item.confidence}%</span>
                        <span className="history-time">{item.time}</span>
                      </div>
                    </div>
                  ))}
                </div>
              )}
              <div className="history-stats">
                <span>Total gestures: <b>{gestureHistory.length}</b></span>
                {gestureHistory.length>0 && <span>Avg confidence: <b>{Math.round(gestureHistory.reduce((s,g)=>s+g.confidence,0)/gestureHistory.length)}%</b></span>}
              </div>
            </div>
          )}
        </aside>
      </div>
    </div>
  );
}

/* ── Text Output full view ────────────────────────────────────────────── */
function TextOutputView({ words, onClear, onCopy, onSave, wordCount, charCount, sentenceCount }){
  return(
    <div className="view-card">
      <div className="view-hdr">
        <span className="view-title">TEXT OUTPUT</span>
        <div className="view-actions">
          <button className="qa-btn small" onClick={onCopy}><CopyIco/> Copy</button>
          <button className="qa-btn small" onClick={onSave}><SaveIco/> Save</button>
          <button className="btn-clear small" onClick={onClear}><TrashIco/> Clear</button>
        </div>
      </div>
      <div className="view-stats-row">
        <div className="stat-box"><span className="stat-ico">T</span><div><div className="stat-lbl">Words</div><div className="stat-val">{wordCount}</div></div></div>
        <div className="stat-box"><span className="stat-ico">Aa</span><div><div className="stat-lbl">Characters</div><div className="stat-val">{charCount}</div></div></div>
        <div className="stat-box"><span className="stat-ico">¶</span><div><div className="stat-lbl">Sentences</div><div className="stat-val">{sentenceCount}</div></div></div>
      </div>
      <div className="view-text-box">
        {words.length===0
          ? <span className="rp-empty">No text recognized yet. Go to Live SLR and start signing.</span>
          : words.map((w,i)=>(<span key={w.id} className={`view-word ${i===words.length-1?'new':''}`}>{w.text.replace(/_/g,' ')}{' '}</span>))
        }
      </div>
    </div>
  );
}

/* ── Settings view ─────────────────────────────────────────────────────── */
function SettingsView({ backendUrl, confThresh, modelLoaded, emailReady, appName }){
  return(
    <div className="view-card">
      <div className="view-hdr"><span className="view-title">SETTINGS</span></div>
      <div className="settings-list">
        <SettingRow label="App Name" value={appName}/>
        <SettingRow label="Backend URL" value={backendUrl} mono/>
        <SettingRow label="Confidence Threshold" value={`${(confThresh*100).toFixed(0)}%`}/>
        <SettingRow label="Model Status" value={modelLoaded?'Loaded ✅':'Not Loaded ❌'}/>
        <SettingRow label="Email Alerts" value={emailReady?'Enabled ✅':'Disabled — configure .env'}/>
        <SettingRow label="Sequence Length" value="30 frames"/>
        <SettingRow label="Feature Vector Size" value="1662"/>
        <SettingRow label="Supported Signs" value="hello, thank_you, please, yes, no, sorry, help, good, bad, stop"/>
      </div>
      <div className="settings-note">
        ⚙️ Configuration is managed via <code>.env</code> files in the backend and frontend folders.
        Edit those files and restart the servers to apply changes.
      </div>
    </div>
  );
}
function SettingRow({ label, value, mono }){
  return(
    <div className="setting-row">
      <span className="setting-label">{label}</span>
      <span className={`setting-value ${mono?'mono':''}`}>{value}</span>
    </div>
  );
}

/* ── Help view ─────────────────────────────────────────────────────────── */
function HelpView(){
  const steps = [
    ["1. Start the backend", "Run uvicorn main:app --host 0.0.0.0 --port 8000 --reload in the backend folder."],
    ["2. Check status", "Sidebar should show Camera, Model, Connection all green before starting."],
    ["3. Click START LIVE SLR", "Allow camera access when your browser prompts you."],
    ["4. Perform ASL signs", "Hold each sign steady for about 1 second — the system needs 30 frames to recognize it."],
    ["5. Watch the text build", "Recognized words appear in the RECOGNIZED TEXT panel on the right."],
    ["6. Use the AI Assistant", "Click Analyze Signs to get a full sentence and intent from Gemini."],
    ["7. Urgency alerts", "If you sign HELP, an email alert can be sent automatically to your emergency contact."],
  ];
  return(
    <div className="view-card">
      <div className="view-hdr"><span className="view-title">HELP & USAGE GUIDE</span></div>
      <div className="help-steps">
        {steps.map(([title,desc],i)=>(
          <div key={i} className="help-step">
            <div className="help-step-title">{title}</div>
            <div className="help-step-desc">{desc}</div>
          </div>
        ))}
      </div>
      <div className="settings-note">
        💡 Tip: Good lighting and a plain background significantly improve recognition accuracy.
      </div>
    </div>
  );
}

/* ── About view ────────────────────────────────────────────────────────── */
function AboutView({ appName }){
  return(
    <div className="view-card">
      <div className="view-hdr"><span className="view-title">ABOUT {appName.toUpperCase()}</span></div>
      <div className="about-body">
        <p className="about-p">
          <b>{appName}</b> is a real-time American Sign Language (ASL) recognition system
          that converts hand gestures into text using computer vision and deep learning,
          enhanced with an agentic AI layer for natural language understanding.
        </p>
        <div className="about-grid">
          <div className="about-item"><div className="about-item-label">Model</div><div className="about-item-val">LSTM · 93.3% test accuracy</div></div>
          <div className="about-item"><div className="about-item-label">Landmarks</div><div className="about-item-val">MediaPipe Hands + Pose</div></div>
          <div className="about-item"><div className="about-item-label">Agent</div><div className="about-item-val">Google Gemini 2.5 Flash</div></div>
          <div className="about-item"><div className="about-item-label">Signs Supported</div><div className="about-item-val">10 ASL words</div></div>
        </div>
        <p className="about-p">
          Built to help deaf and mute individuals communicate more easily, with automatic
          urgency detection that can alert a caregiver when the user signs for help.
        </p>
      </div>
    </div>
  );
}

function SysRow({ label, val, green, plain }) {
  return (
    <div className="sys-row">
      <span className="sys-lbl">{label}</span>
      <span className={`sys-val ${plain?'plain':green?'green':'red'}`}>
        {!plain && <span className={`sys-dot ${green?'green':'red'}`}/>}{val}
      </span>
    </div>
  );
}
function QABtn({ icon, label, sub, onClick }) {
  return (
    <button className="qa-btn" onClick={onClick}>
      <span className="qa-ico">{icon}</span>
      <span className="qa-txt">{label}<br/><span className="qa-sub">{sub}</span></span>
    </button>
  );
}
function NavIcon({ id }) {
  const icons = {
    live:    <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><rect x="2" y="7" width="15" height="13" rx="2"/><path d="M17 12l5-3v10l-5-3"/></svg>,
    text:    <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M4 7h16M4 12h10M4 17h12"/></svg>,
    history: <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 3"/></svg>,
    settings:<svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><circle cx="12" cy="12" r="3"/><path d="M12 2v2M12 20v2M4.22 4.22l1.42 1.42M18.36 18.36l1.42 1.42M2 12h2M20 12h2M4.22 19.78l1.42-1.42M18.36 5.64l1.42-1.42"/></svg>,
    help:    <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><circle cx="12" cy="12" r="9"/><path d="M9 9a3 3 0 016 1c0 2-3 3-3 3M12 17h.01"/></svg>,
    about:   <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><circle cx="12" cy="12" r="9"/><path d="M12 8h.01M12 12v4"/></svg>,
  };
  return <span className="nav-ico">{icons[id]}</span>;
}
const PlayIco  = () => <svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><circle cx="12" cy="12" r="10"/><polygon points="10,8 16,12 10,16" fill="currentColor"/></svg>;
const StopIco  = () => <svg width="17" height="17" viewBox="0 0 24 24" fill="currentColor"><rect x="6" y="6" width="12" height="12" rx="2"/></svg>;
const TrashIco = () => <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><polyline points="3 6 5 6 21 6"/><path d="M19 6l-1 14a2 2 0 01-2 2H8a2 2 0 01-2-2L5 6"/></svg>;
const CopyIco  = () => <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><rect x="9" y="9" width="13" height="13" rx="2"/><path d="M5 15H4a2 2 0 01-2-2V4a2 2 0 012-2h9a2 2 0 012 2v1"/></svg>;
const SaveIco  = () => <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M19 21H5a2 2 0 01-2-2V5a2 2 0 012-2h11l5 5v11a2 2 0 01-2 2z"/><polyline points="17 21 17 13 7 13 7 21"/><polyline points="7 3 7 8 15 8"/></svg>;
const MsgIco   = () => <svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M21 15a2 2 0 01-2 2H7l-4 4V5a2 2 0 012-2h14a2 2 0 012 2z"/></svg>;
