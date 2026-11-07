import { useEffect, useRef, useState } from 'react';
import { Activity, AlertCircle, ArrowLeft, ArrowRight, BookOpen, Camera, Check, Copy, Database, Hand, History, Menu, Mic, Plus, Search, Square, Trash2, Undo2, Volume2, X } from 'lucide-react';
import { FilesetResolver, HandLandmarker } from '@mediapipe/tasks-vision';
import { api } from './services/api';

type Hist = { id: number; text: string; date: string };
type Stats = { sample_count: number; class_count: number; per_label?: Record<string, number> };
const defaultLabels = ['Hello','Thank You','Yes','No','Please','Sorry','Help','Stop','Good','Bad','Love','Water','Food','Friend','Name','Mother','Father','Welcome','I','You'];
const lessons = [
  { name:'Hello', meaning:'A friendly greeting', level:'Beginner', instruction:'Choose a verified sign form for your regional sign language.' },
  { name:'Thank You', meaning:'Express gratitude', level:'Beginner', instruction:'Check the handshape and movement in a trusted learning resource.' },
  { name:'Please', meaning:'A polite request', level:'Beginner', instruction:'Practice the selected sign slowly and comfortably.' },
  { name:'Help', meaning:'Ask for support', level:'Beginner', instruction:'Learn the sign with its appropriate context and regional variation.' },
  { name:'Yes', meaning:'Affirmative response', level:'Beginner', instruction:'Repeat the sign and compare with a verified demonstration.' },
  { name:'Sorry', meaning:'Apologize or express regret', level:'Intermediate', instruction:'Practice the movement and facial expression used in your language.' },
];
const nav = [
  { name:'Home', short:'Home', icon:Hand },
  { name:'Live Sign Translator', short:'Translator', icon:Camera },
  { name:'Sign Language Learning', short:'Learn', icon:BookOpen },
  { name:'Dataset Collection', short:'Dataset', icon:Database },
  { name:'Model Performance', short:'Performance', icon:Activity },
  { name:'Translation History', short:'History', icon:History },
];

export default function App() {
  const [page,setPage]=useState('Home');
  const [mobile,setMobile]=useState(false);
  const [running,setRunning]=useState(false);
  const [status,setStatus]=useState('Camera is off');
  const [sentence,setSentence]=useState('');
  const [prediction,setPrediction]=useState('—');
  const [confidence,setConfidence]=useState(0);
  const [modelReady,setModelReady]=useState(false);
  const [labels,setLabels]=useState(defaultLabels);
  const [stats,setStats]=useState<Stats>({sample_count:0,class_count:0});
  const [history,setHistory]=useState<Hist[]>(()=>JSON.parse(localStorage.getItem('gestura-history')||'[]'));
  const [newLabel,setNewLabel]=useState('');
  const [collectLabel,setCollectLabel]=useState('Hello');
  const [notice,setNotice]=useState('');
  const [metrics,setMetrics]=useState<any>(null);
  const [query,setQuery]=useState('');
  const [category,setCategory]=useState('All levels');
  const video=useRef<HTMLVideoElement|null>(null);
  const stream=useRef<MediaStream|null>(null);
  const hand=useRef<HandLandmarker|null>(null);
  const timer=useRef<number>();
  const lastPrediction=useRef('');
  const lastTime=useRef(0);
  const stable=useRef<{label:string,count:number}>({label:'',count:0});
  const collecting=useRef(false);
  const collected=useRef(0);
  const [collectingUI,setCollectingUI]=useState(false);
  const [collectedUI,setCollected]=useState(0);

  const go=(next:string)=>{setPage(next);setMobile(false);window.scrollTo({top:0,behavior:'smooth'});};
  const refresh=async()=>{
    try {
      const [v,s,m,h]=await Promise.all([api<{labels:string[]}>('/vocabulary'),api<Stats>('/dataset/stats'),api<any>('/model/performance'),api<any>('/health')]);
      setLabels(v.labels);setStats(s);setMetrics(m);setModelReady(h.model_ready);
    } catch { setMetrics(null); }
  };
  useEffect(()=>{void refresh();},[]);
  useEffect(()=>{localStorage.setItem('gestura-history',JSON.stringify(history));},[history]);
  const saveHistory=()=>{const text=sentence.trim();if(!text)return;setHistory(items=>[{id:Date.now(),text,date:new Date().toLocaleString()},...items].slice(0,80));setNotice('Saved to this browser’s history');};
  const speak=()=>{if(!sentence.trim())return;if(!('speechSynthesis'in window)){setNotice('Speech synthesis is unavailable in this browser');return;}speechSynthesis.cancel();speechSynthesis.speak(new SpeechSynthesisUtterance(sentence));};
  const copy=async(text=sentence)=>{if(!text)return;try{await navigator.clipboard.writeText(text);setNotice('Copied to clipboard');}catch{setNotice('Clipboard access is unavailable in this browser');}};
  const stop=(update=true)=>{if(timer.current)clearInterval(timer.current);stream.current?.getTracks().forEach(track=>track.stop());stream.current=null;if(video.current)video.current.srcObject=null;setRunning(false);if(update)setStatus('Camera is off');};
  const processFrame=async()=>{
    if(!video.current||!hand.current||video.current.readyState<2)return;
    try {
      const frameResult=hand.current.detectForVideo(video.current,performance.now());
      if(!frameResult.landmarks?.length){stable.current={label:'',count:0};lastPrediction.current='';return;}
      const points=frameResult.landmarks[0],base=points[0];
      const landmarks=points.flatMap(point=>[point.x-base.x,point.y-base.y,point.z-base.z]);
      if(page==='Dataset Collection'){
        if(collecting.current){await api('/dataset/samples',{method:'POST',body:JSON.stringify({label:collectLabel,landmarks})});collected.current++;setCollected(collected.current);if(collected.current%5===0)void refresh();}
        return;
      }
      if(!modelReady)return;
      const now=Date.now();if(now-lastTime.current<450)return;
      const predictionResult=await api<{label:string,confidence:number}>('/recognize',{method:'POST',body:JSON.stringify({landmarks})});
      if(stable.current.label===predictionResult.label)stable.current.count++;else stable.current={label:predictionResult.label,count:1};
      if(stable.current.count>=3&&predictionResult.confidence>.62&&lastPrediction.current!==predictionResult.label){
        setPrediction(predictionResult.label);setConfidence(predictionResult.confidence);lastPrediction.current=predictionResult.label;lastTime.current=now;
        setSentence(value=>value+(value&&!/\s$/.test(value)?' ':'')+predictionResult.label);
      }
    } catch(error){if(error instanceof Error&&error.message.includes('No trained model'))setModelReady(false);}
  };
  const start=async()=>{
    try {
      setStatus('Requesting camera access…');
      stream.current=await navigator.mediaDevices.getUserMedia({video:{facingMode:'user',width:640,height:480},audio:false});
      if(video.current){video.current.srcObject=stream.current;await video.current.play();}
      setRunning(true);setStatus('Loading hand tracking…');
      if(!hand.current){
        const files=await FilesetResolver.forVisionTasks('https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.22/wasm');
        hand.current=await HandLandmarker.createFromOptions(files,{baseOptions:{modelAssetPath:'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task',delegate:'GPU'},runningMode:'VIDEO',numHands:1});
      }
      setStatus(modelReady?'Camera ready · show one sign':'Camera ready · collect samples and train to enable recognition');
      timer.current=window.setInterval(()=>void processFrame(),160);
    } catch(error:any) {
      setStatus(error.name==='NotAllowedError'?'Camera permission denied. Allow access in browser settings.':`Camera unavailable: ${error.message||'check device and permissions'}`);stop(false);
    }
  };
  const beginCollect=()=>{collected.current=0;setCollected(0);collecting.current=true;setCollectingUI(true);if(!running)void start();};
  const endCollect=()=>{collecting.current=false;setCollectingUI(false);void refresh();};
  useEffect(()=>()=>stop(false),[]);
  const addLabel=async()=>{try{await api('/labels',{method:'POST',body:JSON.stringify({name:newLabel})});setNewLabel('');void refresh();setNotice('Gesture label added');}catch(error:any){setNotice(error.message);}};
  const trainModel=async()=>{try{setNotice('Training model…');const result=await api<any>('/train',{method:'POST'});setMetrics(result);setModelReady(true);setNotice(`Model trained · held-out accuracy ${(result.testing_accuracy*100).toFixed(1)}%`);}catch(error:any){setNotice(error.message);}};
  const filteredLessons=lessons.filter(item=>item.name.toLowerCase().includes(query.toLowerCase())&&(category==='All levels'||item.level===category));

  return <div className="app">
    <header className="top">
      <button className="brand" onClick={()=>go('Home')} aria-label="GESTURA home"><span className="brand-mark"><Hand size={18}/></span><span className="brand-copy"><b>GESTURA</b><small>GESTURES THAT SPEAK.</small></span></button>
      <nav className="top-nav" aria-label="Main navigation">{nav.slice(0,4).map(item=><button key={item.name} className={page===item.name?'nav-active':''} onClick={()=>go(item.name)}>{item.short}</button>)}</nav>
      <button className="primary header-action" onClick={()=>go('Live Sign Translator')}>Start translating</button>
      <button className="mobile-menu" onClick={()=>setMobile(value=>!value)} aria-label={mobile?'Close menu':'Open menu'}>{mobile?<X size={19}/>:<Menu size={19}/>}</button>
    </header>
    {mobile&&<nav className="mobile-nav">{nav.map(item=><button key={item.name} onClick={()=>go(item.name)}><item.icon size={16}/>{item.short}</button>)}</nav>}
    <main>
      {page==='Home'?<>
        <section className="hero">
          <div className="hero-copy">
            <div className="eyebrow">AI-POWERED SIGN LANGUAGE TRANSLATION</div>
            <h1>Gestures that speak.</h1>
            <p className="lead">Communicate beyond words. Translate sign language gestures into text and speech using real-time hand recognition.</p>
            <div className="hero-actions"><button className="primary" onClick={()=>go('Live Sign Translator')}>Start translating <ArrowRight size={16}/></button><button className="secondary" onClick={()=>go('Sign Language Learning')}>Explore sign language</button></div>
            <p className="hero-note">No account required. Your camera stays on your device.</p>
          </div>
          <div className="hero-visual" aria-label="Line illustration of a hand"><div className="hand-illustration"><Hand size={112} strokeWidth={1.2}/></div><p>Hand landmark recognition</p></div>
        </section>
        <section className="home-section how-section"><div className="section-heading"><span className="eyebrow">HOW IT WORKS</span><h2>Three steps to a translation</h2></div><div className="steps"><article><span className="step-number">01</span><h3>Show a sign</h3><p>Perform a hand gesture in front of your camera.</p></article><article><span className="step-number">02</span><h3>Recognize</h3><p>The system detects hand landmarks and checks your trained model.</p></article><article><span className="step-number">03</span><h3>Translate</h3><p>View the result as text and use speech output when you’re ready.</p></article></div></section>
        <section className="home-section features-section"><div className="section-heading"><span className="eyebrow">FEATURES</span><h2>Tools for everyday practice</h2></div><div className="feature-grid"><button onClick={()=>go('Live Sign Translator')}><Camera/><b>Real-time translation</b><span>Translate recognized signs into editable text.</span></button><button onClick={()=>go('Model Performance')}><Activity/><b>Gesture recognition</b><span>Review model status and measured results.</span></button><button onClick={()=>go('Live Sign Translator')}><Volume2/><b>Text-to-speech</b><span>Read your translation aloud with browser speech.</span></button><button onClick={()=>go('Sign Language Learning')}><BookOpen/><b>Sign language learning</b><span>Browse vocabulary and practice with the camera.</span></button></div></section>
        <section className="about-section"><div><span className="eyebrow">ABOUT GESTURA</span><h2>Communication should be easier to share.</h2></div><p>GESTURA is an account-free project for exploring sign language translation and practice. It processes camera frames locally, while its gesture model learns from samples collected on your device. Sign forms vary across languages and regions.</p></section>
      </>:<div className="workspace">
        <aside className="sidebar"><span className="side-title">GESTURA</span>{nav.slice(1).map(item=><button key={item.name} className={page===item.name?'side-active':''} onClick={()=>go(item.name)}><item.icon size={16}/>{item.short}</button>)}<div className="side-note"><AlertCircle size={15}/><span>Signs vary by language and region. Confirm vocabulary for your chosen sign language.</span></div></aside>
        <div className="content"><div className="page-top"><div><button className="back-home" onClick={()=>go('Home')}><ArrowLeft size={14}/> Home</button><h1>{page==='Live Sign Translator'?'Sign Language Translator':page}</h1></div><span className="local-badge">No account required</span></div>
          {page==='Live Sign Translator'&&<>
            <p className="page-intro">Frame one hand clearly. The camera stays on this device; recognition requires a trained model.</p>
            <div className="translator-grid"><section className="panel camera-panel"><div className="panel-heading"><div><span className="tiny-label">CAMERA PREVIEW</span><h2>Show a sign</h2></div><span className={'status-pill '+(running?'on':'')}>{running?'Camera on':'Camera off'}</span></div><div className="video-box">{running?<video ref={video} muted playsInline/>:<div className="camera-placeholder"><Camera size={28}/><b>Camera preview</b><span>Position your hand in the center of the frame.</span></div>}<div className="camera-caption">{status}</div></div><div className="camera-controls">{running?<button className="stop-btn" onClick={()=>stop()}><Square size={14}/> Stop camera</button>:<button className="primary" onClick={()=>void start()}><Camera size={15}/> Start camera</button>}<span>Frames are processed locally.</span></div></section>
              <section className="panel result-panel"><div className="panel-heading"><div><span className="tiny-label">RECOGNITION STATUS</span><h2>{modelReady?'Model ready':'Model not trained'}</h2></div></div><div className="prediction-box"><span className="tiny-label">RECOGNIZED GESTURE</span><strong>{prediction}</strong><div className="confidence-row"><span>Model confidence</span><b>{confidence?`${Math.round(confidence*100)}%`:'—'}</b></div><div className="meter"><i style={{width:`${confidence*100}%`}}/></div></div>{!modelReady&&<p className="model-hint"><AlertCircle size={15}/> Collect labeled samples and train a model to enable predictions.</p>}</section></div>
            <section className="panel sentence-panel"><div className="panel-heading"><div><span className="tiny-label">TRANSLATED TEXT</span><h2>Edit your translation</h2></div><span className="word-count">{sentence.trim()?sentence.trim().split(/\s+/).length:0} words</span></div><textarea value={sentence} onChange={event=>setSentence(event.target.value)} placeholder="Recognized words will appear here. You can also type or edit text."/><div className="sentence-actions"><div><button className="action" onClick={()=>setSentence(value=>value.trim().split(/\s+/).slice(0,-1).join(' '))}><Undo2/> Undo</button><button className="action" onClick={()=>setSentence('')}><Trash2/> Clear</button></div><div><button className="action" onClick={()=>void copy()}><Copy/> Copy</button><button className="action" onClick={speak}><Volume2/> Speak</button><button className="primary small" onClick={saveHistory}>Save session</button></div></div></section>
          </>}
          {page==='Sign Language Learning'&&<><div className="notice-box"><AlertCircle size={16}/><span><b>Choose a regional sign language.</b> Starter labels are generic names; their sign forms differ between ASL, BSL, ISL, and other languages. Confirm lessons with a trusted regional resource.</span></div><div className="learning-toolbar"><label className="search-box"><Search size={16}/><input value={query} onChange={event=>setQuery(event.target.value)} placeholder="Search vocabulary"/></label><select aria-label="Filter by category" value={category} onChange={event=>setCategory(event.target.value)}><option>All levels</option><option>Beginner</option><option>Intermediate</option><option>Advanced</option></select><select aria-label="Sign language"><option>Select a sign language…</option><option>American Sign Language (ASL)</option><option>British Sign Language (BSL)</option><option>Indian Sign Language (ISL)</option><option>Other / custom</option></select></div><div className="lesson-grid">{filteredLessons.map(item=><article className="lesson-card" key={item.name}><div className="lesson-art"><Hand size={34}/><span>Illustrative placeholder</span></div><div className="lesson-body"><div className="lesson-meta"><span>{item.level}</span></div><h3>{item.name}</h3><p><b>Meaning:</b> {item.meaning}</p><p>{item.instruction}</p><button className="text-link" onClick={()=>{setCollectLabel(item.name);go('Live Sign Translator');}}>Practice with camera <ArrowRight size={14}/></button></div></article>)}</div>{filteredLessons.length===0&&<div className="empty-state">No vocabulary matches that search.</div>}</>}
          {page==='Dataset Collection'&&<><p className="page-intro">Collect landmark samples, manage supported gesture classes, and train a model from your data. Camera video is not stored.</p><div className="notice-box"><Database size={16}/><span>Training requires at least two labels with samples. Collect varied examples for more useful evaluation.</span></div><div className="dataset-grid"><section className="panel"><div className="panel-heading"><div><span className="tiny-label">SAMPLE COLLECTION</span><h2>Collect gesture data</h2></div><span className="status-pill">{collectingUI?'Collecting':'Ready'}</span></div><label className="field-label">Gesture class<select value={collectLabel} onChange={event=>setCollectLabel(event.target.value)}>{labels.map(label=><option key={label}>{label}</option>)}</select></label><div className="collection-counter"><strong>{collectedUI}</strong><span>samples this session</span></div><div className="collection-actions">{!collectingUI?<button className="primary" onClick={beginCollect}><Camera size={15}/> Start collecting</button>:<button className="stop-btn" onClick={endCollect}><Square size={14}/> Stop collection</button>}<span>{running?'Camera on':'Camera off'}</span></div><div className="camera-mini">{running?<video ref={element=>{video.current=element;if(element&&stream.current){element.srcObject=stream.current;void element.play().catch(()=>{});}}} muted playsInline/>:<Camera size={22}/>}<span>{running?'Camera active':'Camera preview appears here when started'}</span></div></section>
              <section className="panel"><div className="panel-heading"><div><span className="tiny-label">SUPPORTED CLASSES</span><h2>Gesture labels</h2></div><span className="word-count">{labels.length} labels</span></div><div className="add-label"><input value={newLabel} onChange={event=>setNewLabel(event.target.value)} placeholder="New gesture name" onKeyDown={event=>event.key==='Enter'&&void addLabel()}/><button className="primary small" onClick={()=>void addLabel()}><Plus size={14}/> Add</button></div><div className="label-table"><div className="label-row label-heading"><span>Gesture</span><span>Samples</span></div>{labels.map(label=><div className="label-row" key={label}><span>{label}</span><span>{stats.per_label?.[label]||0}</span></div>)}</div><div className="dataset-foot">{stats.sample_count} total samples · {stats.class_count} classes with samples</div></section></div><section className="panel train-panel"><div><span className="tiny-label">MODEL TRAINING</span><h2>Train and evaluate</h2><p>Accuracy is measured from a held-out split of the collected dataset.</p></div><button className="primary" onClick={()=>void trainModel()}><Activity size={15}/> Train model</button></section></>}
          {page==='Model Performance'&&<><p className="page-intro">Performance values are shown only after a successful training run and are based on your collected data.</p><div className="metrics-grid"><div className="metric-card"><span className="tiny-label">MODEL STATUS</span><strong>{modelReady?'Ready':'Not trained'}</strong></div><div className="metric-card"><span className="tiny-label">DATASET SAMPLES</span><strong>{stats.sample_count}</strong></div><div className="metric-card"><span className="tiny-label">GESTURE CLASSES</span><strong>{stats.class_count}</strong></div><div className="metric-card"><span className="tiny-label">TEST ACCURACY</span><strong>{metrics?.testing_accuracy!=null?`${(metrics.testing_accuracy*100).toFixed(1)}%`:'—'}</strong></div></div><section className="panel performance-panel"><div className="panel-heading"><div><span className="tiny-label">EVALUATION</span><h2>Measured model performance</h2></div><button className="secondary small" onClick={()=>void refresh()}>Refresh</button></div>{metrics?.testing_accuracy!=null?<><div className="accuracy-line"><span>Held-out testing accuracy</span><b>{(metrics.testing_accuracy*100).toFixed(1)}%</b></div><div className="meter big"><i style={{width:`${metrics.testing_accuracy*100}%`}}/></div><p>Measured from {metrics.sample_count} samples across {metrics.class_count} gesture classes.</p></>:<div className="empty-state"><Activity size={25}/><b>No measured results yet</b><span>Collect samples for at least two classes, then train a model.</span><button className="text-link" onClick={()=>go('Dataset Collection')}>Open dataset collection <ArrowRight size={14}/></button></div>}</section></>}
          {page==='Translation History'&&<><div className="history-toolbar"><div><span className="tiny-label">SAVED IN THIS BROWSER</span><h2>Recent translations</h2></div>{history.length>0&&<button className="action danger" onClick={()=>setHistory([])}><Trash2/> Clear history</button>}</div>{history.length===0?<div className="panel empty-state"><History size={25}/><b>No saved translations</b><span>Save a session from the translator to find it here.</span><button className="primary small" onClick={()=>go('Live Sign Translator')}>Open translator</button></div>:<div className="history-list">{history.map(item=><article className="panel history-card" key={item.id}><div className="history-text"><p>{item.text}</p><small>{item.date}</small></div><button className="icon-btn" aria-label="Copy translation" onClick={()=>void copy(item.text)}><Copy size={16}/></button><button className="icon-btn danger" aria-label="Delete translation" onClick={()=>setHistory(items=>items.filter(entry=>entry.id!==item.id))}><Trash2 size={16}/></button></article>)}</div>}</>}
        </div>
      </div>}
    </main>
    <footer className="footer"><button className="brand" onClick={()=>go('Home')}><span className="brand-mark"><Hand size={17}/></span><span className="brand-copy"><b>GESTURA</b><small>GESTURES THAT SPEAK.</small></span></button><span>Communication begins with understanding.</span></footer>
    {notice&&<button className="toast" onClick={()=>setNotice('')}><Check size={15}/>{notice}<X size={13}/></button>}
  </div>;
}
