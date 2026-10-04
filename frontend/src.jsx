import React, {useEffect, useRef, useState} from 'react';
import {createRoot} from 'react-dom/client';
import './style.css';

const workspaces=[['universal','Universal Restoration','Restore noise, blur, and occlusion with one autoencoder.'],['hard','Hard-Routed Restoration','Classify the image and select a specialist. Clean predictions pass through unchanged.'],['soft','Soft Mixture-of-Experts','Blend the unchanged input with noise, blur, and occlusion specialists using learned weights.'],['sketch','Object-to-Sketch Generator','Generate a pencil, technical ink, or charcoal sketch from your photograph.']];
const styles=['Fine Pencil','Technical Ink','Tonal Charcoal'];
const styleDescriptions=['Fine contours and light pencil shading.','High-contrast edges and bold ink contours.','Soft tonal shading and charcoal textures.'];
const conditions=[['clean','Use image as uploaded'],['salt_and_pepper','Salt-and-pepper noise'],['gaussian_blur','Gaussian blur'],['rectangular_occlusion','Rectangular occlusion']];
function Icon({name='flower'}){
 const paths={flower:'M12 8c-5-8-10-2-5 2-8 1-5 9 1 6-2 8 7 8 8 1 6 4 10-4 3-6 5-5-2-10-7-3Z',upload:'M12 16V4m-5 5 5-5 5 5M4 16v4h16v-4',controls:'M4 6h16M4 12h16M4 18h16M8 3v6m8 0v6m-7 0v6',image:'M3 3h18v18H3Zm0 14 6-6 5 5 3-3 4 4M15 7h.01',download:'M12 3v12m-5-5 5 5 5-5M4 17v4h16v-4',spark:'m12 2 3 7 7 3-7 3-3 7-3-7-7-3 7-3Z',check:'m4 12 5 5L20 6'};
 return <svg className="icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true"><path d={paths[name]||paths.flower}/></svg>;
}
function App(){
 const [workspace,setWorkspace]=useState('universal'),[sketchStyle,setSketchStyle]=useState(1);
 const [file,setFile]=useState(null),[samples,setSamples]=useState([]),[sample,setSample]=useState(''),[preview,setPreview]=useState('');
 const [condition,setCondition]=useState('clean'),[prob,setProb]=useState(.08),[kernel,setKernel]=useState(5),[sigma,setSigma]=useState(1.5),[area,setArea]=useState(.2),[boxes,setBoxes]=useState(2);
 const [result,setResult]=useState(null),[error,setError]=useState(''),[busy,setBusy]=useState(false),[ready,setReady]=useState(false);
 const inputRef=useRef(null);const sketch=workspace==='sketch';const index=workspaces.findIndex(w=>w[0]===workspace);
 useEffect(()=>{Promise.all([fetch('/api/samples').then(r=>{if(!r.ok)throw Error();return r.json();}),fetch('/api/health').then(r=>r.json())]).then(([s,h])=>{setSamples(s);setReady(['model_ready','task2_ready','task3_ready','task4_ready'].every(k=>h[k]));}).catch(()=>setError('Could not connect to the backend.'));},[]);
 useEffect(()=>{if(!file){setPreview('');return;}const url=URL.createObjectURL(file);setPreview(url);return()=>URL.revokeObjectURL(url);},[file]);
 function chooseFile(next){setFile(next||null);setSample('');setResult(null);setError('');}
 function chooseSample(value){setSample(value);setFile(null);if(inputRef.current)inputRef.current.value='';setResult(null);setError('');}
 async function run(e){
  e.preventDefault();setBusy(true);setError('');setResult(null);
  const data=new FormData();if(file)data.append('file',file);else if(sample)data.append('sample_id',sample);
  Object.entries(sketch?{style:sketchStyle}:{condition,probability:prob,kernel,sigma,area_ratio:area,num_boxes:boxes,seed:42}).forEach(([k,v])=>data.append(k,v));
  try{const response=await fetch(sketch?'/api/object-to-sketch':workspace==='soft'?'/api/soft-moe-restoration':workspace==='hard'?'/api/hard-routed-restoration':'/api/universal-restoration',{method:'POST',body:data});const body=await response.json();if(!response.ok)throw Error(typeof body.detail==='string'?body.detail:JSON.stringify(body.detail));setResult(body);}catch(e){setError(e.message);}finally{setBusy(false);}
 }
 const slider=(title,value,set,min,max,step)=><label className="slider-label"><span>{title}<strong>{value}</strong></span><input type="range" min={min} max={max} step={step} value={value} onChange={e=>{set(Number(e.target.value));setResult(null);}}/></label>;
 const panels=result?[[sketch?'Original Photograph':'Input Image',result.input,'RGB · 128 × 128'],[sketch?'Generated Sketch':'Restored Image',result.output,sketch?result.style_name:'Model output'],...(result.error_map?[['Clean Reference',result.original,'Ground truth'],['Absolute Error Map',result.error_map,'Pixel residual']]:[])]:[];
 return <div className="studio">
  <header className="studio-header"><a className="brand" href="#" aria-label="Vision and Generative AI Lab"><span className="brand-symbol"><Icon/></span><span><strong>CS4065 // Vision &amp;<br/>Generative AI Lab</strong><small>Image Restoration and Sketch Generator</small></span></a>
   <nav aria-label="Restoration workspace">{workspaces.map(([key,title],i)=><button key={key} type="button" disabled={busy} aria-pressed={workspace===key} onClick={()=>{setWorkspace(key);setResult(null);setError('');}}><span className="nav-number">0{i+1}.</span> {title}</button>)}</nav>
  </header>
  <div className="system-strip"><span className={'badge '+(ready?'green':'')}><span className="status-dot"/>{ready?'MODELS READY':'CONNECTING'}</span><span>Workspace 0{index+1} / {workspaces[index][1]}</span><span className="runtime">ONNX Runtime · CPU</span><span className="resolution">128 × 128 RGB</span></div>
  <main className="studio-main"><div className="workspace-intro"><h1>{workspaces[index][1]}</h1><p>{workspaces[index][2]}</p></div>
   <div className="workspace-grid"><form onSubmit={run} className="control-panel card"><fieldset disabled={busy}>
    <div className="card-heading"><h2><Icon name="controls"/>{sketch?'Synthesis Controls':'Inference Controls'}</h2><span className="badge">{busy?'RUNNING':'READY / IDLE'}</span></div>
    <div className="control-group"><div className="section-label"><span>1. Input Image Source</span><span>{file?'Upload Mode':sample?'Sample Mode':'Choose Input'}</span></div>
     <label className="select-label">Select a clean sample<select value={sample} onChange={e=>chooseSample(e.target.value)}><option value="">Select a flower sample</option>{samples.map(s=><option key={s.id} value={s.id}>Flower {s.id}</option>)}</select></label>
     <label className="upload-zone" onDragOver={e=>e.preventDefault()} onDrop={e=>{e.preventDefault();if(!busy)chooseFile(e.dataTransfer.files[0]);}}><Icon name="upload"/><strong>{file?file.name:'Upload your photograph'}</strong><span>Choose a file or drop it here · up to 5 MB</span><input ref={inputRef} type="file" accept="image/*" onChange={e=>chooseFile(e.target.files[0])}/></label>
     {preview&&<div className="upload-preview"><img src={preview} alt="Selected photograph"/><span>Selected photograph<br/><small>Resized to 128 × 128 for inference</small></span></div>}
    </div>
    <div className="control-group"><div className="section-label"><span>2. {sketch?'Synthesis Style':'Simulated Degradation'}</span><span>{sketch?'3 styles':'Optional'}</span></div>

     <div className="choice-list">{(sketch?styles.map((v,i)=>[i+1,v]):conditions).map(([v,label])=><label key={v} className={'choice '+((sketch?sketchStyle:condition)===v?'selected':'')}><input type="radio" name={sketch?'style':'condition'} value={v} checked={(sketch?sketchStyle:condition)===v} onChange={()=>{sketch?setSketchStyle(v):setCondition(v);setResult(null);}}/><span>{label}{sketch&&<small>{styleDescriptions[v-1]}</small>}</span>{(sketch?sketchStyle:condition)===v&&<em>ACTIVE</em>}</label>)}</div>
     {!sketch&&condition!=='clean'&&<div className="parameter-box">{condition==='salt_and_pepper'&&slider('Noise probability',prob,setProb,.02,.15,.01)}{condition==='gaussian_blur'&&<><label className="select-label">Kernel size<select value={kernel} onChange={e=>{setKernel(Number(e.target.value));setResult(null);}}>{[3,5,7].map(k=><option key={k} value={k}>{k} × {k}</option>)}</select></label>{slider('Standard deviation (σ)',sigma,setSigma,.5,2.5,.1)}</>}{condition==='rectangular_occlusion'&&<>{slider('Covered area',area,setArea,.1,.35,.01)}{slider('Rectangles',boxes,setBoxes,1,3,1)}</>}</div>}
    </div>
    <button data-testid="run" className="primary-button" disabled={busy||(!file&&!sample)}><Icon name="spark"/>{busy?(sketch?'Generating…':'Restoring…'):(sketch?'Generate sketch':'Restore image')}</button><p className="control-note">{sketch?'Style-conditioned GAN generator':'Deterministic corruption seed: 42'}</p>
   </fieldset><p role="alert" className="error-message">{error}</p></form>
   <section className="results-panel card" aria-label="Inference results" aria-busy={busy}>
    {!result?<div className="empty-state"><div className="empty-icon"><Icon name={sketch?'spark':'image'}/></div><span className="badge">{busy?'INFERENCE IN PROGRESS':'WORKSPACE READY'}</span><h2>{busy?(sketch?'Creating your sketch':'Restoring your image'):'Bring your image into focus'}</h2><p>{busy?'Processing your image with the selected trained model.':sketch?'Select a sample or upload a photograph, choose your drawing style, and generate a sketch.':'Select a sample or upload a photograph, choose an optional degradation, and restore your image.'}</p><div className="empty-grid"><span>01 / {sketch?'Photograph':'Input Image'}</span><span>02 / {sketch?'Generated Sketch':'Restored Image'}</span></div></div>:<>
     <div className="result-toolbar"><span className="badge success"><Icon name="check"/>{sketch?'Generation Successful':'Inference Complete'}</span><span className="badge">Inference time: <strong>{result.inference_ms.toFixed(1)} ms</strong></span>{sketch&&<span className="badge">Style: <strong>{result.style_name}</strong></span>}<a className="download-button" href={result.output} download={sketch?'sketch.png':'restored.png'}><Icon name="download"/>Download result</a></div>
     {result.smoke_model&&<p className="error-message">Development model — quality evaluation is pending.</p>}
     {result.routing_probabilities&&<div className="routing-panel"><div className="section-label"><span>{result.soft_mixture?'Expert Contribution Weights':'Routing Probabilities'}</span><span>{result.identity_bypass?'Identity bypass':result.predicted_condition.replaceAll('_',' ')}</span></div><div className="routing-grid">{Object.entries(result.routing_probabilities).map(([name,p])=><div key={name}><div className="weight-label"><span>{name.replaceAll('_',' ')}</span><strong>{(p*100).toFixed(1)}%</strong></div><progress max="1" value={p} aria-label={name+(result.soft_mixture?' contribution weight':' probability')}/></div>)}</div></div>}
     <div className="image-grid">{panels.map(([title,src,note],i)=><figure className={'image-panel '+(i===1?'output-panel':'')} key={title}><figcaption><strong><span className="status-dot"/>{i+1}. {title}</strong><span className="image-tag">{note}</span></figcaption><div className="image-frame"><img src={src} alt={title}/></div><div className="image-footer"><span>{i===3?'Absolute pixel difference':i===2?'Known clean source':i===1?sketch?'Style-conditioned synthesis':'Restoration prediction':'Model input'}</span><span>{i===1?'OUTPUT':'128 × 128'}</span></div></figure>)}</div>
     <div className="result-notes"><div><span>MODEL</span><strong>{['Universal autoencoder','Classifier + specialist','Soft mixture of experts','Conditional GAN'][index]}</strong></div><div><span>OUTPUT RESOLUTION</span><strong>128 × 128 <small>RGB</small></strong></div><div><span>EXECUTION</span><strong>ONNX Runtime <small>CPU</small></strong></div></div>
     {!sketch&&<details className="corruption-details"><summary>Applied corruption settings</summary><pre>{JSON.stringify(result.corruption,null,2)}</pre></details>}
    </>}
   </section></div>
  </main><footer className="studio-footer"><span>CS4065 Deep Generative Computer Vision Research Bench</span><span>Muhammad Shaffan Ahmad · 23i-0673</span></footer>
 </div>;
}
createRoot(document.getElementById('root')).render(<App/>);
