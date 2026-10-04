import React, {useEffect, useState} from 'react';
import {createRoot} from 'react-dom/client';
import './style.css';

function App(){
  const [workspace,setWorkspace]=useState('universal'), [sketchStyle,setSketchStyle]=useState(1);
  const [file,setFile]=useState(null), [samples,setSamples]=useState([]), [sample,setSample]=useState('');
  const [condition,setCondition]=useState('clean'), [prob,setProb]=useState(.08), [kernel,setKernel]=useState(5);
  const [sigma,setSigma]=useState(1.5), [area,setArea]=useState(.2), [boxes,setBoxes]=useState(2);
  const [result,setResult]=useState(null), [error,setError]=useState(''), [busy,setBusy]=useState(false);
  useEffect(()=>{fetch('/api/samples').then(r=>r.json()).then(setSamples).catch(()=>setError('Could not connect to the backend.'));},[]);
  async function run(e){
    e.preventDefault();setBusy(true);setError('');setResult(null);
    const data=new FormData();if(file)data.append('file',file);else if(sample)data.append('sample_id',sample);
    Object.entries(workspace==='sketch'?{style:sketchStyle}:{condition,probability:prob,kernel,sigma,area_ratio:area,num_boxes:boxes,seed:42}).forEach(([k,v])=>data.append(k,v));
    try{const response=await fetch(workspace==='sketch'?'/api/object-to-sketch':workspace==='soft'?'/api/soft-moe-restoration':workspace==='hard'?'/api/hard-routed-restoration':'/api/universal-restoration',{method:'POST',body:data});const body=await response.json();if(!response.ok)throw new Error(typeof body.detail==='string'?body.detail:JSON.stringify(body.detail));setResult(body);}
    catch(e){setError(e.message);}finally{setBusy(false);}
  }
  const slider=(title,value,set,min,max,step)=> <label className="block mt-5">{title}: {value}<input className="block w-full mt-2 accent-emerald-700" type="range" min={min} max={max} step={step} value={value} onChange={e=>set(Number(e.target.value))}/></label>;
  return <main className="min-h-screen bg-slate-50 text-slate-900 p-5 md:p-10"><div className="max-w-6xl mx-auto">
    <p className="text-emerald-700 font-semibold">Generative AI · Restoration</p>
    <nav aria-label="Restoration workspace" className="flex flex-wrap gap-3 mt-4">{[['universal','Universal Restoration'],['hard','Hard-Routed Restoration'],['soft','Soft Mixture-of-Experts'],['sketch','Object-to-Sketch Generator']].map(([key,title])=><button key={key} type="button" disabled={busy} aria-pressed={workspace===key} onClick={()=>{setWorkspace(key);setResult(null);setError('');}} className={'rounded-lg px-4 py-2 border '+(workspace===key?'bg-emerald-700 text-white':'bg-white')}>{title}</button>)}</nav>
    <h1 className="text-3xl font-bold mt-4">{workspace==='sketch'?'Object-to-Sketch Generator':workspace==='soft'?'Soft Mixture-of-Experts Restoration':workspace==='hard'?'Hard-Routed Restoration':'Universal Restoration'}</h1><p className="text-slate-600 mt-3">{workspace==='sketch'?'Choose pencil, technical ink, or charcoal to generate a sketch from your photograph.':workspace==='soft'?'Blend the unchanged input with noise, blur, and occlusion specialists using learned weights.':workspace==='hard'?'Classify the input, then select a noise, blur, or occlusion specialist. Clean predictions pass through unchanged.':'Restore noise, blur, and occlusion with one autoencoder.'}</p>
    <div className="grid md:grid-cols-[300px_1fr] gap-6 mt-8"><form onSubmit={run} className="bg-white border rounded-2xl p-6">
      <label className="font-medium block">Upload an image<input type="file" accept="image/*" className="block mt-3 w-full text-sm" onChange={e=>{setFile(e.target.files[0]||null);setSample('');setResult(null);}}/></label>
      <label className="block mt-5">Or choose a clean sample<select className="block border rounded p-2 w-full mt-2" value={sample} onChange={e=>{setSample(e.target.value);setFile(null);setResult(null);}}><option value="">Select sample</option>{samples.map(s=><option key={s.id} value={s.id}>Flower {s.id}</option>)}</select></label>
      {workspace==='sketch'?<label className="block mt-5">Sketch style<select className="block border rounded p-2 w-full mt-2" value={sketchStyle} onChange={e=>setSketchStyle(Number(e.target.value))}>{['Fine Pencil','Technical Ink','Tonal Charcoal'].map((name,i)=><option key={name} value={i+1}>{name}</option>)}</select></label>:<>
      <label className="block mt-5">Corruption<select className="block border rounded p-2 w-full mt-2" value={condition} onChange={e=>setCondition(e.target.value)}><option value="clean">Use image as uploaded</option><option value="salt_and_pepper">Salt-and-pepper noise</option><option value="gaussian_blur">Gaussian blur</option><option value="rectangular_occlusion">Rectangular occlusion</option></select></label>
      {condition==='salt_and_pepper'&&slider('Noise probability',prob,setProb,.02,.15,.01)}
      {condition==='gaussian_blur'&&<>{slider('Blur sigma',sigma,setSigma,.5,2.5,.1)}<label className="block mt-5">Kernel<select className="ml-3 border p-1" value={kernel} onChange={e=>setKernel(Number(e.target.value))}>{[3,5,7].map(k=><option key={k}>{k}</option>)}</select></label></>}
      {condition==='rectangular_occlusion'&&<>{slider('Covered area',area,setArea,.1,.35,.01)}{slider('Rectangles',boxes,setBoxes,1,3,1)}</>}
      </>}
      <button disabled={busy||(!file&&!sample)} className="mt-7 w-full rounded-lg bg-emerald-700 text-white p-3 disabled:opacity-40">{busy?(workspace==='sketch'?'Generating…':'Restoring…'):(workspace==='sketch'?'Generate sketch':'Restore image')}</button><p role="alert" className="text-red-700 mt-4">{error}</p>
    </form><section className="bg-white border rounded-2xl p-6">
      {!result?<div className="min-h-72 flex items-center justify-center text-slate-500">Your input and restored image will appear here.</div>:<>
        {result.smoke_model&&<p className="mb-4 text-amber-800">Development model — quality evaluation is pending.</p>}
        {result.routing_probabilities&&<div className="mb-6"><p className="font-medium">{result.soft_mixture?'Largest contribution: ':'Selected route: '}{result.predicted_condition.replaceAll('_',' ')}{result.identity_bypass?' (identity bypass)':''}</p><div className="grid sm:grid-cols-2 gap-3 mt-3">{Object.entries(result.routing_probabilities).map(([name,p])=><div key={name}><div className="flex justify-between text-sm"><span>{name.replaceAll('_',' ')}</span><span>{(p*100).toFixed(1)}%</span></div><progress className="w-full accent-emerald-700" max="1" value={p} aria-label={name+(result.soft_mixture?' contribution weight':' probability')}/></div>)}</div></div>}
        <div className="grid sm:grid-cols-2 gap-5">{[[result.sketch_mode?'Photograph':'Input',result.input],[result.sketch_mode?'Generated '+result.style_name:'Restored',result.output],...(result.error_map?[['Clean reference',result.original],['Absolute error',result.error_map]]:[])].map(([title,src])=><figure key={title}><figcaption className="font-medium mb-2">{title}</figcaption><img className="w-full rounded-lg border" src={src} alt={title}/></figure>)}</div>
        <div className="mt-6 flex flex-wrap items-center justify-between gap-4"><span>{result.inference_ms.toFixed(1)} ms inference</span><a className="bg-slate-900 text-white rounded-lg px-4 py-2" href={result.output} download={result.sketch_mode?'sketch.png':'restored.png'}>Download result</a></div>
        {!result.sketch_mode&&<details className="mt-5 text-sm"><summary>Applied corruption settings</summary><pre className="mt-2 overflow-auto">{JSON.stringify(result.corruption,null,2)}</pre></details>}
      </>}
    </section></div></div></main>;
}
createRoot(document.getElementById('root')).render(<App/>);
