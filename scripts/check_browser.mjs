// Exercise the production build through Chrome DevTools, without extra packages.
import fs from 'node:fs';
const output = 'runs/integration/browser';
const appUrl = process.env.APP_URL || 'http://127.0.0.1:18080';
fs.mkdirSync(output, {recursive:true});
const pages = await (await fetch('http://127.0.0.1:9227/json/list')).json();
const page = pages.find(p=>p.type==='page');
const ws = new WebSocket(page.webSocketDebuggerUrl);
await new Promise((resolve,reject)=>{ws.onopen=resolve;ws.onerror=reject;});
let sequence=0;
const pending=new Map(), errors=[];
ws.onmessage=({data})=>{
  const m=JSON.parse(data);
  if(m.id){const p=pending.get(m.id);pending.delete(m.id);m.error?p.reject(new Error(JSON.stringify(m.error))):p.resolve(m.result);}
  if(m.method==='Runtime.exceptionThrown')errors.push(m.params.exceptionDetails.text);
};
const send=(method,params={})=>new Promise((resolve,reject)=>{const id=++sequence;pending.set(id,{resolve,reject});ws.send(JSON.stringify({id,method,params}));});
async function evaluate(expression){
  const value=await send('Runtime.evaluate',{expression,returnByValue:true,awaitPromise:true});
  if(value.exceptionDetails)throw new Error(JSON.stringify(value.exceptionDetails));
  return value.result.value;
}
async function waitFor(expression){
  for(let i=0;i<100;i++){if(await evaluate(`Boolean(${expression})`))return;await new Promise(r=>setTimeout(r,100));}
  throw new Error('Timed out: '+expression);
}
const assert=(condition,message)=>{if(!condition)throw new Error(message);};
try{
  await send('Runtime.enable');await send('Page.enable');
  await send('Emulation.setDeviceMetricsOverride',{width:1365,height:1000,deviceScaleFactor:1,mobile:false});
  await send('Page.navigate',{url:appUrl});
  await waitFor("document.querySelectorAll('nav button').length===4 && document.querySelector('form select')?.options.length>1");
  await evaluate("(()=>{const s=document.querySelector('form select');s.value=s.options[1].value;s.dispatchEvent(new Event('change',{bubbles:true}));})()");
  const checks=[];
  for(let index=0;index<4;index++){
    await evaluate(`document.querySelectorAll('nav button')[${index}].click()`);
    await waitFor(`document.querySelectorAll('nav button')[${index}].getAttribute('aria-pressed')==='true'`);
    if(index<3){
      await evaluate("(()=>{const s=document.querySelectorAll('form select')[1];s.value='gaussian_blur';s.dispatchEvent(new Event('change',{bubbles:true}));})()");
    }
    const styleOutputs=[];
    for(const style of index===3?[1,2,3]:[null]){
      if(style)await evaluate(`(()=>{const s=document.querySelectorAll('form select')[1];s.value='${style}';s.dispatchEvent(new Event('change',{bubbles:true}));})()`);
      await waitFor("!document.querySelector('form button').disabled");
      await evaluate("document.querySelector('form button').click()");
      await waitFor("!document.querySelector('form button').disabled && (document.querySelector('figure img') || document.querySelector('[role=alert]').textContent)");
      const state=await evaluate("({error:document.querySelector('[role=alert]').textContent, title:document.querySelector('h1').textContent,figures:document.querySelectorAll('figure').length, output:document.querySelectorAll('figure img')[1]?.src,download:document.querySelector('a[download]')?.download,weights:document.querySelectorAll('progress').length,smoke:document.body.textContent.includes('Development model')})");
      assert(!state.error, state.error);
      assert(state.output?.startsWith('data:image/png;base64,'),'Missing image');
      assert(!state.smoke,'Unexpected smoke model warning');
      assert(state.download===(index===3?'sketch.png':'restored.png'),'Download missing');
      assert(state.figures===(index===3?2:4),'Reference panels incorrect');
      assert(state.weights===(index===1||index===2?4:0),'Routing weights missing');
      styleOutputs.push(state.output);
      checks.push({workspace:index,style,title:state.title,figures:state.figures,weights:state.weights,passed:true});
    }
    if(index===3)assert(new Set(styleOutputs).size===3,'Style selector returned identical outputs');
    for(const [label,width,height,mobile] of [['desktop',1365,1000,false],['mobile',390,844,true]]){
      await send('Emulation.setDeviceMetricsOverride',{width,height,deviceScaleFactor:1,mobile});
      assert(await evaluate('document.documentElement.scrollWidth <= window.innerWidth'),'Horizontal overflow');
      const screenshot=await send('Page.captureScreenshot',{format:'png',captureBeyondViewport:true});
      fs.writeFileSync(`${output}/workspace_${index+1}_${label}.png`,Buffer.from(screenshot.data,'base64'));
    }
    await send('Emulation.setDeviceMetricsOverride',{width:1365,height:1000,deviceScaleFactor:1,mobile:false});
  }
  assert(errors.length===0,'Browser exceptions: '+errors.join(', '));
  fs.writeFileSync(`${output}/results.json`,JSON.stringify({passed:true,checks,browserExceptions:errors,viewports:[1365,390]},null,2));
  console.log('Production browser checks passed: all four workspaces, all three sketch styles, desktop/mobile screenshots.');
}finally{ws.close();}
