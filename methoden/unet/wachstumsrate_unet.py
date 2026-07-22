# -*- coding: utf-8 -*-
"""
wachstumsrate_unet.py - U-Net-v2-Viewer, ZOOMBAR, mit aggressivem Entrauschen:
(A) raeumlicher Median ueber s + (B) Zeit-Median ueber Frames + (C) Rate-Limiter
(Delta_max pro Schritt). Zoom+Pan in Bild UND Plot (Mausrad/ziehen/Doppelklick).
"""
import os, sys, json, glob, shutil
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(HERE)), "gemeinsam"))
import pfade

UN = json.load(open(os.path.join(HERE, "data.json")))
G = pfade.geometrie()
sys.path.insert(0, pfade.FRAMEDIFF)
import serie_eis as se
import cv2
sub = slice(None, None, 2)
nx = [None if not np.isfinite(v) else round(float(v), 4) for v in G["outx"][sub]]
ny = [None if not np.isfinite(v) else round(float(v), 4) for v in G["outy"][sub]]
rx = [round(float(v), 2) for v in G["x"][sub]]
ry = [round(float(v), 2) for v in G["y"][sub]]
# Nulllinie d0 (Frame 0) + Luecken fuellen -> Referenz-Offset ueberall definiert
f0 = cv2.imread(sorted(glob.glob(os.path.join(pfade.frames_dir(), "*.png")))[0], 0).astype(np.float32)
d0a = se.detektiere(f0, G["x"], G["y"], G["outx"], G["outy"])[sub]
ok = np.isfinite(d0a)
if ok.sum() >= 2:
    d0a = np.interp(np.arange(len(d0a)), np.where(ok)[0], d0a[ok])
d0 = [round(float(v), 3) for v in d0a]
n = min(len(nx), len(UN["frames"][0]["ix"]))

OUT = os.path.join(HERE, "wachstumsrate"); os.makedirs(os.path.join(OUT, "frames"), exist_ok=True)
for k, fp in enumerate(sorted(glob.glob(os.path.join(pfade.frames_dir(), "*.png")))):
    shutil.copy(fp, os.path.join(OUT, "frames", f"{k:04d}.png"))

DATA = {"crop_w": UN["crop_w"], "crop_h": UN["crop_h"], "px_per_mm": UN.get("px_per_mm", 13.9),
        "s": UN["s"][:n], "nx": nx[:n], "ny": ny[:n], "rx": rx[:n], "ry": ry[:n], "d0": d0[:n],
        "frames": [{"file": f"frames/{k:04d}.png", "name": UN["frames"][k]["name"],
                    "ix": UN["frames"][k]["ix"][:n], "iy": UN["frames"][k]["iy"][:n],
                    "dicke": UN["frames"][k]["dicke"][:n]} for k in range(len(UN["frames"]))]}

HTML = r"""<!DOCTYPE html>
<html lang="de"><head><meta charset="utf-8"><title>U-Net v2 - Entrauschen (zoombar)</title>
<style>
 body{font-family:system-ui,Arial,sans-serif;background:#111;color:#eee;margin:0;padding:16px}
 h1{font-size:18px;margin:0 0 4px}.sub{color:#aaa;font-size:13px;margin:0 0 12px}.hint{color:#777;font-size:12px}
 .wrap{display:flex;gap:16px;flex-wrap:wrap}.panel{background:#1c1c1c;border:1px solid #333;border-radius:8px;padding:10px}
 canvas{background:#000;display:block;max-width:100%;cursor:crosshair}
 .ctrl{display:flex;align-items:center;gap:12px;margin-top:12px;flex-wrap:wrap}
 button{background:#FF8C00;border:0;color:#111;font-weight:600;padding:6px 12px;border-radius:6px;cursor:pointer}
 input[type=range]{width:150px}label{font-size:13px}.val{color:#FF8C00;font-weight:600}.info{font-size:12px;color:#999;margin-top:6px}
</style></head><body>
<h1>U-Net v2 &mdash; Entrauschen (Median s + Zeit-Median + &Delta;max/Schritt)</h1>
<p class="sub">Rohkurve grau, gefiltert orange. <span class="hint">Mausrad = Zoom auf Cursor, ziehen = verschieben, Doppelklick = reset (Bild &amp; Plot).</span></p>
<div class="wrap">
 <div class="panel"><canvas id="film" width="760" height="560"></canvas>
  <div class="info">Frame <span id="fidx" class="val">0</span>/<span id="fmax"></span> <span id="fname"></span></div></div>
 <div class="panel"><canvas id="plot" width="600" height="560"></canvas>
  <div class="info">max Dicke gefiltert: <span id="dmx" class="val">0</span> <span id="einh">px</span><span id="mrk" style="color:#5cc8ff"></span></div></div>
</div>
<div class="ctrl panel">
 <button id="play">&#9654; Play</button><input type="range" id="slider" min="0" value="0" style="width:220px">
 <label>&Delta;max <input type="range" id="dmax" min="0" max="4" step="0.1" value="0.8"><span id="dmv" class="val">0.8</span></label>
 <label>Gl&auml;tten s <input type="range" id="sm" min="0" max="25" step="1" value="6"><span id="smv" class="val">6</span></label>
 <label>Zeit-Med <input type="range" id="tm" min="0" max="6" step="1" value="2"><span id="tmv" class="val">2</span></label>
 <label>L&uuml;cken s <input type="range" id="lk" min="0" max="40" step="1" value="10"><span id="lkv" class="val">10</span></label>
 <label><input type="checkbox" id="mono"> nur wachsen</label>
 <label><input type="checkbox" id="rz"> Rate zuerst</label>
 <label><input type="checkbox" id="roh" checked> Roh</label>
 <label><input type="checkbox" id="unit"> mm</label>
 <button id="rf">Bild reset</button><button id="rp">Plot reset</button>
 <label>Tempo <input type="range" id="speed" min="1" max="20" value="6" style="width:80px"></label>
</div>
<script>
/*DATA*/
const F=DATA.frames, S=DATA.s, NX=DATA.nx, NY=DATA.ny, RX=DATA.rx, RY=DATA.ry, D0=DATA.d0, W=DATA.crop_w, H=DATA.crop_h, PPM=DATA.px_per_mm;
const film=document.getElementById('film'), fx=film.getContext('2d');
const plot=document.getElementById('plot'), px=plot.getContext('2d');
const imgs=F.map(f=>{const im=new Image();im.src=f.file;return im;});
let cur=0, playing=false, timer=null, FILT=[], mi=null;      // mi = markierte Station (synchron)
const el=id=>document.getElementById(id);
el('slider').max=F.length-1; el('fmax').textContent=F.length-1;
let SMAX=1; S.forEach(v=>{if(v!=null&&v>SMAX)SMAX=v;});
let DMAX=1; F.forEach(f=>f.dicke.forEach(v=>{if(v!=null&&v>DMAX)DMAX=v;}));
const MM=()=>el('unit').checked;

// ---- Filter (aggressiv kombinierbar) ----
function med(a){const b=a.filter(v=>v!=null).sort((x,y)=>x-y); return b.length?b[b.length>>1]:null;}
function spatMed(row,w){ if(w<=0)return row.slice(); const n=row.length,o=new Array(n);
 for(let i=0;i<n;i++){const buf=[]; for(let j=i-w;j<=i+w;j++)if(j>=0&&j<n)buf.push(row[j]); o[i]=med(buf);} return o;}
function bridge(row,maxgap){                                 // kleine Luecken (<=maxgap) linear ueberbruecken
 if(maxgap<=0)return row.slice(); const n=row.length,o=row.slice(); let i=0;
 while(i<n){ if(o[i]==null){ let j=i; while(j<n&&o[j]==null)j++;
   if(i>0&&j<n&&(j-i)<=maxgap){ const a=o[i-1],b=o[j]; for(let k=i;k<j;k++)o[k]=a+(b-a)*(k-i+1)/(j-i+1); }
   i=j; } else i++; } return o;
}
function buildFilt(){
 const dmax=+el('dmax').value, mono=el('mono').checked, ws=+el('sm').value, wt=+el('tm').value, lk=+el('lk').value, rz=el('rz').checked;
 const T=F.length, N=F[0].dicke.length;
 const spatMedAll=g=>g.map(r=>spatMed(r,ws));               // raeumlicher Median je Frame
 const tempMedAll=g=>{ if(wt<=0)return g.map(r=>r.slice()); const o=g.map(r=>r.slice());   // Zeit-Median je Station
   for(let i=0;i<N;i++)for(let t=0;t<T;t++){const buf=[]; for(let u=t-wt;u<=t+wt;u++)if(u>=0&&u<T)buf.push(g[u][i]); o[t][i]=med(buf);} return o; };
 const rateAll=g=>{ const o=g.map(r=>r.slice());            // Rate-Limiter je Station, pro Schritt
   for(let i=0;i<N;i++){ let prev=null; for(let t=0;t<T;t++){ const z=g[t][i];
     if(z==null){o[t][i]=null;continue;} if(prev==null){o[t][i]=z;prev=z;continue;}
     const lo=mono?prev:prev-dmax, hi=prev+dmax; o[t][i]=Math.max(lo,Math.min(hi,z)); prev=o[t][i]; } } return o; };
 let g=F.map(f=>bridge(f.dicke,lk));                        // (0) Luecken ueberbruecken
 if(rz){ g=rateAll(g); g=spatMedAll(g); g=tempMedAll(g); }  // "Rate zuerst"
 else  { g=spatMedAll(g); g=tempMedAll(g); g=rateAll(g); }  // Default: Mediane zuerst, Rate zuletzt
 FILT=g;
}

// ---- Film Zoom/Pan ----
const fbase=film.width/W; let fv={z:1,ox:0,oy:(film.height-H*fbase)/2};
function fReset(){fv={z:1,ox:0,oy:(film.height-H*fbase)/2};}
function drawFilm(){
 fx.setTransform(1,0,0,1,0,0); fx.clearRect(0,0,film.width,film.height);
 const im=imgs[cur]; if(im.complete)fx.drawImage(im,fv.ox,fv.oy,W*fbase*fv.z,H*fbase*fv.z);
 const FL=FILT[cur]; fx.strokeStyle='#FF3030'; fx.lineWidth=1.8; fx.beginPath(); let st=false;
 for(let i=0;i<FL.length;i++){ const fl=FL[i];               // Linie voll aus Geometrie (auch ueberbrueckt)
   if(fl==null||NX[i]==null||D0[i]==null){st=false;continue;}
   const dd=D0[i]+fl, sx=RX[i]+NX[i]*dd, sy=RY[i]+NY[i]*dd;
   const X=fv.ox+sx*fbase*fv.z, Y=fv.oy+sy*fbase*fv.z;
   if(!st){fx.moveTo(X,Y);st=true;}else fx.lineTo(X,Y);}
 fx.stroke();
 if(mi!=null){ const fl=FL[mi];                             // synchroner Marker im Bild
   if(fl!=null&&NX[mi]!=null&&D0[mi]!=null){ const dd=D0[mi]+fl;
     const X=fv.ox+(RX[mi]+NX[mi]*dd)*fbase*fv.z, Y=fv.oy+(RY[mi]+NY[mi]*dd)*fbase*fv.z;
     fx.strokeStyle='#5cc8ff'; fx.lineWidth=2; fx.beginPath(); fx.arc(X,Y,6,0,7); fx.stroke();
     fx.fillStyle='#5cc8ff'; fx.beginPath(); fx.arc(X,Y,2.5,0,7); fx.fill(); } }
 el('fidx').textContent=cur; el('fname').textContent=F[cur].name;
}
// ---- Plot Zoom/Pan ----
const mL=48,mB=32,mT=12,mR=12; let pv=null;
function pReset(){pv={sMin:0,sMax:SMAX,dMin:0,dMax:(MM()?DMAX/PPM:DMAX)*1.1};}
function pw(){return plot.width-mL-mR;} function ph(){return plot.height-mT-mB;}
function Xs(s){return mL+pw()*(1-(s-pv.sMin)/(pv.sMax-pv.sMin));}
function Yd(d){const u=MM()?d/PPM:d; return mT+ph()*(1-(u-pv.dMin)/(pv.dMax-pv.dMin));}
function drawPlot(){
 if(!pv)pReset(); const w=plot.width,h=plot.height,mm=MM();
 px.clearRect(0,0,w,h); px.strokeStyle='#444'; px.beginPath(); px.moveTo(mL,mT); px.lineTo(mL,h-mB); px.lineTo(w-mR,h-mB); px.stroke();
 px.fillStyle='#999'; px.font='11px sans-serif'; px.fillText('Eisdicke ['+(mm?'mm':'px')+']',6,mT+8); px.fillText('Bogenlaenge s (gespiegelt)',mL+4,h-8);
 px.textAlign='right';
 for(let g=0;g<=4;g++){const yy=mT+ph()*g/4,val=pv.dMin+(pv.dMax-pv.dMin)*(1-g/4);
   px.fillStyle='#777'; px.fillText(val.toFixed(mm?2:1),mL-6,yy+4); px.strokeStyle='#222'; px.beginPath();px.moveTo(mL,yy);px.lineTo(w-mR,yy);px.stroke();}
 px.textAlign='left'; px.save(); px.beginPath(); px.rect(mL,mT,pw(),ph()); px.clip();
 function kurve(get,col,lw){ px.strokeStyle=col; px.lineWidth=lw; px.beginPath(); let st=false;
   for(let i=0;i<S.length;i++){const v=get(i),sv=S[i]; if(v==null||sv==null){st=false;continue;}
     const X=Xs(sv),Y=Yd(v); if(!st){px.moveTo(X,Y);st=true;}else px.lineTo(X,Y);} px.stroke(); }
 if(el('roh').checked) kurve(i=>F[cur].dicke[i],'#666',1);
 kurve(i=>FILT[cur][i],'#FF8C00',2.4); px.restore();
 if(mi!=null&&S[mi]!=null){ const xm=Xs(S[mi]);            // synchroner Marker im Plot
   px.strokeStyle='#5cc8ff'; px.lineWidth=1; px.setLineDash([4,4]); px.beginPath(); px.moveTo(xm,mT); px.lineTo(xm,h-mB); px.stroke(); px.setLineDash([]);
   const fv2=FILT[cur][mi]; if(fv2!=null){ const ym=Yd(fv2); px.fillStyle='#5cc8ff'; px.beginPath(); px.arc(xm,ym,4,0,7); px.fill(); } }
 let dm=0; FILT[cur].forEach(v=>{if(v!=null&&v>dm)dm=v;});
 el('dmx').textContent=(mm?dm/PPM:dm).toFixed(mm?2:1); el('einh').textContent=mm?'mm':'px';
 el('mrk').textContent = (mi!=null&&S[mi]!=null&&FILT[cur][mi]!=null)
   ? '  |  s='+S[mi].toFixed(0)+' px, Dicke='+(mm?(FILT[cur][mi]/PPM).toFixed(2)+' mm':FILT[cur][mi].toFixed(1)+' px') : '';
}
function render(){drawFilm();drawPlot();el('slider').value=cur;}
function refilter(){buildFilt();render();}
function step(){cur=(cur+1)%F.length;render();}
// Film-Interaktion
film.addEventListener('wheel',e=>{e.preventDefault(); const r=film.getBoundingClientRect(),mx=(e.clientX-r.left)*film.width/r.width,my=(e.clientY-r.top)*film.height/r.height;
 const f=e.deltaY<0?1.25:0.8,pz=fv.z; fv.z=Math.min(40,Math.max(0.2,fv.z*f));
 fv.ox=mx-((mx-fv.ox)/(fbase*pz))*fbase*fv.z; fv.oy=my-((my-fv.oy)/(fbase*pz))*fbase*fv.z; drawFilm();},{passive:false});
let fdrag=null; film.addEventListener('mousedown',e=>{fdrag=[e.clientX,e.clientY];});
window.addEventListener('mousemove',e=>{ if(!fdrag)return; const r=film.getBoundingClientRect();
 fv.ox+=(e.clientX-fdrag[0])*film.width/r.width; fv.oy+=(e.clientY-fdrag[1])*film.height/r.height; fdrag=[e.clientX,e.clientY]; drawFilm();});
window.addEventListener('mouseup',()=>{fdrag=null;}); film.addEventListener('dblclick',()=>{fReset();drawFilm();});
// Plot-Interaktion
plot.addEventListener('wheel',e=>{e.preventDefault(); if(!pv)pReset(); const r=plot.getBoundingClientRect(),mx=(e.clientX-r.left)*plot.width/r.width,my=(e.clientY-r.top)*plot.height/r.height;
 const sAt=pv.sMin+(pv.sMax-pv.sMin)*(1-(mx-mL)/pw()), dAt=pv.dMin+(pv.dMax-pv.dMin)*(1-(my-mT)/ph()), f=e.deltaY<0?0.8:1.25;
 pv.sMin=sAt+(pv.sMin-sAt)*f; pv.sMax=sAt+(pv.sMax-sAt)*f; pv.dMin=dAt+(pv.dMin-dAt)*f; pv.dMax=dAt+(pv.dMax-dAt)*f; drawPlot();},{passive:false});
let pdrag=null; plot.addEventListener('mousedown',e=>{pdrag=[e.clientX,e.clientY];});
window.addEventListener('mousemove',e=>{ if(!pdrag||!pv)return; const r=plot.getBoundingClientRect();
 const ds=(pv.sMax-pv.sMin)/pw()*((e.clientX-pdrag[0])*plot.width/r.width), dd=(pv.dMax-pv.dMin)/ph()*((e.clientY-pdrag[1])*plot.height/r.height);
 pv.sMin+=ds; pv.sMax+=ds; pv.dMin-=dd; pv.dMax-=dd; pdrag=[e.clientX,e.clientY]; drawPlot();});
window.addEventListener('mouseup',()=>{pdrag=null;}); plot.addEventListener('dblclick',()=>{pReset();drawPlot();});
// synchroner Bogenlaengen-Marker (Hover in Plot ODER Bild -> Punkt in beiden)
function markAt(i){mi=i; drawFilm(); drawPlot();}
plot.addEventListener('mousemove',e=>{ if(pdrag||!pv)return; const r=plot.getBoundingClientRect();
 const mx=(e.clientX-r.left)*plot.width/r.width, sv=pv.sMin+(pv.sMax-pv.sMin)*(1-(mx-mL)/pw());
 let best=null,bd=1e18; for(let i=0;i<S.length;i++){if(S[i]==null)continue; const d=Math.abs(S[i]-sv); if(d<bd){bd=d;best=i;}} markAt(best);});
film.addEventListener('mousemove',e=>{ if(fdrag)return; const r=film.getBoundingClientRect();
 const mx=(e.clientX-r.left)*film.width/r.width, my=(e.clientY-r.top)*film.height/r.height;
 const sx=(mx-fv.ox)/(fbase*fv.z), sy=(my-fv.oy)/(fbase*fv.z);
 let best=null,bd=1e18; for(let i=0;i<RX.length;i++){if(RX[i]==null)continue; const dx=RX[i]-sx,dy=RY[i]-sy,d=dx*dx+dy*dy; if(d<bd){bd=d;best=i;}} markAt(best);});
// Steuerung
el('play').onclick=function(){playing=!playing; this.innerHTML=playing?'&#10073;&#10073; Pause':'&#9654; Play';
 if(playing){timer=setInterval(step,1000/(+el('speed').value));}else clearInterval(timer);};
el('speed').oninput=function(){if(playing){clearInterval(timer);timer=setInterval(step,1000/(+this.value));}};
el('slider').oninput=function(){cur=+this.value;render();};
el('dmax').oninput=function(){el('dmv').textContent=(+this.value).toFixed(1);refilter();};
el('sm').oninput=function(){el('smv').textContent=this.value;refilter();};
el('tm').oninput=function(){el('tmv').textContent=this.value;refilter();};
el('lk').oninput=function(){el('lkv').textContent=this.value;refilter();};
el('rz').onchange=refilter;
el('mono').onchange=refilter; el('roh').onchange=render;
el('unit').onchange=()=>{pReset();render();};
el('rf').onclick=()=>{fReset();drawFilm();}; el('rp').onclick=()=>{pReset();drawPlot();};
let loaded=0; imgs.forEach(im=>im.onload=()=>{if(++loaded===imgs.length){buildFilt();render();}});
buildFilt(); render();
</script></body></html>"""

open(os.path.join(OUT, "viewer.html"), "w", encoding="utf-8").write(
    HTML.replace("/*DATA*/", "const DATA = " + json.dumps(DATA) + ";"))
print(f"-> {OUT}/viewer.html  ({len(DATA['frames'])} Frames, N={n})")
