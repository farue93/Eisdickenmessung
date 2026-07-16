# -*- coding: utf-8 -*-
"""
vergleich4.py - EIN zoombarer Viewer mit ALLEN 4 Messungen auf Serie 174444:
Frame-Diff (gruen), Canny (cyan), U-Net (rot), SAM (magenta).
Film mit 4 Eislinien + Plot Dicke ueber Bogenlaenge s; Zoom+Pan in Bild UND Plot.
Voraussetzung: fd/cn/sam/un data.json vorhanden (un = nach train.py + apply_unet.py).
"""
import os, sys, json, glob, shutil

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
import pfade                                   # relative Repo-Aufloesung
SRC = {
    "fd":  pfade.framediff_json(),
    "cn":  pfade.canny_json(),
    "un":  os.path.join(pfade.REPO, "methoden", "unet", "data.json"),
    "sam": os.path.join(pfade.REPO, "methoden", "sam", "data.json"),
    "hed": pfade.hed_json(),
}
FRAMES = pfade.frames_dir()
OUT    = os.path.join(BASE, "vergleich"); os.makedirs(os.path.join(OUT, "frames"), exist_ok=True)

for k, p in SRC.items():
    if not os.path.exists(p):
        raise SystemExit(f"[!] fehlt: {p}\n    (un -> erst train.py + apply_unet.py laufen lassen)")
D = {k: json.load(open(p)) for k, p in SRC.items()}
n = min(len(D[k]["frames"]) for k in D)
for i in range(n):
    shutil.copy(sorted(glob.glob(os.path.join(FRAMES, "*.png")))[i], os.path.join(OUT, "frames", f"{i:04d}.png"))

combo = {"crop_w": D["fd"]["crop_w"], "crop_h": D["fd"]["crop_h"], "px_per_mm": D["fd"].get("px_per_mm", 13.9),
         "s": {k: D[k]["s"] for k in D},
         "frames": [dict({"file": f"frames/{i:04d}.png", "name": D["fd"]["frames"][i]["name"]},
                         **{k: {"ix": D[k]["frames"][i]["ix"], "iy": D[k]["frames"][i]["iy"],
                                "dicke": D[k]["frames"][i]["dicke"]} for k in D})
                    for i in range(n)]}
HTML = r"""<!DOCTYPE html>
<html lang="de"><head><meta charset="utf-8"><title>4-Methoden-Vergleich - 174444</title>
<style>
 body{font-family:system-ui,Arial,sans-serif;background:#111;color:#eee;margin:0;padding:16px}
 h1{font-size:18px;margin:0 0 4px} .sub{color:#aaa;font-size:13px;margin:0 0 12px}
 .wrap{display:flex;gap:16px;flex-wrap:wrap}
 .panel{background:#1c1c1c;border:1px solid #333;border-radius:8px;padding:10px}
 canvas{background:#000;display:block;max-width:100%;cursor:crosshair}
 .ctrl{display:flex;align-items:center;gap:14px;margin-top:12px;flex-wrap:wrap}
 button{background:#FF8C00;border:0;color:#111;font-weight:600;padding:6px 12px;border-radius:6px;cursor:pointer}
 input[type=range]{width:260px} label{font-size:14px}
 .info{font-size:12px;color:#999;margin-top:6px} .val{color:#FF8C00;font-weight:600}
 .fd{color:#38d15b}.cn{color:#2ec7d6}.un{color:#ff5555}.sam{color:#e05cff}.hed{color:#ffb300} .hint{font-size:12px;color:#777}
</style></head><body>
<h1>4-Methoden-Vergleich &mdash; Serie 260402-174444</h1>
<p class="sub">Frame-Diff <span class="fd">&#9632;</span> Canny <span class="cn">&#9632;</span> U-Net <span class="un">&#9632;</span> SAM <span class="sam">&#9632;</span> HED <span class="hed">&#9632;</span>
 &nbsp;&mdash;&nbsp; <span class="hint">Mausrad = Zoom, ziehen = verschieben, Doppelklick = reset (Bild UND Plot)</span></p>
<div class="wrap">
 <div class="panel"><canvas id="film" width="740" height="560"></canvas>
  <div class="info">Frame <span id="fidx" class="val">0</span>/<span id="fmax"></span> <span id="fname"></span></div></div>
 <div class="panel"><canvas id="plot" width="600" height="560"></canvas>
  <div class="info">max Dicke: FD <span id="vfd" class="fd">0</span> CN <span id="vcn" class="cn">0</span> UN <span id="vun" class="un">0</span> SAM <span id="vsam" class="sam">0</span> HED <span id="vhed" class="hed">0</span> <span id="einh">px</span></div></div>
</div>
<div class="ctrl panel">
 <button id="play">&#9654; Play</button><input type="range" id="slider" min="0" value="0">
 <label><input type="checkbox" id="tfd" checked> FD</label>
 <label><input type="checkbox" id="tcn" checked> Canny</label>
 <label><input type="checkbox" id="tun" checked> U-Net</label>
 <label><input type="checkbox" id="tsam" checked> SAM</label>
 <label><input type="checkbox" id="thed" checked> HED</label>
 <label><input type="checkbox" id="smooth"> gl&auml;tten</label>
 <label><input type="checkbox" id="unit"> mm</label>
 <button id="rf">Bild reset</button><button id="rp">Plot reset</button>
 <label>Tempo <input type="range" id="speed" min="1" max="20" value="6" style="width:90px"></label>
</div>
<script>
/*DATA*/
const F=DATA.frames, SA=DATA.s, W=DATA.crop_w, H=DATA.crop_h, PPM=DATA.px_per_mm;
const KEYS=['fd','cn','un','sam','hed'], COL={fd:'#38d15b',cn:'#2ec7d6',un:'#ff3030',sam:'#e05cff',hed:'#ffb300'};
const film=document.getElementById('film'), fx=film.getContext('2d');
const plot=document.getElementById('plot'), px=plot.getContext('2d');
let cur=0, playing=false, timer=null;
const imgs=F.map(f=>{const im=new Image();im.src=f.file;return im;});
const on=k=>document.getElementById('t'+k).checked, MM=()=>document.getElementById('unit').checked, SM=()=>document.getElementById('smooth').checked;
document.getElementById('slider').max=F.length-1; document.getElementById('fmax').textContent=F.length-1;
let SMAX=1; KEYS.forEach(k=>SA[k].forEach(v=>{if(v!=null&&v>SMAX)SMAX=v;}));
let DMAX=1; F.forEach(f=>KEYS.forEach(k=>f[k].dicke.forEach(v=>{if(v!=null&&v>DMAX)DMAX=v;})));
function medsmooth(a,win){const n=a.length,out=a.slice(),h=win>>1;
 for(let i=0;i<n;i++){if(a[i]==null){out[i]=null;continue;}const b=[];
   for(let j=i-h;j<=i+h;j++)if(j>=0&&j<n&&a[j]!=null)b.push(a[j]); b.sort((p,q)=>p-q);out[i]=b[b.length>>1];}return out;}
const fbase=film.width/W; let fv={z:1,ox:0,oy:(film.height-H*fbase)/2};
function fReset(){fv={z:1,ox:0,oy:(film.height-H*fbase)/2};}
function s2c(ix,iy){return [fv.ox+ix*fbase*fv.z, fv.oy+iy*fbase*fv.z];}
function drawFilm(){
 fx.setTransform(1,0,0,1,0,0); fx.clearRect(0,0,film.width,film.height);
 const im=imgs[cur]; if(im.complete) fx.drawImage(im,fv.ox,fv.oy,W*fbase*fv.z,H*fbase*fv.z);
 KEYS.forEach(k=>{ if(!on(k))return; const f=F[cur];
   const IX=SM()?medsmooth(f[k].ix,9):f[k].ix, IY=SM()?medsmooth(f[k].iy,9):f[k].iy;
   fx.strokeStyle=COL[k]; fx.lineWidth=1.6; fx.beginPath(); let st=false;
   for(let i=0;i<IX.length;i++){const X=IX[i],Y=IY[i]; if(X==null||Y==null){st=false;continue;}
     const c=s2c(X,Y); if(!st){fx.moveTo(c[0],c[1]);st=true;}else fx.lineTo(c[0],c[1]);} fx.stroke(); });
 document.getElementById('fidx').textContent=cur; document.getElementById('fname').textContent=F[cur].name;
}
const mL=48,mB=32,mT=12,mR=12; let pv=null;
function pReset(){pv={sMin:0,sMax:SMAX,dMin:0,dMax:(MM()?DMAX/PPM:DMAX)*1.08};}
function pw(){return plot.width-mL-mR;} function ph(){return plot.height-mT-mB;}
function Xs(s){return mL+pw()*(1-(s-pv.sMin)/(pv.sMax-pv.sMin));}
function Yd(d){return mT+ph()*(1-(d-pv.dMin)/(pv.dMax-pv.dMin));}
function drawPlot(){
 if(!pv)pReset(); const w=plot.width,h=plot.height,mm=MM(); px.clearRect(0,0,w,h);
 px.strokeStyle='#444'; px.beginPath(); px.moveTo(mL,mT); px.lineTo(mL,h-mB); px.lineTo(w-mR,h-mB); px.stroke();
 px.fillStyle='#999'; px.font='11px sans-serif';
 px.fillText('Eisdicke ['+(mm?'mm':'px')+']',6,mT+8); px.fillText('Bogenlaenge s [px] (gespiegelt)',mL+4,h-8);
 px.textAlign='right';
 for(let g=0;g<=4;g++){const yy=mT+ph()*g/4,val=pv.dMin+(pv.dMax-pv.dMin)*(1-g/4);
   px.fillStyle='#777'; px.fillText(val.toFixed(mm?2:1),mL-6,yy+4);
   px.strokeStyle='#222'; px.beginPath();px.moveTo(mL,yy);px.lineTo(w-mR,yy);px.stroke();}
 px.textAlign='center';
 for(let g=0;g<=4;g++){const xx=mL+pw()*g/4,val=pv.sMin+(pv.sMax-pv.sMin)*(1-g/4);
   px.fillStyle='#777'; px.fillText(val.toFixed(0),xx,h-mB+14);}
 px.textAlign='left'; px.save(); px.beginPath(); px.rect(mL,mT,pw(),ph()); px.clip();
 const f=F[cur], vmax={fd:0,cn:0,un:0,sam:0,hed:0};
 KEYS.forEach(k=>{ if(!on(k))return; const S=SA[k], DK=SM()?medsmooth(f[k].dicke,9):f[k].dicke;
   px.strokeStyle=COL[k]; px.lineWidth=2; px.beginPath(); let st=false;
   for(let i=0;i<DK.length;i++){const v=DK[i],sv=S[i]; if(v==null||sv==null){st=false;continue;}
     if(v>vmax[k])vmax[k]=v; const dv=mm?v/PPM:v,X=Xs(sv),Y=Yd(dv);
     if(!st){px.moveTo(X,Y);st=true;}else px.lineTo(X,Y);} px.stroke(); });
 px.restore(); const uv=x=>(mm?x/PPM:x).toFixed(mm?2:1);
 KEYS.forEach(k=>document.getElementById('v'+k).textContent=uv(vmax[k]));
 document.getElementById('einh').textContent=mm?'mm':'px';
}
function render(){drawFilm();drawPlot();document.getElementById('slider').value=cur;}
function step(){cur=(cur+1)%F.length;render();}
film.addEventListener('wheel',e=>{e.preventDefault();
 const r=film.getBoundingClientRect(),mx=(e.clientX-r.left)*film.width/r.width,my=(e.clientY-r.top)*film.height/r.height;
 const f=e.deltaY<0?1.25:0.8,pz=fv.z; fv.z=Math.min(40,Math.max(0.2,fv.z*f));
 fv.ox=mx-((mx-fv.ox)/(fbase*pz))*fbase*fv.z; fv.oy=my-((my-fv.oy)/(fbase*pz))*fbase*fv.z; drawFilm();
},{passive:false});
let fdrag=null; film.addEventListener('mousedown',e=>{fdrag=[e.clientX,e.clientY];});
window.addEventListener('mousemove',e=>{ if(!fdrag)return; const r=film.getBoundingClientRect();
 fv.ox+=(e.clientX-fdrag[0])*film.width/r.width; fv.oy+=(e.clientY-fdrag[1])*film.height/r.height; fdrag=[e.clientX,e.clientY]; drawFilm();});
window.addEventListener('mouseup',()=>{fdrag=null;});
film.addEventListener('dblclick',()=>{fReset();drawFilm();});
plot.addEventListener('wheel',e=>{e.preventDefault(); if(!pv)pReset();
 const r=plot.getBoundingClientRect(),mx=(e.clientX-r.left)*plot.width/r.width,my=(e.clientY-r.top)*plot.height/r.height;
 const sAt=pv.sMin+(pv.sMax-pv.sMin)*(1-(mx-mL)/pw()), dAt=pv.dMin+(pv.dMax-pv.dMin)*(1-(my-mT)/ph()), f=e.deltaY<0?0.8:1.25;
 pv.sMin=sAt+(pv.sMin-sAt)*f; pv.sMax=sAt+(pv.sMax-sAt)*f; pv.dMin=dAt+(pv.dMin-dAt)*f; pv.dMax=dAt+(pv.dMax-dAt)*f; drawPlot();
},{passive:false});
let pdrag=null; plot.addEventListener('mousedown',e=>{pdrag=[e.clientX,e.clientY];});
window.addEventListener('mousemove',e=>{ if(!pdrag||!pv)return; const r=plot.getBoundingClientRect();
 const ds=(pv.sMax-pv.sMin)/pw()*((e.clientX-pdrag[0])*plot.width/r.width), dd=(pv.dMax-pv.dMin)/ph()*((e.clientY-pdrag[1])*plot.height/r.height);
 pv.sMin+=ds; pv.sMax+=ds; pv.dMin-=dd; pv.dMax-=dd; pdrag=[e.clientX,e.clientY]; drawPlot();});
window.addEventListener('mouseup',()=>{pdrag=null;});
plot.addEventListener('dblclick',()=>{pReset();drawPlot();});
document.getElementById('play').onclick=function(){playing=!playing; this.innerHTML=playing?'&#10073;&#10073; Pause':'&#9654; Play';
 if(playing){const fps=+document.getElementById('speed').value;timer=setInterval(step,1000/fps);}else clearInterval(timer);};
document.getElementById('speed').oninput=function(){if(playing){clearInterval(timer);timer=setInterval(step,1000/(+this.value));}};
document.getElementById('slider').oninput=function(){cur=+this.value;render();};
['tfd','tcn','tun','tsam','thed','smooth'].forEach(id=>document.getElementById(id).onchange=render);
document.getElementById('unit').onchange=()=>{pReset();render();};
document.getElementById('rf').onclick=()=>{fReset();drawFilm();}; document.getElementById('rp').onclick=()=>{pReset();drawPlot();};
let loaded=0; imgs.forEach(im=>im.onload=()=>{if(++loaded===imgs.length)render();}); render();
</script></body></html>"""

open(os.path.join(OUT, "viewer.html"), "w", encoding="utf-8").write(
    HTML.replace("/*DATA*/", "const DATA = " + json.dumps(combo) + ";"))
print(f"-> {OUT}/viewer.html ({n} Frames, {len(D)} Methoden)")
