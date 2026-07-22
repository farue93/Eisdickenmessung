# -*- coding: utf-8 -*-
"""
wachstumsrate_fd.py - interaktiver Frame-Diff-Viewer mit einstellbarer
MAX. WACHSTUMSRATE PRO SCHRITT (Rate-Limiter).

Regler Delta_max [px/Frame] begrenzt live, wie stark sich die Eisdicke von einem
Frame zum naechsten aendern darf: |d_t - d_(t-1)| <= Delta_max, pro Station, pro
Schritt (nicht kumulativ ueber den Versuch). Krasse Ausreisser (Reflex-Spikes)
werden so gekappt. Wirkung sofort in Bild (Eislinie) UND Plot (Dickenkurve).

Self-contained: liegt im frame-differencing-Ordner, findet die eigene Serie
(serie_*/data.json) und die Laser-Normalen (aus der Vorverarbeitung) selbst.
"""
import os, sys, json, glob
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def erste_serie():
    for d in sorted(glob.glob(os.path.join(HERE, "serie_*"))):
        if os.path.exists(os.path.join(d, "data.json")):
            return d
    raise SystemExit("Keine serie_*/data.json gefunden. Erst serie_eis.py laufen lassen.")


def finde_geruest(stem):
    """ROI/Laserlinie-npz suchen (lokal 'pre processing/Bild 0/output' oder Repo 'preprocessing/output')."""
    for c in ("../pre processing/Bild 0/output", "../preprocessing/output",
              "../../preprocessing/output", "../pre processing/output"):
        p = os.path.normpath(os.path.join(HERE, c))
        if os.path.exists(os.path.join(p, stem + "_roi.npz")):
            return p
    raise SystemExit("Vorverarbeitungs-Output (<stem>_roi.npz) nicht gefunden.")


SER = erste_serie()
FD = json.load(open(os.path.join(SER, "data.json")))
STEM = FD["frames"][0]["name"]
GER = finde_geruest(STEM)
L = np.load(os.path.join(GER, STEM + "_laserlinie.npz"))
R = np.load(os.path.join(GER, STEM + "_roi.npz"))
vz = float(R["aussen_vorzeichen"])
outx, outy = vz * L["nx"], vz * L["ny"]                     # Aussen-Normalen (volle Laenge)
sub = slice(None, None, 2)                                  # gleiche Ausduennung wie serie_eis
nx = [None if not np.isfinite(v) else round(float(v), 4) for v in outx[sub]]
ny = [None if not np.isfinite(v) else round(float(v), 4) for v in outy[sub]]
n = min(len(nx), len(FD["frames"][0]["ix"]))
serie_base = os.path.basename(SER)

DATA = {"crop_w": FD["crop_w"], "crop_h": FD["crop_h"], "px_per_mm": FD.get("px_per_mm", 13.9),
        "s": FD["s"][:n], "nx": nx[:n], "ny": ny[:n],
        "frames": [{"file": f"../{serie_base}/frames/{k:04d}.png", "name": FD["frames"][k]["name"],
                    "ix": FD["frames"][k]["ix"][:n], "iy": FD["frames"][k]["iy"][:n],
                    "dicke": FD["frames"][k]["dicke"][:n]} for k in range(len(FD["frames"]))]}

HTML = r"""<!DOCTYPE html>
<html lang="de"><head><meta charset="utf-8"><title>Frame-Diff - Wachstumsrate pro Schritt</title>
<style>
 body{font-family:system-ui,Arial,sans-serif;background:#111;color:#eee;margin:0;padding:16px}
 h1{font-size:18px;margin:0 0 4px}.sub{color:#aaa;font-size:13px;margin:0 0 12px}
 .wrap{display:flex;gap:16px;flex-wrap:wrap}.panel{background:#1c1c1c;border:1px solid #333;border-radius:8px;padding:10px}
 canvas{background:#000;display:block;max-width:100%}
 .ctrl{display:flex;align-items:center;gap:14px;margin-top:12px;flex-wrap:wrap}
 button{background:#FF8C00;border:0;color:#111;font-weight:600;padding:6px 12px;border-radius:6px;cursor:pointer}
 input[type=range]{width:220px}label{font-size:14px}.val{color:#FF8C00;font-weight:600}.info{font-size:12px;color:#999;margin-top:6px}
</style></head><body>
<h1>Frame-Differencing &mdash; max. Wachstumsrate <u>pro Schritt</u></h1>
<p class="sub">&Delta;max begrenzt die &Auml;nderung <b>von Frame zu Frame</b> je Station: |d<sub>t</sub>&minus;d<sub>t&minus;1</sub>| &le; &Delta;max. Rohkurve grau, gefiltert orange.</p>
<div class="wrap">
 <div class="panel"><canvas id="film" width="720" height="540"></canvas>
  <div class="info">Frame <span id="fidx" class="val">0</span>/<span id="fmax"></span> <span id="fname"></span></div></div>
 <div class="panel"><canvas id="plot" width="560" height="380"></canvas>
  <div class="info">max Dicke gefiltert: <span id="dmx" class="val">0</span> <span id="einh">px</span></div></div>
</div>
<div class="ctrl panel">
 <button id="play">&#9654; Play</button><input type="range" id="slider" min="0" value="0">
 <label>&Delta;max/Schritt <input type="range" id="dmax" min="0" max="4" step="0.1" value="1.0"><span id="dmv" class="val">1.0</span> px</label>
 <label><input type="checkbox" id="mono"> nur wachsen</label>
 <label><input type="checkbox" id="roh" checked> Rohkurve zeigen</label>
 <label><input type="checkbox" id="unit"> mm</label>
 <label>Tempo <input type="range" id="speed" min="1" max="20" value="6" style="width:90px"></label>
</div>
<script>
/*DATA*/
const F=DATA.frames, S=DATA.s, NX=DATA.nx, NY=DATA.ny, W=DATA.crop_w, H=DATA.crop_h, PPM=DATA.px_per_mm;
const film=document.getElementById('film'), fx=film.getContext('2d');
const plot=document.getElementById('plot'), px=plot.getContext('2d');
const imgs=F.map(f=>{const im=new Image();im.src=f.file;return im;});
let cur=0, playing=false, timer=null, FILT=[];
const scale=Math.min(1,720/Math.max(W,H)); film.width=Math.round(W*scale); film.height=Math.round(H*scale);
document.getElementById('slider').max=F.length-1; document.getElementById('fmax').textContent=F.length-1;
const el=id=>document.getElementById(id);
let SMAX=1; S.forEach(v=>{if(v!=null&&v>SMAX)SMAX=v;});
let DMAX=1; F.forEach(f=>f.dicke.forEach(v=>{if(v!=null&&v>DMAX)DMAX=v;}));
const MM=()=>el('unit').checked;

function buildFilt(){                                        // pro Station ueber die Zeit: Schritt begrenzen
 const dmax=+el('dmax').value, mono=el('mono').checked, T=F.length, N=F[0].dicke.length;
 const cols=[];
 for(let i=0;i<N;i++){ let prev=null; const c=new Array(T);
   for(let t=0;t<T;t++){ const z=F[t].dicke[i];
     if(z==null){c[t]=null;continue;}
     if(prev==null){c[t]=z; prev=z; continue;}              // erster gueltiger Wert = uebernehmen
     const lo=mono?prev:prev-dmax, hi=prev+dmax;            // erlaubtes Fenster um den Vorwert
     const v=Math.max(lo,Math.min(hi,z)); c[t]=v; prev=v; } // Schritt auf +-dmax kappen
   cols.push(c); }
 FILT=Array.from({length:T},(_,t)=>cols.map(c=>c[t]));
}
function drawFilm(){
 fx.clearRect(0,0,film.width,film.height); const im=imgs[cur];
 if(im.complete)fx.drawImage(im,0,0,film.width,film.height);
 const f=F[cur], FL=FILT[cur]; fx.strokeStyle='#FF3030'; fx.lineWidth=1.6; fx.beginPath(); let st=false;
 for(let i=0;i<f.ix.length;i++){ const raw=f.dicke[i], fl=FL[i];
   if(f.ix[i]==null||raw==null||fl==null||NX[i]==null){st=false;continue;}
   const d=fl-raw, X=(f.ix[i]+NX[i]*d)*scale, Y=(f.iy[i]+NY[i]*d)*scale;
   if(!st){fx.moveTo(X,Y);st=true;}else fx.lineTo(X,Y);}
 fx.stroke(); el('fidx').textContent=cur; el('fname').textContent=f.name;
}
function drawPlot(){
 const w=plot.width,h=plot.height,mL=46,mB=28,mT=10,mR=10,mm=MM();
 px.clearRect(0,0,w,h); px.strokeStyle='#444'; px.beginPath(); px.moveTo(mL,mT); px.lineTo(mL,h-mB); px.lineTo(w-mR,h-mB); px.stroke();
 const YMAX=(mm?DMAX/PPM:DMAX)*1.1||1;
 px.fillStyle='#999'; px.font='11px sans-serif'; px.fillText('Eisdicke ['+(mm?'mm':'px')+']',6,mT+8); px.fillText('Bogenlaenge s (gespiegelt)',mL+4,h-8);
 for(let g=0;g<=4;g++){const yy=mT+(h-mT-mB)*g/4,val=YMAX*(1-g/4); px.fillStyle='#777'; px.fillText(val.toFixed(mm?2:1),mL-34,yy+4);
   px.strokeStyle='#222'; px.beginPath();px.moveTo(mL,yy);px.lineTo(w-mR,yy);px.stroke();}
 const Xs=s=>mL+(w-mL-mR)*(1-s/SMAX), Yd=d=>mT+(h-mT-mB)*(1-(mm?d/PPM:d)/YMAX);
 function kurve(get,col,lw){ px.strokeStyle=col; px.lineWidth=lw; px.beginPath(); let st=false;
   for(let i=0;i<S.length;i++){const v=get(i),sv=S[i]; if(v==null||sv==null){st=false;continue;}
     const X=Xs(sv),Y=Yd(v); if(!st){px.moveTo(X,Y);st=true;}else px.lineTo(X,Y);} px.stroke(); }
 if(el('roh').checked) kurve(i=>F[cur].dicke[i],'#666',1);
 kurve(i=>FILT[cur][i],'#FF8C00',2.2);
 let dm=0; FILT[cur].forEach(v=>{if(v!=null&&v>dm)dm=v;});
 el('dmx').textContent=(mm?dm/PPM:dm).toFixed(mm?2:1); el('einh').textContent=mm?'mm':'px';
}
function render(){drawFilm();drawPlot();el('slider').value=cur;}
function refilter(){buildFilt();render();}
function step(){cur=(cur+1)%F.length;render();}
el('play').onclick=function(){playing=!playing; this.innerHTML=playing?'&#10073;&#10073; Pause':'&#9654; Play';
 if(playing){timer=setInterval(step,1000/(+el('speed').value));}else clearInterval(timer);};
el('speed').oninput=function(){if(playing){clearInterval(timer);timer=setInterval(step,1000/(+this.value));}};
el('slider').oninput=function(){cur=+this.value;render();};
el('dmax').oninput=function(){el('dmv').textContent=(+this.value).toFixed(1);refilter();};
el('mono').onchange=refilter; el('roh').onchange=render; el('unit').onchange=render;
let loaded=0; imgs.forEach(im=>im.onload=()=>{if(++loaded===imgs.length){buildFilt();render();}});
buildFilt(); render();
</script></body></html>"""

OUT = os.path.join(HERE, "wachstumsrate"); os.makedirs(OUT, exist_ok=True)
open(os.path.join(OUT, "viewer.html"), "w", encoding="utf-8").write(
    HTML.replace("/*DATA*/", "const DATA = " + json.dumps(DATA) + ";"))
print(f"-> {OUT}/viewer.html  (Serie {serie_base}, {len(DATA['frames'])} Frames, N={n})")
