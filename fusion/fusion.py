# -*- coding: utf-8 -*-
"""
fusion.py - Methoden-Fusion (Ensemble): pro Station der MEDIAN aus
Frame-Diff, Canny, HED und U-Net. Der Median verwirft Einzelausreisser
automatisch -> robuster Konsens-Verlauf, unabhaengig von einer Einzelmethode.
(SAM ist raus - es misst hier nichts Sinnvolles.)

Die vier Methoden haben teils EIGENE Referenzlinien (Canny/HED) mit anderer
Bogenlaengen-Achse. Deshalb werden alle Kurven ueber die normierte Bogenlaenge
auf das Frame-Diff-Gitter (= Laserlinie) RESAMPLED und dort verglichen.
"""
import os, sys, json, glob, shutil
import numpy as np
import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "gemeinsam"))
import pfade
import serie_eis as se                                     # pfade legt 'frame differencing' in den Pfad

# ---- Methoden-Ergebnisse laden ----
fd = json.load(open(pfade.framediff_json()))
cn = json.load(open(pfade.canny_json()))
hed = json.load(open(pfade.hed_json()))
un = json.load(open(os.path.join(pfade.REPO, "methoden", "unet", "data.json")))
G = pfade.geometrie()
sub = slice(None, None, 2)                                 # gleiche Ausduennung wie serie_eis


def arr(a):
    return np.array([np.nan if v is None else float(v) for v in a], float)


def resample(ss, dd, ts):
    """dd(ss) ueber NORMIERTE Bogenlaenge auf das Zielgitter ts abtasten."""
    ss, dd = arr(ss), arr(dd)
    m = np.isfinite(ss) & np.isfinite(dd)
    if m.sum() < 2:
        return np.full(len(ts), np.nan)
    su = ss[m] / np.nanmax(ss[m]); tu = arr(ts) / np.nanmax(arr(ts))
    return np.interp(tu, su, dd[m])                        # ausserhalb: Randwert


# ---- gemeinsames Gitter = Frame-Diff (Laserlinie) ----
fd_s = fd["s"]
n = min(len(fd_s), len(G["x"][sub]))
rx = [round(float(v), 2) for v in G["x"][sub][:n]]
ry = [round(float(v), 2) for v in G["y"][sub][:n]]
nx = [round(float(v), 4) for v in G["outx"][sub][:n]]
ny = [round(float(v), 4) for v in G["outy"][sub][:n]]

# Nulllinie d0 (Frame 0) einmal detektieren -> volle Linien-Rekonstruktion moeglich
f0 = cv2.imread(sorted(glob.glob(os.path.join(pfade.frames_dir(), "*.png")))[0], 0).astype(np.float32)
d0 = se.detektiere(f0, G["x"], G["y"], G["outx"], G["outy"])[sub][:n]
d0 = [None if not np.isfinite(v) else round(float(v), 3) for v in d0]


def nl(a):
    return [None if not np.isfinite(v) else round(float(v), 3) for v in a]


frames = []
for k in range(len(fd["frames"])):
    fd_d = arr(fd["frames"][k]["dicke"])[:n]
    un_d = arr(un["frames"][k]["dicke"])
    un_on = un_d[:n] if len(un_d) == len(fd_d) else resample(un["s"], un["frames"][k]["dicke"], fd_s[:n])
    cn_on = resample(cn["s"], cn["frames"][k]["dicke"], fd_s[:n])
    hed_on = resample(hed["s"], hed["frames"][k]["dicke"], fd_s[:n])
    stack = np.vstack([fd_d, un_on, cn_on, hed_on])        # (4, N) auf gemeinsamem Gitter
    cnt = np.sum(np.isfinite(stack), axis=0)
    fused = np.nanmedian(stack, axis=0)
    fused[cnt < 2] = np.nan                                # Median nur wenn >=2 Methoden da
    frames.append({"name": fd["frames"][k]["name"], "fused": nl(fused),
                   "fd": nl(fd_d), "cn": nl(cn_on), "hed": nl(hed_on), "un": nl(un_on)})

# ---- Frames kopieren (self-contained) ----
OUT = HERE; os.makedirs(os.path.join(OUT, "frames"), exist_ok=True)
srcf = sorted(glob.glob(os.path.join(pfade.frames_dir(), "*.png")))
for k in range(len(frames)):
    shutil.copy(srcf[k], os.path.join(OUT, "frames", f"{k:04d}.png"))
for k in range(len(frames)):
    frames[k]["file"] = f"frames/{k:04d}.png"

DATA = {"crop_w": fd["crop_w"], "crop_h": fd["crop_h"], "px_per_mm": fd.get("px_per_mm", 13.9),
        "s": fd_s[:n], "rx": rx, "ry": ry, "nx": nx, "ny": ny, "d0": d0, "frames": frames}

HTML = r"""<!DOCTYPE html>
<html lang="de"><head><meta charset="utf-8"><title>Methoden-Fusion (Median)</title>
<style>
 body{font-family:system-ui,Arial,sans-serif;background:#111;color:#eee;margin:0;padding:16px}
 h1{font-size:18px;margin:0 0 4px}.sub{color:#aaa;font-size:13px;margin:0 0 12px}
 .wrap{display:flex;gap:16px;flex-wrap:wrap}.panel{background:#1c1c1c;border:1px solid #333;border-radius:8px;padding:10px}
 canvas{background:#000;display:block;max-width:100%}
 .ctrl{display:flex;align-items:center;gap:14px;margin-top:12px;flex-wrap:wrap}
 button{background:#FF8C00;border:0;color:#111;font-weight:600;padding:6px 12px;border-radius:6px;cursor:pointer}
 input[type=range]{width:240px}label{font-size:14px}.val{color:#FF8C00;font-weight:600}.info{font-size:12px;color:#999;margin-top:6px}
 .fd{color:#38d15b}.cn{color:#2ec7d6}.hed{color:#ffb300}.un{color:#ff5555}.fus{color:#FF8C00}
</style></head><body>
<h1>Methoden-Fusion &mdash; Median aus 4 Methoden</h1>
<p class="sub">Pro Station Median aus Frame-Diff <span class="fd">&#9632;</span> Canny <span class="cn">&#9632;</span> HED <span class="hed">&#9632;</span> U-Net <span class="un">&#9632;</span> &rarr; Konsens <span class="fus">&#9632; fett</span>. Einzelausrei&szlig;er fallen raus.</p>
<div class="wrap">
 <div class="panel"><canvas id="film" width="720" height="540"></canvas>
  <div class="info">Frame <span id="fidx" class="val">0</span>/<span id="fmax"></span> <span id="fname"></span></div></div>
 <div class="panel"><canvas id="plot" width="560" height="380"></canvas>
  <div class="info">max Fusion: <span id="dmx" class="val">0</span> <span id="einh">px</span></div></div>
</div>
<div class="ctrl panel">
 <button id="play">&#9654; Play</button><input type="range" id="slider" min="0" value="0">
 <label><input type="checkbox" id="einzel" checked> Einzelmethoden zeigen</label>
 <label><input type="checkbox" id="unit"> mm</label>
 <label>Tempo <input type="range" id="speed" min="1" max="20" value="6" style="width:90px"></label>
</div>
<script>
/*DATA*/
const F=DATA.frames, S=DATA.s, RX=DATA.rx, RY=DATA.ry, NX=DATA.nx, NY=DATA.ny, D0=DATA.d0;
const W=DATA.crop_w, H=DATA.crop_h, PPM=DATA.px_per_mm;
const film=document.getElementById('film'), fx=film.getContext('2d');
const plot=document.getElementById('plot'), px=plot.getContext('2d');
const imgs=F.map(f=>{const im=new Image();im.src=f.file;return im;});
let cur=0, playing=false, timer=null;
const scale=Math.min(1,720/Math.max(W,H)); film.width=Math.round(W*scale); film.height=Math.round(H*scale);
const el=id=>document.getElementById(id);
el('slider').max=F.length-1; el('fmax').textContent=F.length-1;
let SMAX=1; S.forEach(v=>{if(v!=null&&v>SMAX)SMAX=v;});
let DMAX=1; F.forEach(f=>f.fused.forEach(v=>{if(v!=null&&v>DMAX)DMAX=v;}));
const MM=()=>el('unit').checked;
const COL={fd:'#38d15b',cn:'#2ec7d6',hed:'#ffb300',un:'#ff5555'};

function drawFilm(){
 fx.clearRect(0,0,film.width,film.height); const im=imgs[cur];
 if(im.complete)fx.drawImage(im,0,0,film.width,film.height);
 const f=F[cur]; fx.strokeStyle='#FF8C00'; fx.lineWidth=2; fx.beginPath(); let st=false;
 for(let i=0;i<f.fused.length;i++){ const fu=f.fused[i], d0=D0[i];
   if(fu==null||d0==null||NX[i]==null){st=false;continue;}
   const d=d0+fu, X=(RX[i]+NX[i]*d)*scale, Y=(RY[i]+NY[i]*d)*scale;
   if(!st){fx.moveTo(X,Y);st=true;}else fx.lineTo(X,Y);}
 fx.stroke(); el('fidx').textContent=cur; el('fname').textContent=f.name;
}
function drawPlot(){
 const w=plot.width,h=plot.height,mL=46,mB=28,mT=10,mR=10,mm=MM();
 px.clearRect(0,0,w,h); px.strokeStyle='#444'; px.beginPath(); px.moveTo(mL,mT); px.lineTo(mL,h-mB); px.lineTo(w-mR,h-mB); px.stroke();
 const YMAX=(mm?DMAX/PPM:DMAX)*1.15||1;
 px.fillStyle='#999'; px.font='11px sans-serif'; px.fillText('Eisdicke ['+(mm?'mm':'px')+']',6,mT+8); px.fillText('Bogenlaenge s (gespiegelt)',mL+4,h-8);
 for(let g=0;g<=4;g++){const yy=mT+(h-mT-mB)*g/4,val=YMAX*(1-g/4); px.fillStyle='#777'; px.fillText(val.toFixed(mm?2:1),mL-34,yy+4);
   px.strokeStyle='#222'; px.beginPath();px.moveTo(mL,yy);px.lineTo(w-mR,yy);px.stroke();}
 const Xs=s=>mL+(w-mL-mR)*(1-s/SMAX), Yd=d=>mT+(h-mT-mB)*(1-(mm?d/PPM:d)/YMAX);
 function kurve(a,col,lw){ px.strokeStyle=col; px.lineWidth=lw; px.beginPath(); let st=false;
   for(let i=0;i<S.length;i++){const v=a[i],sv=S[i]; if(v==null||sv==null){st=false;continue;}
     const X=Xs(sv),Y=Yd(v); if(!st){px.moveTo(X,Y);st=true;}else px.lineTo(X,Y);} px.stroke(); }
 const f=F[cur];
 if(el('einzel').checked){ ['fd','cn','hed','un'].forEach(k=>kurve(f[k],COL[k],1)); }
 kurve(f.fused,'#FF8C00',2.6);                             // Fusion fett
 let dm=0; f.fused.forEach(v=>{if(v!=null&&v>dm)dm=v;});
 el('dmx').textContent=(mm?dm/PPM:dm).toFixed(mm?2:1); el('einh').textContent=mm?'mm':'px';
}
function render(){drawFilm();drawPlot();el('slider').value=cur;}
function step(){cur=(cur+1)%F.length;render();}
el('play').onclick=function(){playing=!playing; this.innerHTML=playing?'&#10073;&#10073; Pause':'&#9654; Play';
 if(playing){timer=setInterval(step,1000/(+el('speed').value));}else clearInterval(timer);};
el('speed').oninput=function(){if(playing){clearInterval(timer);timer=setInterval(step,1000/(+this.value));}};
el('slider').oninput=function(){cur=+this.value;render();};
el('einzel').onchange=render; el('unit').onchange=render;
let loaded=0; imgs.forEach(im=>im.onload=()=>{if(++loaded===imgs.length)render();});
render();
</script></body></html>"""
open(os.path.join(OUT, "viewer.html"), "w", encoding="utf-8").write(
    HTML.replace("/*DATA*/", "const DATA = " + json.dumps(DATA) + ";"))
print(f"-> {os.path.join(OUT, 'viewer.html')}  ({len(frames)} Frames, N={n})")

# ---- Endwert-Vergleich (Konsens vs. Einzelmethoden) ----
def endmed(key):
    dk = arr(frames[-1][key]); e = dk[np.isfinite(dk) & (dk > 1.0)]
    return float(np.median(e)) if e.size else 0.0
print("Endwerte px:  Fusion %.1f | FD %.1f | Canny %.1f | HED %.1f | U-Net %.1f"
      % (endmed("fused"), endmed("fd"), endmed("cn"), endmed("hed"), endmed("un")))
