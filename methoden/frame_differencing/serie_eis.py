# -*- coding: utf-8 -*-
"""
serie_eis.py - Komplette Eisreihe auswerten + HTML-Viewer.

Ablauf:
  1. Bild-0-Gerüst laden (Laserlinie x,y,s,nx,ny + ROI-Band/Bbox + Aussenrichtung).
  2. Alle Frames der Serie:
       - mit dem Bild-0-Band croppen (alles ausserhalb schwarz, auf Bbox),
       - Eislinie entlang der Laser-Normalen detektieren (Versatz d_N(s),
         Halbwerts-Schwerpunkt der hellen Laser-Stelle).
  3. Nulllinie = d_0 (Frame 0). Eisdicke(s,N) = d_N(s) - d_0(s).
  4. cropped PNGs + data (in viewer.html eingebettet) -> Film + Dickenkurve,
     Eislinie umschaltbar.

Geometrie (Bogenlänge s, Normalen) stammt IMMER von der festen Laserlinie.
Liest nur; schreibt nur nach OUT_DIR.
"""
import os, glob, json, argparse
import numpy as np
import cv2

# ---- Pfade (CLI-überschreibbar; Defaults relativ zum Repo -> push-tauglich) ----
HERE        = os.path.dirname(os.path.abspath(__file__))
# Vorverarbeitungs-Output liegt im Schwesterordner "pre processing/output":
GERUEST_DIR = os.path.normpath(os.path.join(HERE, "..", "..", "preprocessing", "output"))
def _erste_serie(here):
    _inp = os.path.normpath(os.path.join(here, "..", "..", "input"))          # Serien-Ordner (../input)
    for d in sorted(glob.glob(os.path.join(_inp, "*"))):
        if os.path.isdir(d) and glob.glob(os.path.join(d, "*.tif")):
            return d                                                     # erste Serie mit *.tif
    return _inp
SERIE_DIR   = _erste_serie(HERE)   # erste Bilderserie in ../input (dorthin die Serien ablegen)
STEM        = "2026-04-02_17-51-15-328_image0000000"      # Frame 0 (Default: erstes *.tif der Serie)
OUT_DIR     = "serie_260402-174444"                       # Ausgabeordner (Bilder + Viewer)
FRAMES_DIR  = os.path.join(OUT_DIR, "frames")

# ---- Detektion ----
SUCH_INNEN   = 25     # px Suchweite nach innen
SUCH_AUSSEN  = 60     # px Suchweite nach aussen (Eis ~<=30px; eng gegen ferne Reflexe)
MIN_KONTRAST = 40     # min. Peak-über-Hintergrund, sonst kein Laser hier
GLATT_S      = 9      # Median-Glättung der Dicke über s
EXPORT_STEP  = 2      # nur jeden n-ten Punkt exportieren (kleinere data)
PX_PER_MM    = 13.9   # CAD-kalibriert (Registrierung Laserlinie<->Slat-DXF)
MAX_FRAMES   = None   # None = alle; sonst Zahl (zum Testen)


def lade_grau(pfad):
    """Bild robust als float32-Graustufe laden (oder None bei Fehler)."""
    img = cv2.imread(pfad, cv2.IMREAD_GRAYSCALE)   # Standardweg: direkt als Graustufe laden
    if img is not None:
        return img.astype(np.float32)              # -> float (H,W), Werte 0..255
    try:
        import tifffile                            # Fallback für TIFs, die cv2 nicht öffnet
        a = tifffile.imread(pfad)
        if a.ndim == 3:                            # falls 3 Kanäle vorliegen ...
            a = a[..., :3].mean(2)                 # ... zu einem Graukanal mitteln
        return None if a.size == 0 else a.astype(np.float32)
    except Exception:
        return None                                # wirklich nicht ladbar


def bilinear(A, X, Y):
    """Intensität an nicht-ganzzahligen Positionen X,Y (Arrays) bilinear interpolieren."""
    H, W = A.shape                                 # Bildmaße
    x0 = np.floor(X).astype(np.int32); y0 = np.floor(Y).astype(np.int32)  # linker/oberer Nachbar
    x1 = x0 + 1; y1 = y0 + 1                        # rechter/unterer Nachbar
    ok = (x0 >= 0) & (y0 >= 0) & (x1 < W) & (y1 < H)  # liegt das 2x2-Fenster ganz im Bild?
    xc0 = np.clip(x0, 0, W-1); xc1 = np.clip(x1, 0, W-1)  # Indizes ins Bild klippen (kein Absturz am Rand)
    yc0 = np.clip(y0, 0, H-1); yc1 = np.clip(y1, 0, H-1)
    wx = X - x0; wy = Y - y0                        # Nachkomma-Anteile = Interpolationsgewichte (0..1)
    v = (A[yc0,xc0]*(1-wx)*(1-wy) + A[yc0,xc1]*wx*(1-wy)   # gewichtete Summe der 4 Nachbarpixel
         + A[yc1,xc0]*(1-wx)*wy + A[yc1,xc1]*wx*wy)
    return np.where(ok, v, 0.0)                    # außerhalb des Bildes -> 0


def detektiere(frame, x, y, outx, outy):
    """Versatz d(s) der hellen Laser-Stelle entlang der Aussennormale,
    als Halbwerts-Schwerpunkt um den hellsten Punkt. NaN = keine Linie."""
    t = np.arange(-SUCH_INNEN, SUCH_AUSSEN + 1, 1.0)  # Offsets entlang der Normale: [-25..60] (T=86)
    X = x[:, None] + outx[:, None] * t[None, :]    # (N,T) Abtast-x je Station & Offset
    Y = y[:, None] + outy[:, None] * t[None, :]    # (N,T) Abtast-y
    P = bilinear(frame, X, Y)                      # (N, T) Intensitätsprofile quer zum Laser
    innen = t < 0                                  # bool-Maske der inneren Offsets (t<0, laserabgewandt)
    d = np.full(len(x), np.nan)                    # Ergebnis (N,), zunächst überall NaN
    for i in range(P.shape[0]):                    # jede Station (Bogenlängen-Punkt) einzeln
        prof = P[i]                                # ein Profil (T,): dunkel .. hell(Laser) .. dunkel
        bg = np.median(prof[innen]) if innen.any() else float(prof.min())  # Hintergrund (innere Seite)
        pk = int(np.argmax(prof))                  # Index des hellsten Punkts im Profil
        if prof[pk] - bg < MIN_KONTRAST:           # Kontrast Peak-über-Hintergrund zu klein?
            continue                                # zu schwach -> kein Laser (d[i] bleibt NaN)
        schw = bg + 0.5 * (prof[pk] - bg)           # Halbwertsschwelle (halbe Höhe über bg)
        lo = pk                                    # vom Peak nach innen laufen ...
        while lo - 1 >= 0 and prof[lo-1] > schw:
            lo -= 1                                # ... solange noch über der Halbwertsschwelle
        hi = pk                                    # vom Peak nach außen laufen ...
        while hi + 1 < len(prof) and prof[hi+1] > schw:
            hi += 1                                # ... solange noch über der Schwelle
        w = prof[lo:hi+1] - bg                      # Gewichte = Helligkeit über Hintergrund (nur der Lauf)
        d[i] = float(np.sum(w * t[lo:hi+1]) / np.sum(w))  # hintergrundbereinigter Schwerpunkt = Offset [px]
    return d                                        # (N,) Laser-Offset je Station, NaN wo keiner


def median_glatt(a, k):
    """Median-Glättung von a über ein gleitendes Fenster der Breite k (NaN-tolerant)."""
    if k < 3:
        return a                                    # zu kleines Fenster -> unverändert
    r = k // 2; out = a.copy()                       # Halbfenster r; Kopie fürs Ergebnis
    for i in range(len(a)):                          # jeden Punkt neu setzen
        f = a[max(0, i-r):min(len(a), i+r+1)]        # Fenster [i-r .. i+r]
        f = f[~np.isnan(f)]                          # NaNs herausnehmen
        if f.size:
            out[i] = np.median(f)                    # robuster Median (dämpft Reflex-Ausreißer)
    return out


def nan_liste(a):
    """Array JSON-tauglich machen: NaN/None -> None, sonst auf 2 Stellen runden (kleinere Datei)."""
    return [None if (v is None or np.isnan(v)) else round(float(v), 2) for v in a]


def main():
    global SERIE_DIR, STEM, GERUEST_DIR, OUT_DIR, FRAMES_DIR
    ap = argparse.ArgumentParser(description="Frame Differencing (Serie): Eisdicke ueber eine Bilderserie.")
    ap.add_argument("serie", nargs="?", default=SERIE_DIR, help="Ordner mit der Bilderserie (*.tif)")
    ap.add_argument("--geruest", default=GERUEST_DIR, help="Vorverarbeitungs-Output (laserlinie/roi npz)")
    ap.add_argument("--stem", default=None, help="Frame-0-Stamm (Default: erstes *.tif der Serie)")
    ap.add_argument("--out", default=None, help="Ausgabeordner (Default: serie_<Ordner-Suffix>)")
    a = ap.parse_args()
    SERIE_DIR, GERUEST_DIR = a.serie, a.geruest
    _fr = sorted(glob.glob(os.path.join(SERIE_DIR, "*.tif")))
    STEM = a.stem or (os.path.splitext(os.path.basename(_fr[0]))[0] if _fr else STEM)
    OUT_DIR = a.out or ("serie_" + os.path.basename(os.path.normpath(SERIE_DIR)).split("_")[-1])
    FRAMES_DIR = os.path.join(OUT_DIR, "frames")
    os.makedirs(FRAMES_DIR, exist_ok=True)             # Ausgabeordner frames/ anlegen
    L = np.load(os.path.join(GERUEST_DIR, STEM + "_laserlinie.npz"))  # Laser-Gerüst (aus Frame 0)
    R = np.load(os.path.join(GERUEST_DIR, STEM + "_roi.npz"))         # ROI-Band + Bbox
    x, y, s = L["x"], L["y"], L["s"]                   # Referenzpunkte (N,) + Bogenlänge (N,)
    nx, ny = L["nx"], L["ny"]                          # Einheits-Normalen je Punkt
    vz = float(R["aussen_vorzeichen"]); outx, outy = vz*nx, vz*ny  # Normalen nach AUSSEN orientieren
    maske = R["maske"]; bbox = tuple(int(v) for v in R["bbox"])    # bool-Band (Vollbild) + Crop-Rechteck
    y0, y1, x0, x1 = bbox                              # Crop-Grenzen

    frames = sorted(glob.glob(os.path.join(SERIE_DIR, "*.tif")))
    if MAX_FRAMES:
        frames = frames[:MAX_FRAMES]
    print(f"{len(frames)} Frames, Crop {y1-y0}x{x1-x0}")

    d0 = None                                          # Nulllinie (wird in Frame 0 gesetzt)
    sub = slice(None, None, EXPORT_STEP)               # "jeder EXPORT_STEP-te Punkt" (Ausdünnung)
    s_exp = nan_liste(s[sub])                          # ausgedünnte s-Achse für die data.json
    frame_daten = []                                   # sammelt je Frame ein Dict

    for k, fp in enumerate(frames):
        img = lade_grau(fp)
        if img is None:
            print(f"  Frame {k}: defekt, übersprungen"); continue

        # --- Crop (maskiert, Bbox) ---
        masked = np.zeros_like(img)
        masked[maske] = img[maske]
        crop = masked[y0:y1, x0:x1].astype(np.uint8)
        png = os.path.join(FRAMES_DIR, f"{k:04d}.png")
        cv2.imwrite(png, crop)

        # --- Eislinie detektieren ---
        d = detektiere(img, x, y, outx, outy)
        if d0 is None:
            d0 = d.copy()                            # Nulllinie = Frame 0
        dicke = median_glatt(d - d0, GLATT_S)        # Eisdicke = d_N - d_0

        # --- Eislinie in Crop-Koordinaten (zum Zeichnen) ---
        gut = ~np.isnan(d)
        ix = np.where(gut, x + outx*np.nan_to_num(d) - x0, np.nan)
        iy = np.where(gut, y + outy*np.nan_to_num(d) - y0, np.nan)

        frame_daten.append({
            "file": f"frames/{k:04d}.png",
            "name": os.path.basename(fp).replace(".tif", ""),
            "ix": nan_liste(ix[sub]), "iy": nan_liste(iy[sub]),
            "dicke": nan_liste(dicke[sub]),
        })
        eis = np.isfinite(dicke) & (dicke > 1.0)     # echte Eisdicke (> 1px)
        gn = int(eis.sum())
        mx = np.nanmax(dicke) if np.isfinite(dicke).any() else float("nan")
        print(f"  Frame {k:2d}: Eis auf {gn/len(s)*100:4.0f}% der Bogenlänge  "
              f"max {mx:5.1f}px  median {np.nanmedian(dicke[eis]) if gn else 0:4.1f}px")

    daten = {                                          # alles, was der Viewer braucht:
        "bbox": [x0, y0, x1, y1],                      # Crop-Lage im Vollbild
        "crop_w": x1-x0, "crop_h": y1-y0,              # Crop-Größe (Canvas)
        "px_per_mm": PX_PER_MM,                        # Kalibrierung für die mm-Anzeige
        "s": s_exp,                                    # Bogenlängen-Achse
        "frames": frame_daten,                         # pro Frame: file, name, ix, iy, dicke
    }
    with open(os.path.join(OUT_DIR, "data.json"), "w") as f:
        json.dump(daten, f)
    schreibe_html(daten)
    print(f"-> {OUT_DIR}/viewer.html ({len(frame_daten)} Frames)")


def schreibe_html(daten):
    """Messdaten in das HTML-Template einbetten -> eigenständige viewer.html (server-los)."""
    js = "const DATA = " + json.dumps(daten) + ";"     # Daten als JS-Variable serialisieren
    html = HTML_TEMPLATE.replace("/*DATA*/", js)       # Platzhalter /*DATA*/ im Template ersetzen
    with open(os.path.join(OUT_DIR, "viewer.html"), "w", encoding="utf-8") as f:
        f.write(html)                                  # fertige HTML schreiben


HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="de"><head><meta charset="utf-8">
<title>Eisdickenmessung - Film</title>
<style>
 body{font-family:system-ui,Arial,sans-serif;background:#111;color:#eee;margin:0;padding:16px}
 h1{font-size:18px;margin:0 0 12px}
 .wrap{display:flex;gap:16px;flex-wrap:wrap}
 .panel{background:#1c1c1c;border:1px solid #333;border-radius:8px;padding:10px}
 canvas{background:#000;display:block;max-width:100%}
 .ctrl{display:flex;align-items:center;gap:12px;margin-top:10px;flex-wrap:wrap}
 button{background:#FF8C00;border:0;color:#111;font-weight:600;padding:6px 14px;border-radius:6px;cursor:pointer}
 input[type=range]{width:320px}
 label{font-size:14px}
 .info{font-size:13px;color:#aaa;margin-top:6px}
 .val{color:#FF8C00;font-weight:600}
</style></head><body>
<h1>Eisdickenmessung &mdash; Reihe 260402-174444</h1>
<div class="wrap">
 <div class="panel">
  <canvas id="film"></canvas>
  <div class="info">Frame <span id="fidx" class="val">0</span>/<span id="fmax"></span>
    &nbsp; <span id="fname"></span></div>
 </div>
 <div class="panel">
  <canvas id="plot" width="560" height="360"></canvas>
  <div class="info">max Eisdicke: <span id="dmax" class="val">0</span> px
    (<span id="dmaxmm" class="val">0</span> mm)</div>
 </div>
</div>
<div class="ctrl panel" style="margin-top:14px">
 <button id="play">&#9654; Play</button>
 <input type="range" id="slider" min="0" value="0">
 <label><input type="checkbox" id="toggle" checked> Eislinie einzeichnen</label>
 <label><input type="checkbox" id="smooth"> Ausrei&szlig;er gl&auml;tten</label>
 <label><input type="checkbox" id="unit"> in mm statt px</label>
 <label>Tempo <input type="range" id="speed" min="1" max="30" value="8" style="width:120px"></label>
</div>
<script>
/*DATA*/
const F=DATA.frames, S=DATA.s, W=DATA.crop_w, H=DATA.crop_h;
const film=document.getElementById('film'), fx=film.getContext('2d');
const plot=document.getElementById('plot'), px=plot.getContext('2d');
let cur=0, playing=false, timer=null;
// Anzeigegröße begrenzen
const scale=Math.min(1, 720/Math.max(W,H));
film.width=Math.round(W*scale); film.height=Math.round(H*scale);
document.getElementById('slider').max=F.length-1;
document.getElementById('fmax').textContent=F.length-1;
// Bilder vorladen
const imgs=F.map(f=>{const im=new Image();im.src=f.file;return im;});
// max Dicke über alle Frames (für feste Plot-Skala)
let DMAX=1;
F.forEach(f=>f.dicke.forEach(v=>{if(v!=null&&v>DMAX)DMAX=v;}));
const PPMg=DATA.ppm||null;              // ortsabhängiger Maßstab px/mm je Station
let DMAX_MM=0.1;
F.forEach(f=>f.dicke.forEach((v,i)=>{if(v!=null){const p=PPMg?PPMg[i]:DATA.px_per_mm;
  if(v/p>DMAX_MM)DMAX_MM=v/p;}}));
const MM=()=>document.getElementById('unit').checked;   // true = Anzeige in mm
let SMAX=1; S.forEach(v=>{if(v!=null&&v>SMAX)SMAX=v;});
// Robuste Median-Glättung (ignoriert null); entfernt Reflexband-Ausreißer
function medsmooth(a,win){
 const n=a.length, out=a.slice(), h=win>>1;
 for(let i=0;i<n;i++){ if(a[i]==null){out[i]=null;continue;}
   const buf=[]; for(let j=i-h;j<=i+h;j++){ if(j>=0&&j<n&&a[j]!=null)buf.push(a[j]); }
   buf.sort((p,q)=>p-q); out[i]=buf[buf.length>>1]; }
 return out;
}
const SM=()=>document.getElementById('smooth').checked;

function drawFilm(){
 const im=imgs[cur];
 fx.clearRect(0,0,film.width,film.height);
 if(im.complete) fx.drawImage(im,0,0,film.width,film.height);
 if(document.getElementById('toggle').checked){
   const f=F[cur]; fx.strokeStyle='#FF3030'; fx.lineWidth=1.5; fx.beginPath();
   const IX=SM()?medsmooth(f.ix,9):f.ix, IY=SM()?medsmooth(f.iy,9):f.iy;
   let started=false;
   for(let i=0;i<IX.length;i++){
     const X=IX[i],Y=IY[i];
     if(X==null||Y==null){started=false;continue;}
     const sx=X*scale, sy=Y*scale;
     if(!started){fx.moveTo(sx,sy);started=true;} else fx.lineTo(sx,sy);
   }
   fx.stroke();
 }
 document.getElementById('fidx').textContent=cur;
 document.getElementById('fname').textContent=F[cur].name;
}
function drawPlot(){
 const f=F[cur], w=plot.width, h=plot.height, mL=46,mB=28,mT=10,mR=10;
 px.clearRect(0,0,w,h);
 // Achsen
 px.strokeStyle='#444'; px.fillStyle='#999'; px.font='11px sans-serif';
 px.beginPath(); px.moveTo(mL,mT); px.lineTo(mL,h-mB); px.lineTo(w-mR,h-mB); px.stroke();
 const mm=MM(), YMAX=mm?DMAX_MM:DMAX, DEZ=mm?2:0;
 px.fillText('Eisdicke ['+(mm?'mm':'px')+']',6,mT+8);
 px.fillText('Bogenlänge s [px] (gespiegelt)',mL+4,h-8);
 // y-Gitter
 for(let g=0;g<=4;g++){const yy=mT+(h-mT-mB)*g/4; const val=YMAX*(1-g/4);
   px.fillStyle='#777'; px.fillText(val.toFixed(DEZ),mL-34,yy+4);
   px.strokeStyle='#222'; px.beginPath();px.moveTo(mL,yy);px.lineTo(w-mR,yy);px.stroke();}
 // Kurve
 px.strokeStyle='#FF8C00'; px.lineWidth=2; px.beginPath(); let st=false; let dm=0, dmm=0;
 const DK=SM()?medsmooth(f.dicke,9):f.dicke;
 const PPM=DATA.ppm||null;   // ortsabhängiger Maßstab px/mm je Station
 for(let i=0;i<DK.length;i++){
   const v=DK[i], sv=S[i];
   if(v==null||sv==null){st=false;continue;}
   if(v>dm)dm=v;
   const pmm=PPM?PPM[i]:DATA.px_per_mm; if(v/pmm>dmm)dmm=v/pmm;
   const pv=mm?v/pmm:v;                                          // Kurvenwert je Einheit
   const X=mL+(w-mL-mR)*(1-sv/SMAX), Y=mT+(h-mT-mB)*(1-pv/YMAX);  // x gespiegelt
   if(!st){px.moveTo(X,Y);st=true;}else px.lineTo(X,Y);
 }
 px.stroke();
 document.getElementById('dmax').textContent=dm.toFixed(1);
 document.getElementById('dmaxmm').textContent=dmm.toFixed(2);
}
function render(){drawFilm();drawPlot();document.getElementById('slider').value=cur;}
function step(){cur=(cur+1)%F.length;render();}
document.getElementById('play').onclick=function(){
 playing=!playing; this.innerHTML=playing?'&#10073;&#10073; Pause':'&#9654; Play';
 if(playing){const fps=+document.getElementById('speed').value;
   timer=setInterval(step,1000/fps);} else clearInterval(timer);
};
document.getElementById('speed').oninput=function(){
 if(playing){clearInterval(timer);timer=setInterval(step,1000/(+this.value));}
};
document.getElementById('slider').oninput=function(){cur=+this.value;render();};
document.getElementById('toggle').onchange=render;
document.getElementById('smooth').onchange=render;
document.getElementById('unit').onchange=render;
let loaded=0; imgs.forEach(im=>im.onload=()=>{if(++loaded===imgs.length)render();});
render();
</script></body></html>"""


if __name__ == "__main__":
    main()
