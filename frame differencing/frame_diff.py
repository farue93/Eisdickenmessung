# -*- coding: utf-8 -*-
"""
frame_diff.py - Eisdicke ueber Bogenlaenge per Frame Differencing.

Prinzip (fixes Setup, Bild 0 = eisfreies erstes Frame der Serie):
  1. D = I_frame - I_0  (vorzeichenbehaftet) -> loescht alles Statische,
     uebrig bleibt nur die neue, nach aussen verschobene Eis-Laserlinie.
  2. Variante 1: entlang jeder Bild-0-Normale (aussen) das D-Profil abtasten,
     aeussersten signifikanten positiven Peak intensitaetsgewichtet
     lokalisieren -> Eisdicke(s) als senkrechte Verschiebung ueber Bogenlaenge.

Liest (nur):
  - Serie: G:/Meine Ablage/Uni/VCXU.2-241M_700011810054_260402-174444/*.tif
  - Bild-0-Geruest: ../output/<stem>_laserlinie.npz  (x,y,s,nx,ny,bogenlaenge)
                    ../output/<stem>_roi.npz         (aussen_vorzeichen, maske, bbox)
Schreibt nur in ./output/.
"""
import os, glob, argparse
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---- Pfade (CLI-überschreibbar; Defaults relativ zum Repo -> push-tauglich) ----
HERE = os.path.dirname(os.path.abspath(__file__))
GERUEST_DIR = os.path.normpath(os.path.join(HERE, "..", "pre processing", "Bild 0", "output"))
def _erste_serie(here):
    _inp = os.path.normpath(os.path.join(here, "..", "input"))          # Serien-Ordner (../input)
    for d in sorted(glob.glob(os.path.join(_inp, "*"))):
        if os.path.isdir(d) and glob.glob(os.path.join(d, "*.tif")):
            return d
    return _inp
SERIE_DIR = _erste_serie(HERE)   # erste Bilderserie in ../input (dorthin die Serien ablegen)
STEM = "2026-04-02_17-51-15-328_image0000000"          # Frame 0 (Default: erstes *.tif der Serie)
OUT_DIR = "output"

# ---- Parameter ----
SUCH_AUSSEN = 90      # px Suchweite nach aussen (plausible Eisdicke dieser Phase)
SUCH_INNEN  = 25      # px Suchweite nach innen (Basislinie/Rauschschaetzung)
SCHRITT     = 1.0     # px Abtastschritt entlang der Normale
RAUSCH_K    = 5.0     # Schwelle = K * robustes Sigma von D im Band
MIN_RUN     = 3       # min. zusammenhaengende px ueber Schwelle (Eis-Echo)
GLATT_S     = 15      # Median-Fenster zur Glaettung der Dicke ueber s
ZIEL_FRAME  = -1      # Index des Zielframes (-1 = letztes)


def lade_grau(pfad):
    """Robust: erst cv2, dann tifffile-Fallback. None bei defektem Frame."""
    img = cv2.imread(pfad, cv2.IMREAD_GRAYSCALE)   # Standardweg: direkt als Graustufe laden
    if img is not None:
        return img.astype(np.float32)              # -> float (H,W), Werte 0..255
    try:
        import tifffile                            # Fallback für TIFs, die cv2 nicht öffnet
        a = tifffile.imread(pfad)
        if a.ndim == 3:                            # falls 3 Kanäle ...
            a = a[..., :3].mean(2)                 # ... zu einem Graukanal mitteln
        if a.size == 0:
            return None
        return a.astype(np.float32)
    except Exception:
        return None                                # wirklich nicht ladbar


def letztes_gueltiges(frames, ab_index):
    """Vom gewuenschten Index rueckwaerts das erste ladbare Frame."""
    n = len(frames)
    start = ab_index if ab_index >= 0 else n + ab_index  # -1 -> letztes Frame (n-1)
    for i in range(start, -1, -1):             # vom Ziel rückwärts suchen ...
        im = lade_grau(frames[i])
        if im is not None:                     # ... bis ein Frame ladbar ist
            return i, frames[i], im            # Index, Pfad, Bild zurückgeben
    raise RuntimeError("kein ladbares Zielframe gefunden")


def bilinear(D, X, Y):
    """Bilineare Abtastung von D an (X,Y) (beliebige Form), 0 ausserhalb."""
    H, W = D.shape                                 # Bildmaße
    x0 = np.floor(X).astype(np.int32); y0 = np.floor(Y).astype(np.int32)  # linker/oberer Nachbar
    x1 = x0 + 1; y1 = y0 + 1                        # rechter/unterer Nachbar
    ok = (x0 >= 0) & (y0 >= 0) & (x1 < W) & (y1 < H)  # liegt das 2x2-Fenster im Bild?
    xc0 = np.clip(x0, 0, W - 1); xc1 = np.clip(x1, 0, W - 1)  # Indizes klippen (Randschutz)
    yc0 = np.clip(y0, 0, H - 1); yc1 = np.clip(y1, 0, H - 1)
    wx = X - x0; wy = Y - y0                        # Nachkomma-Anteile = Interpolationsgewichte
    val = (D[yc0, xc0] * (1 - wx) * (1 - wy) + D[yc0, xc1] * wx * (1 - wy)   # gewichtete Summe
           + D[yc1, xc0] * (1 - wx) * wy + D[yc1, xc1] * wx * wy)            # der 4 Nachbarpixel
    return np.where(ok, val, 0.0)                  # außerhalb -> 0


def robustes_sigma(werte):
    """Robuste Streuungsschätzung über die MAD (ausreißerfest statt Standardabweichung)."""
    med = np.median(werte)                         # Median (robuster Mittelwert)
    mad = np.median(np.abs(werte - med))           # Median der Absolutabweichungen (MAD)
    return 1.4826 * mad + 1e-6                     # MAD -> sigma-Äquivalent (Normalverteilung)


def dicke_aus_profil(prof, t, schwelle):
    """Staerkster positiver Peak im Aussenbereich = Eis-Echo (Dipol-Rotlobe),
    lokalisiert per intensitaetsgew. Schwerpunkt des zusammenhaengenden Runs.
    prof, t: 1D entlang der Normale (innen<0 ... aussen>0).
    Rueckgabe: Dicke (px) oder NaN, wenn kein Eis."""
    aussen = t >= 0                          # nur der Außenbereich (dort wächst Eis)
    if not np.any((prof > schwelle) & aussen):
        return np.nan                        # nichts über der Schwelle -> kein Eis
    p = np.where(aussen, prof, -np.inf)      # inneren Bereich ausblenden (-inf)
    pk = int(np.argmax(p))                   # staerkster positiver Peak (aussen)
    if prof[pk] <= schwelle:
        return np.nan                        # selbst der Peak ist nur Rauschen
    # zusammenhaengenden Run um den Peak waehlen (solange > Schwelle)
    lo = pk
    while lo - 1 >= 0 and prof[lo - 1] > schwelle:
        lo -= 1                              # Run nach innen ausdehnen
    hi = pk
    while hi + 1 < len(prof) and prof[hi + 1] > schwelle:
        hi += 1                              # Run nach außen ausdehnen
    if hi - lo + 1 < MIN_RUN:
        return np.nan                        # zu kurzer Run -> Rauschspitze, verwerfen
    g = np.arange(lo, hi + 1)                # Indizes des Runs
    w = prof[g]                              # Gewichte = D-Werte im Run
    t_star = np.sum(w * t[g]) / np.sum(w)    # gewichteter Schwerpunkt = Offset [px]
    return max(t_star, 0.0)                  # negative Ausreißer auf 0 kappen


def messe(D, x, y, outx, outy):
    """Für jede Bogenlängen-Station die Eisdicke aus dem D-Profil bestimmen."""
    t = np.arange(-SUCH_INNEN, SUCH_AUSSEN + SCHRITT, SCHRITT)  # Offsets [-25..90]
    # Abtastkoordinaten: (N Punkte) x (T offsets)
    X = x[:, None] + outx[:, None] * t[None, :]   # (N,T) Abtast-x entlang der Normalen
    Y = y[:, None] + outy[:, None] * t[None, :]   # (N,T) Abtast-y
    P = bilinear(D, X, Y)                     # (N,T) D-Profile (Differenzbild entlang der Normalen)
    # Rauschschwelle aus dem inneren Bereich (dort kein Eis erwartet)
    innen = P[:, t < 0]                       # Profilteil mit t<0 = Basislinie
    sigma = robustes_sigma(innen.ravel())     # robuste Rauschstreuung von D
    schwelle = RAUSCH_K * sigma               # Signifikanzschwelle = K * sigma
    dicke = np.array([dicke_aus_profil(P[i], t, schwelle) for i in range(P.shape[0])])  # je Station
    return dicke, schwelle, sigma


def median_glatt(a, k):
    """Median-Glättung über ein gleitendes Fenster der Breite k (NaN-tolerant)."""
    if k < 3:
        return a                              # zu kleines Fenster -> unverändert
    n = len(a); r = k // 2; out = a.copy()     # Halbfenster r; Ergebnis-Kopie
    for i in range(n):                         # jeden Punkt neu setzen
        lo = max(0, i - r); hi = min(n, i + r + 1)  # Fenstergrenzen
        fenster = a[lo:hi]
        fenster = fenster[~np.isnan(fenster)]  # NaNs herausnehmen
        if fenster.size:
            out[i] = np.median(fenster)        # robuster Median (dämpft Ausreißer)
    return out


def diff_heatmap(D, bbox, schwelle):
    """Signiertes Differenzbild faerben (rot=neu hell, blau=verloren)."""
    y0, y1, x0, x1 = bbox
    crop = D[y0:y1, x0:x1]                    # Differenzbild auf das ROI-Band zuschneiden
    sc = max(schwelle * 4, 1.0)              # Farbskala (Sättigung bei ~4*Schwelle)
    pos = np.clip(crop / sc, 0, 1)          # positive Differenz (neu hell) -> 0..1
    neg = np.clip(-crop / sc, 0, 1)         # negative Differenz (verschwunden) -> 0..1
    img = np.zeros((*crop.shape, 3), np.uint8)   # leeres BGR-Bild
    img[..., 2] = (pos * 255).astype(np.uint8)   # R = neu hell (Eis-Echo = Rotlobe des Dipols)
    img[..., 0] = (neg * 255).astype(np.uint8)   # B = verschwunden (Gegenlobe an der alten Linie)
    return img


def main():
    global SERIE_DIR, STEM, GERUEST_DIR, OUT_DIR, ZIEL_FRAME
    ap = argparse.ArgumentParser(description="Frame Differencing (Dipol D=I_N-I_0) auf einer Bilderserie.")
    ap.add_argument("serie", nargs="?", default=SERIE_DIR, help="Ordner mit der Bilderserie (*.tif)")
    ap.add_argument("--geruest", default=GERUEST_DIR, help="Vorverarbeitungs-Output (laserlinie/roi npz)")
    ap.add_argument("--stem", default=None, help="Frame-0-Stamm (Default: erstes *.tif der Serie)")
    ap.add_argument("--out", default=None, help="Ausgabeordner (Default: output)")
    ap.add_argument("--ziel", type=int, default=ZIEL_FRAME, help="Ziel-Frame-Index (-1 = letztes)")
    a = ap.parse_args()
    SERIE_DIR, GERUEST_DIR, ZIEL_FRAME = a.serie, a.geruest, a.ziel
    _fr = sorted(glob.glob(os.path.join(SERIE_DIR, "*.tif")))
    STEM = a.stem or (os.path.splitext(os.path.basename(_fr[0]))[0] if _fr else STEM)
    OUT_DIR = a.out or OUT_DIR
    os.makedirs(OUT_DIR, exist_ok=True)

    # ---- Geruest laden ----
    L = np.load(os.path.join(GERUEST_DIR, STEM + "_laserlinie.npz"))  # Laser-Gerüst (Frame 0)
    R = np.load(os.path.join(GERUEST_DIR, STEM + "_roi.npz"))         # ROI-Band + Bbox
    x, y, s = L["x"], L["y"], L["s"]           # Referenzpunkte (N,) + Bogenlänge
    nx, ny = L["nx"], L["ny"]                  # Einheits-Normalen je Punkt
    vz = float(R["aussen_vorzeichen"])         # +1/-1: welche Seite ist "außen"
    outx, outy = vz * nx, vz * ny              # Normalen nach außen orientieren
    bbox = tuple(int(v) for v in R["bbox"])    # Crop-Rechteck

    # ---- Frames ----
    frames = sorted(glob.glob(os.path.join(SERIE_DIR, "*.tif")))  # alle Frame-Dateien
    ref = os.path.join(SERIE_DIR, STEM + ".tif")  # Referenz = Frame 0 (eisfrei)
    I0 = lade_grau(ref)                        # Referenzbild I_0
    zi, ziel, IN = letztes_gueltiges(frames, ZIEL_FRAME)  # Zielbild I_N (Standard: letztes Frame)
    print(f"Referenz : {os.path.basename(ref)}")
    print(f"Ziel     : {os.path.basename(ziel)}  (Index {zi}/{len(frames)-1})")
    D = IN - I0                                # DIFFERENZBILD: alles Statische fällt weg

    dicke, schwelle, sigma = messe(D, x, y, outx, outy)  # Eisdicke je Station aus D
    dicke_g = median_glatt(dicke, GLATT_S)     # Dicke über s glätten
    gueltig = ~np.isnan(dicke_g)               # wo wurde Eis erkannt?
    print(f"D-Rauschen sigma={sigma:.2f}  Schwelle={schwelle:.2f}")
    print(f"Eis erkannt auf {gueltig.mean()*100:.0f}% der Bogenlaenge  "
          f"max {np.nanmax(dicke_g):.1f}px  median {np.nanmedian(dicke_g[gueltig]):.1f}px")

    zname = os.path.splitext(os.path.basename(ziel))[0]  # Namensstamm für die Ausgabedateien

    # ---- 1) Differenz-Heatmap (Band-Crop) ----
    hm = diff_heatmap(D, bbox, schwelle)       # D rot/blau einfärben
    cv2.imwrite(os.path.join(OUT_DIR, f"diff_{zname}.png"), hm)

    # ---- 2) Overlay: Bild-0-Linie (gruen) + Eislinie (rot) ----
    ov = cv2.cvtColor(IN.astype(np.uint8), cv2.COLOR_GRAY2BGR)  # Zielbild als Farbbild
    cv2.polylines(ov, [np.stack([x, y], 1).round().astype(np.int32)],
                  False, (0, 220, 0), 2)       # Frame-0-Laserlinie grün einzeichnen
    eis_x = x + outx * np.nan_to_num(dicke_g)  # Eislinie = Referenz + Dicke entlang der Normale
    eis_y = y + outy * np.nan_to_num(dicke_g)
    seg = np.stack([eis_x, eis_y], 1)[gueltig].round().astype(np.int32)  # nur gültige Punkte
    if seg.size:
        cv2.polylines(ov, [seg], False, (0, 0, 255), 2)  # Eislinie rot einzeichnen
    y0, y1, x0, x1 = bbox
    cv2.imwrite(os.path.join(OUT_DIR, f"overlay_{zname}.png"), ov[y0:y1, x0:x1])  # Band-Crop speichern

    # ---- 3) Dicke ueber Bogenlaenge ----
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(s, dicke, color="0.7", lw=0.8, label="roh")
    ax.plot(s, dicke_g, color="C3", lw=1.8, label=f"Median {GLATT_S}px")
    ax.set_xlabel("Bogenlaenge s [px]"); ax.set_ylabel("Eisdicke [px]")
    ax.set_title(f"Eisdicke ueber Bogenlaenge  ({zname})")
    ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, f"dicke_{zname}.png"), dpi=120)
    plt.close(fig)

    np.savez_compressed(os.path.join(OUT_DIR, f"dicke_{zname}.npz"),   # Messdaten sichern
                        s=s, dicke_px=dicke, dicke_glatt=dicke_g,
                        schwelle=schwelle, sigma=sigma)
    print(f"-> diff_{zname}.png, overlay_{zname}.png, dicke_{zname}.png/.npz")


if __name__ == "__main__":
    main()
