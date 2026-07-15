# -*- coding: utf-8 -*-
"""
crop_roi.py - ROI-Band um die gefittete Laserlinie.

Legt ein gerundetes Band ("Padding") um die Laserlinie:
  - kleines Padding zur Innenseite (Profilkoerper)
  - grosses Padding zur Aussenseite (dort waechst das Eis)
Die Aussenseite wird automatisch bestimmt (weg vom Kurven-Schwerpunkt =
konvexe Seite der Profil-Vorderkante) und ist per Flag umschaltbar.

Ausgabe je Bild:
  *_crop_grenze.png  - Originalbild mit eingezeichneter Crop-Grenze (innen/aussen)
  *_crop_maske.png   - nur das Band, Rest schwarz, VOLLE Bildgroesse
                       (globale Pixelkoordinaten bleiben erhalten)
  *_roi.npz          - maske (bool, HxW), bbox, padding, aussen-richtung

Nur lesend auf raw/. Schreibt nur nach output/.
"""
import os, glob
import numpy as np
import cv2

RAW_DIR = "raw"
OUT_DIR = "output"

# ---- einstellbare Parameter ----
PAD_INNEN   = 60      # px Padding zur Innenseite (Profilkoerper) -> klein
PAD_AUSSEN  = 320     # px Padding zur Aussenseite (Eiswachstum)  -> gross
AUSSEN_FLIP = False   # True kehrt die automatisch erkannte Aussenseite um
CAP_SEGMENTE = 24     # Aufloesung der runden Endkappen
NUR_MASKE   = False   # True: nur *_crop_maske.png schreiben (kein Grenzbild/npz)
# Darstellung
FARBE_AUSSEN = (0, 200, 255)   # BGR orange  - aussere Grenze
FARBE_INNEN  = (0, 220, 0)     # BGR gruen   - innere Grenze
FARBE_LASER  = (60, 60, 255)   # BGR rot     - Laserlinie
LINIE_DICKE  = 6
FUELL_ALPHA  = 0.18            # Transparenz der Bandfuellung im Grenzbild


def aussen_richtung(x, y, nx, ny):
    """Vorzeichen so, dass die Normale (nx,ny) von der Kurvenmitte WEG zeigt."""
    cx, cy = x.mean(), y.mean()
    # Vektor Punkt -> Schwerpunkt
    to_c_x, to_c_y = cx - x, cy - y
    # Projektion der Normale auf Richtung zum Schwerpunkt, gemittelt
    proj = np.mean(nx * to_c_x + ny * to_c_y)
    # zeigt die Normale im Mittel ZUM Schwerpunkt (proj>0)? dann ist aussen = -Normale
    s = -1.0 if proj > 0 else 1.0
    if AUSSEN_FLIP:
        s = -s
    return s


def bogen_kappe(p_end, p_outer, p_inner, tangent_fwd):
    """Runde Endkappe: Bogen von p_outer nach p_inner um p_end,
    der durch die Vorwaertstangente laeuft. Radius wird interpoliert."""
    a_out = np.arctan2(p_outer[1] - p_end[1], p_outer[0] - p_end[0])
    a_in = np.arctan2(p_inner[1] - p_end[1], p_inner[0] - p_end[0])
    a_f = np.arctan2(tangent_fwd[1], tangent_fwd[0])
    r_out = np.hypot(p_outer[0] - p_end[0], p_outer[1] - p_end[1])
    r_in = np.hypot(p_inner[0] - p_end[0], p_inner[1] - p_end[1])

    # Sweep CCW von a_out nach a_in; pruefen ob a_f darin liegt, sonst CW
    def norm(d):
        return (d + np.pi) % (2 * np.pi) - np.pi
    ccw = norm(a_in - a_out) % (2 * np.pi)          # 0..2pi
    f_rel = norm(a_f - a_out) % (2 * np.pi)
    if f_rel <= ccw:                                 # Vorwaerts liegt im CCW-Bogen
        span = ccw
    else:
        span = ccw - 2 * np.pi                       # CW (negativ)
    pts = []
    for k in range(1, CAP_SEGMENTE):
        t = k / CAP_SEGMENTE
        a = a_out + span * t
        r = r_out + (r_in - r_out) * t
        pts.append([p_end[0] + r * np.cos(a), p_end[1] + r * np.sin(a)])
    return pts


def band_polygon(x, y, nx, ny, s_aussen):
    """Geschlossenes Polygon des Bandes mit runden Endkappen."""
    out = np.stack([nx, ny], 1) * s_aussen          # Einheits-Aussennormale
    P = np.stack([x, y], 1)
    outer = P + out * PAD_AUSSEN
    inner = P - out * PAD_INNEN

    # Tangenten an den Enden (zeigen aus dem Band heraus)
    t_end = P[-1] - P[-3]
    t_start = P[0] - P[2]

    poly = list(outer)
    poly += bogen_kappe(P[-1], outer[-1], inner[-1], t_end)
    poly += list(inner[::-1])
    poly += bogen_kappe(P[0], inner[0], outer[0], t_start)
    return np.array(poly, dtype=np.float32), outer, inner


def verarbeite(npz_pfad, raw_pfad):
    name = os.path.basename(npz_pfad).replace("_laserlinie.npz", "")
    d = np.load(npz_pfad)
    x, y, nx, ny = d["x"], d["y"], d["nx"], d["ny"]
    H, W = int(d["H"]), int(d["W"])

    s_aussen = aussen_richtung(x, y, nx, ny)
    poly, outer, inner = band_polygon(x, y, nx, ny, s_aussen)

    # ---- Maske (volle Bildgroesse) ----
    maske = np.zeros((H, W), np.uint8)
    cv2.fillPoly(maske, [np.round(poly).astype(np.int32)], 255)
    maske_b = maske > 0

    # ---- Originalbild laden ----
    img = cv2.imread(raw_pfad, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Bild nicht ladbar: {raw_pfad}")
    if img.shape[0] != H or img.shape[1] != W:
        img = cv2.resize(img, (W, H))

    # ---- Bild 2: nur Band, Rest schwarz, volle Groesse (immer) ----
    masked = np.zeros_like(img)
    masked[maske_b] = img[maske_b]
    m_pfad = os.path.join(OUT_DIR, name + "_crop_maske.png")
    cv2.imwrite(m_pfad, masked)

    if NUR_MASKE:
        print(f"{name}  -> {os.path.basename(m_pfad)}  (nur Maske)")
        return

    # ---- Bild 1: Grenze markiert (mit transparenter Fuellung) ----
    grenze = img.copy()
    overlay = grenze.copy()
    cv2.fillPoly(overlay, [np.round(poly).astype(np.int32)], (255, 180, 0))
    cv2.addWeighted(overlay, FUELL_ALPHA, grenze, 1 - FUELL_ALPHA, 0, grenze)
    cv2.polylines(grenze, [np.round(outer).astype(np.int32)], False, FARBE_AUSSEN, LINIE_DICKE)
    cv2.polylines(grenze, [np.round(inner).astype(np.int32)], False, FARBE_INNEN, LINIE_DICKE)
    laser = np.stack([x, y], 1)
    cv2.polylines(grenze, [np.round(laser).astype(np.int32)], False, FARBE_LASER, 3)
    # Legende
    cv2.putText(grenze, f"aussen (+{PAD_AUSSEN}px)", (60, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 2.2, FARBE_AUSSEN, 5)
    cv2.putText(grenze, f"innen  (-{PAD_INNEN}px)", (60, 220),
                cv2.FONT_HERSHEY_SIMPLEX, 2.2, FARBE_INNEN, 5)
    g_pfad = os.path.join(OUT_DIR, name + "_crop_grenze.png")
    cv2.imwrite(g_pfad, grenze)

    # ---- bbox + npz ----
    ys, xs = np.where(maske_b)
    bbox = (int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1)
    np.savez_compressed(
        os.path.join(OUT_DIR, name + "_roi.npz"),
        maske=maske_b, bbox=np.array(bbox), pad_innen=PAD_INNEN,
        pad_aussen=PAD_AUSSEN, aussen_vorzeichen=s_aussen, H=H, W=W)

    seite = "automatisch" + (" + FLIP" if AUSSEN_FLIP else "")
    print(f"{name}")
    print(f"  Aussenseite: {seite}  (Vorzeichen {s_aussen:+.0f})")
    print(f"  Band: innen {PAD_INNEN}px / aussen {PAD_AUSSEN}px   "
          f"Flaeche {maske_b.sum():,} px")
    print(f"  bbox [y {bbox[0]}:{bbox[1]}, x {bbox[2]}:{bbox[3]}]")
    print(f"  -> {os.path.basename(g_pfad)} + {os.path.basename(m_pfad)} + _roi.npz")


def main():
    import argparse
    global OUT_DIR, RAW_DIR
    ap = argparse.ArgumentParser(description="ROI-Band um die gefittete Laserlinie erzeugen.")
    ap.add_argument("--out", default=OUT_DIR, help="Ordner mit *_laserlinie.npz (Ein-/Ausgabe)")
    ap.add_argument("--raw", default=RAW_DIR, help="Ordner mit den zugehoerigen Rohbildern (*.tif)")
    ap.add_argument("--stem", default=None, help="nur diesen Frame-0-Stamm; leer = alle im --out")
    a = ap.parse_args()
    OUT_DIR, RAW_DIR = a.out, a.raw
    pat = (a.stem + "_laserlinie.npz") if a.stem else "*_laserlinie.npz"
    npzs = sorted(glob.glob(os.path.join(OUT_DIR, pat)))
    if not npzs:
        print(f"Keine {pat} in {OUT_DIR} gefunden.")
        return
    for npz in npzs:
        name = os.path.basename(npz).replace("_laserlinie.npz", "")
        raw = os.path.join(RAW_DIR, name + ".tif")
        if not os.path.exists(raw):
            print(f"  raw fehlt fuer {name} in {RAW_DIR}, ueberspringe")
            continue
        verarbeite(npz, raw)


if __name__ == "__main__":
    main()
