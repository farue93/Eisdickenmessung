# -*- coding: utf-8 -*-
"""
run.py - Alle Bilderserien in input/ auswerten.

Workflow:
  1. Repo ziehen.
  2. Je Messreihe einen Unterordner in  input/  ablegen (voller *.tif).
     Das erste *.tif ist Frame 0 und MUSS eisfrei sein (Referenz/Nulllinie).
  3. python run.py

Je Serie: Vorverarbeitung (Laserlinie + ROI, nur einmal) -> Frame Differencing
-> Canny. Ergebnis: viewer.html je Methode + eine ergebnis.html mit Links.
"""
import os, sys, glob, subprocess

ROOT    = os.path.dirname(os.path.abspath(__file__))
INPUT   = os.path.join(ROOT, "input")
PRE     = os.path.join(ROOT, "pre processing")
GERUEST = os.path.join(PRE, "output")
FD      = os.path.join(ROOT, "frame differencing")
CANNY   = os.path.join(ROOT, "canny ice detection")
PY      = sys.executable


def run(titel, cmd, cwd):
    print(f"\n--- {titel} ---")
    return subprocess.run([PY] + cmd, cwd=cwd).returncode == 0


def eine_serie(serie):
    tifs = sorted(glob.glob(os.path.join(serie, "*.tif")))
    if not tifs:
        return None
    frame0 = tifs[0]
    stem = os.path.splitext(os.path.basename(frame0))[0]
    suffix = os.path.basename(serie).split("_")[-1]
    print(f"\n=== Serie: {os.path.basename(serie)}  (Frame 0: {stem}) ===")

    npz_l = os.path.join(GERUEST, stem + "_laserlinie.npz")
    npz_r = os.path.join(GERUEST, stem + "_roi.npz")
    if not (os.path.exists(npz_l) and os.path.exists(npz_r)):
        run("Vorverarbeitung: Laserlinie (Frame 0)",
            ["laser_pipeline.py", frame0, "--out", GERUEST], PRE)
        run("Vorverarbeitung: ROI-Band",
            ["crop_roi.py", "--out", GERUEST, "--raw", serie, "--stem", stem], PRE)
    else:
        print("  Vorverarbeitung vorhanden -> uebersprungen")

    run("Frame Differencing", ["serie_eis.py", serie], FD)
    run("Canny",              ["canny_eis.py", serie], CANNY)

    return os.path.basename(serie), [
        ("Frame Differencing", os.path.join("frame differencing", f"serie_{suffix}", "viewer.html")),
        ("Canny",              os.path.join("canny ice detection", f"serie_{suffix}", "viewer.html")),
    ]


def main():
    os.makedirs(INPUT, exist_ok=True)
    series = [d for d in sorted(glob.glob(os.path.join(INPUT, "*")))
              if os.path.isdir(d) and glob.glob(os.path.join(d, "*.tif"))]
    if not series:
        print("Keine Bilderserie in input/ gefunden.")
        print("Lege je Messreihe einen Unterordner mit *.tif in 'input/' ab")
        print("(erstes *.tif = eisfreier Frame 0) und starte erneut.")
        return

    ergebnisse = [r for r in (eine_serie(os.path.abspath(s)) for s in series) if r]

    # Ergebnis-Uebersicht (ergebnis.html)
    zeilen = []
    for name, viewer in ergebnisse:
        links = " &nbsp;|&nbsp; ".join(
            f'<a href="{p.replace(os.sep,"/")}">{n}</a>'
            f'{"" if os.path.exists(os.path.join(ROOT,p)) else " (fehlt)"}'
            for n, p in viewer)
        zeilen.append(f"<li><b>{name}</b>: {links}</li>")
    html = (f"<!DOCTYPE html><html lang=de><meta charset=utf-8><title>Eisdickenmessung</title>"
            f"<body style='font-family:system-ui;background:#111;color:#eee;padding:24px'>"
            f"<h1>Eisdickenmessung &mdash; Ergebnisse</h1>"
            f"<ul style='font-size:17px;line-height:2'>{''.join(zeilen)}</ul></body></html>")
    idx = os.path.join(ROOT, "ergebnis.html")
    open(idx, "w", encoding="utf-8").write(html)

    print("\n" + "=" * 60)
    print(f"FERTIG. Uebersicht im Browser oeffnen:\n  {idx}")
    for name, viewer in ergebnisse:
        for n, p in viewer:
            full = os.path.join(ROOT, p)
            print(f"  [{'OK   ' if os.path.exists(full) else 'FEHLT'}] {name} / {n}: {full}")


if __name__ == "__main__":
    main()
