# -*- coding: utf-8 -*-
"""
roi_werkzeug.py - Messbereich am Messtag festlegen (Schritt 9/10 im Ablauf).

WARUM DAS NOETIG IST: Die Panelmaske aus der bisherigen Auswertung gehoert zu
der DAMALIGEN Kameraposition. Sobald die Kameras neu ausgerichtet sind, ist sie
wertlos - sie wuerde den falschen Bildbereich als Bezugsflaeche ausweisen. Der
Messbereich muss also vor Ort neu gesetzt werden, und zwar bevor der erste
Versuchslauf startet.

Bedienung:
  1. Bild waehlen (ein beliebiges Bild aus dem Aufnahmeordner, ohne Eis)
  2. Rechteck aufziehen  -> der Bereich, der ausgewertet wird
  3. Optional: mit dem Pinsel Stoerbereiche im Rechteck ausschliessen
     (Halterungen, Reflexe, Kanalwand)
  4. Speichern -> .npz, in der Live-Oberflaeche unter "Panelmaske" eintragen

Bewusst OHNE SAM: kein 40-MB-Modell, keine Zusatzpakete, kein Fehlschlag im
ungueenstigen Moment. Ein Rechteck plus Ausschluss reicht - die physikalisch
belastbare Groesse ist ohnehin die Flaeche in mm2 aus der Kalibrierung, nicht
der Prozentwert.

Tasten: s speichern · z letzten Pinselstrich zurueck · r Rechteck neu
        p Pinsel/Ausschluss an-aus · [ ] Pinselgroesse · f einpassen · q Ende
"""
import os, sys, ctypes
import numpy as np
import cv2

try:
    ctypes.windll.shcore.SetProcessDpiAwareness(2)
except Exception:
    try:
        ctypes.windll.user32.SetProcessDPIAware()
    except Exception:
        pass

BASE = os.path.dirname(os.path.abspath(__file__))
FENSTER = "Messbereich festlegen"
ANZEIGE_W, ANZEIGE_H = 1500, 820

S = {"bild": None, "H": 0, "W": 0, "zoom": 1.0, "ox": 0.0, "oy": 0.0,
     "rechteck": None, "zieht": False, "start": None,
     "aus": None, "pinsel": False, "malt": False, "radius": 40,
     "verlauf": [], "maus": None, "quelle": ""}


def bild_laden(pfad):
    img = cv2.imread(pfad, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise SystemExit(f"Bild nicht lesbar: {pfad}")
    S["bild"] = img
    S["H"], S["W"] = img.shape
    S["aus"] = np.zeros(img.shape, np.uint8)
    S["rechteck"] = None
    S["verlauf"] = []
    S["quelle"] = pfad
    print(f"geladen: {os.path.basename(pfad)}  {S['W']}x{S['H']}")


# ---------------------------------------------------------------- Ansicht
def einpassen():
    S["zoom"] = min(ANZEIGE_W / S["W"], ANZEIGE_H / S["H"])
    S["ox"] = S["oy"] = 0.0
    klemmen()


def klemmen():
    sw, sh = ANZEIGE_W / S["zoom"], ANZEIGE_H / S["zoom"]
    S["ox"] = (S["W"] - sw) / 2 if S["W"] <= sw else float(np.clip(S["ox"], 0, S["W"] - sw))
    S["oy"] = (S["H"] - sh) / 2 if S["H"] <= sh else float(np.clip(S["oy"], 0, S["H"] - sh))


def zu_bild(cx, cy):
    return (S["ox"] + cx / S["zoom"], S["oy"] + cy / S["zoom"])


def zu_canvas(p):
    return ((p[0] - S["ox"]) * S["zoom"], (p[1] - S["oy"]) * S["zoom"])


def zeichnen():
    if S["bild"] is None:
        return
    klemmen()
    z = S["zoom"]
    M = np.float32([[z, 0, -S["ox"] * z], [0, z, -S["oy"] * z]])
    grau = cv2.warpAffine(S["bild"], M, (ANZEIGE_W, ANZEIGE_H), flags=cv2.INTER_AREA)
    vis = cv2.cvtColor(np.clip(grau.astype(np.float32) * 1.3, 0, 255).astype(np.uint8),
                       cv2.COLOR_GRAY2BGR)

    gueltig = maske_bauen()
    if gueltig is not None:
        m = cv2.warpAffine(gueltig, M, (ANZEIGE_W, ANZEIGE_H), flags=cv2.INTER_NEAREST)
        # alles AUSSERHALB des Messbereichs abdunkeln - so sieht man sofort,
        # was tatsaechlich ausgewertet wird
        vis[m == 0] = (vis[m == 0] * 0.32).astype(np.uint8)

    if S["rechteck"]:
        x0, y0, x1, y1 = S["rechteck"]
        a = tuple(int(v) for v in zu_canvas((x0, y0)))
        b = tuple(int(v) for v in zu_canvas((x1, y1)))
        cv2.rectangle(vis, a, b, (0, 200, 255), 2)

    if S["pinsel"] and S["maus"]:
        cv2.circle(vis, S["maus"], max(2, int(S["radius"] * S["zoom"])), (0, 120, 255), 2)

    flaeche = int(gueltig.sum()) if gueltig is not None else 0
    anteil = 100 * flaeche / (S["W"] * S["H"]) if flaeche else 0
    kopf = (f"{'PINSEL: Stoerbereiche ausschliessen' if S['pinsel'] else 'RECHTECK aufziehen'}"
            f"   Messbereich {flaeche:,} px ({anteil:.1f}% des Bildes)"
            f"   Radius {S['radius']}   Zoom {z:.2f}x")
    cv2.rectangle(vis, (0, 0), (ANZEIGE_W, 28), (0, 0, 0), -1)
    cv2.putText(vis, kopf, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 210, 255), 1, cv2.LINE_AA)
    hilfe = "s speichern | z zurueck | r Rechteck neu | p Pinsel | [ ] Radius | f einpassen | q Ende"
    cv2.rectangle(vis, (0, ANZEIGE_H - 26), (ANZEIGE_W, ANZEIGE_H), (0, 0, 0), -1)
    cv2.putText(vis, hilfe, (10, ANZEIGE_H - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (190, 190, 190), 1, cv2.LINE_AA)
    cv2.imshow(FENSTER, vis)


def maske_bauen():
    if not S["rechteck"]:
        return None
    x0, y0, x1, y1 = S["rechteck"]
    m = np.zeros((S["H"], S["W"]), np.uint8)
    m[y0:y1, x0:x1] = 1
    m[S["aus"] > 0] = 0
    return m


# ---------------------------------------------------------------- Maus
def maus(ereignis, cx, cy, flags, _):
    S["maus"] = (cx, cy)
    if ereignis == cv2.EVENT_MOUSEWHEEL:
        bx, by = zu_bild(cx, cy)
        try:
            hoch = cv2.getMouseWheelDelta(flags) > 0
        except Exception:
            hoch = flags > 0
        S["zoom"] = float(np.clip(S["zoom"] * (1.25 if hoch else 0.8), 0.05, 40))
        S["ox"], S["oy"] = bx - cx / S["zoom"], by - cy / S["zoom"]
        zeichnen(); return

    if S["pinsel"]:
        if ereignis == cv2.EVENT_LBUTTONDOWN:
            S["malt"] = True
            S["verlauf"].append(S["aus"].copy())
            if len(S["verlauf"]) > 20:
                S["verlauf"].pop(0)
        if S["malt"] and ereignis in (cv2.EVENT_LBUTTONDOWN, cv2.EVENT_MOUSEMOVE):
            p = zu_bild(cx, cy)
            cv2.circle(S["aus"], (int(p[0]), int(p[1])), S["radius"], 255, -1)
            zeichnen()
        if ereignis == cv2.EVENT_LBUTTONUP:
            S["malt"] = False
        return

    if ereignis == cv2.EVENT_LBUTTONDOWN:
        S["zieht"] = True; S["start"] = zu_bild(cx, cy)
    elif ereignis == cv2.EVENT_MOUSEMOVE and S["zieht"]:
        p = zu_bild(cx, cy)
        S["rechteck"] = rechteck_aus(S["start"], p); zeichnen()
    elif ereignis == cv2.EVENT_LBUTTONUP and S["zieht"]:
        S["zieht"] = False
        p = zu_bild(cx, cy)
        S["rechteck"] = rechteck_aus(S["start"], p); zeichnen()
    elif ereignis == cv2.EVENT_MOUSEMOVE:
        zeichnen()


def rechteck_aus(a, b):
    x0, x1 = sorted((int(a[0]), int(b[0])))
    y0, y1 = sorted((int(a[1]), int(b[1])))
    x0 = max(0, x0); y0 = max(0, y0)
    x1 = min(S["W"], x1); y1 = min(S["H"], y1)
    return (x0, y0, x1, y1) if x1 - x0 > 10 and y1 - y0 > 10 else None


def speichern():
    m = maske_bauen()
    if m is None:
        print("  kein Rechteck gesetzt - nichts gespeichert"); return
    x0, y0, x1, y1 = S["rechteck"]
    ziel = os.path.join(BASE, "messbereich.npz")
    # gleiches Format wie die bisherige Panelmaske: volle Bildgroesse + bbox
    np.savez(ziel, maske=m, bbox=np.array([x0, y0, x1, y1]),
             flaeche_px=int(m.sum()), quelle=os.path.basename(S["quelle"]))
    vor = os.path.join(BASE, "messbereich_kontrolle.png")
    vis = cv2.cvtColor(np.clip(S["bild"].astype(np.float32) * 1.3, 0, 255).astype(np.uint8),
                       cv2.COLOR_GRAY2BGR)
    vis[m == 0] = (vis[m == 0] * 0.3).astype(np.uint8)
    cv2.rectangle(vis, (x0, y0), (x1, y1), (0, 200, 255), 3)
    klein = cv2.resize(vis, (1400, int(S["H"] * 1400 / S["W"])), interpolation=cv2.INTER_AREA)
    cv2.imwrite(vor, klein)
    print(f"  gespeichert: {ziel}")
    print(f"  Messbereich {int(m.sum()):,} px   Kontrollbild: {vor}")
    print(f"  -> in der Live-Oberflaeche unter 'Panelmaske' eintragen")


def main():
    pfad = sys.argv[1] if len(sys.argv) > 1 else None
    if not pfad:
        import tkinter as tk
        from tkinter import filedialog
        w = tk.Tk(); w.withdraw()
        pfad = filedialog.askopenfilename(
            title="Ein eisfreies Bild aus dem Aufnahmeordner waehlen",
            filetypes=[("Bilder", "*.tif *.tiff *.png *.jpg *.jpeg *.bmp"), ("Alle", "*.*")])
        w.destroy()
    if not pfad:
        return
    bild_laden(pfad)
    cv2.namedWindow(FENSTER, cv2.WINDOW_AUTOSIZE)
    einpassen()
    cv2.setMouseCallback(FENSTER, maus)
    zeichnen()
    print("Rechteck aufziehen, dann s zum Speichern.")
    while True:
        k = cv2.waitKey(20) & 0xFF
        if k == 255:
            if cv2.getWindowProperty(FENSTER, cv2.WND_PROP_VISIBLE) < 1:
                break
            continue
        if k in (ord('q'), 27): break
        elif k == ord('s'): speichern()
        elif k == ord('p'): S["pinsel"] = not S["pinsel"]; zeichnen()
        elif k == ord('r'): S["rechteck"] = None; zeichnen()
        elif k == ord('f'): einpassen(); zeichnen()
        elif k == ord('z') and S["verlauf"]:
            S["aus"] = S["verlauf"].pop(); zeichnen()
        elif k == ord('['): S["radius"] = max(4, S["radius"] - 8); zeichnen()
        elif k == ord(']'): S["radius"] = min(400, S["radius"] + 8); zeichnen()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
