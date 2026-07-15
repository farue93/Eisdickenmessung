# -*- coding: utf-8 -*-
"""
label_tool.py - Laserlinie labeln (laser_v2).

Werkzeuge:
  PINSEL (linke Maustaste): markiert im Radius nur Pixel, deren Helligkeit
     >= "Helligkeit"-Schwelle ist. Die BLAUE Vorschau zeigt live, welche Pixel
     bei einem Klick markiert wuerden (rein nach Helligkeit).
  RADIERER (rechte Maustaste): loescht im Radius, pixelgenau, ohne Gate.
Zoom (Mausrad) + Verschieben (mittlere Taste). Vorbelegt sind die besten
Laserpunkte (rot). Du erweiterst sie zur vollstaendigen Linie.

Tasten: n/p Frame (auto-save) | s speichern | c leeren | g Gate an/aus
        h Vorschau an/aus | 1 = 100% | f einpassen | q/ESC Ende
Radius/Helligkeit auch per Trackbar (oder [ ] und , .).
"""
import os, glob, cv2
import numpy as np

BASE   = os.path.dirname(os.path.abspath(__file__))
FRAMES = os.path.join(BASE, "frames")           # die 5 ausgewaehlten Frames
MASKS  = os.path.join(BASE, "masks")            # vorbelegt + deine Ergaenzungen
WIN    = "laser_v2 labeln  (L=Pinsel  R=Radierer)"
os.makedirs(MASKS, exist_ok=True)

files = sorted(glob.glob(os.path.join(FRAMES, "*.png")))
assert files, f"Keine Frames in {FRAMES} - erst setup_labels.py laufen lassen."

# Ansichts-/Werkzeug-Zustand in EINEM dict (bequem im Callback aenderbar)
S = {"i": 0, "img": None, "mask": None, "H": 0, "W": 0,
     "zoom": 1.0, "ox": 0.0, "oy": 0.0, "tool": None, "last": None, "pan": None}


# ---------------------------------------------------- laden / speichern
def mask_path(i): return os.path.join(MASKS, os.path.basename(files[i]))
def speichern():
    if S["mask"] is not None: cv2.imwrite(mask_path(S["i"]), S["mask"])
def lade(i):
    S["i"] = i % len(files)
    S["img"] = cv2.imread(files[S["i"]], cv2.IMREAD_GRAYSCALE)
    S["H"], S["W"] = S["img"].shape
    mp = mask_path(S["i"])
    m = cv2.imread(mp, cv2.IMREAD_GRAYSCALE) if os.path.exists(mp) else None
    S["mask"] = m if m is not None else np.zeros((S["H"], S["W"]), np.uint8)


# ---------------------------------------------------- Trackbars lesen
def tb(name):
    try: return cv2.getTrackbarPos(name, WIN)
    except cv2.error: return 0
def radius():   return max(1, tb("Radius"))
def schwelle(): return tb("Helligkeit")
def gate_an():  return tb("Gate 0/1") == 1
def hint_an():  return tb("Vorschau 0/1") == 1


# ---------------------------------------------------- malen
def stempel(p0, p1, tool):
    r = radius(); dick = 2 * r - 1                       # r=1 -> 1 px (pixelgenau)
    if tool == "erase":
        cv2.line(S["mask"], p0, p1, 0, dick); return     # radieren ohne Gate
    tmp = np.zeros((S["H"], S["W"]), np.uint8)
    cv2.line(tmp, p0, p1, 255, dick)                     # Pinselspur
    if gate_an():
        tmp[S["img"] < schwelle()] = 0                  # nur helle Pixel behalten
    S["mask"][tmp > 0] = 255


# ---------------------------------------------------- Ansicht (Zoom/Pan)
def winsize():
    try:
        _, _, w, h = cv2.getWindowImageRect(WIN)
        if w > 0 and h > 0: return w, h
    except cv2.error: pass
    return 1280, 900
def win_zu_quelle(mx, my):
    return int(S["ox"] + mx / S["zoom"]), int(S["oy"] + my / S["zoom"])
def overlay():
    g = np.clip(S["img"].astype(np.float32) * 1.8, 0, 255).astype(np.uint8)
    ov = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
    if hint_an():                                       # BLAUE Vorschau: was Gate treffen wuerde
        cand = S["img"] >= schwelle()
        ov[cand] = (0.45 * ov[cand] + np.array([150, 40, 0])).astype(np.uint8)
    m = S["mask"] > 0                                   # bereits markiert = ROT
    ov[m] = (0.30 * ov[m] + np.array([0, 0, 180])).astype(np.uint8)
    return ov
def render():
    winw, winh = winsize()
    vw = int(np.ceil(winw / S["zoom"])) + 1; vh = int(np.ceil(winh / S["zoom"])) + 1
    S["ox"] = float(min(max(0.0, S["ox"]), max(0.0, S["W"] - vw)))
    S["oy"] = float(min(max(0.0, S["oy"]), max(0.0, S["H"] - vh)))
    ox, oy = int(S["ox"]), int(S["oy"])
    sub = overlay()[oy:oy + vh, ox:ox + vw]
    disp = cv2.resize(sub, (winw, winh), interpolation=cv2.INTER_NEAREST)
    n = int((S["mask"] > 0).sum())
    hud = (f"Frame {S['i']+1}/{len(files)}  {os.path.basename(files[S['i']])}   "
           f"Radius {radius()}  Helligkeit {schwelle()}  Gate {'AN' if gate_an() else 'aus'}   "
           f"Maske {n}px  Zoom {S['zoom']:.1f}x")
    cv2.rectangle(disp, (0, 0), (winw, 22), (0, 0, 0), -1)
    cv2.putText(disp, hud, (8, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1, cv2.LINE_AA)
    cv2.imshow(WIN, disp)
def zoom_auf(mx, my, f):
    sx, sy = S["ox"] + mx / S["zoom"], S["oy"] + my / S["zoom"]
    S["zoom"] = float(np.clip(S["zoom"] * f, 0.1, 40.0))
    S["ox"], S["oy"] = sx - mx / S["zoom"], sy - my / S["zoom"]
def einpassen():
    winw, winh = winsize(); S["zoom"] = min(winw / S["W"], winh / S["H"]); S["ox"] = S["oy"] = 0.0


# ---------------------------------------------------- Maus
def maus(event, mx, my, flags, _):
    if event == cv2.EVENT_MOUSEWHEEL:
        zoom_auf(mx, my, 1.25 if flags > 0 else 0.8); render(); return
    if event == cv2.EVENT_LBUTTONDOWN:
        S["tool"] = "paint"; S["last"] = win_zu_quelle(mx, my); stempel(S["last"], S["last"], "paint"); render(); return
    if event == cv2.EVENT_RBUTTONDOWN:
        S["tool"] = "erase"; S["last"] = win_zu_quelle(mx, my); stempel(S["last"], S["last"], "erase"); render(); return
    if event == cv2.EVENT_MBUTTONDOWN:
        S["pan"] = (mx, my); return
    if event == cv2.EVENT_MOUSEMOVE:
        if S["tool"] and S["last"] is not None:
            p = win_zu_quelle(mx, my); stempel(S["last"], p, S["tool"]); S["last"] = p; render()
        elif S["pan"] is not None:
            S["ox"] -= (mx - S["pan"][0]) / S["zoom"]; S["oy"] -= (my - S["pan"][1]) / S["zoom"]
            S["pan"] = (mx, my); render()
        return
    if event in (cv2.EVENT_LBUTTONUP, cv2.EVENT_RBUTTONUP): S["tool"] = None; S["last"] = None
    if event == cv2.EVENT_MBUTTONUP: S["pan"] = None


def main():
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL); cv2.resizeWindow(WIN, 1280, 900)
    cv2.createTrackbar("Radius", WIN, 4, 40, lambda v: None)
    cv2.createTrackbar("Helligkeit", WIN, 150, 255, lambda v: None)
    cv2.createTrackbar("Gate 0/1", WIN, 1, 1, lambda v: None)
    cv2.createTrackbar("Vorschau 0/1", WIN, 1, 1, lambda v: None)
    cv2.setMouseCallback(WIN, maus)
    lade(0); einpassen(); render()
    print("Bereit. L=Pinsel R=Radierer  Rad=Zoom  Mitte=Verschieben  n/p=Frame  s=save  q=Ende")
    while True:
        k = cv2.waitKey(20) & 0xFF
        if k == 255: continue
        if k in (ord('q'), 27): speichern(); break
        elif k == ord('n'): speichern(); lade(S["i"] + 1); render()
        elif k == ord('p'): speichern(); lade(S["i"] - 1); render()
        elif k == ord('s'): speichern(); print("gespeichert:", os.path.basename(mask_path(S["i"])))
        elif k == ord('c'): S["mask"][:] = 0; render()
        elif k == ord('1'): S["zoom"] = 1.0; render()
        elif k == ord('f'): einpassen(); render()
        elif k == ord('g'): cv2.setTrackbarPos("Gate 0/1", WIN, 0 if gate_an() else 1); render()
        elif k == ord('h'): cv2.setTrackbarPos("Vorschau 0/1", WIN, 0 if hint_an() else 1); render()
        elif k == ord('['): cv2.setTrackbarPos("Radius", WIN, max(1, radius() - 1)); render()
        elif k == ord(']'): cv2.setTrackbarPos("Radius", WIN, min(40, radius() + 1)); render()
        elif k == ord(','): cv2.setTrackbarPos("Helligkeit", WIN, max(0, schwelle() - 5)); render()
        elif k == ord('.'): cv2.setTrackbarPos("Helligkeit", WIN, min(255, schwelle() + 5)); render()
    cv2.destroyAllWindows(); print("Fertig. Masken in:", MASKS)


if __name__ == "__main__":
    main()
