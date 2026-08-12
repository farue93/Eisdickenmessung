# -*- coding: utf-8 -*-
"""
live_laser.py - Live-Auswertung der Laserlinie am Versuchs-PC.

Ueberwacht den Aufnahmeordner, segmentiert die Laserlinie mit dem trainierten
U-Net und berechnet die Eisdicke als Verschiebung der Linie gegenueber dem
EISFREIEN Referenzzustand - dieselbe Groesse wie in der Offline-Auswertung
(laser_v2/unet_v2/apply_unet.py).

Zwei Dinge sind hier anders als offline, beide mit Absicht:

1. Die Geometrie (Stuetzstellen entlang der Linie und ihre Normalen) wird aus
   den ersten eisfreien Frames des LAUFENDEN Versuchs abgeleitet. Am Messtag
   ist der Aufbau neu, eine mitgebrachte Geometriedatei waere ungueltig.
   Wer eine bestehende Geometrie hat, kann sie stattdessen laden.

2. Nur das U-Net, kein Methodenvergleich. Die Live-Ansicht beantwortet eine
   Frage: laeuft der Versuch sauber? Der Vergleich gehoert in die
   Offline-Auswertung, wo beliebig oft neu gerechnet werden kann.

Start:  start_laser.bat   oder   python live_laser.py
"""
import os, sys, json, time, threading, queue
import tkinter as tk
from tkinter import ttk

import numpy as np
import cv2

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
from gemeinsam.konfig import Konfig
from gemeinsam.ordnerwache import Ordnerwache, bild_lesen
from gemeinsam.massstab import Massstab
from gemeinsam import geraet as ger
from gemeinsam import gui

STANDARD = {
    "aufnahme_ordner": "",
    "ergebnis_ordner": os.path.join(BASE, "ergebnisse", "laser"),
    "modell": "",
    "geometrie": "",
    "kalibrierung": "",
    "px_pro_mm": 13.9,
    "schwelle": 0.5,
    "jedes_n_te": 1,
    "referenz_frames": 10,
    "such_von": -25.0,
    "such_bis": 60.0,
    "glaettung": 9,
    "min_flaeche": 100,
    "overlays_speichern": True,
    "geraet": "auto",
}


# ----------------------------------------------------------------- Geometrie
def mittellinie(maske, mindest=3):
    """Spaltenweiser Schwerpunkt der Laserlinie -> (x, y) der Stuetzstellen.

    GRENZE DES VERFAHRENS: setzt voraus, dass die Linie je Bildspalte hoechstens
    einmal vorkommt, also y als Funktion von x beschreibbar ist. Fuer eine Linie,
    die sich um die Nase herumbiegt und dabei ueber sich selbst zurueckfaellt,
    trifft das nicht zu - dann die Geometrie aus der Basis-Pipeline laden."""
    H, W = maske.shape
    xs, ys = [], []
    yy = np.arange(H, dtype=np.float32)
    for sx in range(W):
        spalte = maske[:, sx].astype(np.float32)
        if (spalte > 0).sum() < mindest:
            continue
        xs.append(float(sx))
        ys.append(float((yy * spalte).sum() / spalte.sum()))
    return np.array(xs), np.array(ys)


def glatt(a, w):
    if w < 3 or len(a) < w:
        return a
    kern = np.ones(w) / w
    rand = w // 2
    erweitert = np.concatenate([np.full(rand, a[0]), a, np.full(rand, a[-1])])
    return np.convolve(erweitert, kern, mode="valid")[:len(a)]


def geometrie_ableiten(maske, glaettung=31):
    """Stuetzstellen, Normalen und Bogenlaenge aus einer sauberen Laserlinie."""
    x, y = mittellinie(maske)
    if len(x) < 20:
        raise ValueError("zu wenige Linienpunkte - Schwelle, Crop oder Modell pruefen")
    y = glatt(y, glaettung)
    dx = np.gradient(x)
    dy = np.gradient(y)
    laenge = np.hypot(dx, dy)
    laenge[laenge == 0] = 1.0
    # Normale = Tangente um 90 Grad gedreht; Vorzeichen so, dass sie nach oben
    # (kleinere y) zeigt - dorthin waechst das Eis im Bild.
    outx, outy = -dy / laenge, dx / laenge
    if np.mean(outy) > 0:
        outx, outy = -outx, -outy
    s = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(x), np.diff(y)))])
    return dict(x=x, y=y, outx=outx, outy=outy, s=s)


def bilinear(bild, X, Y):
    """Bilineare Abtastung an beliebigen Gleitkommastellen."""
    H, W = bild.shape
    x0 = np.clip(np.floor(X).astype(int), 0, W - 2)
    y0 = np.clip(np.floor(Y).astype(int), 0, H - 2)
    fx, fy = X - x0, Y - y0
    a = bild[y0, x0] * (1 - fx) * (1 - fy) + bild[y0, x0 + 1] * fx * (1 - fy)
    b = bild[y0 + 1, x0] * (1 - fx) * fy + bild[y0 + 1, x0 + 1] * fx * fy
    return a + b


def median_glatt(a, w):
    if w < 3:
        return a
    n, r, aus = len(a), w // 2, a.copy()
    for i in range(n):
        teil = a[max(0, i - r):min(n, i + r + 1)]
        teil = teil[np.isfinite(teil)]
        aus[i] = np.median(teil) if teil.size else np.nan
    return aus


# ----------------------------------------------------------------- Auswertung
class Auswerter:
    def __init__(self, modellpfad, schwelle, geraetwahl, min_flaeche):
        import torch
        from netze.unet_laser import UNet
        self.torch = torch
        self.geraet, self.geraet_text = ger.geraet_waehlen(geraetwahl)
        self.netz = UNet().to(self.geraet)
        self.netz.load_state_dict(torch.load(modellpfad, map_location=self.geraet))
        self.netz.eval()
        self.schwelle = float(schwelle)
        self.min_flaeche = int(min_flaeche)
        self.geo = None

    def maske(self, img):
        """Laserlinienmaske. Randauffuellung auf Vielfache von 8, weil das Netz
        vier Mal halbiert - sonst passen die Skip-Verbindungen nicht."""
        H, W = img.shape
        Hp, Wp = H + (-H) % 8, W + (-W) % 8
        feld = np.pad(img, ((0, Hp - H), (0, Wp - W)))[None, None] / 255.0
        x = self.torch.from_numpy(feld).float().to(self.geraet)
        with self.torch.no_grad():
            p = self.torch.sigmoid(self.netz(x))[0, 0].cpu().numpy()[:H, :W]
        roh = (p > self.schwelle).astype(np.uint8) * 255
        n, lab, st, _ = cv2.connectedComponentsWithStats(roh)
        behalten = np.zeros_like(roh)
        for k in range(1, n):
            if st[k, cv2.CC_STAT_AREA] >= self.min_flaeche:
                behalten[lab == k] = 255
        return behalten

    def versatz(self, maske, von, bis):
        """Schwerpunkt der Linie entlang der Normalen je Stuetzstelle."""
        g = self.geo
        t = np.arange(von, bis + 1e-9, 1.0)
        X = g["x"][:, None] + g["outx"][:, None] * t[None, :]
        Y = g["y"][:, None] + g["outy"][:, None] * t[None, :]
        P = bilinear(maske.astype(np.float32), X, Y)
        d = np.full(len(g["x"]), np.nan)
        for i in range(len(d)):
            m = P[i] > 127
            if m.any():
                w = P[i][m]
                d[i] = float(np.sum(t[m] * w) / np.sum(w))
        return d


# ----------------------------------------------------------------- Oberflaeche
class Anwendung(ttk.Frame):
    def __init__(self, wurzel):
        super().__init__(wurzel, padding=10)
        self.pack(fill="both", expand=True)
        self.kfg = Konfig(os.path.join(BASE, "einstellungen_laser.json"), STANDARD)
        self.warteschlange = queue.Queue()
        self.laeuft = False
        self.ergebnisse = []
        self.massstab = Massstab()
        self._bauen()
        self.after(200, self._abholen)

    def _bauen(self):
        links = ttk.Frame(self); links.pack(side="left", fill="y", padx=(0, 12))
        rechts = ttk.Frame(self); rechts.pack(side="left", fill="both", expand=True)

        a = gui.abschnitt(links, "Ordner")
        self.f_auf = gui.Pfadfeld(a, "Aufnahmeordner", self.kfg["aufnahme_ordner"], "ordner")
        self.f_auf.pack(fill="x", pady=2)
        self.f_erg = gui.Pfadfeld(a, "Ergebnisordner", self.kfg["ergebnis_ordner"], "ordner")
        self.f_erg.pack(fill="x", pady=2)

        b = gui.abschnitt(links, "Modell, Geometrie, Kalibrierung")
        self.f_modell = gui.Pfadfeld(b, "U-Net (.pt)", self.kfg["modell"], "datei",
                                     [("PyTorch-Modell", "*.pt"), ("Alle", "*.*")])
        self.f_modell.pack(fill="x", pady=2)
        self.f_geo = gui.Pfadfeld(b, "Geometrie (.npz, opt.)", self.kfg["geometrie"], "datei",
                                  [("NumPy", "*.npz"), ("Alle", "*.*")])
        self.f_geo.pack(fill="x", pady=2)
        ttk.Label(b, text="leer lassen = aus den Referenzframes ableiten",
                  foreground="#666").pack(anchor="w")
        self.f_kal = gui.Pfadfeld(b, "Kalibrierung (.npz)", self.kfg["kalibrierung"], "datei",
                                  [("NumPy", "*.npz"), ("Alle", "*.*")])
        self.f_kal.pack(fill="x", pady=2)
        self.f_ppm = gui.Feld(b, "px/mm ersatzweise", self.kfg["px_pro_mm"], 8, "",
                              "nur ohne Kalibrierung")
        self.f_ppm.pack(fill="x", pady=2)

        c = gui.abschnitt(links, "Auswertung")
        self.f_ref = gui.Feld(c, "Referenzframes", self.kfg["referenz_frames"], 8, "",
                              "eisfrei am Anfang")
        self.f_ref.pack(fill="x", pady=2)
        self.f_schwelle = gui.Feld(c, "Schwelle", self.kfg["schwelle"], 8)
        self.f_schwelle.pack(fill="x", pady=2)
        self.f_von = gui.Feld(c, "Suchbereich von", self.kfg["such_von"], 8, "px")
        self.f_von.pack(fill="x", pady=2)
        self.f_bis = gui.Feld(c, "Suchbereich bis", self.kfg["such_bis"], 8, "px")
        self.f_bis.pack(fill="x", pady=2)
        self.f_glatt = gui.Feld(c, "Glaettung", self.kfg["glaettung"], 8, "px")
        self.f_glatt.pack(fill="x", pady=2)
        self.f_minfl = gui.Feld(c, "min. Linienflaeche", self.kfg["min_flaeche"], 8, "px")
        self.f_minfl.pack(fill="x", pady=2)
        self.f_n = gui.Feld(c, "jedes N-te Bild", self.kfg["jedes_n_te"], 8, "", "1 = alle")
        self.f_n.pack(fill="x", pady=2)
        self.v_overlay = tk.BooleanVar(value=self.kfg["overlays_speichern"])
        ttk.Checkbutton(c, text="Overlay-Bilder mitspeichern",
                        variable=self.v_overlay).pack(anchor="w", pady=2)
        self.v_geraet = tk.StringVar(value=self.kfg["geraet"])
        z = ttk.Frame(c); z.pack(fill="x", pady=2)
        ttk.Label(z, text="Rechenwerk", width=22, anchor="w").pack(side="left")
        ttk.Combobox(z, textvariable=self.v_geraet, values=["auto", "cpu"],
                     width=8, state="readonly").pack(side="left")

        d = gui.abschnitt(links, "Steuerung")
        self.knopf = ttk.Button(d, text="Start", command=self._umschalten)
        self.knopf.pack(fill="x", pady=2)
        ttk.Button(d, text="Referenz neu setzen", command=self._ref_neu).pack(fill="x", pady=2)
        ttk.Button(d, text="Einstellungen speichern", command=self._speichern).pack(fill="x", pady=2)

        self.bild = gui.Bildflaeche(rechts); self.bild.pack(fill="both", expand=True)
        self.kurve = gui.Verlauf(rechts, titel="max. Eisdicke"); self.kurve.pack(fill="x", pady=(8, 0))
        self.status = gui.Statuszeile(rechts); self.status.pack(fill="x", pady=(8, 0))
        self.status.setzen("Ordner und Modell waehlen, dann Start")

    def _sammeln(self):
        k = self.kfg
        k["aufnahme_ordner"] = self.f_auf.text(); k["ergebnis_ordner"] = self.f_erg.text()
        k["modell"] = self.f_modell.text(); k["geometrie"] = self.f_geo.text()
        k["kalibrierung"] = self.f_kal.text(); k["px_pro_mm"] = self.f_ppm.zahl(13.9)
        k["schwelle"] = self.f_schwelle.zahl(0.5)
        k["such_von"] = self.f_von.zahl(-25.0); k["such_bis"] = self.f_bis.zahl(60.0)
        k["glaettung"] = max(1, self.f_glatt.zahl(9, ganz=True))
        k["min_flaeche"] = max(1, self.f_minfl.zahl(100, ganz=True))
        k["jedes_n_te"] = max(1, self.f_n.zahl(1, ganz=True))
        k["referenz_frames"] = max(1, self.f_ref.zahl(10, ganz=True))
        k["overlays_speichern"] = bool(self.v_overlay.get()); k["geraet"] = self.v_geraet.get()

    def _speichern(self):
        self._sammeln(); self.kfg.speichern()
        self.status.setzen("Einstellungen gespeichert", "ok")

    def _ref_neu(self):
        self._neue_referenz = True
        self.ergebnisse.clear(); self.kurve.leeren()
        self.status.setzen("Referenz wird beim naechsten Bild neu aufgebaut", "warn")

    def _umschalten(self):
        if self.laeuft:
            self.laeuft = False; self.knopf.configure(text="Start")
            self.status.setzen("gestoppt"); return
        self._sammeln(); self.kfg.speichern()
        fehler = self._pruefen()
        if fehler:
            self.status.setzen(fehler, "fehler"); return
        self.laeuft = True; self.knopf.configure(text="Stopp")
        threading.Thread(target=self._arbeiten, daemon=True).start()

    def _pruefen(self):
        """Alles pruefen, BEVOR der Arbeitsfaden startet - sonst schlaegt ein
        Tippfehler erst im Faden fehl, wo er als Traceback statt als
        Statusmeldung erscheint."""
        if not os.path.isdir(self.kfg["aufnahme_ordner"]):
            return "Aufnahmeordner existiert nicht"
        if not os.path.exists(self.kfg["modell"]):
            return "Modelldatei (.pt) nicht gefunden"
        # Ein angegebener, aber nicht vorhandener Geometriepfad wurde bisher
        # stillschweigend uebergangen - die Geometrie waere dann abgeleitet
        # worden, obwohl der Nutzer eine mitbringen wollte.
        if self.kfg["geometrie"] and not os.path.exists(self.kfg["geometrie"]):
            return "Geometriedatei nicht gefunden (Feld leeren, um sie abzuleiten)"
        if self.kfg["kalibrierung"] and not os.path.exists(self.kfg["kalibrierung"]):
            return "Kalibrierdatei nicht gefunden"
        return None

    # -------------------------------------------------- Arbeitsfaden
    def _arbeiten(self):
        m = self.warteschlange.put
        k = self.kfg
        try:
            m(("status", ("Modell wird geladen ...", "normal")))
            aus = Auswerter(k["modell"], k["schwelle"], k["geraet"], k["min_flaeche"])
            self.massstab.laden(k["kalibrierung"])
            os.makedirs(k["ergebnis_ordner"], exist_ok=True)
            wache = Ordnerwache(k["aufnahme_ordner"], ab_bestand=True)

            geladen = False
            if k["geometrie"] and os.path.exists(k["geometrie"]):
                g = np.load(k["geometrie"])
                aus.geo = {n: g[n] for n in ("x", "y", "outx", "outy", "s")}
                geladen = True
            m(("status", (f"{aus.geraet_text} | {self.massstab.quelle} | "
                          + ("Geometrie geladen" if geladen
                             else f"warte auf {k['referenz_frames']} eisfreie Referenzframes"), "ok")))

            ref_masken, d0, zaehler, gemessen = [], None, 0, 0
            while self.laeuft:
                neue = wache.neue()
                if not neue:
                    time.sleep(0.25); continue
                for pfad in neue:
                    if not self.laeuft:
                        break
                    zaehler += 1
                    if zaehler % k["jedes_n_te"] != 0:
                        continue
                    img = bild_lesen(pfad)
                    if img is None:
                        m(("status", (f"nicht lesbar: {os.path.basename(pfad)}", "warn"))); continue

                    maske = aus.maske(img)

                    # --- Referenzphase: eisfreie Frames sammeln ---
                    if d0 is None:
                        ref_masken.append(maske)
                        m(("bild", self._overlay(img, maske, None, aus)))
                        m(("status", (f"Referenz {len(ref_masken)}/{k['referenz_frames']} "
                                      f"- noch NICHT spruehen", "warn")))
                        if len(ref_masken) < k["referenz_frames"]:
                            continue
                        mittel = (np.mean(np.stack(ref_masken), axis=0) > 127).astype(np.uint8) * 255
                        if aus.geo is None:
                            aus.geo = geometrie_ableiten(mittel)
                            np.savez(os.path.join(k["ergebnis_ordner"], "geometrie.npz"), **aus.geo)
                        d0 = aus.versatz(mittel, k["such_von"], k["such_bis"])
                        m(("status", (f"{aus.geraet_text} | Referenz steht "
                                      f"({len(aus.geo['x'])} Stuetzstellen) - Versuch kann starten", "ok")))
                        continue

                    # --- Messphase ---
                    t0 = time.time()
                    d = aus.versatz(maske, k["such_von"], k["such_bis"])
                    dicke_px = median_glatt(d - d0, k["glaettung"])
                    dauer = time.time() - t0

                    ppm = self.massstab.px_pro_mm() or k["px_pro_mm"]
                    dicke_mm = dicke_px / ppm if ppm else dicke_px
                    gut = np.isfinite(dicke_mm)
                    maxd = float(np.nanmax(dicke_mm)) if gut.any() else 0.0
                    mittel_d = float(np.nanmean(dicke_mm)) if gut.any() else 0.0

                    gemessen += 1
                    self.ergebnisse.append({
                        "datei": os.path.basename(pfad), "nr": gemessen,
                        "max_mm": round(maxd, 4), "mittel_mm": round(mittel_d, 4),
                        "px_pro_mm": round(ppm, 3), "zeit": time.strftime("%H:%M:%S"),
                        "dicke_mm": [None if not np.isfinite(v) else round(float(v), 4)
                                     for v in dicke_mm],
                    })
                    self._json_schreiben()

                    ov = self._overlay(img, maske, (aus, d), aus)
                    if k["overlays_speichern"]:
                        klein = cv2.resize(ov, (900, int(ov.shape[0] * 900 / ov.shape[1])),
                                           interpolation=cv2.INTER_AREA)
                        cv2.imwrite(os.path.join(k["ergebnis_ordner"],
                                                 f"overlay_{gemessen:05d}.jpg"), klein,
                                    [cv2.IMWRITE_JPEG_QUALITY, 85])
                    m(("bild", ov)); m(("wert", maxd))
                    m(("status", (f"{aus.geraet_text} | Bild {gemessen} | max {maxd:.3f} mm | "
                                  f"mittel {mittel_d:.3f} mm | {dauer*1000:.0f} ms/Bild", "ok")))
            m(("status", ("gestoppt", "normal")))
        except Exception as e:
            import traceback; traceback.print_exc()
            m(("status", (f"Fehler: {e}", "fehler")))
        finally:
            m(("ende", None))

    def _overlay(self, img, maske, mess, aus):
        vis = cv2.cvtColor(np.clip(img.astype(np.float32) * 1.3, 0, 255).astype(np.uint8),
                           cv2.COLOR_GRAY2BGR)
        vis[maske > 0] = (60, 90, 255)                      # erkannte Linie
        if aus.geo is not None:
            g = aus.geo
            for i in range(0, len(g["x"]), 12):
                cv2.circle(vis, (int(g["x"][i]), int(g["y"][i])), 1, (0, 220, 0), -1)
            if mess is not None:
                _, d = mess
                for i in range(0, len(g["x"]), 12):
                    if np.isfinite(d[i]):
                        px = int(g["x"][i] + g["outx"][i] * d[i])
                        py = int(g["y"][i] + g["outy"][i] * d[i])
                        cv2.circle(vis, (px, py), 1, (0, 255, 255), -1)
        return vis

    def _json_schreiben(self):
        try:
            json.dump({"typ": "laser", "einstellungen": self.kfg.werte,
                       "messwerte": self.ergebnisse},
                      open(os.path.join(self.kfg["ergebnis_ordner"], "messwerte.json"),
                           "w", encoding="utf-8"), indent=1, ensure_ascii=False)
        except Exception:
            pass

    def _abholen(self):
        try:
            while True:
                art, nutz = self.warteschlange.get_nowait()
                if art == "bild":
                    self.bild.zeigen(nutz)
                elif art == "wert":
                    self.kurve.anhaengen(nutz)
                elif art == "status":
                    self.status.setzen(*nutz)
                elif art == "ende":
                    self.laeuft = False; self.knopf.configure(text="Start")
        except queue.Empty:
            pass
        self.after(150, self._abholen)


def main():
    wurzel = tk.Tk()
    wurzel.title("Live-Auswertung Laserlinie")
    wurzel.geometry("1360x820")
    try:
        ttk.Style().theme_use("vista")
    except tk.TclError:
        pass
    Anwendung(wurzel)
    wurzel.mainloop()


if __name__ == "__main__":
    main()
