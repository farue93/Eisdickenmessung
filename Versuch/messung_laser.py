# -*- coding: utf-8 -*-
"""
messung_laser.py - Laserlinienmessung am Versuchs-PC.

Zwei Reiter:

  KALIBRIERUNG   Massstabskarte aus Schachbrettbildern, die aus der
                 Dateiauswahl kommen - oder von Hand ueber eine Strecke
                 bekannter Laenge. Gehoert VOR die Messung: dann steht das
                 Brett noch, es ist Zeit, und ein misslungener Versuch kostet
                 nichts.
  MESSUNG        Livebild, eisfreie Referenz, Messung.

Gemessen wird die Eisdicke als Verschiebung der Laserlinie gegenueber dem
EISFREIEN Referenzzustand - dieselbe Groesse wie in der Offline-Auswertung
(laser_v2/unet_v2/apply_unet.py). Die Nulllage und die Geometrie entstehen aus
den ersten eisfreien Bildern des LAUFENDEN Versuchs; eine mitgebrachte
Geometriedatei waere ungueltig, sobald Kamera oder Laser neu ausgerichtet sind.

Die Umrechnung in Millimeter geschieht AN JEDER STUETZSTELLE EINZELN und in der
Richtung, in der gemessen wird - siehe _in_mm(). Ein globaler px/mm-Wert lag auf
der Testgeometrie um bis zu 41 % daneben, die isotrope Naeherung um 24 %, die
richtungsabhaengige Metrik um 0,7 %.

Bewusst nur das U-Net, kein Methodenvergleich: Die Live-Ansicht beantwortet am
Kanal eine einzige Frage - laeuft der Versuch sauber? Der Vergleich gehoert in
die Offline-Auswertung, wo beliebig oft neu gerechnet werden kann. Die
Rohbilder bleiben unangetastet, es geht also nichts verloren.

Start:  start_laser.bat   oder   python messung_laser.py
"""
import os, sys, json, time, threading, queue
import tkinter as tk
from tkinter import ttk

import numpy as np
import cv2

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
from gemeinsam.konfig import Konfig
from gemeinsam.ordnerwache import Ordnerwache, bild_lesen, AUTO
from gemeinsam.massstab import Massstab
from gemeinsam.kalibrier_tab import KalibrierTab
from gemeinsam.anleitung import AnleitungTab
from gemeinsam.kalibrierung import vorgabe
from gemeinsam import geraet as ger
from gemeinsam import gui

# Vorgabewerte aus dem MITGELIEFERTEN Musterbogen (A4, 10 mm) - so
# koennen Druckvorlage und Eingabefeld nicht auseinanderlaufen.
VORGABE_ECKEN, VORGABE_MM = vorgabe("a4")

BILDWAHL = ["immer das neueste (Frames auslassen)", "jedes Bild", "jedes N-te Bild"]

STANDARD = {
    "aufnahme_ordner": "",
    "unterordner": AUTO,
    "ergebnis_ordner": os.path.join(BASE, "ergebnisse", "laser"),
    "modell": os.path.join(BASE, "modelle", "laser.pt"),
    # --- Kalibrierreiter
    "kalibrierung": "",
    "kalibrierung_art": "",
    "feld_mm": VORGABE_MM,
    "ecken_x": VORGABE_ECKEN[0],
    "ecken_y": VORGABE_ECKEN[1],
    "grad": "automatisch",
    "hand_laenge_mm": 100.0,
    # --- Messreiter
    "px_pro_mm": 13.9,
    "bildwahl": BILDWAHL[0],
    "jedes_n_te": 2,
    "referenz_frames": 10,
    "schwelle": 0.5,
    "such_von": -25.0,
    "such_bis": 60.0,
    "glaettung": 9,
    "min_flaeche": 100,
    "overlays_speichern": True,
    "crops_speichern": False,
    "geraet": "auto",
}


# ----------------------------------------------------------------- Geometrie
def mittellinie(maske, mindest=3):
    """Spaltenweiser Schwerpunkt der Laserlinie -> (x, y) der Stuetzstellen.

    GRENZE DES VERFAHRENS: setzt voraus, dass die Linie je Bildspalte hoechstens
    einmal vorkommt, also y als Funktion von x beschreibbar ist. Fuer eine Linie,
    die sich um die Nase herumbiegt und dabei ueber sich selbst zurueckfaellt,
    trifft das nicht zu."""
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
        raise ValueError("zu wenige Linienpunkte - Schwelle, Ausrichtung oder "
                         "Modell pruefen")
    y = glatt(y, glaettung)
    dx, dy = np.gradient(x), np.gradient(y)
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
        # Netzdefinition liegt MITGELIEFERT in netze/ - der Versuchsordner muss
        # ohne das uebrige Repo lauffaehig sein.
        from netze.unet_laser import UNet
        self.torch = torch
        self.geraet, self.geraet_text = ger.geraet_waehlen(geraetwahl)
        self.netz = UNet().to(self.geraet)
        self.netz.load_state_dict(torch.load(modellpfad, map_location=self.geraet))
        self.netz.eval()
        self.schwelle = float(schwelle)
        self.min_flaeche = int(min_flaeche)
        self.geo = None
        self.d0 = None

    @property
    def referenz_bereit(self):
        return self.geo is not None and self.d0 is not None

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

    def ausschnitt(self, img, rand=40):
        """Bildausschnitt um die Linie herum - das, was die Messung benutzt.

        Fuer 'gecroppte Bilder mitspeichern': die Rohbilder liegen ohnehin im
        Kameraordner, der Ausschnitt ist deutlich kleiner und enthaelt genau
        den Bereich, aus dem der Messwert stammt."""
        if self.geo is None:
            return img
        g = self.geo
        H, W = img.shape
        x0 = int(max(0, g["x"].min() - rand))
        x1 = int(min(W, g["x"].max() + rand))
        y0 = int(max(0, g["y"].min() - rand))
        y1 = int(min(H, g["y"].max() + rand))
        return img[y0:y1, x0:x1]


# ----------------------------------------------------------------- Messreiter
class Messreiter(ttk.Frame):
    def __init__(self, eltern, kfg):
        super().__init__(eltern, padding=10)
        self.kfg = kfg
        self.warteschlange = queue.Queue()
        self.befehle = queue.Queue()
        self.verbunden = False
        self.modus = "vorschau"           # vorschau | referenz | messung
        # Jeder Arbeitsfaden bekommt eine Generation. Wer nicht mehr die
        # aktuelle hat, beendet sich. Ohne das koennen bei schnellem Trennen
        # und Wiederverbinden ZWEI Faeden gleichzeitig laufen: Sie teilen sich
        # die Befehlswarteschlange, aber jeder hat seinen eigenen Auswerter -
        # dann baut der eine die Referenz auf und der andere lehnt den
        # Messbefehl ab, weil ihm die Referenz fehlt.
        self._generation = 0
        self.ergebnisse = []
        self.massstab = Massstab()
        self.lauf_ordner = None
        self._bauen()
        self._massstab_anzeigen(self.massstab.laden(self.kfg["kalibrierung"]))
        self.after(200, self._abholen)

    # -------------------------------------------------- Aufbau
    def _bauen(self):
        links = ttk.Frame(self); links.pack(side="left", fill="y", padx=(0, 12))
        rechts = ttk.Frame(self); rechts.pack(side="left", fill="both", expand=True)

        a = gui.abschnitt(links, "1  Kamera")
        self.f_auf = gui.Pfadfeld(a, "Aufnahmeordner", self.kfg["aufnahme_ordner"], "ordner")
        self.f_auf.pack(fill="x", pady=2)
        self.f_unter = gui.Auswahl(a, "Unterordner", self.kfg["unterordner"],
                                   [AUTO], nachladen=self._unterordner_laden)
        self.f_unter.pack(fill="x", pady=2)
        ttk.Label(a, text="Bilder liegen in Unterordnern des Aufnahmeordners",
                  foreground="#666").pack(anchor="w")
        self.f_erg = gui.Pfadfeld(a, "Ergebnisordner", self.kfg["ergebnis_ordner"], "ordner")
        self.f_erg.pack(fill="x", pady=2)
        self.k_verbinden = ttk.Button(a, text="Kamera verbinden", command=self._verbinden)
        self.k_verbinden.pack(fill="x", pady=(6, 2))

        b = gui.abschnitt(links, "2  Massstab")
        self.f_kal = gui.Pfadfeld(b, "Kalibrierung (.npz)", self.kfg["kalibrierung"],
                                  "datei", [("NumPy", "*.npz"), ("Alle", "*.*")])
        self.f_kal.pack(fill="x", pady=2)
        ttk.Button(b, text="laden", command=self._kal_laden).pack(fill="x", pady=2)
        ttk.Label(b, text="wird im Reiter 'Kalibrierung' erzeugt",
                  foreground="#666").pack(anchor="w")
        self.f_ppm = gui.Feld(b, "px/mm ersatzweise", self.kfg["px_pro_mm"], 8, "",
                              "nur ohne Kalibrierung")
        self.f_ppm.pack(fill="x", pady=2)

        c = gui.abschnitt(links, "3  Eisfrei-Referenz")
        self.f_ref = gui.Feld(c, "Referenzframes", self.kfg["referenz_frames"], 8, "",
                              "eisfrei!")
        self.f_ref.pack(fill="x", pady=2)
        ttk.Button(c, text="Eisfrei-Referenz aufnehmen",
                   command=lambda: self._befehl("referenz")).pack(fill="x", pady=(6, 2))

        d = gui.abschnitt(links, "4  Messung")
        self.f_bildwahl = gui.Auswahl(d, "Bildauswahl", self.kfg["bildwahl"], BILDWAHL, 26)
        self.f_bildwahl.pack(fill="x", pady=2)
        self.f_n = gui.Feld(d, "N (fuer jedes N-te)", self.kfg["jedes_n_te"], 8)
        self.f_n.pack(fill="x", pady=2)
        self.v_overlay = tk.BooleanVar(value=self.kfg["overlays_speichern"])
        ttk.Checkbutton(d, text="Overlay-Bilder mitspeichern",
                        variable=self.v_overlay).pack(anchor="w", pady=2)
        self.v_crops = tk.BooleanVar(value=self.kfg["crops_speichern"])
        ttk.Checkbutton(d, text="gecroppte Bilder mitspeichern",
                        variable=self.v_crops).pack(anchor="w", pady=2)
        self.k_messen = ttk.Button(d, text="Messung starten", command=self._messen)
        self.k_messen.pack(fill="x", pady=(6, 2))

        e = gui.abschnitt(links, "Feinjustierung")
        self.f_schwelle = gui.Feld(e, "Schwelle", self.kfg["schwelle"], 8, "",
                                   "hoeher = strenger")
        self.f_schwelle.pack(fill="x", pady=2)
        self.f_von = gui.Feld(e, "Suchbereich von", self.kfg["such_von"], 8, "px")
        self.f_von.pack(fill="x", pady=2)
        self.f_bis = gui.Feld(e, "Suchbereich bis", self.kfg["such_bis"], 8, "px")
        self.f_bis.pack(fill="x", pady=2)
        self.f_glatt = gui.Feld(e, "Glaettung", self.kfg["glaettung"], 8, "px")
        self.f_glatt.pack(fill="x", pady=2)
        self.f_minfl = gui.Feld(e, "min. Linienflaeche", self.kfg["min_flaeche"], 8, "px")
        self.f_minfl.pack(fill="x", pady=2)
        self.v_geraet = tk.StringVar(value=self.kfg["geraet"])
        z = ttk.Frame(e); z.pack(fill="x", pady=2)
        ttk.Label(z, text="Rechenwerk", width=22, anchor="w").pack(side="left")
        ttk.Combobox(z, textvariable=self.v_geraet, values=["auto", "cpu"],
                     width=8, state="readonly").pack(side="left")
        self.f_modell = gui.Pfadfeld(e, "U-Net (.pt)", self.kfg["modell"], "datei",
                                     [("PyTorch-Modell", "*.pt"), ("Alle", "*.*")])
        self.f_modell.pack(fill="x", pady=2)
        ttk.Button(e, text="Einstellungen uebernehmen",
                   command=self._uebernehmen).pack(fill="x", pady=(6, 2))

        # ---- rechte Spalte
        self.bild = gui.Bildflaeche(rechts); self.bild.pack(fill="both", expand=True)
        self.kurve = gui.Verlauf(rechts, titel="max. Eisdicke mm")
        self.kurve.pack(fill="x", pady=(8, 0))
        zust = ttk.LabelFrame(rechts, text="Zustand", padding=(10, 6))
        zust.pack(fill="x", pady=(8, 0))
        self.z_kamera = gui.Zustand(zust, "Kamera"); self.z_kamera.pack(fill="x")
        self.z_mass = gui.Zustand(zust, "Massstab"); self.z_mass.pack(fill="x")
        self.z_ref = gui.Zustand(zust, "Eisfrei-Referenz"); self.z_ref.pack(fill="x")
        self.z_mess = gui.Zustand(zust, "Messung"); self.z_mess.pack(fill="x")
        self.z_kamera.setzen("nicht verbunden")
        self.z_ref.setzen("fehlt - vor dem Spruehen aufnehmen")
        self.z_mess.setzen("laeuft nicht")
        self.status = gui.Statuszeile(rechts); self.status.pack(fill="x", pady=(8, 0))
        self.status.setzen("Aufnahmeordner waehlen, dann 'Kamera verbinden'")

    # -------------------------------------------------- Einstellungen
    def _sammeln(self):
        k = self.kfg
        k["aufnahme_ordner"] = self.f_auf.text()
        k["unterordner"] = self.f_unter.text() or AUTO
        k["ergebnis_ordner"] = self.f_erg.text()
        k["modell"] = self.f_modell.text()
        k["kalibrierung"] = self.f_kal.text()
        k["px_pro_mm"] = self.f_ppm.zahl(13.9)
        k["bildwahl"] = self.f_bildwahl.text()
        k["jedes_n_te"] = max(1, self.f_n.zahl(2, ganz=True))
        k["referenz_frames"] = max(1, self.f_ref.zahl(10, ganz=True))
        k["schwelle"] = self.f_schwelle.zahl(0.5)
        k["such_von"] = self.f_von.zahl(-25.0)
        k["such_bis"] = self.f_bis.zahl(60.0)
        k["glaettung"] = max(1, self.f_glatt.zahl(9, ganz=True))
        k["min_flaeche"] = max(1, self.f_minfl.zahl(100, ganz=True))
        k["overlays_speichern"] = bool(self.v_overlay.get())
        k["crops_speichern"] = bool(self.v_crops.get())
        k["geraet"] = self.v_geraet.get()
        k.speichern()

    def _uebernehmen(self):
        self._sammeln()
        self.status.setzen("Einstellungen uebernommen und gespeichert", "ok")

    def _unterordner_laden(self):
        ordner = self.f_auf.text()
        if not os.path.isdir(ordner):
            self.status.setzen("Aufnahmeordner existiert nicht", "fehler"); return
        w = Ordnerwache(ordner, AUTO, ab_bestand=True)
        self.f_unter.fuellen([AUTO] + w.ordner_liste())
        self.status.setzen(f"{len(w.ordner_liste())} Unterordner gefunden, "
                           f"aktiv waere: {w.aktiv_kurz}", "ok")

    def kalibrierung_setzen(self, npz):
        """Wird vom Kalibrierreiter gerufen."""
        self.f_kal.setzen(npz)
        self._kal_laden()

    def _kal_laden(self):
        self._sammeln()
        self._massstab_anzeigen(self.massstab.laden(self.kfg["kalibrierung"]))

    def _massstab_anzeigen(self, ok):
        if ok:
            art = self.kfg.werte.get("kalibrierung_art") or "Karte"
            self.z_mass.setzen(f"{self.massstab.quelle} | "
                               f"{self.massstab.px_pro_mm():.2f} px/mm | {art}", "ok")
        else:
            self.z_mass.setzen(f"keine - Ersatzwert {self.kfg['px_pro_mm']} px/mm", "warn")

    # -------------------------------------------------- Steuerung
    def _befehl(self, name):
        if not self.verbunden:
            self.status.setzen("erst 'Kamera verbinden'", "fehler"); return
        self._sammeln()
        self.befehle.put(name)

    def _verbinden(self):
        if self.verbunden:
            self.verbunden = False
            self._generation += 1
            self.k_verbinden.configure(text="Kamera verbinden")
            return
        self._sammeln()
        if not os.path.isdir(self.kfg["aufnahme_ordner"]):
            self.status.setzen("Aufnahmeordner existiert nicht", "fehler"); return
        if not os.path.exists(self.kfg["modell"]):
            self.status.setzen("Modelldatei (.pt) nicht gefunden", "fehler"); return
        if self.kfg["kalibrierung"] and not os.path.exists(self.kfg["kalibrierung"]):
            self.status.setzen("Kalibrierdatei nicht gefunden", "fehler"); return
        self._generation += 1
        while not self.befehle.empty():      # Befehle des alten Fadens verwerfen
            self.befehle.get_nowait()
        self.verbunden = True
        self.k_verbinden.configure(text="Kamera trennen")
        threading.Thread(target=self._arbeiten, args=(self._generation,),
                         name=f"messung-laser-{self._generation}",
                         daemon=True).start()

    def _messen(self):
        if self.modus == "messung":
            self.modus = "vorschau"
            self.k_messen.configure(text="Messung starten")
            self.z_mess.setzen("gestoppt", "warn")
            return
        self._befehl("messung")

    # -------------------------------------------------- Arbeitsfaden
    def _arbeiten(self, generation):
        m = self.warteschlange.put
        k = self.kfg
        aus = None

        def laeuft():
            return self.verbunden and self._generation == generation

        try:
            wache = Ordnerwache(k["aufnahme_ordner"], k["unterordner"], ab_bestand=True)
            m(("kamera", (f"verbunden | Ordner: {wache.aktiv_kurz}", "ok")))
            m(("status", ("Modell wird geladen ...", "normal")))
            aus = Auswerter(k["modell"], k["schwelle"], k["geraet"], k["min_flaeche"])
            m(("status", (f"{aus.geraet_text} | Livebild - Laser ausrichten, "
                          f"dann Eisfrei-Referenz", "ok")))

            ref_masken, puffer = [], []
            zaehler = gemessen = 0
            tempo = None

            while laeuft():
                try:
                    befehl = self.befehle.get_nowait()
                except queue.Empty:
                    befehl = None

                if befehl == "referenz":
                    ref_masken = []
                    aus.geo = aus.d0 = None
                    self.modus = "referenz"
                    m(("ref", (f"sammelt 0/{k['referenz_frames']} - NICHT spruehen", "warn")))
                elif befehl == "messung":
                    if not aus.referenz_bereit:
                        m(("status", ("erst Eisfrei-Referenz aufnehmen", "fehler")))
                    else:
                        self.lauf_ordner = self._lauf_ordner_anlegen(wache, aus)
                        self.ergebnisse, gemessen = [], 0
                        m(("leeren", None))
                        self.modus = "messung"
                        m(("mess", (f"laeuft -> {os.path.basename(self.lauf_ordner)}", "ok")))
                        m(("knopf", "Messung stoppen"))

                # ---- Bild holen
                if not puffer:
                    if k["bildwahl"] == BILDWAHL[0]:
                        p = wache.neueste()
                        puffer = [p] if p else []
                    else:
                        puffer = wache.neue()
                if not puffer:
                    time.sleep(0.2)
                    continue
                pfad = puffer.pop(0)
                zaehler += 1
                if k["bildwahl"] == BILDWAHL[2] and zaehler % k["jedes_n_te"] != 0:
                    continue
                img = bild_lesen(pfad)
                if img is None:
                    m(("status", (f"nicht lesbar: {os.path.basename(pfad)}", "warn")))
                    continue

                # Live einstellbar: waehrend die Anzeige laeuft nachjustieren zu
                # koennen ist am Kanal mehr wert als ein Neustart je Aenderung.
                aus.schwelle = k["schwelle"]
                aus.min_flaeche = k["min_flaeche"]
                maske = aus.maske(img)

                # ---- Referenzphase
                if self.modus == "referenz":
                    ref_masken.append(maske)
                    m(("bild", self._overlay(img, maske, None, aus)))
                    if len(ref_masken) < k["referenz_frames"]:
                        m(("ref", (f"sammelt {len(ref_masken)}/{k['referenz_frames']} "
                                   f"- NICHT spruehen", "warn")))
                        continue
                    mittel = (np.mean(np.stack(ref_masken), axis=0) > 127).astype(np.uint8) * 255
                    try:
                        aus.geo = geometrie_ableiten(mittel)
                    except ValueError as e:
                        self.modus = "vorschau"
                        m(("ref", (str(e), "fehler")))
                        continue
                    aus.d0 = aus.versatz(mittel, k["such_von"], k["such_bis"])
                    self.modus = "vorschau"
                    m(("ref", (f"steht - {len(aus.geo['x'])} Stuetzstellen, "
                               f"{k['referenz_frames']} Frames", "ok")))
                    m(("status", ("Referenz steht - Messung starten, dann spruehen", "ok")))
                    continue

                # ---- Vorschau
                if self.modus != "messung":
                    m(("bild", self._overlay(img, maske, None, aus)))
                    m(("kamera", (f"Ordner {wache.aktiv_kurz} | {os.path.basename(pfad)}"
                                  + (f" | {wache.uebersprungen} uebersprungen"
                                     if wache.uebersprungen else ""), "ok")))
                    continue

                # ---- Messphase
                t0 = time.time()
                d = aus.versatz(maske, k["such_von"], k["such_bis"])
                dicke_px = median_glatt(d - aus.d0, k["glaettung"])
                dauer = time.time() - t0
                tempo = dauer if tempo is None else 0.7 * tempo + 0.3 * dauer

                dicke_mm, ppm_text = self._in_mm(dicke_px, aus)
                gut = np.isfinite(dicke_mm)
                maxd = float(np.nanmax(dicke_mm)) if gut.any() else 0.0
                mittel_d = float(np.nanmean(dicke_mm)) if gut.any() else 0.0

                gemessen += 1
                self.ergebnisse.append({
                    "datei": os.path.basename(pfad), "quelle": wache.aktiv_kurz,
                    "nr": gemessen, "max_mm": round(maxd, 4),
                    "mittel_mm": round(mittel_d, 4), "massstab": ppm_text,
                    "zeit": time.strftime("%H:%M:%S"),
                    "dicke_mm": [None if not np.isfinite(v) else round(float(v), 4)
                                 for v in dicke_mm],
                })
                self._json_schreiben()

                ov = self._overlay(img, maske, d, aus)
                if k["overlays_speichern"]:
                    klein = cv2.resize(ov, (900, int(ov.shape[0] * 900 / ov.shape[1])),
                                       interpolation=cv2.INTER_AREA)
                    cv2.imwrite(os.path.join(self.lauf_ordner,
                                             f"overlay_{gemessen:05d}.jpg"), klein,
                                [cv2.IMWRITE_JPEG_QUALITY, 85])
                if k["crops_speichern"]:
                    os.makedirs(os.path.join(self.lauf_ordner, "crops"), exist_ok=True)
                    cv2.imwrite(os.path.join(self.lauf_ordner, "crops",
                                             f"crop_{gemessen:05d}.png"),
                                aus.ausschnitt(img))
                m(("bild", ov)); m(("wert", maxd))
                m(("status", (f"{aus.geraet_text} | Bild {gemessen} | max {maxd:.3f} mm | "
                              f"mittel {mittel_d:.3f} mm | {ppm_text} | "
                              f"{tempo*1000:.0f} ms/Bild"
                              + (f" | {wache.uebersprungen} uebersprungen"
                                 if wache.uebersprungen else ""), "ok")))
            if self._generation == generation:
                m(("status", ("Kamera getrennt", "normal")))
        except Exception as e:
            import traceback; traceback.print_exc()
            if self._generation == generation:
                m(("status", (f"Fehler: {e}", "fehler")))
        finally:
            # Ein abgeloester Faden darf die Oberflaeche nicht zuruecksetzen -
            # sonst schaltet er den gerade gestarteten Nachfolger wieder aus.
            if self._generation == generation:
                m(("ende", None))

    # -------------------------------------------------- Teilschritte
    def _in_mm(self, dicke_px, aus):
        """Dicke in Pixeln -> Millimeter, an JEDER Stuetzstelle mit dem dort
        geltenden Massstab.

        Die Eisdicke ist die Verschiebung der Linie entlang der Normalen: von
        der eisfreien Nulllage d0 bis zur gemessenen Lage. Genau diese Strecke
        wird durch die Massstabskarte geschickt, Pixel fuer Pixel. Ein globaler
        px/mm-Wert waere am Bildrand systematisch falsch - dort, wo die Kamera
        am schraegsten auf die Nase blickt und wo der interessante Teil des
        Eisansatzes liegt.

        Ohne Kalibrierung bleibt der eingetragene Ersatzwert; die Statuszeile
        weist das aus, damit niemand Pixel fuer Millimeter haelt."""
        g = aus.geo
        if self.massstab.vorhanden and g is not None:
            start_x = g["x"] + g["outx"] * aus.d0
            start_y = g["y"] + g["outy"] * aus.d0
            ende_x = start_x + g["outx"] * dicke_px
            ende_y = start_y + g["outy"] * dicke_px
            laenge = self.massstab.laenge_mm(np.stack([start_x, start_y], axis=-1),
                                             np.stack([ende_x, ende_y], axis=-1))
            # laenge_mm liefert Betraege - das Vorzeichen der Verschiebung
            # (Eis waechst nach aussen, Rueckgang nach innen) muss erhalten
            # bleiben, sonst wuerde jede Abweichung als Eis gezaehlt.
            mm = np.sign(dicke_px) * laenge
            lok = self.massstab.px_pro_mm_lokal(g["x"], g["y"])
            return mm, f"Karte {lok.min():.2f}-{lok.max():.2f} px/mm"
        ppm = self.kfg["px_pro_mm"]
        return (dicke_px / ppm if ppm else dicke_px), f"Ersatzwert {ppm} px/mm"

    def _lauf_ordner_anlegen(self, wache, aus):
        name = time.strftime("%Y%m%d_%H%M%S")
        if wache.aktiv_kurz != ".":
            name += "_" + os.path.basename(wache.aktiv_kurz)
        p = os.path.join(self.kfg["ergebnis_ordner"], name)
        os.makedirs(p, exist_ok=True)
        # Die abgeleitete Geometrie gehoert zum Lauf: ohne sie sind die
        # gespeicherten Dickenprofile spaeter keiner Bogenlaenge zuzuordnen.
        if aus.geo is not None:
            np.savez(os.path.join(p, "geometrie.npz"), **aus.geo)
        return p

    def _overlay(self, img, maske, d, aus):
        vis = cv2.cvtColor(np.clip(img.astype(np.float32) * 1.3, 0, 255).astype(np.uint8),
                           cv2.COLOR_GRAY2BGR)
        vis[maske > 0] = (60, 90, 255)                      # erkannte Linie
        if aus.geo is not None:
            g = aus.geo
            for i in range(0, len(g["x"]), 12):
                cv2.circle(vis, (int(g["x"][i]), int(g["y"][i])), 1, (0, 220, 0), -1)
            if d is not None:
                for i in range(0, len(g["x"]), 12):
                    if np.isfinite(d[i]):
                        px = int(g["x"][i] + g["outx"][i] * d[i])
                        py = int(g["y"][i] + g["outy"][i] * d[i])
                        cv2.circle(vis, (px, py), 1, (0, 255, 255), -1)
        return vis

    def _json_schreiben(self):
        if not self.lauf_ordner:
            return
        try:
            json.dump({"typ": "laser", "einstellungen": self.kfg.werte,
                       "messwerte": self.ergebnisse},
                      open(os.path.join(self.lauf_ordner, "messwerte.json"),
                           "w", encoding="utf-8"), indent=1, ensure_ascii=False)
        except Exception:
            pass

    # -------------------------------------------------- GUI-Aktualisierung
    def _abholen(self):
        try:
            while True:
                art, nutz = self.warteschlange.get_nowait()
                if art == "bild":
                    self.bild.zeigen(nutz)
                elif art == "wert":
                    self.kurve.anhaengen(nutz)
                elif art == "leeren":
                    self.kurve.leeren()
                elif art == "status":
                    self.status.setzen(*nutz)
                elif art == "kamera":
                    self.z_kamera.setzen(*nutz)
                elif art == "ref":
                    self.z_ref.setzen(*nutz)
                elif art == "mess":
                    self.z_mess.setzen(*nutz)
                elif art == "knopf":
                    self.k_messen.configure(text=nutz)
                elif art == "ende":
                    self.verbunden = False
                    self.modus = "vorschau"
                    self.k_verbinden.configure(text="Kamera verbinden")
                    self.k_messen.configure(text="Messung starten")
                    self.z_kamera.setzen("nicht verbunden")
                    self.z_mess.setzen("laeuft nicht")
        except queue.Empty:
            pass
        self.after(150, self._abholen)


def main():
    wurzel = tk.Tk()
    wurzel.title("Laserlinienmessung")
    wurzel.geometry("1480x940")
    try:
        ttk.Style().theme_use("vista")
    except tk.TclError:
        pass
    kfg = Konfig(os.path.join(BASE, "einstellungen_laser.json"), STANDARD)

    reiter = ttk.Notebook(wurzel)
    reiter.pack(fill="both", expand=True)
    mess = Messreiter(reiter, kfg)
    kalib = KalibrierTab(reiter, kfg, "laser", "Laser", "strecke",
                         bei_uebernahme=mess.kalibrierung_setzen,
                         startordner=lambda: kfg["aufnahme_ordner"])
    reiter.add(kalib, text="Kalibrierung")
    reiter.add(mess, text="Messung")
    reiter.add(AnleitungTab(reiter), text="Anleitung")
    wurzel.mainloop()


if __name__ == "__main__":
    main()
