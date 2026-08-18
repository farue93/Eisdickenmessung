# -*- coding: utf-8 -*-
"""
messung_flaeche.py - Eisflaechenmessung am Versuchs-PC: Kalibrierung,
Messbereich, eisfreie Referenz und Live-Messung in EINER Oberflaeche, fuer
BEIDE Flaechenkameras (unten und oben).

Die beiden Kameras liegen auf je einem Reiter und sind vollstaendig getrennt:
eigener Aufnahmeordner, eigener Massstab, eigener Messbereich, eigene Referenz,
eigener Arbeitsfaden. Beide koennen gleichzeitig laufen; der Reiter waehlt nur,
welche gerade angezeigt wird. Getrennt sein MUESSEN sie, weil jede Kamera ihren
eigenen Blickwinkel hat - ein gemeinsamer Massstab oder ein gemeinsamer
Nullzustand waere fuer beide falsch.

Ablauf je Kamera - die Schritte sind auch die Knoepfe:

  1. Kamera verbinden        Livebild aus dem Aufnahmeordner.
  2. Massstab kalibrieren    Ein Bild mit aufgeklebtem Schachbrett genuegt.
  3. Messbereich festlegen   Rechteck auf dem Panel; Bezugsflaeche der Prozente.
  4. Eisfrei-Referenz        Sauberer Ausgangszustand aus N eisfreien Bildern.
  5. Messung starten         Ab hier darf gespruht werden.

DAS FLAECHENMODELL HAT ZWEI EINGANGSKANAELE: das Bild und die Abweichung vom
eisfreien Ausgangszustand. Der zweite Kanal beantwortet die Frage, die das Bild
allein nicht beantworten kann - ist diese Struktur NEU? Ohne ihn meldete das
Netz auf dem nachweislich eisfreien Startbild im unteren Panelband 22 % Eis,
weil dort ein Rueckstandsband liegt, das wie feines Eisgefuege aussieht.
Deshalb wird ohne Referenz nicht gemessen.

Start:  start_flaeche.bat   oder   python messung_flaeche.py
"""
import os, sys, json, time, threading, queue, subprocess
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
KAMERAS = [("unten", "Kamera unten"), ("oben", "Kamera oben")]


def standard(kennung):
    return {
        "aufnahme_ordner": "",
        "unterordner": AUTO,
        "ergebnis_ordner": os.path.join(BASE, "ergebnisse", f"flaeche_{kennung}"),
        "modell": os.path.join(BASE, "modelle", "flaeche.pt"),
        "messbereich": "",
        # --- Kalibrierreiter
        "kalibrierung": "",
        "kalibrierung_art": "",
        "feld_mm": VORGABE_MM,
        "ecken_x": VORGABE_ECKEN[0],
        "ecken_y": VORGABE_ECKEN[1],
        "grad": "automatisch",
        "hand_breite_mm": 100.0,
        "hand_hoehe_mm": 100.0,
        # --- Messreiter
        "bildwahl": BILDWAHL[0],
        "jedes_n_te": 2,
        "referenz_frames": 10,
        "schwelle": 0.5,
        "overlays_speichern": True,
        "crops_speichern": False,
        "geraet": "auto",
    }


# ----------------------------------------------------------------- Auswertung
class Auswerter:
    """Kapselt Modell, Messbereich und Referenzzustand. Laeuft im Arbeitsfaden."""

    def __init__(self, modellpfad, schwelle, geraetwahl):
        import torch
        # Netzdefinition liegt MITGELIEFERT in netze/ - der Versuchsordner muss
        # ohne das uebrige Repo lauffaehig sein.
        from netze.unet_flaeche import UNet, vorhersage_gekachelt
        self.torch = torch
        self._vorhersage = vorhersage_gekachelt
        self.geraet, self.geraet_text = ger.geraet_waehlen(geraetwahl)
        d = torch.load(modellpfad, map_location=self.geraet)
        # Kanalzahl kommt aus der Modelldatei, nicht aus einer Einstellung: so
        # kann nie ein 1-Kanal-Modell mit 2-Kanal-Eingang gefuettert werden.
        self.kanaele = int(d.get("kanaele", 1))
        self.netz = UNet(kanaele=self.kanaele).to(self.geraet)
        self.netz.load_state_dict(d["state"])
        self.netz.eval()
        self.schwelle = float(d.get("schwelle", schwelle))
        self.clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(16, 16))
        self.panel = None
        self.bbox = None
        self.soll = 0
        self._puffer = []
        self.median = None
        self.mad = None

    # ---- Messbereich
    def panel_laden(self, npz_pfad):
        if not npz_pfad or not os.path.exists(npz_pfad):
            return None
        p = np.load(npz_pfad)
        self.bbox = tuple(int(v) for v in p["bbox"])
        maske = cv2.erode(p["maske"],
                          cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (51, 51)))
        x0, y0, x1, y1 = self.bbox
        self.panel = maske[y0:y1, x0:x1] > 0
        # Der Referenzzustand gilt fuer den alten Zuschnitt - mit neuem
        # Messbereich passt er nicht mehr und muss neu aufgenommen werden.
        self.median = self.mad = None
        self._puffer = []
        return int(self.panel.sum())

    def zuschneiden(self, img):
        if self.bbox is None:
            return img
        x0, y0, x1, y1 = self.bbox
        return img[y0:y1, x0:x1]

    # ---- Referenz
    @property
    def referenz_bereit(self):
        return self.kanaele < 2 or self.median is not None

    def referenz_start(self, anzahl):
        self.soll = int(anzahl)
        self._puffer = []
        self.median = self.mad = None

    def referenz_sammeln(self, img):
        """Einen eisfreien Frame hinzufuegen -> (fertig, anzahl)."""
        self._puffer.append(self.zuschneiden(img).astype(np.float32))
        if len(self._puffer) < self.soll:
            return False, len(self._puffer)
        stapel = np.stack(self._puffer)
        self.median = np.median(stapel, axis=0)
        # MAD statt Standardabweichung: unempfindlich gegen einzelne Ausreisser,
        # etwa einen Frame, in dem doch schon etwas zu sehen war.
        self.mad = np.median(np.abs(stapel - self.median), axis=0)
        self._puffer = []
        return True, self.soll

    def _abweichung(self, crop):
        """Robuster z-Wert gegen den Referenzzustand, auf 0..1 gedeckelt.
        Bewusst auf dem ROHBILD - CLAHE arbeitet je Bild und je Kachel, zwei
        behandelte Aufnahmen stehen danach nicht mehr auf gemeinsamer Skala."""
        z = np.abs(crop.astype(np.float32) - self.median) / (1.4826 * self.mad + 3.0)
        return np.clip(z / 8.0, 0.0, 1.0).astype(np.float32)

    # ---- Messung
    def auswerten(self, img):
        """-> (eismaske, crop). Vorverarbeitung identisch zum Training: CLAHE,
        dann auf 0..1 - sonst sieht das Netz etwas anderes als gelernt."""
        crop = self.zuschneiden(img)
        norm = self.clahe.apply(crop).astype(np.float32) / 255.0
        eingang = norm if self.kanaele < 2 else np.stack([norm, self._abweichung(crop)])
        p = self._vorhersage(self.netz, eingang, self.geraet)
        eis = p > self.schwelle
        if self.panel is not None and self.panel.shape == eis.shape:
            eis &= self.panel
        return eis, crop

    @property
    def bezugsflaeche_px(self):
        return int(self.panel.sum()) if self.panel is not None else None


def overlay_bauen(crop, eis):
    vis = cv2.cvtColor(np.clip(crop.astype(np.float32) * 1.35, 0, 255).astype(np.uint8),
                       cv2.COLOR_GRAY2BGR)
    ov = vis.copy()
    ov[eis] = (255, 90, 30)
    return cv2.addWeighted(vis, 0.55, ov, 0.45, 0)


# ----------------------------------------------------------------- Ein Reiter
class Kamerapanel(ttk.Frame):
    """Eine vollstaendige Kamera: Einstellungen, Anzeige, eigener Arbeitsfaden."""

    def __init__(self, eltern, kennung, titel):
        super().__init__(eltern, padding=10)
        self.kennung, self.titel = kennung, titel
        self.kfg = Konfig(os.path.join(BASE, f"einstellungen_flaeche_{kennung}.json"),
                          standard(kennung))
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
        self.letzter_pfad = None
        self._bauen()
        self._massstab_anzeigen(self.massstab.laden(self.kfg["kalibrierung"]))
        if self.kfg["messbereich"] and os.path.exists(self.kfg["messbereich"]):
            self.z_bereich.setzen(os.path.basename(self.kfg["messbereich"]), "ok")
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
        ttk.Label(b, text=f"wird im Reiter 'Kalibrierung {self.kennung}' erzeugt",
                  foreground="#666").pack(anchor="w")

        c = gui.abschnitt(links, "3  Messbereich")
        self.f_bereich = gui.Pfadfeld(c, "Messbereich (.npz)", self.kfg["messbereich"],
                                      "datei", [("NumPy", "*.npz"), ("Alle", "*.*")])
        self.f_bereich.pack(fill="x", pady=2)
        ttk.Button(c, text="Messbereich festlegen ...",
                   command=self._bereich_setzen).pack(fill="x", pady=(6, 2))
        ttk.Button(c, text="Datei uebernehmen",
                   command=lambda: self._befehl("bereich")).pack(fill="x", pady=2)

        d = gui.abschnitt(links, "4  Eisfrei-Referenz")
        self.f_ref = gui.Feld(d, "Referenzframes", self.kfg["referenz_frames"], 8, "",
                              "eisfrei!")
        self.f_ref.pack(fill="x", pady=2)
        ttk.Button(d, text="Eisfrei-Referenz aufnehmen",
                   command=lambda: self._befehl("referenz")).pack(fill="x", pady=(6, 2))

        e = gui.abschnitt(links, "5  Messung")
        self.f_bildwahl = gui.Auswahl(e, "Bildauswahl", self.kfg["bildwahl"], BILDWAHL, 26)
        self.f_bildwahl.pack(fill="x", pady=2)
        self.f_n = gui.Feld(e, "N (fuer jedes N-te)", self.kfg["jedes_n_te"], 8)
        self.f_n.pack(fill="x", pady=2)
        self.v_overlay = tk.BooleanVar(value=self.kfg["overlays_speichern"])
        ttk.Checkbutton(e, text="Overlay-Bilder mitspeichern",
                        variable=self.v_overlay).pack(anchor="w", pady=2)
        self.v_crops = tk.BooleanVar(value=self.kfg["crops_speichern"])
        ttk.Checkbutton(e, text="gecroppte Bilder mitspeichern",
                        variable=self.v_crops).pack(anchor="w", pady=2)
        self.k_messen = ttk.Button(e, text="Messung starten", command=self._messen)
        self.k_messen.pack(fill="x", pady=(6, 2))

        f = gui.abschnitt(links, "Feinjustierung")
        self.f_schwelle = gui.Feld(f, "Schwelle", self.kfg["schwelle"], 8, "",
                                   "0..1, hoeher = strenger")
        self.f_schwelle.pack(fill="x", pady=2)
        self.v_geraet = tk.StringVar(value=self.kfg["geraet"])
        z3 = ttk.Frame(f); z3.pack(fill="x", pady=2)
        ttk.Label(z3, text="Rechenwerk", width=22, anchor="w").pack(side="left")
        ttk.Combobox(z3, textvariable=self.v_geraet, values=["auto", "cpu"],
                     width=8, state="readonly").pack(side="left")
        self.f_modell = gui.Pfadfeld(f, "U-Net (.pt)", self.kfg["modell"], "datei",
                                     [("PyTorch-Modell", "*.pt"), ("Alle", "*.*")])
        self.f_modell.pack(fill="x", pady=2)
        ttk.Button(f, text="Einstellungen uebernehmen",
                   command=self._uebernehmen).pack(fill="x", pady=(6, 2))

        # ---- rechte Spalte
        self.bild = gui.Bildflaeche(rechts); self.bild.pack(fill="both", expand=True)
        self.kurve = gui.Verlauf(rechts, titel="Bedeckungsgrad %")
        self.kurve.pack(fill="x", pady=(8, 0))
        zust = ttk.LabelFrame(rechts, text=f"Zustand - {self.titel}", padding=(10, 6))
        zust.pack(fill="x", pady=(8, 0))
        self.z_kamera = gui.Zustand(zust, "Kamera"); self.z_kamera.pack(fill="x")
        self.z_mass = gui.Zustand(zust, "Massstab"); self.z_mass.pack(fill="x")
        self.z_bereich = gui.Zustand(zust, "Messbereich"); self.z_bereich.pack(fill="x")
        self.z_ref = gui.Zustand(zust, "Eisfrei-Referenz"); self.z_ref.pack(fill="x")
        self.z_mess = gui.Zustand(zust, "Messung"); self.z_mess.pack(fill="x")
        self.z_kamera.setzen("nicht verbunden")
        self.z_bereich.setzen("nicht gesetzt - Prozente beziehen sich aufs Vollbild", "warn")
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
        k["messbereich"] = self.f_bereich.text()
        k["bildwahl"] = self.f_bildwahl.text()
        k["jedes_n_te"] = max(1, self.f_n.zahl(2, ganz=True))
        k["referenz_frames"] = max(1, self.f_ref.zahl(10, ganz=True))
        k["schwelle"] = self.f_schwelle.zahl(0.5)
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
        """Wird vom zugehoerigen Kalibrierreiter gerufen."""
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
            self.z_mass.setzen("keine - Ausgabe nur in Prozent", "warn")

    # -------------------------------------------------- Messbereich
    def _bereich_setzen(self):
        """Das ROI-Werkzeug auf dem zuletzt empfangenen Bild starten und das
        Ergebnis uebernehmen, sobald das Fenster geschlossen wird."""
        if not self.letzter_pfad:
            self.status.setzen("erst 'Kamera verbinden' - es wird ein Bild gebraucht",
                               "fehler")
            return
        self._sammeln()
        ziel = self.kfg["messbereich"] or os.path.join(
            BASE, f"messbereich_flaeche_{self.kennung}.npz")
        self.status.setzen("ROI-Fenster: Rechteck aufziehen, 's' speichern, 'q' schliessen",
                           "warn")

        def lauf():
            try:
                subprocess.run([sys.executable, os.path.join(BASE, "roi_werkzeug.py"),
                                self.letzter_pfad, ziel], check=False)
            except Exception as e:
                self.warteschlange.put(("status", (f"ROI-Werkzeug: {e}", "fehler")))
                return
            if os.path.exists(ziel):
                self.warteschlange.put(("bereichpfad", ziel))
                self.befehle.put("bereich")
            else:
                self.warteschlange.put(("status", ("kein Messbereich gespeichert", "warn")))

        threading.Thread(target=lauf, daemon=True).start()

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
        if self.kfg["messbereich"] and not os.path.exists(self.kfg["messbereich"]):
            self.status.setzen("Messbereichsdatei nicht gefunden", "fehler"); return
        if self.kfg["kalibrierung"] and not os.path.exists(self.kfg["kalibrierung"]):
            self.status.setzen("Kalibrierdatei nicht gefunden", "fehler"); return
        self._generation += 1
        while not self.befehle.empty():      # Befehle des alten Fadens verwerfen
            self.befehle.get_nowait()
        self.verbunden = True
        self.k_verbinden.configure(text="Kamera trennen")
        threading.Thread(target=self._arbeiten, args=(self._generation,),
                         name=f"messung-flaeche-{self.kennung}-{self._generation}",
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

        def laeuft():
            return self.verbunden and self._generation == generation

        try:
            wache = Ordnerwache(k["aufnahme_ordner"], k["unterordner"], ab_bestand=True)
            m(("kamera", (f"verbunden | Ordner: {wache.aktiv_kurz}", "ok")))
            m(("status", ("Modell wird geladen ...", "normal")))
            aus = Auswerter(k["modell"], k["schwelle"], k["geraet"])
            kanal = f"{aus.kanaele} " + ("Kanal" if aus.kanaele == 1 else "Kanaele")
            if k["messbereich"]:
                n = aus.panel_laden(k["messbereich"])
                m(("bereich", (f"{os.path.basename(k['messbereich'])} | "
                               f"{n:,} px Bezugsflaeche".replace(",", "."), "ok")))
            m(("status", (f"{aus.geraet_text} | {kanal} | Livebild - "
                          f"Massstab, Messbereich und Referenz setzen", "ok")))

            puffer = []
            zaehler = gemessen = 0
            tempo = None

            while laeuft():
                try:
                    befehl = self.befehle.get_nowait()
                except queue.Empty:
                    befehl = None

                if befehl == "bereich":
                    try:
                        n = aus.panel_laden(k["messbereich"])
                    except Exception as e:
                        m(("bereich", (f"nicht lesbar: {e}", "fehler"))); n = None
                    if n:
                        m(("bereich", (f"{os.path.basename(k['messbereich'])} | "
                                       f"{n:,} px Bezugsflaeche".replace(",", "."), "ok")))
                        m(("ref", ("verworfen - neuer Messbereich, Referenz neu aufnehmen",
                                   "fehler")))
                        # Mit dem Messbereich faellt auch die Referenz weg. Eine
                        # laufende Messung MUSS deshalb hier enden - sonst
                        # rechnete der naechste Frame gegen einen Nullzustand,
                        # den es nicht mehr gibt.
                        if self.modus == "messung":
                            self.modus = "vorschau"
                            m(("mess", ("gestoppt - Messbereich geaendert", "warn")))
                            m(("knopf", "Messung starten"))
                        m(("status", ("Messbereich gesetzt - Eisfrei-Referenz jetzt "
                                      "neu aufnehmen", "warn")))
                elif befehl == "referenz":
                    aus.referenz_start(k["referenz_frames"])
                    self.modus = "referenz"
                    m(("ref", (f"sammelt 0/{k['referenz_frames']} - NICHT spruehen", "warn")))
                elif befehl == "messung":
                    if not aus.referenz_bereit:
                        m(("status", ("erst Eisfrei-Referenz aufnehmen", "fehler")))
                    else:
                        self.lauf_ordner = self._lauf_ordner_anlegen(wache)
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
                self.letzter_pfad = pfad
                aus.schwelle = k["schwelle"]
                crop = aus.zuschneiden(img)

                # ---- Referenzphase
                if self.modus == "referenz":
                    fertig, n = aus.referenz_sammeln(img)
                    m(("bild", overlay_bauen(crop, np.zeros(crop.shape, bool))))
                    if not fertig:
                        m(("ref", (f"sammelt {n}/{aus.soll} - NICHT spruehen", "warn")))
                        continue
                    self.modus = "vorschau"
                    m(("ref", (f"steht - {aus.soll} eisfreie Frames", "ok")))
                    m(("status", ("Referenz steht - Messung starten, dann spruehen", "ok")))
                    continue

                # ---- Vorschau
                if self.modus != "messung":
                    if aus.referenz_bereit:
                        eis, crop = aus.auswerten(img)
                        m(("bild", overlay_bauen(crop, eis)))
                    else:
                        m(("bild", overlay_bauen(crop, np.zeros(crop.shape, bool))))
                    m(("kamera", (f"Ordner {wache.aktiv_kurz} | {os.path.basename(pfad)}"
                                  + (f" | {wache.uebersprungen} uebersprungen"
                                     if wache.uebersprungen else ""), "ok")))
                    continue

                # ---- Messphase
                if not aus.referenz_bereit:
                    # Kann nur eintreten, wenn die Referenz waehrend der Messung
                    # ungueltig geworden ist. Lieber hier abfangen als mit einem
                    # halben Nullzustand weiterrechnen.
                    self.modus = "vorschau"
                    m(("mess", ("gestoppt - Referenz fehlt", "warn")))
                    m(("knopf", "Messung starten"))
                    continue
                t0 = time.time()
                eis, crop = aus.auswerten(img)
                dauer = time.time() - t0
                tempo = dauer if tempo is None else 0.7 * tempo + 0.3 * dauer

                bezug = aus.bezugsflaeche_px or eis.size
                anteil = 100.0 * float(eis.sum()) / bezug
                versatz = (aus.bbox[0], aus.bbox[1]) if aus.bbox else (0, 0)
                mm2 = self.massstab.flaeche_mm2(eis, versatz)

                gemessen += 1
                self.ergebnisse.append({
                    "datei": os.path.basename(pfad), "quelle": wache.aktiv_kurz,
                    "nr": gemessen, "anteil_prozent": round(anteil, 3),
                    "flaeche_mm2": None if mm2 is None else round(mm2, 1),
                    "zeit": time.strftime("%H:%M:%S"),
                })
                self._json_schreiben()

                ov = overlay_bauen(crop, eis)
                if k["overlays_speichern"]:
                    klein = cv2.resize(ov, (900, int(ov.shape[0] * 900 / ov.shape[1])),
                                       interpolation=cv2.INTER_AREA)
                    cv2.imwrite(os.path.join(self.lauf_ordner,
                                             f"overlay_{gemessen:05d}.jpg"), klein,
                                [cv2.IMWRITE_JPEG_QUALITY, 85])
                if k["crops_speichern"]:
                    # Der Zuschnitt ist genau das, was das Netz gesehen hat -
                    # verlustfrei als PNG, damit die Offline-Auswertung darauf
                    # aufsetzen kann, ohne den Messbereich neu anzuwenden.
                    os.makedirs(os.path.join(self.lauf_ordner, "crops"), exist_ok=True)
                    cv2.imwrite(os.path.join(self.lauf_ordner, "crops",
                                             f"crop_{gemessen:05d}.png"), crop)
                m(("bild", ov)); m(("wert", anteil))
                m(("status", (f"{aus.geraet_text} | Bild {gemessen} | {anteil:5.2f}%"
                              + (f" | {mm2:.0f} mm2" if mm2 is not None
                                 else " | keine Kalibrierung")
                              + f" | {tempo*1000:.0f} ms/Bild"
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
    def _lauf_ordner_anlegen(self, wache):
        name = time.strftime("%Y%m%d_%H%M%S")
        if wache.aktiv_kurz != ".":
            name += "_" + os.path.basename(wache.aktiv_kurz)
        p = os.path.join(self.kfg["ergebnis_ordner"], name)
        os.makedirs(p, exist_ok=True)
        return p

    def _json_schreiben(self):
        if not self.lauf_ordner:
            return
        try:
            json.dump({"typ": "flaeche", "kamera": self.kennung,
                       "einstellungen": self.kfg.werte, "messwerte": self.ergebnisse},
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
                elif art == "mass":
                    self.z_mass.setzen(*nutz)
                elif art == "bereich":
                    self.z_bereich.setzen(*nutz)
                elif art == "ref":
                    self.z_ref.setzen(*nutz)
                elif art == "mess":
                    self.z_mess.setzen(*nutz)
                elif art == "knopf":
                    self.k_messen.configure(text=nutz)
                elif art == "bereichpfad":
                    self.f_bereich.setzen(nutz); self.kfg["messbereich"] = nutz
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


def _zahl(text, standard):
    try:
        return float(str(text).replace(",", "."))
    except ValueError:
        return standard


def main():
    wurzel = tk.Tk()
    wurzel.title("Eisflaechenmessung - Kamera unten und oben")
    wurzel.geometry("1500x960")
    try:
        ttk.Style().theme_use("vista")
    except tk.TclError:
        pass
    reiter = ttk.Notebook(wurzel)
    reiter.pack(fill="both", expand=True)
    # Je Kamera zwei Reiter, nach Kamera gruppiert: der Ablauf liest sich damit
    # von links nach rechts - erst kalibrieren, dann messen.
    for kennung, titel in KAMERAS:
        mess = Kamerapanel(reiter, kennung, titel)
        kalib = KalibrierTab(reiter, mess.kfg, f"flaeche_{kennung}", titel, "rechteck",
                             bei_uebernahme=mess.kalibrierung_setzen,
                             startordner=lambda m=mess: m.kfg["aufnahme_ordner"])
        reiter.add(kalib, text=f"Kalibrierung {kennung}")
        reiter.add(mess, text=f"Messung {kennung}")
    reiter.add(AnleitungTab(reiter), text="Anleitung")
    wurzel.mainloop()


if __name__ == "__main__":
    main()
