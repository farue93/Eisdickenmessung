# -*- coding: utf-8 -*-
"""
live_flaeche.py - Live-Auswertung der Flaechenansicht am Versuchs-PC.

Ueberwacht den Aufnahmeordner, segmentiert jedes neue Bild mit dem trainierten
U-Net und zeigt Bedeckungsgrad und Eisflaeche fortlaufend an.

Bewusst NUR das U-Net, kein Methodenvergleich: Die Live-Ansicht beantwortet am
Kanal eine einzige Frage - laeuft der Versuch sauber? Mehrere Verfahren kosten
Rechenzeit und Bedienaufwand, ohne dass davon eine Entscheidung abhaengt. Der
Vergleich gehoert in die Offline-Auswertung, wo beliebig oft neu gerechnet
werden kann. Die Rohbilder bleiben unangetastet, es geht also nichts verloren.

Start:  start_flaeche.bat   oder   python live_flaeche.py
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
    "ergebnis_ordner": os.path.join(BASE, "ergebnisse", "flaeche"),
    "modell": "",
    "panel_maske": "",
    "kalibrierung": "",
    "schwelle": 0.5,
    "jedes_n_te": 1,
    "aufnahme_fps": 5.0,
    "referenz_frames": 10,
    "overlays_speichern": True,
    "geraet": "auto",
}


# ----------------------------------------------------------------- Auswertung
class Auswerter:
    """Kapselt Modell und Vorverarbeitung. Laeuft im Arbeitsfaden."""

    def __init__(self, modellpfad, panel_npz, schwelle, geraetwahl, referenz_anzahl=10):
        import torch
        # Netzdefinition liegt MITGELIEFERT in netze/ - der Versuchsordner
        # muss ohne das uebrige Repo lauffaehig sein.
        from netze.unet_flaeche import UNet, vorhersage_gekachelt
        self.torch = torch
        self._vorhersage = vorhersage_gekachelt
        self.geraet, self.geraet_text = ger.geraet_waehlen(geraetwahl)
        d = torch.load(modellpfad, map_location=self.geraet)
        # Kanalzahl kommt aus der Modelldatei, nicht aus einer Einstellung:
        # so kann nie ein 1-Kanal-Modell mit 2-Kanal-Eingang gefuettert werden.
        self.kanaele = int(d.get("kanaele", 1))
        self.netz = UNet(kanaele=self.kanaele).to(self.geraet)
        self.netz.load_state_dict(d["state"])
        self.netz.eval()
        self.schwelle = float(d.get("schwelle", schwelle))
        self.clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(16, 16))

        # Referenzzustand: Median und robuste Streuung der eisfreien Frames
        # DIESES Laufs. Nur so ist die Abweichung "was ist neu hinzugekommen"
        # und nicht "wie sah es an einem anderen Tag aus".
        self.referenz_anzahl = referenz_anzahl if self.kanaele >= 2 else 0
        self._ref_puffer = []
        self.median = None
        self.mad = None

        self.panel = None
        self.bbox = None
        if panel_npz and os.path.exists(panel_npz):
            p = np.load(panel_npz)
            self.bbox = tuple(int(v) for v in p["bbox"])
            maske = cv2.erode(p["maske"],
                              cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (51, 51)))
            x0, y0, x1, y1 = self.bbox
            self.panel = maske[y0:y1, x0:x1] > 0

    def zuschneiden(self, img):
        if self.bbox is None:
            return img
        x0, y0, x1, y1 = self.bbox
        return img[y0:y1, x0:x1]

    @property
    def referenz_bereit(self):
        return self.kanaele < 2 or self.median is not None

    def referenz_sammeln(self, img):
        """Einen eisfreien Frame zur Referenz hinzufuegen.
        -> True, sobald die Referenz steht."""
        self._ref_puffer.append(self.zuschneiden(img).astype(np.float32))
        if len(self._ref_puffer) < self.referenz_anzahl:
            return False
        stapel = np.stack(self._ref_puffer)
        self.median = np.median(stapel, axis=0)
        # MAD statt Standardabweichung: unempfindlich gegen einzelne Ausreisser,
        # etwa einen Frame, in dem doch schon etwas zu sehen war.
        self.mad = np.median(np.abs(stapel - self.median), axis=0)
        self._ref_puffer = []
        return True

    def _abweichung(self, crop):
        """Robuster z-Wert gegen den Referenzzustand, auf 0..1 gedeckelt.
        Bewusst auf dem ROHBILD - CLAHE arbeitet je Bild und je Kachel, zwei
        behandelte Aufnahmen stehen danach nicht mehr auf gemeinsamer Skala."""
        z = np.abs(crop.astype(np.float32) - self.median) / (1.4826 * self.mad + 3.0)
        return np.clip(z / 8.0, 0.0, 1.0).astype(np.float32)

    def auswerten(self, img):
        """-> (eismaske, crop). Vorverarbeitung identisch zum Training:
        CLAHE, dann auf 0..1 - sonst sieht das Netz etwas anderes als gelernt."""
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


# ----------------------------------------------------------------- Oberflaeche
class Anwendung(ttk.Frame):
    def __init__(self, wurzel):
        super().__init__(wurzel, padding=10)
        self.pack(fill="both", expand=True)
        self.kfg = Konfig(os.path.join(BASE, "einstellungen_flaeche.json"), STANDARD)
        self.warteschlange = queue.Queue()
        self.laeuft = False
        self.faden = None
        self.ergebnisse = []
        self.massstab = Massstab()

        self._bauen()
        self.after(200, self._abholen)

    # -------------------------------------------------- Aufbau
    def _bauen(self):
        links = ttk.Frame(self); links.pack(side="left", fill="y", padx=(0, 12))
        rechts = ttk.Frame(self); rechts.pack(side="left", fill="both", expand=True)

        a = gui.abschnitt(links, "Ordner")
        self.f_auf = gui.Pfadfeld(a, "Aufnahmeordner", self.kfg["aufnahme_ordner"], "ordner")
        self.f_auf.pack(fill="x", pady=2)
        self.f_erg = gui.Pfadfeld(a, "Ergebnisordner", self.kfg["ergebnis_ordner"], "ordner")
        self.f_erg.pack(fill="x", pady=2)

        b = gui.abschnitt(links, "Modell und Kalibrierung")
        self.f_modell = gui.Pfadfeld(b, "U-Net (.pt)", self.kfg["modell"], "datei",
                                     [("PyTorch-Modell", "*.pt"), ("Alle", "*.*")])
        self.f_modell.pack(fill="x", pady=2)
        self.f_panel = gui.Pfadfeld(b, "Panelmaske (.npz)", self.kfg["panel_maske"], "datei",
                                    [("NumPy", "*.npz"), ("Alle", "*.*")])
        self.f_panel.pack(fill="x", pady=2)
        self.f_kal = gui.Pfadfeld(b, "Kalibrierung (.npz)", self.kfg["kalibrierung"], "datei",
                                  [("NumPy", "*.npz"), ("Alle", "*.*")])
        self.f_kal.pack(fill="x", pady=2)

        c = gui.abschnitt(links, "Auswertung")
        self.f_schwelle = gui.Feld(c, "Schwelle", self.kfg["schwelle"], 8,
                                   hinweis="0..1, hoeher = strenger")
        self.f_schwelle.pack(fill="x", pady=2)
        self.f_n = gui.Feld(c, "jedes N-te Bild", self.kfg["jedes_n_te"], 8, "", "1 = alle")
        self.f_n.pack(fill="x", pady=2)
        self.f_fps = gui.Feld(c, "Aufnahmerate", self.kfg["aufnahme_fps"], 8, "B/s")
        self.f_fps.pack(fill="x", pady=2)
        self.f_ref = gui.Feld(c, "Referenzframes", self.kfg["referenz_frames"], 8, "",
                              "eisfrei am Anfang")
        self.f_ref.pack(fill="x", pady=2)
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
        ttk.Button(d, text="Werte zuruecksetzen", command=self._zuruecksetzen).pack(fill="x", pady=2)
        ttk.Button(d, text="Einstellungen speichern", command=self._speichern).pack(fill="x", pady=2)

        self.bild = gui.Bildflaeche(rechts); self.bild.pack(fill="both", expand=True)
        self.kurve = gui.Verlauf(rechts, titel="Bedeckungsgrad %"); self.kurve.pack(fill="x", pady=(8, 0))
        self.status = gui.Statuszeile(rechts); self.status.pack(fill="x", pady=(8, 0))
        self.status.setzen("Ordner und Modell waehlen, dann Start")

    # -------------------------------------------------- Steuerung
    def _sammeln(self):
        self.kfg["aufnahme_ordner"] = self.f_auf.text()
        self.kfg["ergebnis_ordner"] = self.f_erg.text()
        self.kfg["modell"] = self.f_modell.text()
        self.kfg["panel_maske"] = self.f_panel.text()
        self.kfg["kalibrierung"] = self.f_kal.text()
        self.kfg["schwelle"] = self.f_schwelle.zahl(0.5)
        self.kfg["jedes_n_te"] = max(1, self.f_n.zahl(1, ganz=True))
        self.kfg["aufnahme_fps"] = self.f_fps.zahl(5.0)
        self.kfg["referenz_frames"] = max(0, self.f_ref.zahl(10, ganz=True))
        self.kfg["overlays_speichern"] = bool(self.v_overlay.get())
        self.kfg["geraet"] = self.v_geraet.get()

    def _speichern(self):
        self._sammeln(); self.kfg.speichern()
        self.status.setzen("Einstellungen gespeichert", "ok")

    def _zuruecksetzen(self):
        self.ergebnisse.clear(); self.kurve.leeren()
        self.status.setzen("Messwerte geleert (Einstellungen unveraendert)")

    def _umschalten(self):
        if self.laeuft:
            self.laeuft = False
            self.knopf.configure(text="Start")
            self.status.setzen("gestoppt")
            return
        self._sammeln(); self.kfg.speichern()
        fehler = self._pruefen()
        if fehler:
            self.status.setzen(fehler, "fehler"); return
        self.laeuft = True
        self.knopf.configure(text="Stopp")
        self.faden = threading.Thread(target=self._arbeiten, daemon=True)
        self.faden.start()

    def _pruefen(self):
        if not os.path.isdir(self.kfg["aufnahme_ordner"]):
            return "Aufnahmeordner existiert nicht"
        if not os.path.exists(self.kfg["modell"]):
            return "Modelldatei (.pt) nicht gefunden"
        if self.kfg["panel_maske"] and not os.path.exists(self.kfg["panel_maske"]):
            return "Panelmaske nicht gefunden"
        if self.kfg["kalibrierung"] and not os.path.exists(self.kfg["kalibrierung"]):
            return "Kalibrierdatei nicht gefunden"
        return None

    # -------------------------------------------------- Arbeitsfaden
    def _arbeiten(self):
        m = self.warteschlange.put
        try:
            m(("status", ("Modell wird geladen ...", "normal")))
            aus = Auswerter(self.kfg["modell"], self.kfg["panel_maske"],
                            self.kfg["schwelle"], self.kfg["geraet"],
                            self.kfg["referenz_frames"])
            self.massstab.laden(self.kfg["kalibrierung"])
            os.makedirs(self.kfg["ergebnis_ordner"], exist_ok=True)

            wache = Ordnerwache(self.kfg["aufnahme_ordner"], ab_bestand=True)
            kanal_text = f"{aus.kanaele} Kanal" + ("" if aus.kanaele == 1 else "e")
            if aus.referenz_anzahl:
                m(("status", (f"{aus.geraet_text} | {kanal_text} | "
                              f"warte auf {aus.referenz_anzahl} EISFREIE Frames "
                              f"fuer die Referenz - noch nicht spruehen!", "warn")))
            else:
                m(("status", (f"{aus.geraet_text} | {kanal_text} | "
                              f"{self.massstab.quelle} | warte auf Bilder ...", "ok")))

            gemessen, zaehler, tempo = 0, 0, None
            while self.laeuft:
                neue = wache.neue()
                if not neue:
                    time.sleep(0.25); continue
                for pfad in neue:
                    if not self.laeuft:
                        break
                    zaehler += 1
                    if zaehler % self.kfg["jedes_n_te"] != 0:
                        continue
                    img = bild_lesen(pfad)
                    if img is None:
                        m(("status", (f"nicht lesbar: {os.path.basename(pfad)}", "warn")))
                        continue

                    # Solange die Referenz fehlt, wird NICHT ausgewertet - ohne
                    # sauberen Ausgangszustand waere der zweite Kanal bedeutungslos.
                    if not aus.referenz_bereit:
                        fertig = aus.referenz_sammeln(img)
                        haben = aus.referenz_anzahl if fertig else len(aus._ref_puffer)
                        if fertig:
                            m(("status", (f"Referenz steht ({aus.referenz_anzahl} Frames) | "
                                          f"{aus.geraet_text} | {self.massstab.quelle} | "
                                          f"bereit - jetzt kann gespruehen werden", "ok")))
                        else:
                            m(("status", (f"Referenz: {haben}/{aus.referenz_anzahl} eisfreie "
                                          f"Frames - noch nicht spruehen!", "warn")))
                        m(("bild", overlay_bauen(aus.zuschneiden(img),
                                                 np.zeros(aus.zuschneiden(img).shape, bool))))
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
                    eintrag = {"datei": os.path.basename(pfad), "nr": gemessen,
                               "anteil_prozent": round(anteil, 3),
                               "flaeche_mm2": None if mm2 is None else round(mm2, 1),
                               "zeit": time.strftime("%H:%M:%S")}
                    self.ergebnisse.append(eintrag)

                    ov = overlay_bauen(crop, eis)
                    if self.kfg["overlays_speichern"]:
                        klein = cv2.resize(ov, (900, int(ov.shape[0] * 900 / ov.shape[1])),
                                           interpolation=cv2.INTER_AREA)
                        cv2.imwrite(os.path.join(self.kfg["ergebnis_ordner"],
                                                 f"overlay_{gemessen:05d}.jpg"), klein,
                                    [cv2.IMWRITE_JPEG_QUALITY, 85])
                    self._json_schreiben()

                    text = (f"{aus.geraet_text} | Bild {gemessen} | {anteil:5.2f}%"
                            + (f" | {mm2:.0f} mm2" if mm2 is not None else " | keine Kalibrierung")
                            + f" | {tempo*1000:.0f} ms/Bild")
                    m(("bild", ov)); m(("wert", anteil)); m(("status", (text, "ok")))
            m(("status", ("gestoppt", "normal")))
        except Exception as e:
            import traceback; traceback.print_exc()
            m(("status", (f"Fehler: {e}", "fehler")))
        finally:
            m(("ende", None))

    def _json_schreiben(self):
        try:
            json.dump({"typ": "flaeche", "einstellungen": self.kfg.werte,
                       "messwerte": self.ergebnisse},
                      open(os.path.join(self.kfg["ergebnis_ordner"], "messwerte.json"),
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
                elif art == "status":
                    self.status.setzen(*nutz)
                elif art == "ende":
                    self.laeuft = False
                    self.knopf.configure(text="Start")
        except queue.Empty:
            pass
        self.after(150, self._abholen)


def main():
    wurzel = tk.Tk()
    wurzel.title("Live-Auswertung Flaeche")
    wurzel.geometry("1360x780")
    try:
        ttk.Style().theme_use("vista")
    except tk.TclError:
        pass
    Anwendung(wurzel)
    wurzel.mainloop()


if __name__ == "__main__":
    main()
