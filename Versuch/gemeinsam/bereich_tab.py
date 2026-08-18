# -*- coding: utf-8 -*-
"""
bereich_tab.py - Messbereich festlegen: Bild waehlen, Vorschlag, anpassen.

Derselbe Dreischritt wie beim Kalibrierreiter, und aus demselben Grund:

  1. BILD WAEHLEN. Aus der Dateiauswahl, ein eisfreies Bild dieser Kamera.
     Vor der Messung, nicht mittendrin.
  2. VORSCHLAG. Laeuft sofort nach der Bildwahl. Der Vorschlag ist die groesste
     zusammenhaengende helle Flaeche - auf der Testreihe trifft er das von Hand
     gepruefte Ergebnis mit IoU 0,87.
  3. ANPASSEN. Ecken ziehen, innen ziehen zum Verschieben, oder ganz neu
     aufziehen. Mit dem Pinsel lassen sich Stoerstellen im Rechteck
     ausschliessen - Halterungen, Reflexe, Kanalwand.

Der Messbereich legt fest, WORAUF sich der Bedeckungsgrad bezieht. Er gehoert
deshalb ins Protokoll, und zwar mit Groesse und Lage - ein Prozentwert ohne
seine Bezugsflaeche ist nicht deutbar. Die Zustandszeile nennt beides.
"""
import os
import tkinter as tk
from tkinter import ttk, filedialog

import numpy as np
import cv2

from . import messbereich as mb
from . import gui

BILDTYPEN = [("Bilder", "*.tif *.tiff *.png *.jpg *.jpeg *.bmp"), ("Alle", "*.*")]


class BereichTab(ttk.Frame):
    def __init__(self, eltern, kfg, kennung, titel, bei_uebernahme=None,
                 startordner=None):
        super().__init__(eltern, padding=10)
        self.kfg, self.kennung, self.titel = kfg, kennung, titel
        self.bei_uebernahme = bei_uebernahme
        self.startordner = startordner
        self.grau = None
        self.bildpfad = ""
        self.ausschluss = None       # uint8, >0 = ausgeschlossen
        self.verlauf = []            # fuer Rueckgaengig
        self.bericht = None
        self._ausschliessen = True
        self._bauen()

    # ------------------------------------------------------------- Aufbau
    def _bauen(self):
        links = ttk.Frame(self); links.pack(side="left", fill="y", padx=(0, 12))
        rechts = ttk.Frame(self); rechts.pack(side="left", fill="both", expand=True)

        a = gui.abschnitt(links, "1  Bild")
        ttk.Button(a, text="Bild waehlen ...", command=self._bild_waehlen).pack(fill="x", pady=2)
        self.l_bild = ttk.Label(a, text="kein Bild gewaehlt", foreground="#666",
                                wraplength=300, justify="left")
        self.l_bild.pack(anchor="w", pady=(2, 0))
        ttk.Label(a, text="ein EISFREIES Bild dieser Kamera",
                  foreground="#666").pack(anchor="w")

        b = gui.abschnitt(links, "2  Vorschlag")
        ttk.Button(b, text="Vorschlag neu berechnen",
                   command=self._vorschlagen).pack(fill="x", pady=2)
        ttk.Label(b, text="groesste zusammenhaengende helle Flaeche",
                  foreground="#666", wraplength=300).pack(anchor="w")

        c = gui.abschnitt(links, "3  Anpassen")
        ttk.Label(c, text="Ecken ziehen = Groesse\ninnen ziehen = verschieben\n"
                          "ausserhalb ziehen = neu aufziehen",
                  foreground="#666", justify="left").pack(anchor="w")
        ttk.Button(c, text="Rechteck bearbeiten",
                   command=self._rechteck_modus).pack(fill="x", pady=(6, 2))
        z = ttk.Frame(c); z.pack(fill="x", pady=2)
        ttk.Label(z, text="Pinsel", width=10, anchor="w").pack(side="left")
        self.v_radius = tk.IntVar(value=40)
        ttk.Spinbox(z, from_=5, to=400, increment=5, width=6,
                    textvariable=self.v_radius,
                    command=self._radius).pack(side="left")
        ttk.Label(z, text="px").pack(side="left", padx=(4, 0))
        z2 = ttk.Frame(c); z2.pack(fill="x", pady=2)
        ttk.Button(z2, text="ausschliessen",
                   command=lambda: self._pinsel(True)).pack(side="left", fill="x", expand=True)
        ttk.Button(z2, text="freigeben",
                   command=lambda: self._pinsel(False)).pack(side="left", fill="x",
                                                             expand=True, padx=(4, 0))
        ttk.Button(c, text="letzten Strich zurueck",
                   command=self._zurueck).pack(fill="x", pady=2)
        ttk.Button(c, text="Ausschluss ganz loeschen",
                   command=self._ausschluss_leeren).pack(fill="x", pady=2)

        d = gui.abschnitt(links, "4  Uebernehmen")
        ttk.Button(d, text="Fuer die Messung uebernehmen",
                   command=self._uebernehmen).pack(fill="x", pady=(2, 2))
        ttk.Label(d, text="wird gespeichert und im Messreiter eingetragen",
                  foreground="#666", wraplength=300).pack(anchor="w")

        # ---- rechte Spalte
        kopf = ttk.Frame(rechts); kopf.pack(fill="x")
        ttk.Button(kopf, text="einpassen",
                   command=lambda: self.leinwand.einpassen()).pack(side="left")
        self.l_werkzeug = ttk.Label(kopf, text="", foreground="#B4530A")
        self.l_werkzeug.pack(side="left", padx=8)

        self.leinwand = gui.Bildleinwand(rechts, bei_form=self._form_fertig,
                                         bei_pinsel=self._malen)
        self.leinwand.pack(fill="both", expand=True, pady=(6, 0))

        zust = ttk.LabelFrame(rechts, text=f"Messbereich - {self.titel}",
                              padding=(10, 6))
        zust.pack(fill="x", pady=(8, 0))
        self.z_bereich = gui.Zustand(zust, "Rechteck"); self.z_bereich.pack(fill="x")
        self.z_gilt = gui.Zustand(zust, "gilt fuer die Messung"); self.z_gilt.pack(fill="x")
        self.z_bereich.setzen("noch kein Bild gewaehlt")
        alt = self.kfg.werte.get("messbereich", "")
        if alt and os.path.exists(alt):
            self.z_gilt.setzen(os.path.basename(alt) + " (aus einem frueheren Lauf - "
                               "pruefen, ob er zur jetzigen Kamera passt)", "warn")
        else:
            self.z_gilt.setzen("nicht gesetzt - Prozente bezoegen sich aufs Vollbild",
                               "warn")
        self.status = gui.Statuszeile(rechts); self.status.pack(fill="x", pady=(8, 0))
        self.status.setzen("Eisfreies Bild dieser Kamera waehlen")

    # ------------------------------------------------------------- Bild
    def _bild_waehlen(self):
        start = self.startordner() if callable(self.startordner) else self.startordner
        p = filedialog.askopenfilename(
            title=f"Eisfreies Bild fuer den Messbereich ({self.titel})",
            initialdir=start if start and os.path.isdir(start) else os.getcwd(),
            filetypes=BILDTYPEN)
        if not p:
            return
        img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if img is None:
            self.status.setzen(f"Bild nicht lesbar: {os.path.basename(p)}", "fehler")
            return
        self.grau, self.bildpfad = img, p
        self.ausschluss = np.zeros(img.shape, np.uint8)
        self.verlauf = []
        self.l_bild.configure(text=f"{os.path.basename(p)}   "
                                   f"{img.shape[1]}x{img.shape[0]} px")
        self.leinwand.maske = self.ausschluss
        self.leinwand.bild_setzen(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR))
        self.update_idletasks()
        self._vorschlagen()

    def _vorschlagen(self):
        if self.grau is None:
            self.status.setzen("erst ein Bild waehlen", "fehler"); return
        self.status.setzen("suche die Panelflaeche ...", "normal")
        self.update_idletasks()
        v = mb.vorschlagen(self.grau)
        if v is None:
            H, W = self.grau.shape
            v = (int(W * 0.1), int(H * 0.1), int(W * 0.9), int(H * 0.6))
            self.status.setzen("keine klare Flaeche gefunden - grobes Rechteck "
                               "gesetzt, bitte anpassen", "warn")
        else:
            self.status.setzen("Vorschlag gesetzt - Ecken ziehen zum Anpassen", "ok")
        self._rechteck_modus(behalten=False)
        self.leinwand.form_setzen(v)
        self._form_fertig(v)

    # ------------------------------------------------------------- Werkzeuge
    def _rechteck_modus(self, behalten=True):
        if self.grau is None:
            self.status.setzen("erst ein Bild waehlen", "fehler"); return
        self.leinwand.werkzeug("rechteck", griffe=True, behalten=behalten)
        self.l_werkzeug.configure(text="Rechteck: Ecken ziehen")

    def _pinsel(self, ausschliessen):
        if self.grau is None:
            self.status.setzen("erst ein Bild waehlen", "fehler"); return
        self._ausschliessen = ausschliessen
        self.leinwand.pinselradius = int(self.v_radius.get())
        self.leinwand.werkzeug("pinsel", behalten=True)
        self.l_werkzeug.configure(
            text="Pinsel: " + ("ausschliessen" if ausschliessen else "freigeben"))
        self.status.setzen("im Bild ziehen; danach 'Rechteck bearbeiten', um die "
                           "Ecken wieder anzufassen", "warn")

    def _radius(self):
        self.leinwand.pinselradius = int(self.v_radius.get())

    def _malen(self, bx, by, radius, neu):
        if self.ausschluss is None:
            return
        if neu:                       # ein Strich = ein Schritt zurueck
            self.verlauf.append(self.ausschluss.copy())
            self.verlauf = self.verlauf[-12:]
        cv2.circle(self.ausschluss, (int(bx), int(by)), int(radius),
                   1 if self._ausschliessen else 0, -1)
        self._form_fertig(self.leinwand.form(), still=True)

    def _zurueck(self):
        if not self.verlauf:
            self.status.setzen("nichts zurueckzunehmen"); return
        self.ausschluss[:] = self.verlauf.pop()
        self.leinwand.maske = self.ausschluss
        self.leinwand._zeichnen()
        self._form_fertig(self.leinwand.form(), still=True)

    def _ausschluss_leeren(self):
        if self.ausschluss is None:
            return
        self.verlauf.append(self.ausschluss.copy())
        self.ausschluss[:] = 0
        self.leinwand._zeichnen()
        self._form_fertig(self.leinwand.form(), still=True)
        self.status.setzen("Ausschluss geloescht")

    # ------------------------------------------------------------- Anzeige
    def _form_fertig(self, form, still=False):
        if form is None or self.grau is None:
            return
        x0, y0, x1, y1 = form
        flaeche = mb.maske_bauen(self.grau.shape, form, self.ausschluss)
        px = int(flaeche.sum())
        raus = (x1 - x0) * (y1 - y0) - px
        punkt = lambda v: f"{v:,}".replace(",", ".")
        text = (f"{x1-x0}x{y1-y0} ab ({x0},{y0}) | {punkt(px)} px"
                + (f" | {punkt(raus)} px ausgeschlossen" if raus > 0 else ""))
        klein = px < 0.01 * self.grau.size
        self.z_bereich.setzen(text, "fehler" if klein else "ok")
        if klein and not still:
            self.status.setzen("Messbereich ist sehr klein - beabsichtigt?", "warn")

    # ------------------------------------------------------------- Uebernahme
    def _uebernehmen(self):
        form = self.leinwand.form()
        if self.grau is None or form is None:
            self.status.setzen("erst Bild waehlen und Rechteck setzen", "fehler"); return
        ziel = os.path.join(self.kfg["ergebnis_ordner"], "messbereich",
                            f"{self.kennung}_messbereich.npz")
        b = mb.speichern(ziel, self.grau, form, self.ausschluss,
                         quelle=os.path.basename(self.bildpfad))
        if not b["ok"]:
            self.status.setzen(b["fehler"], "fehler"); return
        self.bericht = b
        self.kfg["messbereich"] = b["npz"]
        self.kfg.speichern()
        self.z_gilt.setzen(f"{os.path.basename(b['npz'])} | {mb.bericht_text(b)}", "ok")
        if self.bei_uebernahme:
            self.bei_uebernahme(b["npz"])
        self.status.setzen(f"uebernommen - Kontrollbild: {b['png']}", "ok")
