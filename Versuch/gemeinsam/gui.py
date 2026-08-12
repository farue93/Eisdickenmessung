# -*- coding: utf-8 -*-
"""
gui.py - Wiederverwendbare Bausteine fuer die beiden Live-Oberflaechen.

Bewusst tkinter: gehoert zu Python, braucht keinen Browser, keinen Server und
keine Installation auf den Versuchs-PCs. Am Messtag zaehlt, dass es startet.
"""
import os
import tkinter as tk
from tkinter import ttk, filedialog

import numpy as np
from PIL import Image, ImageTk

FARBE_OK = "#1E7A4C"
FARBE_WARN = "#B4530A"
FARBE_FEHLER = "#A62B1F"


class Feld(ttk.Frame):
    """Beschriftete Eingabezeile mit optionaler Einheit und Hinweistext."""

    def __init__(self, eltern, text, wert, breite=10, einheit="", hinweis=""):
        super().__init__(eltern)
        ttk.Label(self, text=text, width=22, anchor="w").pack(side="left")
        self.var = tk.StringVar(value=str(wert))
        ttk.Entry(self, textvariable=self.var, width=breite).pack(side="left")
        if einheit:
            ttk.Label(self, text=einheit, width=7, anchor="w").pack(side="left")
        if hinweis:
            ttk.Label(self, text=hinweis, foreground="#666").pack(side="left")

    def zahl(self, standard=0.0, ganz=False):
        try:
            w = float(self.var.get().replace(",", "."))
            return int(w) if ganz else w
        except ValueError:
            return standard

    def text(self):
        return self.var.get().strip()

    def setzen(self, wert):
        self.var.set(str(wert))


class Pfadfeld(ttk.Frame):
    """Pfadeingabe mit Durchsuchen-Knopf. modus: 'ordner' | 'datei'."""

    def __init__(self, eltern, text, wert, modus="ordner", typen=None):
        super().__init__(eltern)
        self.modus, self.typen = modus, typen or [("Alle Dateien", "*.*")]
        ttk.Label(self, text=text, width=22, anchor="w").pack(side="left")
        self.var = tk.StringVar(value=wert)
        ttk.Entry(self, textvariable=self.var, width=46).pack(side="left", fill="x", expand=True)
        ttk.Button(self, text="...", width=3, command=self._waehlen).pack(side="left", padx=(4, 0))

    def _waehlen(self):
        start = self.var.get() if os.path.exists(self.var.get()) else os.getcwd()
        if self.modus == "ordner":
            p = filedialog.askdirectory(initialdir=start)
        else:
            p = filedialog.askopenfilename(initialdir=os.path.dirname(start) or start,
                                           filetypes=self.typen)
        if p:
            self.var.set(os.path.normpath(p))

    def text(self):
        return self.var.get().strip()

    def setzen(self, wert):
        self.var.set(wert)


class Bildflaeche(ttk.Frame):
    """Zeigt ein BGR- oder Graubild, passt es in die verfuegbare Flaeche ein."""

    def __init__(self, eltern, breite=760, hoehe=300):
        super().__init__(eltern)
        self.leinwand = tk.Canvas(self, width=breite, height=hoehe,
                                  background="#111", highlightthickness=0)
        self.leinwand.pack(fill="both", expand=True)
        self._bild = None

    def zeigen(self, bgr):
        if bgr is None:
            return
        b = self.leinwand.winfo_width() or 760
        h = self.leinwand.winfo_height() or 300
        if bgr.ndim == 2:
            bgr = np.stack([bgr] * 3, axis=-1)
        bh, bw = bgr.shape[:2]
        f = min(b / bw, h / bh)
        neu = (max(1, int(bw * f)), max(1, int(bh * f)))
        rgb = bgr[..., ::-1]
        bild = Image.fromarray(rgb.astype(np.uint8)).resize(neu, Image.BILINEAR)
        self._bild = ImageTk.PhotoImage(bild)      # Referenz halten, sonst weg
        self.leinwand.delete("all")
        self.leinwand.create_image(b // 2, h // 2, image=self._bild)


class Verlauf(ttk.Frame):
    """Einfacher Kurvenverlauf ueber die Zeit (ohne matplotlib)."""

    def __init__(self, eltern, hoehe=150, farbe="#E07A0C", titel=""):
        super().__init__(eltern)
        self.titel, self.farbe = titel, farbe
        self.leinwand = tk.Canvas(self, height=hoehe, background="#1b1b1b",
                                  highlightthickness=0)
        self.leinwand.pack(fill="both", expand=True)
        self.werte = []

    def anhaengen(self, wert):
        self.werte.append(float(wert))
        self.zeichnen()

    def leeren(self):
        self.werte.clear()
        self.zeichnen()

    def zeichnen(self):
        c = self.leinwand
        c.delete("all")
        b = c.winfo_width() or 700
        h = c.winfo_height() or 150
        rand = 26
        for i in range(5):
            y = rand * 0.4 + i * (h - rand) / 4
            c.create_line(0, y, b, y, fill="#2e2e2e")
        if len(self.werte) < 2:
            c.create_text(8, 12, anchor="w", fill="#888",
                          text=self.titel or "wartet auf Daten")
            return
        hoch = max(self.werte) or 1.0
        n = len(self.werte)
        punkte = []
        for i, v in enumerate(self.werte):
            punkte += [i / (n - 1) * b, (h - rand * 0.5) - v / hoch * (h - rand)]
        c.create_line(*punkte, fill=self.farbe, width=2)
        c.create_text(8, 12, anchor="w", fill="#aaa",
                      text=f"{self.titel}  jetzt {self.werte[-1]:.2f}   max {hoch:.2f}")


class Statuszeile(ttk.Frame):
    def __init__(self, eltern):
        super().__init__(eltern)
        self.var = tk.StringVar(value="bereit")
        self.label = ttk.Label(self, textvariable=self.var, anchor="w")
        self.label.pack(side="left", fill="x", expand=True)

    def setzen(self, text, art="normal"):
        self.var.set(text)
        self.label.configure(foreground={"ok": FARBE_OK, "warn": FARBE_WARN,
                                         "fehler": FARBE_FEHLER}.get(art, ""))


def abschnitt(eltern, titel):
    r = ttk.LabelFrame(eltern, text=titel, padding=(10, 6))
    r.pack(fill="x", pady=(0, 8))
    return r
