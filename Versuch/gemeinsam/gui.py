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


class Auswahl(ttk.Frame):
    """Beschriftetes Auswahlfeld. nachladen: Knopf, der die Liste neu holt -
    noetig, weil die Kamera Unterordner erst waehrend des Versuchs anlegt."""

    def __init__(self, eltern, text, wert, werte, breite=30, nachladen=None):
        super().__init__(eltern)
        ttk.Label(self, text=text, width=22, anchor="w").pack(side="left")
        self.var = tk.StringVar(value=wert)
        self.box = ttk.Combobox(self, textvariable=self.var, values=list(werte),
                                width=breite, state="readonly")
        self.box.pack(side="left", fill="x", expand=True)
        if nachladen:
            ttk.Button(self, text="↻", width=3,
                       command=nachladen).pack(side="left", padx=(4, 0))

    def fuellen(self, werte, behalten=True):
        alt = self.var.get()
        self.box.configure(values=list(werte))
        if not (behalten and alt in werte):
            self.var.set(werte[0] if werte else "")

    def text(self):
        return self.var.get()

    def setzen(self, wert):
        self.var.set(wert)


class Zustand(ttk.Frame):
    """Einzeiliger Zustandsanzeiger fuer einen Arbeitsschritt: Punkt, Titel,
    Ergebnistext. Damit ist auf einen Blick zu sehen, welche Schritte erledigt
    sind - am Messtag der Unterschied zwischen 'sieht aus wie bereit' und
    'ist bereit'."""

    def __init__(self, eltern, titel):
        super().__init__(eltern)
        self.punkt = ttk.Label(self, text="●", foreground=FARBE_FEHLER, width=2)
        self.punkt.pack(side="left")
        ttk.Label(self, text=titel, width=20, anchor="w").pack(side="left")
        self.var = tk.StringVar(value="offen")
        ttk.Label(self, textvariable=self.var, anchor="w").pack(side="left", fill="x", expand=True)

    def setzen(self, text, art="fehler"):
        self.var.set(text)
        self.punkt.configure(foreground={"ok": FARBE_OK, "warn": FARBE_WARN,
                                         "fehler": FARBE_FEHLER}.get(art, "#888"))


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


class Bildleinwand(ttk.Frame):
    """Bildanzeige zum Arbeiten: Zoom mit dem Mausrad, Verschieben mit der
    rechten Taste, und zwei Zeichenwerkzeuge - Rechteck und Strecke.

    Der Zoom ist nicht Zierde: Ein 4096 px breites Bild passt bei rund 900 px
    Anzeigebreite nur im Verhaeltnis 1:4,5 hinein. Eine von Hand gezogene
    Strecke waere damit auf 4-5 Bildpixel genau - fuer eine Massstabsmessung zu
    grob. Mit Zoom wird sie pixelgenau.
    """

    def __init__(self, eltern, breite=900, hoehe=520, bei_form=None):
        super().__init__(eltern)
        self.breite, self.hoehe, self.bei_form = breite, hoehe, bei_form
        self.leinwand = tk.Canvas(self, width=breite, height=hoehe,
                                  background="#111", highlightthickness=0,
                                  cursor="crosshair")
        self.leinwand.pack(fill="both", expand=True)
        self.quelle = None
        self.zoom, self.ox, self.oy = 1.0, 0, 0
        self._x0 = self._y0 = 0
        self._bild = None
        self.art = None              # None | "rechteck" | "strecke"
        self.punkte = []             # gesetzte Punkte in BILDkoordinaten
        self._zieht = False
        self._schiebt = None

        self.leinwand.bind("<ButtonPress-1>", self._links_ab)
        self.leinwand.bind("<B1-Motion>", self._links_zieh)
        self.leinwand.bind("<ButtonRelease-1>", self._links_auf)
        self.leinwand.bind("<ButtonPress-3>", self._rechts_ab)
        self.leinwand.bind("<B3-Motion>", self._rechts_zieh)
        self.leinwand.bind("<MouseWheel>", self._rad)
        self.leinwand.bind("<Configure>", lambda e: self._zeichnen())

    # ---------------------------------------------------------- Bild
    def bild_setzen(self, bgr, einpassen=True):
        neu = self.quelle is None or bgr.shape[:2] != self.quelle.shape[:2]
        self.quelle = bgr if bgr.ndim == 3 else np.stack([bgr] * 3, axis=-1)
        if einpassen and neu:
            self.einpassen()
        else:
            self._zeichnen()

    def einpassen(self):
        if self.quelle is None:
            return
        cb = self.leinwand.winfo_width() or self.breite
        ch = self.leinwand.winfo_height() or self.hoehe
        H, W = self.quelle.shape[:2]
        self.zoom = min(cb / W, ch / H)
        self.ox = self.oy = 0
        self._zeichnen()

    # ---------------------------------------------------------- Werkzeug
    def werkzeug(self, art):
        self.art = art
        self.punkte = []
        self._zeichnen()

    def form_loeschen(self):
        self.punkte = []
        self._zeichnen()

    def form(self):
        """Rechteck als (x0,y0,x1,y1) oder Strecke als ((x0,y0),(x1,y1)) in
        Bildkoordinaten, sonst None."""
        if len(self.punkte) < 2:
            return None
        (ax, ay), (bx, by) = self.punkte[0], self.punkte[1]
        if self.art == "rechteck":
            return (int(min(ax, bx)), int(min(ay, by)),
                    int(max(ax, bx)), int(max(ay, by)))
        return ((ax, ay), (bx, by))

    # ---------------------------------------------------------- Umrechnung
    def _c2b(self, cx, cy):
        return (self._x0 + cx / self.zoom, self._y0 + cy / self.zoom)

    def _b2c(self, bx, by):
        return ((bx - self._x0) * self.zoom, (by - self._y0) * self.zoom)

    # ---------------------------------------------------------- Maus
    def _links_ab(self, e):
        if self.quelle is None or self.art is None:
            return
        self.punkte = [self._c2b(e.x, e.y)]
        self._zieht = True

    def _links_zieh(self, e):
        if not self._zieht:
            return
        p = self._c2b(e.x, e.y)
        self.punkte = [self.punkte[0], p]
        self._zeichnen()

    def _links_auf(self, e):
        if not self._zieht:
            return
        self._zieht = False
        if len(self.punkte) == 2 and self.bei_form:
            self.bei_form(self.form())

    def _rechts_ab(self, e):
        self._schiebt = (e.x, e.y, self.ox, self.oy)

    def _rechts_zieh(self, e):
        if not self._schiebt:
            return
        x, y, ox, oy = self._schiebt
        self.ox = ox - (e.x - x) / self.zoom
        self.oy = oy - (e.y - y) / self.zoom
        self._zeichnen()

    def _rad(self, e):
        if self.quelle is None:
            return
        bx, by = self._c2b(e.x, e.y)                # Punkt unter dem Zeiger
        faktor = 1.25 if e.delta > 0 else 1 / 1.25
        H, W = self.quelle.shape[:2]
        cb = self.leinwand.winfo_width() or self.breite
        klein = min(cb / W, (self.leinwand.winfo_height() or self.hoehe) / H)
        self.zoom = max(klein * 0.9, min(16.0, self.zoom * faktor))
        # denselben Bildpunkt wieder unter den Zeiger legen
        self.ox = bx - e.x / self.zoom
        self.oy = by - e.y / self.zoom
        self._zeichnen()

    # ---------------------------------------------------------- Anzeige
    def _zeichnen(self):
        c = self.leinwand
        c.delete("all")
        if self.quelle is None:
            return
        cb = c.winfo_width() or self.breite
        ch = c.winfo_height() or self.hoehe
        H, W = self.quelle.shape[:2]
        sicht_b, sicht_h = cb / self.zoom, ch / self.zoom
        self.ox = max(-sicht_b * 0.2, min(self.ox, W - sicht_b * 0.8))
        self.oy = max(-sicht_h * 0.2, min(self.oy, H - sicht_h * 0.8))
        x0, y0 = int(max(0, self.ox)), int(max(0, self.oy))
        x1 = int(min(W, np.ceil(self.ox + sicht_b)))
        y1 = int(min(H, np.ceil(self.oy + sicht_h)))
        if x1 <= x0 or y1 <= y0:
            return
        self._x0, self._y0 = x0, y0

        aus = self.quelle[y0:y1, x0:x1]
        nb = max(1, int(round((x1 - x0) * self.zoom)))
        nh = max(1, int(round((y1 - y0) * self.zoom)))
        import cv2
        klein = cv2.resize(aus, (nb, nh),
                           interpolation=cv2.INTER_AREA if self.zoom < 1
                           else cv2.INTER_NEAREST)
        self._bild = ImageTk.PhotoImage(Image.fromarray(klein[..., ::-1]))
        c.create_image(0, 0, anchor="nw", image=self._bild)

        if len(self.punkte) == 2:
            (ax, ay), (bx, by) = self.punkte
            cax, cay = self._b2c(ax, ay)
            cbx, cby = self._b2c(bx, by)
            if self.art == "rechteck":
                c.create_rectangle(cax, cay, cbx, cby, outline="#FFD24A", width=2)
            else:
                c.create_line(cax, cay, cbx, cby, fill="#FFD24A", width=2)
                for px, py in ((cax, cay), (cbx, cby)):
                    c.create_oval(px-4, py-4, px+4, py+4, outline="#FFD24A", width=2)
        c.create_text(6, ch - 8, anchor="sw", fill="#999",
                      text=f"Zoom {self.zoom:.2f}x   Rad = zoomen, rechte Taste = schieben")


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
