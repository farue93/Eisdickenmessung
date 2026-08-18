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

    GRIFF_PX = 12                # Fangbereich der Eckanfasser, in Anzeigepixeln

    def __init__(self, eltern, breite=900, hoehe=520, bei_form=None,
                 bei_pinsel=None):
        super().__init__(eltern)
        self.breite, self.hoehe, self.bei_form = breite, hoehe, bei_form
        self.bei_pinsel = bei_pinsel
        self.leinwand = tk.Canvas(self, width=breite, height=hoehe,
                                  background="#111", highlightthickness=0,
                                  cursor="crosshair")
        self.leinwand.pack(fill="both", expand=True)
        self.quelle = None
        self.maske = None            # uint8, >0 wird eingefaerbt (Ausschluss)
        self.zoom, self.ox, self.oy = 1.0, 0, 0
        self._x0 = self._y0 = 0
        self._bild = None
        self.art = None              # None | "rechteck" | "strecke" | "pinsel"
        self.griffe = False          # Eckanfasser zeigen und ziehbar machen
        self.pinselradius = 40
        self.punkte = []             # gesetzte Punkte in BILDkoordinaten
        self._zieht = False
        self._griff = None           # Index der gepackten Ecke, oder "innen"
        self._schiebt = None

        self.leinwand.bind("<ButtonPress-1>", self._links_ab)
        self.leinwand.bind("<B1-Motion>", self._links_zieh)
        self.leinwand.bind("<ButtonRelease-1>", self._links_auf)
        self.leinwand.bind("<ButtonPress-3>", self._rechts_ab)
        self.leinwand.bind("<B3-Motion>", self._rechts_zieh)
        self.leinwand.bind("<MouseWheel>", self._rad)
        self.leinwand.bind("<Configure>", self._eingerichtet)
        self._einpassen_noetig = False

    def _groesse(self):
        """Leinwandgroesse -> (breite, hoehe, schon_eingerichtet).

        Vor dem ersten Anzeigen meldet Tk die Breite 1, nicht 0 - ein
        'winfo_width() or 900' faellt darauf herein und rechnet den Zoom auf
        einem Pixel aus. Das Bild erschien dann als Punkt, bis jemand
        'einpassen' drueckte."""
        cb, ch = self.leinwand.winfo_width(), self.leinwand.winfo_height()
        if cb <= 1 or ch <= 1:
            return self.breite, self.hoehe, False
        return cb, ch, True

    def _eingerichtet(self, _=None):
        if self._einpassen_noetig:
            self.einpassen()
        else:
            self._zeichnen()

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
        cb, ch, fertig = self._groesse()
        H, W = self.quelle.shape[:2]
        self.zoom = min(cb / W, ch / H)
        self.ox = self.oy = 0
        # Solange die Leinwand ihre echte Groesse noch nicht kennt, bleibt der
        # Wunsch offen und wird beim ersten <Configure> nachgeholt.
        self._einpassen_noetig = not fertig
        self._zeichnen()

    # ---------------------------------------------------------- Werkzeug
    def werkzeug(self, art, griffe=False, behalten=False):
        """behalten=True laesst eine vorhandene Form stehen - noetig, wenn ein
        Vorschlag angezeigt und danach angepasst werden soll."""
        self.art = art
        self.griffe = griffe
        if not behalten:
            self.punkte = []
        self._zeichnen()

    def form_loeschen(self):
        self.punkte = []
        self._zeichnen()

    def form_setzen(self, form):
        """Rechteck von aussen vorgeben (Vorschlag)."""
        if form is None:
            self.punkte = []
        else:
            x0, y0, x1, y1 = form
            self.punkte = [(float(x0), float(y0)), (float(x1), float(y1))]
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

    def _ecken(self):
        """Die vier Ecken des Rechtecks in Bildkoordinaten, feste Reihenfolge:
        oben links, oben rechts, unten rechts, unten links."""
        f = self.form()
        if f is None or self.art != "rechteck":
            return []
        x0, y0, x1, y1 = f
        return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]

    def _griff_finden(self, cx, cy):
        """Welche Ecke liegt unter dem Zeiger? -> Index, 'innen' oder None."""
        for i, (bx, by) in enumerate(self._ecken()):
            gx, gy = self._b2c(bx, by)
            if abs(gx - cx) <= self.GRIFF_PX and abs(gy - cy) <= self.GRIFF_PX:
                return i
        f = self.form()
        if f and self.art == "rechteck":
            bx, by = self._c2b(cx, cy)
            if f[0] <= bx <= f[2] and f[1] <= by <= f[3]:
                return "innen"
        return None

    # ---------------------------------------------------------- Umrechnung
    def _c2b(self, cx, cy):
        return (self._x0 + cx / self.zoom, self._y0 + cy / self.zoom)

    def _b2c(self, bx, by):
        return ((bx - self._x0) * self.zoom, (by - self._y0) * self.zoom)

    # ---------------------------------------------------------- Maus
    def _links_ab(self, e):
        if self.quelle is None or self.art is None:
            return
        if self.art == "pinsel":
            self._zieht = True
            self._malen(e, neu=True)
            return
        # Anfasser haben Vorrang: erst anpassen, dann neu aufziehen. Ohne das
        # wuerde jeder Klick den Vorschlag verwerfen, statt ihn zu aendern.
        if self.griffe:
            g = self._griff_finden(e.x, e.y)
            if g is not None:
                self._griff = g
                self._start = (self._c2b(e.x, e.y), list(self.punkte))
                self._zieht = True
                return
        self._griff = None
        self.punkte = [self._c2b(e.x, e.y)]
        self._zieht = True

    def _links_zieh(self, e):
        if not self._zieht:
            return
        if self.art == "pinsel":
            self._malen(e); return
        p = self._c2b(e.x, e.y)
        if self._griff is None:
            self.punkte = [self.punkte[0], p]
        elif self._griff == "innen":
            (sx, sy), alt = self._start
            dx, dy = p[0] - sx, p[1] - sy
            self.punkte = [(a[0] + dx, a[1] + dy) for a in alt]
        else:
            # Die gepackte Ecke wandert mit; die diagonal gegenueberliegende
            # bleibt liegen und spannt das Rechteck neu auf.
            ecken = self._ecken()
            fest = ecken[(self._griff + 2) % 4]
            self.punkte = [fest, p]
        self._zeichnen()

    def _links_auf(self, e):
        if not self._zieht:
            return
        self._zieht = False
        if self.art == "pinsel":
            return
        self._griff = None
        if len(self.punkte) == 2 and self.bei_form:
            self.bei_form(self.form())

    def _malen(self, e, neu=False):
        """neu=True beim Aufsetzen des Stiftes - daran erkennt der Empfaenger
        den Beginn eines Striches und kann ihn als Ganzes zuruecknehmen."""
        if self.bei_pinsel:
            bx, by = self._c2b(e.x, e.y)
            self.bei_pinsel(bx, by, self.pinselradius, neu)
            self._zeichnen()

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
        cb, ch, _ = self._groesse()
        klein = min(cb / W, ch / H)
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
        cb, ch, _ = self._groesse()
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
        if self.maske is not None and self.maske.shape[:2] == (H, W):
            mk = cv2.resize(self.maske[y0:y1, x0:x1], (nb, nh),
                            interpolation=cv2.INTER_NEAREST)
            treffer = mk > 0
            if treffer.any():
                klein = klein.copy()
                klein[treffer] = (klein[treffer] * 0.45
                                  + np.array([40, 40, 210]) * 0.55).astype(np.uint8)
        self._bild = ImageTk.PhotoImage(Image.fromarray(klein[..., ::-1]))
        c.create_image(0, 0, anchor="nw", image=self._bild)

        if len(self.punkte) == 2:
            (ax, ay), (bx, by) = self.punkte
            cax, cay = self._b2c(ax, ay)
            cbx, cby = self._b2c(bx, by)
            if self.art == "rechteck":
                c.create_rectangle(cax, cay, cbx, cby, outline="#FFD24A", width=2)
                if self.griffe:
                    for ex, ey in self._ecken():
                        gx, gy = self._b2c(ex, ey)
                        c.create_rectangle(gx-6, gy-6, gx+6, gy+6,
                                           outline="#FFD24A", fill="#1b1b1b", width=2)
            else:
                c.create_line(cax, cay, cbx, cby, fill="#FFD24A", width=2)
                for px, py in ((cax, cay), (cbx, cby)):
                    c.create_oval(px-4, py-4, px+4, py+4, outline="#FFD24A", width=2)

        hinweis = "Rad = zoomen, rechte Taste = schieben"
        if self.art == "pinsel":
            hinweis = f"Pinsel {self.pinselradius} px   " + hinweis
        elif self.griffe:
            hinweis = "Ecken ziehen = anpassen, innen ziehen = verschieben   " + hinweis
        c.create_text(6, ch - 8, anchor="sw", fill="#999",
                      text=f"Zoom {self.zoom:.2f}x   {hinweis}")


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
