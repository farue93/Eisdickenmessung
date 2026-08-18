# -*- coding: utf-8 -*-
"""
anleitung.py - Die Anleitung IN der Messoberflaeche.

Gelesen wird README.md, dieselbe Datei, die auch ausserhalb gilt. Das ist der
ganze Punkt: Eine Anleitung, die als zweiter Text im Code steht, weicht nach der
dritten Aenderung von der Datei ab, und dann glaubt man am Messtag der
falschen. Hier gibt es nur eine Quelle.

Gerendert wird eine kleine Teilmenge von Markdown - genau das, was die Datei
benutzt: Ueberschriften, Absaetze, Listen, Tabellen, Zitatbloecke, Code und
Fettdruck. Kein Paket, kein Browser; am Messtag zaehlt, dass es startet.

Links steht das Inhaltsverzeichnis aus den Ueberschriften, rechts der Text.
"""
import os
import re
import tkinter as tk
from tkinter import ttk

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STANDARD_DATEI = os.path.join(BASE, "README.md")


class AnleitungTab(ttk.Frame):
    def __init__(self, eltern, datei=None, suchwort=None):
        super().__init__(eltern, padding=10)
        self.datei = datei or STANDARD_DATEI
        self.marken = []                 # (Ueberschrift, Textmarke)
        self._bauen()
        self.laden(suchwort)

    # ------------------------------------------------------------- Aufbau
    def _bauen(self):
        links = ttk.Frame(self); links.pack(side="left", fill="y", padx=(0, 12))
        rechts = ttk.Frame(self); rechts.pack(side="left", fill="both", expand=True)

        ttk.Label(links, text="Inhalt", font=("Segoe UI", 10, "bold")).pack(anchor="w")
        rahmen = ttk.Frame(links); rahmen.pack(fill="both", expand=True, pady=(4, 6))
        self.liste = tk.Listbox(rahmen, width=34, height=30, exportselection=False,
                                activestyle="none", borderwidth=0,
                                highlightthickness=1)
        self.liste.pack(side="left", fill="both", expand=True)
        rolle = ttk.Scrollbar(rahmen, orient="vertical", command=self.liste.yview)
        rolle.pack(side="left", fill="y")
        self.liste.configure(yscrollcommand=rolle.set)
        self.liste.bind("<<ListboxSelect>>", self._springen)
        ttk.Button(links, text="Anleitung neu laden",
                   command=lambda: self.laden()).pack(fill="x")
        ttk.Button(links, text="im Editor oeffnen",
                   command=self._oeffnen).pack(fill="x", pady=(4, 0))

        z = ttk.Frame(rechts); z.pack(fill="x")
        ttk.Label(z, text="Suchen").pack(side="left")
        self.v_suche = tk.StringVar()
        e = ttk.Entry(z, textvariable=self.v_suche, width=30)
        e.pack(side="left", padx=(6, 4))
        e.bind("<Return>", lambda ev: self._suchen())
        ttk.Button(z, text="weiter", command=self._suchen).pack(side="left")
        self.l_treffer = ttk.Label(z, text="", foreground="#666")
        self.l_treffer.pack(side="left", padx=8)

        rahmen2 = ttk.Frame(rechts); rahmen2.pack(fill="both", expand=True, pady=(6, 0))
        self.text = tk.Text(rahmen2, wrap="word", padx=16, pady=12,
                            borderwidth=0, highlightthickness=1,
                            font=("Segoe UI", 10), background="#FCFCFC")
        self.text.pack(side="left", fill="both", expand=True)
        rolle2 = ttk.Scrollbar(rahmen2, orient="vertical", command=self.text.yview)
        rolle2.pack(side="left", fill="y")
        self.text.configure(yscrollcommand=rolle2.set)
        self._marken_bauen()

    def _marken_bauen(self):
        t = self.text
        t.tag_configure("h1", font=("Segoe UI", 17, "bold"), spacing1=16, spacing3=8)
        t.tag_configure("h2", font=("Segoe UI", 13, "bold"), spacing1=16, spacing3=6,
                        foreground="#1B4F72")
        t.tag_configure("h3", font=("Segoe UI", 11, "bold"), spacing1=10, spacing3=4)
        t.tag_configure("p", spacing3=6)
        t.tag_configure("li", lmargin1=22, lmargin2=38, spacing3=3)
        t.tag_configure("zitat", lmargin1=14, lmargin2=14, spacing1=6, spacing3=6,
                        background="#F1F5F9", foreground="#243B53")
        t.tag_configure("tab", font=("Consolas", 9), lmargin1=14, lmargin2=14)
        t.tag_configure("kopf", font=("Consolas", 9, "bold"), lmargin1=14)
        t.tag_configure("fett", font=("Segoe UI", 10, "bold"))
        t.tag_configure("code", font=("Consolas", 9), background="#EEF1F4")
        t.tag_configure("linie", spacing1=8, spacing3=8, foreground="#BBB")
        t.tag_configure("fund", background="#FFE9A8")

    # ------------------------------------------------------------- Laden
    def laden(self, suchwort=None):
        self.text.configure(state="normal")
        self.text.delete("1.0", "end")
        self.liste.delete(0, "end")
        self.marken = []
        try:
            roh = open(self.datei, encoding="utf-8").read()
        except OSError as e:
            self.text.insert("end", f"Anleitung nicht lesbar: {e}\n\n"
                                    f"Erwartet: {self.datei}", "p")
            self.text.configure(state="disabled")
            return
        self._schreiben(roh)
        self.text.configure(state="disabled")
        if suchwort:
            self.v_suche.set(suchwort)
            self._suchen()

    def _schreiben(self, roh):
        zeilen = roh.splitlines()
        i, in_tabelle = 0, False
        absatz = []

        def absatz_leeren():
            if absatz:
                self._formatiert("".join(absatz).strip() + "\n", "p")
                absatz.clear()

        while i < len(zeilen):
            z = zeilen[i]
            k = z.strip()

            if k.startswith("|"):                       # Tabelle
                absatz_leeren()
                block = []
                while i < len(zeilen) and zeilen[i].strip().startswith("|"):
                    block.append(zeilen[i].strip())
                    i += 1
                self._tabelle(block)
                in_tabelle = False
                continue
            i += 1

            if not k:
                absatz_leeren()
                continue
            if k.startswith("### "):
                absatz_leeren(); self._ueberschrift(k[4:], "h3"); continue
            if k.startswith("## "):
                absatz_leeren(); self._ueberschrift(k[3:], "h2"); continue
            if k.startswith("# "):
                absatz_leeren(); self._ueberschrift(k[2:], "h1"); continue
            if set(k) <= set("-=*") and len(k) >= 3:
                absatz_leeren()
                self.text.insert("end", "─" * 60 + "\n", "linie"); continue
            if k.startswith(">"):
                absatz_leeren()
                self._formatiert(k.lstrip("> ").rstrip() + "\n", "zitat"); continue
            if re.match(r"^([-*]|\d+\.)\s+", k):
                absatz_leeren()
                self._formatiert("• " + re.sub(r"^([-*]|\d+\.)\s+", "", k) + "\n", "li")
                continue
            absatz.append(k + " ")
        absatz_leeren()

    def _ueberschrift(self, text, tag):
        marke = f"m{len(self.marken)}"
        self.text.insert("end", text + "\n", tag)
        self.text.mark_set(marke, f"end-{len(text)+2}c")
        self.text.mark_gravity(marke, "left")
        einzug = {"h1": "", "h2": "  ", "h3": "      "}[tag]
        self.marken.append((einzug + text, marke))
        self.liste.insert("end", einzug + text)

    def _tabelle(self, block):
        """Markdown-Tabelle in feste Spalten setzen - in einem Text-Widget ist
        Monospace die einzige verlaessliche Ausrichtung."""
        reihen = []
        for z in block:
            zellen = [c.strip() for c in z.strip("|").split("|")]
            if all(set(c) <= set("-: ") and c for c in zellen):
                continue                                # Trennzeile
            reihen.append([re.sub(r"[*`]", "", c) for c in zellen])
        if not reihen:
            return
        spalten = max(len(r) for r in reihen)
        reihen = [r + [""] * (spalten - len(r)) for r in reihen]
        breiten = [max(len(r[s]) for r in reihen) for s in range(spalten)]
        breiten = [min(b, 48) for b in breiten]
        for n, r in enumerate(reihen):
            zeile = "  ".join(r[s][:breiten[s]].ljust(breiten[s]) for s in range(spalten))
            self.text.insert("end", zeile.rstrip() + "\n", "kopf" if n == 0 else "tab")
        self.text.insert("end", "\n", "tab")

    def _formatiert(self, text, grundtag):
        """**fett** und `code` als Auszeichnung uebernehmen, Rest normal."""
        rest = text
        muster = re.compile(r"\*\*(.+?)\*\*|`(.+?)`")
        while True:
            t = muster.search(rest)
            if not t:
                self.text.insert("end", rest, grundtag); break
            self.text.insert("end", rest[:t.start()], grundtag)
            if t.group(1) is not None:
                self.text.insert("end", t.group(1), (grundtag, "fett"))
            else:
                self.text.insert("end", t.group(2), (grundtag, "code"))
            rest = rest[t.end():]

    # ------------------------------------------------------------- Bedienung
    def _springen(self, _=None):
        w = self.liste.curselection()
        if not w:
            return
        self.text.see(self.marken[w[0]][1])
        self.text.yview_scroll(-1, "units")

    def _suchen(self):
        wort = self.v_suche.get().strip()
        self.text.tag_remove("fund", "1.0", "end")
        if not wort:
            self.l_treffer.configure(text=""); return
        start = self.text.index("insert +1c")
        stelle = self.text.search(wort, start, "end", nocase=True)
        if not stelle:
            stelle = self.text.search(wort, "1.0", "end", nocase=True)
        if not stelle:
            self.l_treffer.configure(text="nicht gefunden"); return
        ende = f"{stelle}+{len(wort)}c"
        self.text.tag_add("fund", stelle, ende)
        self.text.mark_set("insert", ende)
        self.text.see(stelle)
        anzahl = 0
        pos = "1.0"
        while True:
            pos = self.text.search(wort, pos, "end", nocase=True)
            if not pos:
                break
            anzahl += 1
            pos = f"{pos}+1c"
        self.l_treffer.configure(text=f"{anzahl} Treffer")

    def _oeffnen(self):
        try:
            os.startfile(self.datei)                     # nur Windows
        except Exception as e:
            self.l_treffer.configure(text=f"nicht zu oeffnen: {e}")


if __name__ == "__main__":
    w = tk.Tk()
    w.title("Anleitung")
    w.geometry("1200x800")
    AnleitungTab(w).pack(fill="both", expand=True)
    w.mainloop()
