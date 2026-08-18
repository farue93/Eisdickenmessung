# -*- coding: utf-8 -*-
"""
kalibrier_tab.py - Der Kalibrierreiter, gemeinsam fuer Laser und Flaeche.

Ergebnis ist eine MASSSTABSKARTE: fuer JEDEN Bildpunkt der lokale Umrechnungs-
faktor Pixel -> Millimeter. Ein einzelner px/mm-Wert genuegt nicht, weil die
Kamera schraeg auf eine gekruemmte Flaeche blickt - ein Pixel am Rand deckt ein
Vielfaches der Flaeche eines Pixels in der Bildmitte ab. Die Messung liefert
damit spaeter mm bzw. mm^2, nicht Pixel.

Der Ablauf ist dreistufig, und die Reihenfolge hat einen Grund:

  1. BILDER LADEN. Aus der Dateiauswahl, vor der Messung. Dann steht das Brett
     noch, es ist Zeit, und ein misslungener Versuch kostet nichts.

     MEHRERE Bilder sind der Normalfall, nicht die Ausnahme: Ein einzelnes
     Brett stuetzt nur den Bildbereich, den es bedeckt - typisch wenige Prozent
     der Bildflaeche; ueberall sonst setzt die Karte den Verlauf nur fort. Wird
     dasselbe Brett nacheinander an mehrere Stellen geklebt und jedes Mal
     aufgenommen, gehen alle Stuetzstellen in EINEN gemeinsamen Fit ein. Die
     Liste ist jederzeit aenderbar: Bilder dazu, einzelne wieder raus, neu
     rechnen.

  2. AUTOMATISCH. Schachbretter suchen, Karte fitten. Laeuft nach jeder
     Aenderung der Liste von selbst.

  3. VON HAND. Zwei Dinge, die die Automatik nicht kann:
       - Suchbereich einschraenken, wenn ausser dem Brett noch etwas
         Schachbrettartiges im Bild liegt (Gitter, Spiegelung im nassen Panel).
         Der Detektor findet dann unter Umstaenden das Falsche und meldet
         trotzdem Erfolg. Der Suchbereich gehoert JE BILD, weil das Brett in
         jeder Aufnahme woanders liegt.
       - Massstab direkt messen, wenn gar kein Brett da ist oder wenn das
         Ergebnis unabhaengig geprueft werden soll.

Die Handmessung ist je Strang eine andere, weil die Messgroesse eine andere
ist: Beim Laser wird eine LAENGE gemessen (Verschiebung quer zur Linie), dort
ist das Werkzeug eine Strecke. Bei der Flaeche wird eine FLAECHE gemessen, dort
ist es ein Rechteck. Ein Rechteck fuer den Laser waere Scheingenauigkeit, eine
Strecke fuer die Flaeche eine unbelegte Annahme ueber die zweite Richtung.

Liegt bereits eine automatische Kalibrierung vor, dient die Handmessung als
GEGENPROBE: der Reiter zeigt, um wie viel Prozent beide auseinanderliegen. Ein
Unterschied von mehreren Prozent heisst fast immer, dass die eingetragene
Feldgroesse des Bretts nicht der gedruckten entspricht.
"""
import os
import tkinter as tk
from tkinter import ttk, filedialog

import numpy as np
import cv2

from . import kalibrierung as kal
from . import gui

BILDTYPEN = [("Bilder", "*.tif *.tiff *.png *.jpg *.jpeg *.bmp"), ("Alle", "*.*")]


class KalibrierTab(ttk.Frame):
    def __init__(self, eltern, kfg, kennung, titel, modus, bei_uebernahme=None,
                 startordner=None):
        """modus: 'strecke' (Laser) oder 'rechteck' (Flaeche).
        bei_uebernahme(npz_pfad) wird gerufen, wenn der Massstab gilt."""
        super().__init__(eltern, padding=10)
        self.kfg, self.kennung, self.titel = kfg, kennung, titel
        self.modus = modus
        self.bei_uebernahme = bei_uebernahme
        self.startordner = startordner
        self.bilder = []            # [{name, pfad, grau, suchbereich}]
        self.auto = None            # Bericht der automatischen Kalibrierung
        self.hand = None            # Bericht der Handmessung
        self.aktiv = None           # welcher Bericht gilt
        self._welches = None
        self._hand_form = None
        self._bauen()

    # ------------------------------------------------------- Bildliste
    @property
    def index(self):
        w = self.liste.curselection()
        return w[0] if w else (0 if self.bilder else None)

    @property
    def grau(self):
        i = self.index
        return self.bilder[i]["grau"] if i is not None else None

    def _liste_fuellen(self, auswahl=None):
        self.liste.delete(0, "end")
        for b in self.bilder:
            marke = "  [Suchbereich]" if b.get("suchbereich") else ""
            self.liste.insert("end", b["name"] + marke)
        if self.bilder:
            i = min(auswahl if auswahl is not None else 0, len(self.bilder) - 1)
            self.liste.selection_clear(0, "end")
            self.liste.selection_set(i)
        self.l_anzahl.configure(
            text=f"{len(self.bilder)} Bild(er) geladen"
                 + ("  -  mehr Positionen = mehr belegte Bildflaeche"
                    if 0 < len(self.bilder) < 3 else ""))

    # ------------------------------------------------------------- Aufbau
    def _bauen(self):
        links = ttk.Frame(self); links.pack(side="left", fill="y", padx=(0, 12))
        rechts = ttk.Frame(self); rechts.pack(side="left", fill="both", expand=True)

        a = gui.abschnitt(links, "1  Schachbrettbilder")
        rahmen = ttk.Frame(a); rahmen.pack(fill="x", pady=2)
        self.liste = tk.Listbox(rahmen, height=6, exportselection=False,
                                activestyle="none")
        self.liste.pack(side="left", fill="both", expand=True)
        rolle = ttk.Scrollbar(rahmen, orient="vertical", command=self.liste.yview)
        rolle.pack(side="left", fill="y")
        self.liste.configure(yscrollcommand=rolle.set)
        self.liste.bind("<<ListboxSelect>>", lambda e: self._anzeigen())
        self.l_anzahl = ttk.Label(a, text="0 Bild(er) geladen", foreground="#666")
        self.l_anzahl.pack(anchor="w")
        ttk.Button(a, text="Bilder hinzufuegen ...",
                   command=self._bilder_laden).pack(fill="x", pady=(6, 2))
        z0 = ttk.Frame(a); z0.pack(fill="x", pady=2)
        ttk.Button(z0, text="Entfernen",
                   command=self._bild_entfernen).pack(side="left", fill="x", expand=True)
        ttk.Button(z0, text="Alle entfernen",
                   command=self._alle_entfernen).pack(side="left", fill="x",
                                                      expand=True, padx=(4, 0))
        ttk.Label(a, text="dasselbe Brett an mehreren Stellen aufnehmen",
                  foreground="#666", wraplength=300).pack(anchor="w", pady=(2, 0))

        b = gui.abschnitt(links, "2  Schachbrett (automatisch)")
        z = ttk.Frame(b); z.pack(fill="x", pady=2)
        ttk.Label(z, text="Feldgroesse", width=18, anchor="w").pack(side="left")
        self.f_feld = ttk.Entry(z, width=8); self.f_feld.pack(side="left")
        self.f_feld.insert(0, str(self.kfg["feld_mm"]))
        ttk.Label(z, text="mm  gemessen!", foreground="#666").pack(side="left", padx=(4, 0))
        z2 = ttk.Frame(b); z2.pack(fill="x", pady=2)
        ttk.Label(z2, text="innere Ecken", width=18, anchor="w").pack(side="left")
        self.f_ex = ttk.Entry(z2, width=5); self.f_ex.pack(side="left")
        self.f_ex.insert(0, str(self.kfg["ecken_x"]))
        ttk.Label(z2, text="x").pack(side="left", padx=2)
        self.f_ey = ttk.Entry(z2, width=5); self.f_ey.pack(side="left")
        self.f_ey.insert(0, str(self.kfg["ecken_y"]))
        ttk.Label(b, text="6x9 Felder = 5x8 innere Ecken",
                  foreground="#666").pack(anchor="w")
        z3 = ttk.Frame(b); z3.pack(fill="x", pady=2)
        ttk.Label(z3, text="Polynomgrad", width=18, anchor="w").pack(side="left")
        self.v_grad = tk.StringVar(value=str(self.kfg["grad"]))
        ttk.Combobox(z3, textvariable=self.v_grad,
                     values=["automatisch", "1", "2"], width=12,
                     state="readonly").pack(side="left")
        ttk.Label(b, text="automatisch: ab 3 Brettern entscheidet die Kreuzprobe",
                  foreground="#666", wraplength=300).pack(anchor="w")
        ttk.Button(b, text="Eckenzahl aus Bild bestimmen",
                   command=self._ecken_raten).pack(fill="x", pady=(6, 2))
        ttk.Button(b, text="Suchbereich fuer dieses Bild",
                   command=lambda: self._werkzeug("suchbereich")).pack(fill="x", pady=2)
        ttk.Button(b, text="Suchbereich aufheben",
                   command=self._such_weg).pack(fill="x", pady=2)
        ttk.Button(b, text="Automatisch kalibrieren",
                   command=self._automatisch).pack(fill="x", pady=(6, 2))

        c = gui.abschnitt(links, "3  Massstab von Hand")
        if self.modus == "strecke":
            ttk.Label(c, text="Strecke bekannter Laenge ziehen",
                      foreground="#666").pack(anchor="w")
            z4 = ttk.Frame(c); z4.pack(fill="x", pady=2)
            ttk.Label(z4, text="reale Laenge", width=18, anchor="w").pack(side="left")
            self.f_laenge = ttk.Entry(z4, width=8); self.f_laenge.pack(side="left")
            self.f_laenge.insert(0, str(self.kfg.werte.get("hand_laenge_mm", 100.0)))
            ttk.Label(z4, text="mm").pack(side="left", padx=(4, 0))
            ttk.Button(c, text="Strecke ziehen",
                       command=lambda: self._werkzeug("hand")).pack(fill="x", pady=(6, 2))
        else:
            ttk.Label(c, text="Rechteck bekannter Groesse ziehen",
                      foreground="#666").pack(anchor="w")
            z4 = ttk.Frame(c); z4.pack(fill="x", pady=2)
            ttk.Label(z4, text="reale Groesse", width=18, anchor="w").pack(side="left")
            self.f_breite = ttk.Entry(z4, width=7); self.f_breite.pack(side="left")
            self.f_breite.insert(0, str(self.kfg.werte.get("hand_breite_mm", 100.0)))
            ttk.Label(z4, text="x").pack(side="left", padx=2)
            self.f_hoehe = ttk.Entry(z4, width=7); self.f_hoehe.pack(side="left")
            self.f_hoehe.insert(0, str(self.kfg.werte.get("hand_hoehe_mm", 100.0)))
            ttk.Label(z4, text="mm").pack(side="left", padx=(4, 0))
            ttk.Button(c, text="Rechteck ziehen",
                       command=lambda: self._werkzeug("hand")).pack(fill="x", pady=(6, 2))
        ttk.Button(c, text="Handmessung auswerten",
                   command=self._von_hand).pack(fill="x", pady=2)
        self.l_gegen = ttk.Label(c, text="", foreground="#666", wraplength=300,
                                 justify="left")
        self.l_gegen.pack(anchor="w", pady=(2, 0))

        d = gui.abschnitt(links, "4  Uebernehmen")
        self.v_quelle = tk.StringVar(value="auto")
        ttk.Radiobutton(d, text="Karte aus den Schachbrettbildern",
                        variable=self.v_quelle, value="auto",
                        command=self._quelle_gewechselt).pack(anchor="w")
        ttk.Radiobutton(d, text="Handmessung (ortsunabhaengig)",
                        variable=self.v_quelle, value="hand",
                        command=self._quelle_gewechselt).pack(anchor="w")
        ttk.Button(d, text="Fuer die Messung uebernehmen",
                   command=self._uebernehmen).pack(fill="x", pady=(6, 2))

        # ---- rechte Spalte
        kopf = ttk.Frame(rechts); kopf.pack(fill="x")
        self.v_overlay = tk.BooleanVar(value=True)
        ttk.Checkbutton(kopf, text="Overlay anzeigen", variable=self.v_overlay,
                        command=self._anzeigen).pack(side="left")
        ttk.Button(kopf, text="einpassen",
                   command=lambda: self.leinwand.einpassen()).pack(side="left", padx=8)
        self.l_werkzeug = ttk.Label(kopf, text="", foreground="#B4530A")
        self.l_werkzeug.pack(side="left", padx=8)

        self.leinwand = gui.Bildleinwand(rechts, bei_form=self._form_fertig)
        self.leinwand.pack(fill="both", expand=True, pady=(6, 0))

        zust = ttk.LabelFrame(rechts, text=f"Ergebnis - {self.titel}", padding=(10, 6))
        zust.pack(fill="x", pady=(8, 0))
        self.z_auto = gui.Zustand(zust, "aus Bildern"); self.z_auto.pack(fill="x")
        self.z_hand = gui.Zustand(zust, "von Hand"); self.z_hand.pack(fill="x")
        self.z_gilt = gui.Zustand(zust, "gilt fuer die Messung"); self.z_gilt.pack(fill="x")
        self.z_auto.setzen("noch keine Bilder geladen")
        self.z_hand.setzen("nicht gemessen", "warn")
        self.z_gilt.setzen("kein Massstab - Messung liefe nur in Pixel/Prozent", "warn")
        self.status = gui.Statuszeile(rechts); self.status.pack(fill="x", pady=(8, 0))
        self.status.setzen("Schachbrettbilder laden - je mehr Positionen, "
                           "desto groesser der belegte Bereich")

    # ------------------------------------------------------------- Bilder
    def _bilder_laden(self):
        start = self.startordner() if callable(self.startordner) else self.startordner
        pfade = filedialog.askopenfilenames(
            title="Schachbrettbilder waehlen (eisfrei) - Mehrfachauswahl moeglich",
            initialdir=start if start and os.path.isdir(start) else os.getcwd(),
            filetypes=BILDTYPEN)
        if not pfade:
            return
        neu, schlecht = 0, []
        for p in pfade:
            img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
            if img is None:
                schlecht.append(os.path.basename(p)); continue
            self.bilder.append(dict(name=os.path.basename(p), pfad=p, grau=img,
                                    suchbereich=None))
            neu += 1
        self._liste_fuellen(len(self.bilder) - 1)
        if schlecht:
            self.status.setzen(f"nicht lesbar: {', '.join(schlecht)}", "fehler")
        if neu:
            self._nach_aenderung(f"{neu} Bild(er) geladen")

    def _bild_entfernen(self):
        i = self.index
        if i is None:
            return
        weg = self.bilder.pop(i)["name"]
        self._liste_fuellen(max(0, i - 1))
        self._nach_aenderung(f"{weg} entfernt")

    def _alle_entfernen(self):
        self.bilder = []
        self.auto = self.aktiv = None
        self._liste_fuellen()
        self.leinwand.quelle = None
        self.z_auto.setzen("noch keine Bilder geladen")
        self.status.setzen("alle Bilder entfernt")

    def _nach_aenderung(self, text):
        """Nach jeder Aenderung der Liste neu rechnen - der Massstab soll immer
        zu dem gehoeren, was gerade in der Liste steht."""
        self.leinwand.werkzeug(None)
        self.l_werkzeug.configure(text="")
        if self.bilder:
            self.status.setzen(text + " - wird neu gerechnet ...", "normal")
            self.update_idletasks()
            self._automatisch()
        else:
            self._anzeigen()

    # ------------------------------------------------------------- Werkzeuge
    def _werkzeug(self, welches):
        if not self.bilder:
            self.status.setzen("erst Schachbrettbilder laden", "fehler"); return
        self._welches = welches
        if welches == "suchbereich":
            self.leinwand.werkzeug("rechteck")
            self.l_werkzeug.configure(
                text=f"Suchbereich fuer {self.bilder[self.index]['name']}")
        else:
            self.leinwand.werkzeug("rechteck" if self.modus == "rechteck" else "strecke")
            self.l_werkzeug.configure(
                text="Rechteck aufziehen" if self.modus == "rechteck" else "Strecke ziehen")
        self.status.setzen("im Bild ziehen - Mausrad zoomt, rechte Taste schiebt", "warn")

    def _form_fertig(self, form):
        if form is None or not self.bilder:
            return
        if self._welches == "suchbereich":
            i = self.index
            self.bilder[i]["suchbereich"] = form
            self._liste_fuellen(i)
            self.status.setzen(f"Suchbereich {form[2]-form[0]}x{form[3]-form[1]} px "
                               f"fuer {self.bilder[i]['name']} - jetzt "
                               f"'Automatisch kalibrieren'", "ok")
        else:
            self._hand_form = form
            self.status.setzen("Form gesetzt - jetzt 'Handmessung auswerten'", "ok")

    def _such_weg(self):
        i = self.index
        if i is None:
            return
        self.bilder[i]["suchbereich"] = None
        self._liste_fuellen(i)
        self.leinwand.werkzeug(None)
        self.l_werkzeug.configure(text="")
        self.status.setzen(f"Suchbereich fuer {self.bilder[i]['name']} aufgehoben")

    # ------------------------------------------------------------- Schritte
    def _ziele(self, endung):
        ordner = os.path.join(self.kfg["ergebnis_ordner"], "kalibrierung")
        return os.path.join(ordner, f"{self.kennung}_{endung}")

    def _ecken_raten(self, still=False):
        """Eckenzahl aus dem ausgewaehlten Bild bestimmen und eintragen."""
        i = self.index
        if i is None:
            if not still:
                self.status.setzen("erst Schachbrettbilder laden", "fehler")
            return False
        if not still:
            self.status.setzen("suche die Eckenzahl ...", "normal")
            self.update_idletasks()
        ecken, stufe = kal.ecken_raten(self.bilder[i]["grau"],
                                       suchbereich=self.bilder[i].get("suchbereich"))
        if ecken is None:
            if not still:
                self.status.setzen("keine bekannte Eckenzahl gefunden - Brett ganz "
                                   "im Bild? Suchbereich aufziehen?", "fehler")
            return False
        self.f_ex.delete(0, "end"); self.f_ex.insert(0, str(ecken[0]))
        self.f_ey.delete(0, "end"); self.f_ey.insert(0, str(ecken[1]))
        self._werte_sichern()
        self.status.setzen(f"{ecken[0]}x{ecken[1]} innere Ecken gefunden "
                           f"(ueber {stufe}) und eingetragen", "ok")
        return True

    def _automatisch(self, _zweiter_versuch=False):
        if not self.bilder:
            self.status.setzen("erst Schachbrettbilder laden", "fehler"); return
        self._werte_sichern()
        self.status.setzen(f"suche Schachbretter in {len(self.bilder)} Bild(ern) ...",
                           "normal")
        self.update_idletasks()
        b = kal.kalibrieren(self.bilder,
                            (self.kfg["ecken_x"], self.kfg["ecken_y"]),
                            self.kfg["feld_mm"], grad=self.kfg["grad"],
                            npz_pfad=self._ziele("kalibrierung.npz"),
                            png_pfad=self._ziele("kontrolle.png"),
                            name=self.titel, anzeige=self.index or 0)
        if not b["ok"]:
            # Die verwechselte Eckenzahl ist der haeufigste Grund. Statt nur zu
            # melden, einmal selbst nachsehen - und wenn etwas gefunden wird,
            # direkt damit weiterrechnen.
            if not _zweiter_versuch and self._ecken_raten(still=True):
                self.status.setzen(f"Eckenzahl korrigiert auf "
                                   f"{self.kfg['ecken_x']}x{self.kfg['ecken_y']} - "
                                   f"rechne neu ...", "warn")
                self.update_idletasks()
                return self._automatisch(_zweiter_versuch=True)
            self.auto = None
            self.z_auto.setzen(b["fehler"][:110], "fehler")
            self.status.setzen(b["fehler"], "fehler")
            self._quelle_gewechselt()
            return
        self.auto = b
        art = "ok" if (b["restfehler_prozent"] < 3 and b["gestuetzt_prozent"] > 25) else "warn"
        self.z_auto.setzen(kal.zusammenfassung(b), art)
        # Nicht erkannte Bilder sind kein Abbruch, aber sie muessen auffallen -
        # sonst kalibriert man mit der Haelfte der Positionen, ohne es zu merken.
        schlecht = [e["name"] for e in b["bilder"] if not e["ok"]]
        if self.v_quelle.get() == "auto":
            self._quelle_gewechselt()
        self._gegenprobe()
        self.status.setzen(
            (f"NICHT erkannt: {', '.join(schlecht)}  -  " if schlecht else "")
            + f"Kontrollbild: {b['png']}",
            "warn" if schlecht else art)

    def _von_hand(self):
        if not self.bilder:
            self.status.setzen("erst Schachbrettbilder laden", "fehler"); return
        if self._hand_form is None:
            self.status.setzen("erst eine Form im Bild ziehen", "fehler"); return
        self._werte_sichern()
        grau = self.grau
        shape = grau.shape
        if self.modus == "strecke":
            p0, p1 = self._hand_form
            b = kal.manuell_strecke(p0, p1, self.kfg["hand_laenge_mm"], shape, grau,
                                    npz_pfad=self._ziele("hand.npz"),
                                    png_pfad=self._ziele("hand_kontrolle.png"),
                                    name=self.titel)
        else:
            b = kal.manuell_rechteck(self._hand_form, self.kfg["hand_breite_mm"],
                                     self.kfg["hand_hoehe_mm"], shape, grau,
                                     npz_pfad=self._ziele("hand.npz"),
                                     png_pfad=self._ziele("hand_kontrolle.png"),
                                     name=self.titel)
        if not b["ok"]:
            self.hand = None
            self.z_hand.setzen(b["fehler"], "fehler")
            self.status.setzen(b["fehler"], "fehler"); return
        self.hand = b
        self.z_hand.setzen(kal.zusammenfassung(b), "ok")
        if self.v_quelle.get() == "hand":
            self._quelle_gewechselt()
        self._gegenprobe()

    def _gegenprobe(self):
        """Handmessung gegen die automatische Karte halten."""
        if not (self.auto and self.hand and self._hand_form is not None):
            self.l_gegen.configure(text="")
            return
        from .massstab import Massstab
        ms = Massstab(self.auto["npz"])
        if self.modus == "strecke":
            abw = kal.gegenprobe(ms.karte, self._hand_form, "strecke",
                                 soll_mm=self.kfg["hand_laenge_mm"])
        else:
            x0, y0, x1, y1 = self._hand_form
            abw = kal.gegenprobe(ms.karte, [(x0, y0), (x1, y0), (x1, y1), (x0, y1)],
                                 "rechteck",
                                 soll_mm2=self.kfg["hand_breite_mm"] * self.kfg["hand_hoehe_mm"])
        if abw is None:
            return
        text = (f"Gegenprobe: die Karte liefert fuer dieselbe Form {abw:+.2f}%.")
        if abs(abw) > 3:
            text += ("  Mehr als 3% - meist stimmt die eingetragene Feldgroesse "
                     "nicht mit dem Ausdruck ueberein. Nachmessen!")
        self.l_gegen.configure(text=text,
                               foreground=gui.FARBE_FEHLER if abs(abw) > 3 else "#666")

    # ------------------------------------------------------------- Anzeige
    def _quelle_gewechselt(self):
        self.aktiv = self.auto if self.v_quelle.get() == "auto" else self.hand
        self._anzeigen()

    def _anzeigen(self):
        grau = self.grau
        if grau is None:
            return
        farbig = cv2.cvtColor(grau, cv2.COLOR_GRAY2BGR)
        if not self.v_overlay.get() or self.aktiv is None:
            self.leinwand.bild_setzen(farbig, einpassen=False)
            return
        z = self.aktiv.get("_zeichnung")
        if z and tuple(z["karte"].shape) == tuple(grau.shape):
            # Overlay auf das GERADE GEWAEHLTE Bild neu zeichnen, ohne die
            # Erkennung zu wiederholen - so passt es beim Blaettern immer.
            bild = kal.kontrollbild(grau, z["ec_liste"], z["ecken"], z["punkte"],
                                    z["karte"], z["kopfzeilen"])
        else:
            bild = self.aktiv["bild"]
        self.leinwand.bild_setzen(bild, einpassen=False)

    # ------------------------------------------------------------- Uebernahme
    def _uebernehmen(self):
        if self.aktiv is None:
            self.status.setzen("es gibt noch keinen gueltigen Massstab", "fehler"); return
        npz = self.aktiv["npz"]
        self.kfg["kalibrierung"] = npz
        self.kfg["kalibrierung_art"] = self.aktiv["art"]
        self.kfg.speichern()
        self.z_gilt.setzen(f"{os.path.basename(npz)} | {kal.zusammenfassung(self.aktiv)}",
                           "ok")
        if self.bei_uebernahme:
            self.bei_uebernahme(npz)
        self.status.setzen("Massstab uebernommen - im Messreiter eingetragen", "ok")

    # ------------------------------------------------------------- Werte
    def _werte_sichern(self):
        k = self.kfg
        k["feld_mm"] = _zahl(self.f_feld.get(), 10.0)
        k["ecken_x"] = int(_zahl(self.f_ex.get(), 5))
        k["ecken_y"] = int(_zahl(self.f_ey.get(), 8))
        k["grad"] = self.v_grad.get()
        if self.modus == "strecke":
            k["hand_laenge_mm"] = _zahl(self.f_laenge.get(), 100.0)
        else:
            k["hand_breite_mm"] = _zahl(self.f_breite.get(), 100.0)
            k["hand_hoehe_mm"] = _zahl(self.f_hoehe.get(), 100.0)
        k.speichern()


def _zahl(text, standard):
    try:
        return float(str(text).replace(",", "."))
    except ValueError:
        return standard
