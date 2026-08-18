# -*- coding: utf-8 -*-
"""
ordnerwache.py - Neue Bilder in einem Kameraordner erkennen.

Drei Dinge, die am Messtag zaehlen:

1. WANN IST EINE DATEI FERTIG? Die Kamerasoftware legt die Datei an und fuellt
   sie danach; wer sofort liest, bekommt ein halbes Bild oder gar nichts.
   cv2.imread meldet das nicht als Fehler, sondern gibt stillschweigend None
   zurueck - dieser Fehler hat uns in der Auswertung schon einen kompletten
   Serienlauf gekostet. Deshalb gilt eine Datei erst als fertig, wenn ihre
   Groesse ueber zwei Abfragen hinweg unveraendert und groesser als null ist.

2. WELCHER UNTERORDNER IST GERADE DRAN? Die Kamera legt je Lauf einen eigenen
   Unterordner an. 'automatisch' waehlt den zuletzt beschriebenen - startet die
   Kamera einen neuen Lauf, wechselt die Wache von selbst mit, ohne dass am
   Versuchs-PC jemand etwas umstellen muss.

3. LIVE ODER LUECKENLOS? Bei einer Live-Ueberwachung zaehlt der aktuelle
   Zustand, nicht die Vollstaendigkeit: rechnet die Auswertung langsamer als die
   Kamera aufnimmt, muss sie Bilder ueberspringen duerfen, sonst laeuft die
   Anzeige der Wirklichkeit immer weiter hinterher. Dafuer gibt es neueste().
   Fuer eine luecklose Auswertung gibt es neue().
"""
import os
import time


ENDUNGEN = (".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp")

AUTO = "automatisch (neuester)"
MAX_TIEFE = 2            # Unterordner-Tiefe, die durchsucht wird
ORDNER_INTERVALL = 2.0   # s zwischen zwei Pruefungen, welcher Ordner aktiv ist


class Ordnerwache:
    def __init__(self, wurzel, unterordner=AUTO, endungen=ENDUNGEN, ab_bestand=True):
        """ab_bestand=True: was beim Start schon im Ordner liegt, gilt als
        gesehen. Beim Wechsel auf einen NEUEN Unterordner gilt das nicht - dort
        beginnt ein neuer Lauf, dessen Bilder alle ausgewertet werden sollen."""
        self.wurzel = os.path.normpath(wurzel)
        self.wahl = unterordner or AUTO
        self.endungen = tuple(e.lower() for e in endungen)
        self.gesehen = set()
        self.groessen = {}
        self.uebersprungen = 0
        self._geprueft = 0.0
        self.aktiv = self._bestimmen()
        if ab_bestand:
            self.gesehen = set(self._dateien())

    # ------------------------------------------------------------ Unterordner
    def ordner_liste(self):
        """Unterordner unterhalb der Wurzel (relative Pfade), fuer die Auswahl
        in der Oberflaeche."""
        aus = []
        for weg, dirs, _ in os.walk(self.wurzel):
            if weg[len(self.wurzel):].count(os.sep) >= MAX_TIEFE:
                dirs[:] = []
                continue
            for d in sorted(dirs):
                aus.append(os.path.relpath(os.path.join(weg, d), self.wurzel))
        return aus

    def _hat_bilder(self, ordner):
        try:
            return any(f.lower().endswith(self.endungen) for f in os.listdir(ordner))
        except OSError:
            return False

    @staticmethod
    def _zeit(pfad):
        try:
            return os.path.getmtime(pfad)
        except OSError:
            return -1.0

    def _bestimmen(self):
        """Welcher Ordner wird ueberwacht?

        Bei 'automatisch' der zuletzt beschriebene Ordner, der ueberhaupt Bilder
        enthaelt. Bewertet wird die Aenderungszeit des ORDNERS, nicht die aller
        Dateien darin - das bleibt auch bei zehntausend Bildern auf einem
        Netzlaufwerk billig."""
        if self.wahl and self.wahl != AUTO:
            p = os.path.join(self.wurzel, self.wahl)
            return p if os.path.isdir(p) else self.wurzel
        bester, beste_zeit = None, -1.0
        for rel in [None] + self.ordner_liste():
            p = self.wurzel if rel is None else os.path.join(self.wurzel, rel)
            if not self._hat_bilder(p):
                continue
            t = self._zeit(p)
            if t > beste_zeit:
                bester, beste_zeit = p, t
        return bester or self.wurzel

    def _ordner_pruefen(self):
        """Hat die Kamera einen neuen Lauf begonnen? Nur alle paar Sekunden,
        nicht bei jeder Abfrage."""
        jetzt = time.time()
        if jetzt - self._geprueft < ORDNER_INTERVALL:
            return False
        self._geprueft = jetzt
        neu = self._bestimmen()
        if neu == self.aktiv:
            return False
        self.aktiv = neu
        self.gesehen.clear()      # neuer Lauf: alles darin ist auszuwerten
        self.groessen.clear()
        return True

    @property
    def aktiv_kurz(self):
        rel = os.path.relpath(self.aktiv, self.wurzel)
        return "." if rel == os.curdir else rel

    # ------------------------------------------------------------ Bilder
    def _dateien(self):
        try:
            return sorted(f for f in os.listdir(self.aktiv)
                          if f.lower().endswith(self.endungen))
        except OSError:
            return []

    def neue(self):
        """Alle fertig geschriebenen, noch nicht ausgelieferten Bilder in
        Namensreihenfolge (volle Pfade)."""
        self._ordner_pruefen()
        fertig = []
        for name in self._dateien():
            if name in self.gesehen:
                continue
            pfad = os.path.join(self.aktiv, name)
            try:
                groesse = os.path.getsize(pfad)
            except OSError:
                continue
            if groesse > 0 and self.groessen.get(name) == groesse:
                self.gesehen.add(name)
                self.groessen.pop(name, None)
                fertig.append(pfad)
            else:
                self.groessen[name] = groesse     # naechste Runde nochmal pruefen
        return fertig

    def neueste(self):
        """Nur das aktuellste fertige Bild; alles davor gilt als erledigt.

        Das ist der Live-Betrieb: Was zaehlt, ist der Zustand JETZT. Wie viele
        Bilder dabei uebersprungen wurden, steht in .uebersprungen und gehoert
        in die Statuszeile - sonst bliebe unbemerkt, dass die Auswertung der
        Kamera nicht folgen kann."""
        alle = self.neue()
        if not alle:
            return None
        self.uebersprungen += len(alle) - 1
        return alle[-1]

    def zuruecksetzen(self, ab_bestand=True):
        self.aktiv = self._bestimmen()
        self.gesehen = set(self._dateien()) if ab_bestand else set()
        self.groessen.clear()
        self.uebersprungen = 0

    def letztes_bild(self):
        """Das zuletzt geschriebene Bild im aktiven Ordner - unabhaengig davon,
        ob es schon ausgewertet wurde. Fuer die Kalibrierung, die auf einem
        stehenden Bild arbeitet."""
        self._ordner_pruefen()
        namen = self._dateien()
        return os.path.join(self.aktiv, namen[-1]) if namen else None


def bild_lesen(pfad, versuche=5):
    """imread mit Wiederholung. Zusaetzliche Absicherung gegen kurzzeitig
    nicht lesbare Dateien (Netzlaufwerk, Virenscanner, Synchronisation)."""
    import cv2
    for i in range(versuche):
        img = cv2.imread(pfad, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            return img
        time.sleep(0.15 * (i + 1))
    return None
