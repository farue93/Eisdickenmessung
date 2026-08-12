# -*- coding: utf-8 -*-
"""
ordnerwache.py - Neue Bilddateien in einem Ordner erkennen.

Der kritische Punkt ist NICHT das Finden neuer Dateien, sondern zu erkennen,
wann eine Datei FERTIG GESCHRIEBEN ist. Die Kamerasoftware legt die Datei an
und fuellt sie danach; wer sofort liest, bekommt ein halbes Bild oder gar
nichts. cv2.imread meldet das nicht als Fehler, sondern gibt stillschweigend
None zurueck - dieser Fehler hat uns in der Auswertung schon einen kompletten
Serienlauf gekostet.

Deshalb gilt eine Datei erst als fertig, wenn ihre Groesse ueber zwei
Abfragen hinweg unveraendert und groesser als null ist.
"""
import os
import time


ENDUNGEN = (".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp")


class Ordnerwache:
    def __init__(self, ordner, endungen=ENDUNGEN, ab_bestand=True):
        """ab_bestand=True: bereits vorhandene Dateien gelten als gesehen und
        werden nicht ausgewertet - beim Start soll nur Neues verarbeitet
        werden. Fuer die Auswertung eines fertigen Laufs auf False setzen."""
        self.ordner = ordner
        self.endungen = tuple(e.lower() for e in endungen)
        self.gesehen = set()
        self.groessen = {}
        if ab_bestand:
            self.gesehen = set(self._kandidaten())

    def _kandidaten(self):
        try:
            return [f for f in os.listdir(self.ordner)
                    if f.lower().endswith(self.endungen)]
        except OSError:
            return []

    def neue(self):
        """Liste fertig geschriebener, noch nicht ausgelieferter Dateien
        (vollstaendige Pfade, in Dateinamen-Reihenfolge)."""
        fertig = []
        for name in self._kandidaten():
            if name in self.gesehen:
                continue
            pfad = os.path.join(self.ordner, name)
            try:
                groesse = os.path.getsize(pfad)
            except OSError:
                continue
            vorher = self.groessen.get(name)
            if groesse > 0 and vorher == groesse:
                self.gesehen.add(name)
                self.groessen.pop(name, None)
                fertig.append(pfad)
            else:
                self.groessen[name] = groesse      # naechste Runde nochmal pruefen
        return sorted(fertig)

    def zuruecksetzen(self, ab_bestand=True):
        self.gesehen = set(self._kandidaten()) if ab_bestand else set()
        self.groessen.clear()


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
