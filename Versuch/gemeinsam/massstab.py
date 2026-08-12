# -*- coding: utf-8 -*-
"""
massstab.py - Kalibrierung anwenden: Pixel -> Millimeter.

Nutzt die Maszstabskarte aus kalibrierung_schachbrett.py. Ohne Kalibrierung
laeuft alles weiter, dann aber ausschliesslich in Pixel und Prozent - die
Oberflaeche zeigt das deutlich an, damit am Ende niemand Pixelwerte fuer
Millimeter haelt.
"""
import os
import numpy as np


class Massstab:
    def __init__(self, npz_pfad=None):
        self.karte = None            # mm^2 je Pixel, volle Bildgroesse
        self.quelle = "keine Kalibrierung - Ausgabe nur in Pixel/Prozent"
        self.restfehler = None
        if npz_pfad:
            self.laden(npz_pfad)

    def laden(self, npz_pfad):
        if not npz_pfad or not os.path.exists(npz_pfad):
            self.karte = None
            self.quelle = "keine Kalibrierung - Ausgabe nur in Pixel/Prozent"
            return False
        try:
            d = np.load(npz_pfad)
            koef, shape, grad = d["koeffizienten"], tuple(d["shape"]), int(d["grad"])
            self.karte = self._karte_bauen(koef, shape, grad)
            self.restfehler = float(d["restfehler_prozent"]) if "restfehler_prozent" in d else None
            rest = f", Restfehler {self.restfehler:.2f}%" if self.restfehler is not None else ""
            self.quelle = f"{os.path.basename(npz_pfad)}{rest}"
            return True
        except Exception as e:
            self.karte = None
            self.quelle = f"Kalibrierung nicht lesbar ({e}) - nur Pixel/Prozent"
            return False

    @staticmethod
    def _karte_bauen(koef, shape, grad):
        H, W = shape
        yy, xx = np.mgrid[0:H, 0:W]
        x = xx / W * 2 - 1
        y = yy / H * 2 - 1
        spalten = [x.ravel()**i * y.ravel()**j
                   for i in range(grad + 1) for j in range(grad + 1 - i)]
        A = np.stack(spalten, axis=-1)
        return np.exp(A @ koef).reshape(H, W).astype(np.float32)

    def flaeche_mm2(self, maske, versatz=(0, 0)):
        """Reale Flaeche der True-Pixel in mm^2, oder None ohne Kalibrierung.
        versatz = (x0, y0) des Crops im Vollbild, damit die Karte trotz
        Zuschnitt an der richtigen Stelle ausgelesen wird."""
        if self.karte is None:
            return None
        x0, y0 = versatz
        h, w = maske.shape
        aus = self.karte[y0:y0 + h, x0:x0 + w]
        if aus.shape != maske.shape:
            return None                     # Crop passt nicht zur Kalibrierung
        return float(aus[maske].sum())

    def px_pro_mm(self, versatz=(0, 0), shape=None):
        """Linearer Massstab im Median - nur zur Anzeige."""
        if self.karte is None:
            return None
        k = self.karte
        if shape is not None:
            x0, y0 = versatz
            k = k[y0:y0 + shape[0], x0:x0 + shape[1]]
        return float(np.median(1.0 / np.sqrt(np.maximum(k, 1e-12))))

    @property
    def vorhanden(self):
        return self.karte is not None
