# -*- coding: utf-8 -*-
"""
massstab.py - Kalibrierung anwenden: Pixel -> Millimeter.

Nutzt die Massstabskarte aus gemeinsam/kalibrierung.py. Ohne Kalibrierung
laeuft alles weiter, dann aber ausschliesslich in Pixel und Prozent - die
Oberflaeche zeigt das deutlich an, damit am Ende niemand Pixelwerte fuer
Millimeter haelt.
"""
import os
import numpy as np


class Massstab:
    def __init__(self, npz_pfad=None):
        self.karte = None            # mm^2 je Pixel, volle Bildgroesse
        self.metrik = None           # (gxx, gxy, gyy) fuer Laengen je Richtung
        self.quelle = "keine Kalibrierung - Ausgabe nur in Pixel/Prozent"
        self.restfehler = None
        if npz_pfad:
            self.laden(npz_pfad)

    def laden(self, npz_pfad):
        if not npz_pfad or not os.path.exists(npz_pfad):
            self.karte = None
            self.metrik = None
            self.quelle = "keine Kalibrierung - Ausgabe nur in Pixel/Prozent"
            return False
        try:
            d = np.load(npz_pfad)
            koef, shape, grad = d["koeffizienten"], tuple(d["shape"]), int(d["grad"])
            # grenzen = kleinste/groesste GEMESSENE Skala. Damit wird dieselbe
            # Kappung angewendet wie beim Erzeugen der Karte; sonst wuerde die
            # Auswertung ausserhalb des Schachbretts andere Werte benutzen als
            # das Kontrollbild gezeigt hat.
            grenzen = tuple(d["grenzen"]) if "grenzen" in d else None
            self.karte = self._karte_bauen(koef, shape, grad, grenzen)
            # Metrik ist optional: Handmessungen und aeltere Dateien haben
            # keine. Dann werden Laengen isotrop aus der Flaechenskala
            # genaehert - siehe laenge_mm.
            if "metrik_koeffizienten" in d:
                from gemeinsam.kalibrierung import metrik_bauen
                self.metrik = metrik_bauen(d["metrik_koeffizienten"], shape, grad)
            else:
                self.metrik = None
            self.restfehler = float(d["restfehler_prozent"]) if "restfehler_prozent" in d else None
            rest = f", Restfehler {self.restfehler:.2f}%" if self.restfehler is not None else ""
            self.quelle = f"{os.path.basename(npz_pfad)}{rest}"
            return True
        except Exception as e:
            self.karte = None
            self.metrik = None
            self.quelle = f"Kalibrierung nicht lesbar ({e}) - nur Pixel/Prozent"
            return False

    @staticmethod
    def _karte_bauen(koef, shape, grad, grenzen=None):
        H, W = shape
        yy, xx = np.mgrid[0:H, 0:W]
        x = xx / W * 2 - 1
        y = yy / H * 2 - 1
        spalten = [x.ravel()**i * y.ravel()**j
                   for i in range(grad + 1) for j in range(grad + 1 - i)]
        A = np.stack(spalten, axis=-1)
        karte = np.exp(A @ koef).reshape(H, W)
        if grenzen is not None:
            from gemeinsam.kalibrierung import KAPPUNG
            lo, hi = float(grenzen[0]), float(grenzen[1])
            karte = np.clip(karte, lo / KAPPUNG, hi * KAPPUNG)
        return karte.astype(np.float32)

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

    def laenge_mm(self, p0, p1, schritte=64):
        """Laenge gerader Strecken in mm, PIXELWEISE und RICHTUNGSABHAENGIG.

        p0, p1: (N,2)-Arrays mit Anfangs- und Endpunkt je Strecke in Pixeln.
        -> (N,) Laengen in mm, NaN wo die Eingabe NaN ist.

        Zwei Dinge, die ein einzelner px/mm-Wert nicht kann:

        ORT: Der Massstab ist ortsabhaengig. Ueber ein Bildfeld schwankt er je
        nach Blickwinkel um Zehnerprozente; ein Median waere am Rand
        systematisch falsch - gerade dort, wo bei der Laserlinie der
        interessante Teil der Nase liegt. Deshalb wird entlang jeder Strecke
        abgetastet.

        RICHTUNG: Blickt die Kamera schraeg auf die Flaeche, ist die
        Verkuerzung in den beiden Hauptrichtungen verschieden. sqrt(Flaechen-
        skala) ist nur ihr geometrisches Mittel; eine Laenge in EINER Richtung
        weicht davon um bis zu der Anisotropie ab. Mit der Metrik G ist die
        Laenge eines Pixelvektors v exakt sqrt(v^T G v).

        Ohne Metrik in der Kalibrierdatei (Handmessung, aeltere Datei) wird
        isotrop aus der Flaechenskala genaehert."""
        if self.karte is None:
            return None
        p0 = np.asarray(p0, float).reshape(-1, 2)
        p1 = np.asarray(p1, float).reshape(-1, 2)
        H, W = self.karte.shape
        vx, vy = p1[:, 0] - p0[:, 0], p1[:, 1] - p0[:, 1]
        d = np.hypot(vx, vy)
        gueltig = np.isfinite(d) & np.isfinite(p0).all(1) & np.isfinite(p1).all(1)
        aus = np.full(len(d), np.nan)
        if not gueltig.any():
            return aus
        n = int(np.clip(np.ceil(np.nanmax(d[gueltig])), 2, schritte))
        t = np.linspace(0.0, 1.0, n)[None, :]
        a, b = p0[gueltig], p1[gueltig]
        X = a[:, 0:1] + t * (b[:, 0:1] - a[:, 0:1])
        Y = a[:, 1:2] + t * (b[:, 1:2] - a[:, 1:2])
        xi = np.clip(np.round(X).astype(int), 0, W - 1)
        yi = np.clip(np.round(Y).astype(int), 0, H - 1)
        if self.metrik is None:
            mm_je_px = np.sqrt(self.karte[yi, xi])
        else:
            laenge = np.where(d[gueltig] > 0, d[gueltig], 1.0)
            ux = (vx[gueltig] / laenge)[:, None]        # Einheitsrichtung
            uy = (vy[gueltig] / laenge)[:, None]
            gxx, gxy, gyy = self.metrik
            mm_je_px = np.sqrt(np.maximum(
                gxx[yi, xi] * ux**2 + 2 * gxy[yi, xi] * ux * uy + gyy[yi, xi] * uy**2,
                0.0))
        aus[gueltig] = d[gueltig] * mm_je_px.mean(axis=1)
        return aus

    def px_pro_mm_lokal(self, x, y):
        """Linearer Massstab an einzelnen Bildstellen -> px/mm je Punkt
        (isotrope Naeherung, nur zur Anzeige)."""
        if self.karte is None:
            return None
        H, W = self.karte.shape
        xi = np.clip(np.round(np.asarray(x, float)).astype(int), 0, W - 1)
        yi = np.clip(np.round(np.asarray(y, float)).astype(int), 0, H - 1)
        return 1.0 / np.sqrt(self.karte[yi, xi])

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
