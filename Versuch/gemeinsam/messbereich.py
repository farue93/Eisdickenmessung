# -*- coding: utf-8 -*-
"""
messbereich.py - Den Messbereich vorschlagen und speichern.

Der Messbereich legt fest, WORAUF sich der Bedeckungsgrad bezieht. Er muss vor
Ort neu gesetzt werden: Eine Maske aus einem frueheren Aufbau gehoert zur
damaligen Kameraposition und weist nach dem Neuausrichten den falschen
Bildbereich als Bezugsflaeche aus.

Der Vorschlag nimmt einem den ersten Wurf ab, mehr nicht - er ist ausdruecklich
zum Anpassen gedacht. Verfahren: Otsu auf dem verkleinerten Bild, groesste
zusammenhaengende helle Flaeche, Loecher schliessen, umschliessendes Rechteck.

Warum das reicht und warum nicht mehr:
Das Panel ist im Bild die grosse zusammenhaengende helle Flaeche - dunkler
Kanalhintergrund ringsum. Auf der Testreihe trifft der Vorschlag das von Hand
gepruefte SAM-Ergebnis mit IoU 0,87; er sitzt am selben oberen Rand und ist
etwas schmaler. Genau dafuer gibt es die Anfasser.

Bewusst OHNE SAM: kein 40-MB-Modell, keine Zusatzpakete, kein Fehlschlag im
unguenstigen Moment. Die physikalisch belastbare Groesse ist ohnehin die
Flaeche in mm^2 aus der Kalibrierung; der Prozentwert bezieht sich auf dieses
Rechteck und ist als solcher zu protokollieren.
"""
import os
import numpy as np
import cv2

KLEIN = 0.2          # Verkleinerung fuer die Suche


def vorschlagen(grau, klein=KLEIN):
    """Groesste zusammenhaengende helle Flaeche -> (x0, y0, x1, y1) im
    Vollbild, oder None."""
    if grau is None or grau.size == 0:
        return None
    k = cv2.resize(grau, (0, 0), fx=klein, fy=klein, interpolation=cv2.INTER_AREA)
    weich = cv2.GaussianBlur(k, (0, 0), 3)
    _, binaer = cv2.threshold(weich, 0, 255,
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Schliessen, damit Reflexe und dunkle Streifen das Panel nicht zerlegen
    binaer = cv2.morphologyEx(binaer, cv2.MORPH_CLOSE,
                              cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21)))
    n, _, st, _ = cv2.connectedComponentsWithStats(binaer)
    if n < 2:
        return None
    i = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    x, y = st[i, cv2.CC_STAT_LEFT], st[i, cv2.CC_STAT_TOP]
    b, h = st[i, cv2.CC_STAT_WIDTH], st[i, cv2.CC_STAT_HEIGHT]
    f = 1.0 / klein
    H, W = grau.shape[:2]
    return (int(max(0, round(x * f))), int(max(0, round(y * f))),
            int(min(W, round((x + b) * f))), int(min(H, round((y + h) * f))))


def maske_bauen(shape, rechteck, ausschluss=None):
    """Rechteck minus ausgeschlossene Stellen -> uint8-Maske in Bildgroesse."""
    m = np.zeros(shape[:2], np.uint8)
    if rechteck is None:
        return m
    x0, y0, x1, y1 = [int(v) for v in rechteck]
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(shape[1], x1), min(shape[0], y1)
    if x1 <= x0 or y1 <= y0:
        return m
    m[y0:y1, x0:x1] = 1
    if ausschluss is not None:
        m[ausschluss > 0] = 0
    return m


def speichern(pfad, grau, rechteck, ausschluss=None, quelle=""):
    """Messbereich ablegen -> Bericht. Format wie bisher: volle Maske + bbox,
    damit die Auswertung nichts umlernen muss."""
    m = maske_bauen(grau.shape, rechteck, ausschluss)
    if not m.any():
        return dict(ok=False, fehler="leerer Messbereich")
    x0, y0, x1, y1 = [int(v) for v in rechteck]
    os.makedirs(os.path.dirname(pfad) or ".", exist_ok=True)
    np.savez(pfad, maske=m, bbox=np.array([x0, y0, x1, y1]),
             flaeche_px=int(m.sum()), quelle=quelle or "")
    png = os.path.splitext(pfad)[0] + "_kontrolle.png"
    cv2.imwrite(png, kontrollbild(grau, m, (x0, y0, x1, y1)))
    return dict(ok=True, fehler=None, npz=pfad, png=png,
                px=int(m.sum()), rechteck=(x0, y0, x1, y1),
                breite=x1 - x0, hoehe=y1 - y0,
                ausgeschlossen=int((x1 - x0) * (y1 - y0) - m.sum()))


def kontrollbild(grau, maske, rechteck, breit=1400):
    vis = cv2.cvtColor(np.clip(grau.astype(np.float32) * 1.3, 0, 255).astype(np.uint8),
                       cv2.COLOR_GRAY2BGR)
    vis[maske == 0] = (vis[maske == 0] * 0.3).astype(np.uint8)
    x0, y0, x1, y1 = rechteck
    cv2.rectangle(vis, (x0, y0), (x1, y1), (0, 200, 255), max(2, grau.shape[1] // 600))
    H, W = grau.shape[:2]
    breit = min(breit, W)
    return cv2.resize(vis, (breit, int(H * breit / W)), interpolation=cv2.INTER_AREA)


def bericht_text(b):
    """Eine Zeile, die den Messbereich vollstaendig beschreibt."""
    if not b.get("ok"):
        return b.get("fehler", "Messbereich fehlgeschlagen")
    punkt = lambda v: f"{v:,}".replace(",", ".")
    x0, y0, _, _ = b["rechteck"]
    aus = (f"{b['breite']}x{b['hoehe']} ab ({x0},{y0}) | "
           f"{punkt(b['px'])} px Bezugsflaeche")
    if b["ausgeschlossen"]:
        aus += f" | {punkt(b['ausgeschlossen'])} px ausgeschlossen"
    return aus
