# -*- coding: utf-8 -*-
"""
schachbrett_drucken.py - Druckvorlage fuer die Kalibrierung erzeugen.

OHNE JEDE ZUSATZBIBLIOTHEK. Die Vorlage muss sich auf jedem Rechner erzeugen
lassen, auch auf einem Versuchs-PC ohne installierte Pakete - deshalb wird das
PDF hier direkt geschrieben. Der Inhalt ist einfach genug dafuer: schwarze
Rechtecke, Linien und Text in Helvetica, einer der 14 PDF-Standardschriften,
die nicht eingebettet werden muessen.

Ausgabe in EXAKTER physischer Groesse. PDF statt Bild, weil ein PDF echte
Millimeterangaben traegt - ein PNG nur Pixel plus eine DPI-Angabe, die viele
Druckertreiber ignorieren.

WICHTIG BEIM DRUCKEN:
  - "Tatsaechliche Groesse" / "100 %" waehlen, NICHT "An Seite anpassen".
    Anpassen skaliert um einige Prozent, und dieser Fehler ginge unbemerkt in
    JEDE spaetere Millimeterangabe ein.
  - Mattes Papier oder matte Klebefolie, kein Glanz.
  - Nach dem Druck die Kontrollstrecke auf dem Blatt nachmessen.

Aufruf:
  python schachbrett_drucken.py                 Standard: 10 mm, 5x8 innere Ecken
  python schachbrett_drucken.py 5               5-mm-Felder
  python schachbrett_drucken.py 10 7 10         10 mm, 7x10 innere Ecken
"""
import os, sys

BASE = os.path.dirname(os.path.abspath(__file__))

FELD_MM = 10.0        # Kantenlaenge eines Feldes
ECKEN = (5, 8)        # INNERE Ecken (Spalten, Zeilen) -> Felder = +1 je Richtung
KOPIEN = 2            # Bretter je Blatt (das zweite als Ersatz)
MESSSTRECKE_MM = 100.0

PT = 72.0 / 25.4      # Punkte je Millimeter
A4 = (210.0, 297.0)   # mm


class Seite:
    """Sammelt PDF-Zeichenbefehle in Millimetern (Ursprung links unten)."""

    def __init__(self):
        self.teile = []

    def rechteck(self, x, y, b, h, grau=0.0, gefuellt=True, strich=0.0):
        self.teile.append(f"{grau:.3f} {'g' if gefuellt else 'G'}")
        if not gefuellt:
            self.teile.append(f"{strich:.2f} w")
        self.teile.append(f"{x*PT:.3f} {y*PT:.3f} {b*PT:.3f} {h*PT:.3f} re "
                          f"{'f' if gefuellt else 'S'}")

    def linie(self, x1, y1, x2, y2, grau=0.0, breite=0.3):
        self.teile.append(f"{grau:.3f} G {breite:.2f} w "
                          f"{x1*PT:.3f} {y1*PT:.3f} m {x2*PT:.3f} {y2*PT:.3f} l S")

    def text(self, x, y, inhalt, groesse=9.0, schrift="F1", grau=0.0):
        sicher = (inhalt.replace("\\", r"\\").replace("(", r"\(").replace(")", r"\)"))
        self.teile.append(f"BT {grau:.3f} g /{schrift} {groesse:.1f} Tf "
                          f"{x*PT:.3f} {y*PT:.3f} Td ({sicher}) Tj ET")

    def text_mittig(self, xm, y, inhalt, groesse=9.0, schrift="F1"):
        # Helvetica-Breiten grob ueber 0,5 * Groesse je Zeichen abgeschaetzt;
        # fuer kurze Beschriftungen genau genug
        breite_pt = len(inhalt) * groesse * 0.5
        self.text(xm - breite_pt / 2 / PT, y, inhalt, groesse, schrift)

    def strom(self):
        return "\n".join(self.teile).encode("cp1252", errors="replace")


def pdf_schreiben(pfad, seite, groesse_mm=A4):
    """Minimales, gueltiges PDF mit einer Seite und zwei Standardschriften."""
    inhalt = seite.strom()
    b, h = groesse_mm[0] * PT, groesse_mm[1] * PT

    objekte = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        f"<< /Type /Pages /Kids [3 0 R] /Count 1 >>".encode(),
        (f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {b:.3f} {h:.3f}] "
         f"/Resources << /Font << /F1 5 0 R /F2 6 0 R /F3 7 0 R >> >> "
         f"/Contents 4 0 R >>").encode(),
        b"<< /Length " + str(len(inhalt)).encode() + b" >>\nstream\n" + inhalt + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold /Encoding /WinAnsiEncoding >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Oblique /Encoding /WinAnsiEncoding >>",
    ]

    aus = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    versaetze = []
    for i, o in enumerate(objekte, start=1):
        versaetze.append(len(aus))
        aus += f"{i} 0 obj\n".encode() + o + b"\nendobj\n"

    xref = len(aus)
    aus += f"xref\n0 {len(objekte)+1}\n".encode()
    aus += b"0000000000 65535 f \n"
    for v in versaetze:
        aus += f"{v:010d} 00000 n \n".encode()
    aus += (f"trailer\n<< /Size {len(objekte)+1} /Root 1 0 R >>\n"
            f"startxref\n{xref}\n%%EOF\n").encode()

    open(pfad, "wb").write(bytes(aus))


def bauen(feld_mm=FELD_MM, ecken=ECKEN, kopien=KOPIEN):
    spalten_f, zeilen_f = ecken[0] + 1, ecken[1] + 1
    brett_b, brett_h = spalten_f * feld_mm, zeilen_f * feld_mm
    s = Seite()
    oben = A4[1]

    s.text(20, oben - 20, "Kalibriermuster Eiswindkanal", 13, "F2")
    zeilen = [
        f"Feldgroesse {feld_mm:g} mm  -  {ecken[0]} x {ecken[1]} innere Ecken  "
        f"({spalten_f} x {zeilen_f} Felder)  -  Brett {brett_b:g} x {brett_h:g} mm",
        "Drucken mit TATSAECHLICHER GROESSE / 100 % - nicht 'An Seite anpassen'.",
        "Mattes Papier oder matte Klebefolie verwenden, kein Glanz.",
        "Nach dem Druck die Kontrollstrecke unten nachmessen und den gemessenen",
        "Wert als feld_mm in kalibrierung_schachbrett.py eintragen.",
    ]
    for i, z in enumerate(zeilen):
        s.text(20, oben - 27 - i * 5, z, 9.5)

    ruhe = max(feld_mm, 8)
    gesamt = kopien * brett_b + (kopien - 1) * 22
    x_start = (A4[0] - gesamt) / 2
    y_unten = oben - 62 - brett_h

    for k in range(kopien):
        x0 = x_start + k * (brett_b + 22)
        for i in range(zeilen_f):
            for j in range(spalten_f):
                if (i + j) % 2 == 0:
                    s.rechteck(x0 + j * feld_mm, y_unten + i * feld_mm,
                               feld_mm, feld_mm, grau=0.0)
        s.rechteck(x0, y_unten, brett_b, brett_h, gefuellt=False, strich=0.3)

        # Eckwinkel ausserhalb der Ruhezone: beim Zuschneiden hier bleiben,
        # der weisse Rand gehoert zum Muster
        for ex, ey, sx, sy in ((x0 - ruhe, y_unten - ruhe, 1, 1),
                               (x0 + brett_b + ruhe, y_unten - ruhe, -1, 1),
                               (x0 - ruhe, y_unten + brett_h + ruhe, 1, -1),
                               (x0 + brett_b + ruhe, y_unten + brett_h + ruhe, -1, -1)):
            s.linie(ex, ey, ex + sx * 6, ey, grau=0.55, breite=0.4)
            s.linie(ex, ey, ex, ey + sy * 6, grau=0.55, breite=0.4)

        s.text_mittig((x0 + brett_b / 2) * PT, y_unten - ruhe - 5,
                      f"{feld_mm:g} mm Felder", 8)

    # Kontrollstrecke
    y_skala = y_unten - 34
    s.linie(20, y_skala, 20 + MESSSTRECKE_MM, y_skala, grau=0.0, breite=0.8)
    for i in range(int(MESSSTRECKE_MM // 10) + 1):
        s.linie(20 + i * 10, y_skala, 20 + i * 10, y_skala + (3.5 if i % 5 == 0 else 2),
                grau=0.0, breite=0.8)
    s.text(20, y_skala - 4.5, f"Kontrollstrecke {MESSSTRECKE_MM:.0f},0 mm", 9, "F2")
    s.text(66, y_skala - 4.5, "nachmessen! Abweichung = Druckerskalierung", 8)

    s.text(20, 18, "Der weisse Rand um jedes Brett gehoert dazu (Ruhezone) - beim "
                   "Zuschneiden an den Eckwinkeln stehen lassen,", 8.5, "F3")
    s.text(20, 13.5, "sonst findet die Eckenerkennung das Muster nicht.", 8.5, "F3")

    pfad = os.path.join(BASE, f"schachbrett_{feld_mm:g}mm_{ecken[0]}x{ecken[1]}.pdf")
    pdf_schreiben(pfad, s)
    return pfad, (brett_b, brett_h)


FORMATE = {"a4": (210.0, 297.0), "a3": (297.0, 420.0)}
RAND_MIN_MM = 12.0     # weisse Ruhezone ringsum, mindestens


def blatt_bauen(feld_mm, formatname):
    """Reiner Musterbogen: nur das Schachbrett, sonst nichts auf dem Blatt.

    Der weisse Rand ist KEIN Gestaltungsrest, sondern Voraussetzung: Ein bis an
    den Blattrand laufendes Muster wird von findChessboardCornersSB nicht
    gefunden, weil die aeusserste Feldreihe keine geschlossene Umgebung hat.

    Die Feldzahl wird so gewaehlt, dass die INNEREN Ecken gerade x ungerade
    sind. Bei gleicher Paritaet in beiden Richtungen ist das Brett um 180 Grad
    mehrdeutig und die Zuordnung kann kippen.
    """
    seite_b, seite_h = FORMATE[formatname]
    nutz_b, nutz_h = seite_b - 2 * RAND_MIN_MM, seite_h - 2 * RAND_MIN_MM
    felder_b = int(nutz_b // feld_mm)
    felder_h = int(nutz_h // feld_mm)

    # innere Ecken = Felder - 1; auf gerade x ungerade bringen
    if (felder_b - 1) % 2 != 0:      # Spalten-Ecken sollen gerade sein
        felder_b -= 1
    if (felder_h - 1) % 2 == 0:      # Zeilen-Ecken sollen ungerade sein
        felder_h -= 1

    brett_b, brett_h = felder_b * feld_mm, felder_h * feld_mm
    x0, y0 = (seite_b - brett_b) / 2, (seite_h - brett_h) / 2

    s = Seite()
    for i in range(felder_h):
        for j in range(felder_b):
            if (i + j) % 2 == 0:
                s.rechteck(x0 + j * feld_mm, y0 + i * feld_mm, feld_mm, feld_mm, grau=0.0)

    ecken = (felder_b - 1, felder_h - 1)
    pfad = os.path.join(BASE, f"muster_{formatname.upper()}_{feld_mm:g}mm_"
                              f"{ecken[0]}x{ecken[1]}.pdf")
    pdf_schreiben(pfad, s, (seite_b, seite_h))
    return pfad, ecken, (brett_b, brett_h), (x0, y0)


def blaetter(feld_mm):
    print(f"Reine Musterboegen, Feldgroesse {feld_mm:g} mm\n")
    for name in ("a4", "a3"):
        pfad, ecken, (bb, bh), (rx, ry) = blatt_bauen(feld_mm, name)
        print(f"{name.upper()}: {ecken[0]} x {ecken[1]} innere Ecken, "
              f"Muster {bb:g} x {bh:g} mm, Rand {rx:.1f} / {ry:.1f} mm")
        print(f"   -> {os.path.basename(pfad)}")
        print(f"      in kalibrierung_schachbrett.py:  ecken=({ecken[0]}, {ecken[1]}), "
              f"feld_mm={feld_mm:g}\n")
    print("Drucken mit TATSAECHLICHER GROESSE / 100 %, mattes Papier.")
    print("Den weissen Rand NICHT wegschneiden - ohne ihn wird das Muster nicht erkannt.")
    print("Feldgroesse nach dem Druck nachmessen und den gemessenen Wert eintragen.")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1].lower() == "blatt":
        blaetter(float(sys.argv[2]) if len(sys.argv) > 2 else FELD_MM)
        sys.exit()

    feld = float(sys.argv[1]) if len(sys.argv) > 1 else FELD_MM
    ecken = (int(sys.argv[2]), int(sys.argv[3])) if len(sys.argv) > 3 else ECKEN
    pfad, (b, h) = bauen(feld, ecken)
    print(f"-> {pfad}")
    print(f"   {ecken[0]} x {ecken[1]} innere Ecken, Feld {feld:g} mm, Brett {b:g} x {h:g} mm")
    print(f"   In kalibrierung_schachbrett.py eintragen:")
    print(f"     ecken=({ecken[0]}, {ecken[1]}), feld_mm={feld:g}")
    print(f"   feld_mm nach dem Druck durch den NACHGEMESSENEN Wert ersetzen.")
