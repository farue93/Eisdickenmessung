# -*- coding: utf-8 -*-
"""
kalibrierung_schachbrett.py - Massstab Pixel -> Millimeter aus Schachbrettbildern.

Liefert je Kameraansicht eine MASSSTABSKARTE: mm^2 reale Oberflaeche je
Bildpixel, ortsabhaengig. Ein einzelner px/mm-Wert genuegt nicht, weil die
Kamera schraeg auf eine gekruemmte Flaeche blickt - ein Pixel am gekruemmten
Rand deckt ein Vielfaches der Flaeche eines Pixels in der Bildmitte ab.

Verfahren:
  1. Schachbrettecken subpixelgenau finden (findChessboardCornersSB).
  2. Je Feld: bekannte reale Flaeche (feld_mm^2) geteilt durch die gemessene
     Pixelflaeche des Vierecks -> lokale Skala in mm^2/px^2.
  3. Diese Stuetzstellen mit einem 2D-Polynom ueber das ganze Bild fortsetzen.
     Gefittet wird der LOGARITHMUS der Skala, weil sie positiv und
     multiplikativ ist; ein Polynom auf dem Rohwert koennte negativ werden.
  4. Kontrollbild schreiben, damit am Messtag in Sekunden sichtbar ist, ob die
     Erkennung sitzt - solange das Brett noch klebt.

Aufruf:
  python kalibrierung_schachbrett.py          # alle Bilder in kalibrierung_bilder/
  python kalibrierung_schachbrett.py test     # Selbsttest mit synthetischem Brett

Dateinamen der Kalibrierbilder:  <ansicht>_<nr>.<endung>
  z.B. flaeche_unten_01.png, flaeche_unten_02.png, laser_01.png
Mehrere Bilder je Ansicht (Brett umgesetzt) werden zu EINEM Fit zusammengefasst;
das ist der Weg, wenn ein Brett das Bildfeld nicht allein ausfuellt.
"""
import os, sys, glob, json, re
import numpy as np
import cv2

BASE = os.path.dirname(os.path.abspath(__file__))
BILDER = os.path.join(BASE, "kalibrierung_bilder")
OUT = os.path.join(BASE, "kalibrierung_ergebnis")

# INNERE Ecken, nicht Felder!  Ein Brett mit 6x9 FELDERN hat 5x8 innere Ecken.
# Das ist der klassische Stolperstein - lieber einmal nachzaehlen.
BRETT = {
    "laser":         dict(ecken=(5, 8), feld_mm=10.0),
    "flaeche_unten": dict(ecken=(5, 8), feld_mm=10.0),
    "flaeche_oben":  dict(ecken=(5, 8), feld_mm=10.0),
}
STANDARD = dict(ecken=(5, 8), feld_mm=10.0)
GRAD = 2          # Polynomgrad der Massstabskarte


def ansicht_aus_name(pfad):
    """flaeche_unten_02.png -> flaeche_unten"""
    name = os.path.splitext(os.path.basename(pfad))[0]
    return re.sub(r"_\d+$", "", name)


def _suchen(bild, ecken):
    flags = cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY
    ok, ec = cv2.findChessboardCornersSB(bild, ecken, flags=flags)
    if ok:
        return ec.reshape(-1, 2)
    ok, ec = cv2.findChessboardCorners(
        bild, ecken, flags=cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE)
    if not ok:
        return None
    kriterium = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.001)
    cv2.cornerSubPix(bild, ec, (11, 11), (-1, -1), kriterium)
    return ec.reshape(-1, 2)


def ecken_finden(grau, ecken):
    """Subpixelgenaue innere Schachbrettecken. Gibt (ecken, verwendete_stufe)
    zurueck, sonst (None, None).

    Die Kontrastaufbereitung ist nicht optional: im dunklen unteren
    Panelbereich (mittlere Helligkeit ~52 statt ~128) findet der Detektor auf
    dem Rohbild NICHTS, mit CLAHE dagegen alle 40 Ecken. Genau dort waechst das
    Eis zuerst - ohne diese Kette waere der Maassstab ausgerechnet im
    wichtigsten Bereich nicht bestimmbar. Die Eckenlage aendert sich dabei
    nicht, Kontrastspreizung ist rein radiometrisch."""
    stufen = [
        ("Rohbild", lambda b: b),
        ("CLAHE 8x8", lambda b: cv2.createCLAHE(3.0, (8, 8)).apply(b)),
        ("CLAHE 16x16", lambda b: cv2.createCLAHE(4.0, (16, 16)).apply(b)),
        ("Histogrammausgleich", cv2.equalizeHist),
    ]
    for name, f in stufen:
        ec = _suchen(f(grau), ecken)
        if ec is not None:
            return ec, name
    return None, None


def vierecksflaeche(p):
    """Flaeche eines Vierecks aus 4 Punkten (Gauss/Schnuersenkelformel)."""
    x, y = p[:, 0], p[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def skalenproben(ec, ecken, feld_mm):
    """Je Schachbrettfeld eine Stuetzstelle: Mittelpunkt (px) und Skala mm^2/px^2.
    Zusaetzlich die Kantenlaengen, um Anisotropie sichtbar zu machen."""
    spalten, zeilen = ecken
    gitter = ec.reshape(zeilen, spalten, 2)
    punkte, skala, kant_x, kant_y = [], [], [], []
    for i in range(zeilen - 1):
        for j in range(spalten - 1):
            quad = np.array([gitter[i, j], gitter[i, j+1],
                             gitter[i+1, j+1], gitter[i+1, j]])
            flaeche_px = vierecksflaeche(quad)
            if flaeche_px < 1e-6:
                continue
            punkte.append(quad.mean(axis=0))
            skala.append(feld_mm**2 / flaeche_px)
            kant_x.append(np.linalg.norm(gitter[i, j+1] - gitter[i, j]) / feld_mm)
            kant_y.append(np.linalg.norm(gitter[i+1, j] - gitter[i, j]) / feld_mm)
    return (np.array(punkte), np.array(skala),
            np.array(kant_x), np.array(kant_y))


def _vandermonde(x, y, grad):
    spalten = [x**i * y**j for i in range(grad + 1) for j in range(grad + 1 - i)]
    return np.stack(spalten, axis=-1)


def fit_karte(punkte, skala, shape, grad=GRAD):
    """2D-Polynomfit auf log(Skala). Gibt Koeffizienten und relativen
    Restfehler zurueck - Letzterer sagt, wie gut die Karte die Messpunkte
    trifft, und gehoert in jede Ergebnisangabe."""
    H, W = shape
    x = punkte[:, 0] / W * 2 - 1          # auf [-1,1] normieren, sonst schlecht konditioniert
    y = punkte[:, 1] / H * 2 - 1
    A = _vandermonde(x, y, grad)
    koef, *_ = np.linalg.lstsq(A, np.log(skala), rcond=None)
    rest = np.exp(A @ koef) / skala - 1
    return koef, float(np.sqrt(np.mean(rest**2)) * 100)


def karte_bauen(koef, shape, grad=GRAD):
    """Volle Massstabskarte mm^2/px^2 fuer jedes Bildpixel."""
    H, W = shape
    yy, xx = np.mgrid[0:H, 0:W]
    x = xx / W * 2 - 1
    y = yy / H * 2 - 1
    A = _vandermonde(x.ravel(), y.ravel(), grad)
    return np.exp(A @ koef).reshape(H, W).astype(np.float32)


def kontrollbild(bild, ec_liste, ecken, punkte, skala, karte, pfad, kopfzeilen):
    """Ein Bild, das am Messtag in Sekunden die drei Fragen beantwortet:
    Wurde erkannt? Ist der Fit gut? Und - am wichtigsten - WO ist die Karte
    durch Messungen gestuetzt und wo nur extrapoliert?"""
    H, W = karte.shape
    vis = bild.copy() if bild.ndim == 3 else cv2.cvtColor(bild, cv2.COLOR_GRAY2BGR)

    # Gestuetzter Bereich = konvexe Huelle aller Stuetzstellen
    gestuetzt = np.zeros((H, W), np.uint8)
    huelle = cv2.convexHull(punkte.astype(np.float32)).astype(np.int32)
    cv2.fillPoly(gestuetzt, [huelle], 1)

    # Massstabskarte einfaerben; ausserhalb der Stuetzstellen deutlich blasser,
    # damit extrapolierte Bereiche nicht wie gemessene aussehen.
    norm = (karte - karte.min()) / max(1e-12, karte.max() - karte.min())
    hm = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS)
    vis = cv2.addWeighted(vis, 0.55, hm, 0.45, 0)
    aussen = gestuetzt == 0
    vis[aussen] = (vis[aussen] * 0.45 + 40).astype(np.uint8)
    cv2.polylines(vis, [huelle], True, (255, 255, 255), 5)

    for ec in ec_liste:
        cv2.drawChessboardCorners(vis, ecken, ec.reshape(-1, 1, 2).astype(np.float32), True)
    for p in punkte:
        cv2.circle(vis, (int(p[0]), int(p[1])), 5, (255, 255, 255), -1)

    # Farbskala mit beschrifteten Stuetzwerten
    bh, bw, rand = 34, 620, 40
    by = H - rand - bh
    for i in range(bw):
        f = i / (bw - 1)
        farbe = cv2.applyColorMap(np.uint8([[int(f * 255)]]), cv2.COLORMAP_VIRIDIS)[0, 0]
        cv2.rectangle(vis, (rand + i, by), (rand + i + 1, by + bh),
                      tuple(int(c) for c in farbe), -1)
    cv2.rectangle(vis, (rand, by), (rand + bw, by + bh), (255, 255, 255), 2)
    for f, wert in ((0.0, karte.min()), (1.0, karte.max())):
        cv2.putText(vis, f"{wert:.5f}", (rand + int(f * bw) - int(f * 120), by + bh + 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.95, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(vis, "mm2 je Pixel", (rand, by - 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.95, (255, 255, 255), 2, cv2.LINE_AA)

    # Kennzahlen oben links auf dunklem Grund
    x0, y0, zh = 40, 40, 46
    kasten = vis[y0:y0 + zh * len(kopfzeilen) + 24, x0:x0 + 1180]
    vis[y0:y0 + zh * len(kopfzeilen) + 24, x0:x0 + 1180] = (kasten * 0.25).astype(np.uint8)
    for i, (text, gut) in enumerate(kopfzeilen):
        farbe = (120, 255, 140) if gut is True else (120, 160, 255) if gut is False else (255, 255, 255)
        cv2.putText(vis, text, (x0 + 16, y0 + 44 + i * zh),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, farbe, 2, cv2.LINE_AA)

    klein = cv2.resize(vis, (1600, int(H * 1600 / W)), interpolation=cv2.INTER_AREA)
    cv2.imwrite(pfad, klein)


def eine_ansicht(ansicht, dateien):
    cfg = BRETT.get(ansicht, STANDARD)
    ecken, feld_mm = tuple(cfg["ecken"]), float(cfg["feld_mm"])
    print(f"\n=== {ansicht} ===  Brett {ecken[0]}x{ecken[1]} innere Ecken, Feld {feld_mm} mm")

    alle_p, alle_s, alle_kx, alle_ky, ec_liste = [], [], [], [], []
    shape, erstes, protokoll = None, None, []
    for fp in dateien:
        bild = cv2.imread(fp, cv2.IMREAD_COLOR)
        if bild is None:
            print(f"  {os.path.basename(fp)}: nicht lesbar"); continue
        grau = cv2.cvtColor(bild, cv2.COLOR_BGR2GRAY)
        if shape is None:
            shape, erstes = grau.shape, bild
        elif grau.shape != shape:
            print(f"  {os.path.basename(fp)}: andere Bildgroesse {grau.shape} - uebersprungen")
            continue
        ec, stufe = ecken_finden(grau, ecken)
        if ec is None:
            print(f"  {os.path.basename(fp)}: KEIN Schachbrett gefunden "
                  f"-> Eckenzahl, Beleuchtung und Reflexe pruefen")
            protokoll.append((f"{os.path.basename(fp)}: NICHT ERKANNT", False))
            continue
        protokoll.append((f"{os.path.basename(fp)}: {len(ec)} Ecken ueber {stufe}", True))
        p, s, kx, ky = skalenproben(ec, ecken, feld_mm)
        alle_p.append(p); alle_s.append(s); alle_kx.append(kx); alle_ky.append(ky)
        ec_liste.append(ec)
        print(f"  {os.path.basename(fp)}: {len(ec)} Ecken ueber '{stufe}', "
              f"{len(s)} Felder, Skala {s.min():.5f} bis {s.max():.5f} mm2/px")

    if not alle_p:
        print("  -> keine verwertbaren Bilder"); return None

    punkte = np.concatenate(alle_p); skala = np.concatenate(alle_s)
    kx = np.concatenate(alle_kx); ky = np.concatenate(alle_ky)
    koef, rest = fit_karte(punkte, skala, shape)
    karte = karte_bauen(koef, shape)

    px_pro_mm = 1.0 / np.sqrt(karte)
    anisotropie = 100 * abs(np.median(kx) - np.median(ky)) / max(1e-9, np.median(kx))
    spanne = 100 * (karte.max() / karte.min() - 1)

    print(f"  Stuetzstellen gesamt: {len(skala)}")
    print(f"  Restfehler des Fits: {rest:.2f}%   (>3% deutet auf verrutschtes "
          f"Brett oder falsche Feldgroesse hin)")
    print(f"  Massstab im Bild: {np.median(px_pro_mm):.2f} px/mm im Median, "
          f"{px_pro_mm.min():.2f} bis {px_pro_mm.max():.2f}")
    print(f"  Flaechenmassstab schwankt ueber das Bild um {spanne:.0f}%")
    print(f"  Anisotropie (x gegen y): {anisotropie:.1f}%")

    os.makedirs(OUT, exist_ok=True)
    np.savez(os.path.join(OUT, f"{ansicht}_kalibrierung.npz"),
             koeffizienten=koef, grad=GRAD, shape=np.array(shape),
             punkte=punkte, skala=skala, feld_mm=feld_mm,
             ecken=np.array(ecken), restfehler_prozent=rest)
    gestuetzt_pct = 100 * cv2.contourArea(
        cv2.convexHull(punkte.astype(np.float32))) / (shape[0] * shape[1])
    kopf = [(f"{ansicht}   {len(ec_liste)} Bild(er), {len(skala)} Felder", None)]
    kopf += protokoll
    kopf += [
        (f"Restfehler des Fits: {rest:.2f}%" + ("  OK" if rest < 3 else "  ZU HOCH - pruefen"),
         rest < 3),
        (f"Massstab: {np.median(px_pro_mm):.2f} px/mm im Median "
         f"({px_pro_mm.min():.2f} bis {px_pro_mm.max():.2f})", None),
        (f"Flaechenmassstab schwankt um {spanne:.0f}% ueber das Bild", None),
        (f"gemessen gestuetzt: {gestuetzt_pct:.0f}% der Bildflaeche "
         f"(weiss umrandet) - Rest ist extrapoliert", gestuetzt_pct > 25),
    ]
    kontrollbild(erstes, ec_liste, ecken, punkte, skala, karte,
                 os.path.join(OUT, f"{ansicht}_kontrolle.png"), kopf)
    print(f"  gemessen gestuetzt: {gestuetzt_pct:.0f}% der Bildflaeche "
          f"- ausserhalb extrapoliert die Karte nur")
    return dict(ansicht=ansicht, bilder=len(ec_liste), felder=int(len(skala)),
                restfehler_prozent=round(rest, 3),
                px_pro_mm_median=round(float(np.median(px_pro_mm)), 3),
                schwankung_prozent=round(float(spanne), 1),
                anisotropie_prozent=round(float(anisotropie), 1))


def selbsttest():
    """Synthetisches Schachbrett mit BEKANNTEM Massstab durch die Pipeline
    schicken. Damit laesst sich vor dem Messtag pruefen, ob die Kette stimmt -
    am Messtag selbst ist dafuer keine Zeit."""
    print("Selbsttest mit synthetischem Brett")
    ecken = (5, 8); feld_mm = 10.0; feld_px = 90
    W, H = 1600, 1200
    brett = np.full((H, W), 255, np.uint8)
    x0, y0 = 300, 200
    for i in range(ecken[1] + 1):
        for j in range(ecken[0] + 1):
            if (i + j) % 2 == 0:
                cv2.rectangle(brett, (x0 + j*feld_px, y0 + i*feld_px),
                              (x0 + (j+1)*feld_px - 1, y0 + (i+1)*feld_px - 1), 0, -1)
    # perspektivisch verzerren -> ortsabhaengiger Massstab, wie beim Schraegblick
    quelle = np.float32([[0, 0], [W, 0], [W, H], [0, H]])
    ziel = np.float32([[80, 40], [W-40, 150], [W-120, H-60], [140, H-120]])
    M = cv2.getPerspectiveTransform(quelle, ziel)
    verzerrt = cv2.warpPerspective(brett, M, (W, H), borderValue=255)

    os.makedirs(BILDER, exist_ok=True)
    tmp = os.path.join(BILDER, "selbsttest_01.png")
    cv2.imwrite(tmp, verzerrt)

    grau = cv2.imread(tmp, cv2.IMREAD_GRAYSCALE)
    ec, stufe = ecken_finden(grau, ecken)
    if ec is None:
        print("  FEHLGESCHLAGEN: Ecken im synthetischen Bild nicht gefunden."); return
    p, s, kx, ky = skalenproben(ec, ecken, feld_mm)
    koef, rest = fit_karte(p, s, grau.shape)
    karte = karte_bauen(koef, grau.shape)

    # Wahrheit: im unverzerrten Brett sind feld_px Pixel = feld_mm
    wahr_unverzerrt = (feld_mm / feld_px) ** 2
    print(f"  {len(ec)} Ecken gefunden, {len(s)} Felder")
    print(f"  Skala gemessen: {s.min():.6f} bis {s.max():.6f} mm2/px")
    print(f"  Skala unverzerrt (Sollwert vor der Perspektive): {wahr_unverzerrt:.6f}")
    print(f"  Restfehler des Fits: {rest:.2f}%")

    # Kernpruefung: rekonstruierte reale Brettflaeche gegen den Sollwert
    soll_mm2 = (ecken[0] - 1) * (ecken[1] - 1) * feld_mm**2
    maske = np.zeros(grau.shape, np.uint8)
    huelle = cv2.convexHull(ec.astype(np.float32)).astype(np.int32)
    cv2.fillPoly(maske, [huelle], 1)
    ist_mm2 = float(karte[maske > 0].sum())
    abw = 100 * (ist_mm2 / soll_mm2 - 1)
    print(f"  Brettflaeche: Soll {soll_mm2:.1f} mm2, ueber die Karte {ist_mm2:.1f} mm2 "
          f"-> Abweichung {abw:+.2f}%")
    print("  ERGEBNIS:", "OK" if abs(abw) < 2 and rest < 2 else "PRUEFEN")
    os.remove(tmp)


def main():
    if not os.path.isdir(BILDER):
        os.makedirs(BILDER, exist_ok=True)
        print(f"Ordner angelegt: {BILDER}\nKalibrierbilder dort ablegen als "
              f"<ansicht>_<nr>.png, z.B. flaeche_unten_01.png")
        return
    dateien = sorted(sum([glob.glob(os.path.join(BILDER, f"*{e}"))
                          for e in (".png", ".jpg", ".jpeg", ".tif", ".tiff")], []))
    if not dateien:
        print(f"Keine Bilder in {BILDER}"); return

    nach_ansicht = {}
    for fp in dateien:
        nach_ansicht.setdefault(ansicht_aus_name(fp), []).append(fp)

    bericht = []
    for ansicht, fs in sorted(nach_ansicht.items()):
        e = eine_ansicht(ansicht, fs)
        if e:
            bericht.append(e)
    if bericht:
        os.makedirs(OUT, exist_ok=True)
        json.dump(bericht, open(os.path.join(OUT, "bericht.json"), "w"), indent=1)
        print(f"\n-> {OUT}  (Kontrollbilder ansehen, solange das Brett noch klebt!)")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        selbsttest()
    else:
        main()
