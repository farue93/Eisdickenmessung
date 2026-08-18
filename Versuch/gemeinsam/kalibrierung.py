# -*- coding: utf-8 -*-
"""
kalibrierung.py - Massstab Pixel -> Millimeter aus EINEM Schachbrettbild.

Liefert eine MASSSTABSKARTE: mm^2 reale Oberflaeche je Bildpixel, ortsabhaengig.
Ein einzelner px/mm-Wert genuegt nicht, weil die Kamera schraeg auf eine
gekruemmte Flaeche blickt - ein Pixel am gekruemmten Rand deckt ein Vielfaches
der Flaeche eines Pixels in der Bildmitte ab.

Verfahren:
  1. Schachbrettecken subpixelgenau finden (findChessboardCornersSB).
  2. Je Feld: bekannte reale Flaeche (feld_mm^2) geteilt durch die gemessene
     Pixelflaeche des Vierecks -> lokale Skala in mm^2/px^2.
  3. Diese Stuetzstellen mit einem 2D-Polynom ueber das ganze Bild fortsetzen.
     Gefittet wird der LOGARITHMUS der Skala, weil sie positiv und multiplikativ
     ist; ein Polynom auf dem Rohwert koennte negativ werden.

EIN BILD STATT MEHRERER - und was daraus folgt:
Mit einem einzigen Brett ist nur der Bildbereich gestuetzt, den das Brett
bedeckt; ausserhalb setzt die Karte fort, ohne dort gemessen zu haben. Zwei
Konsequenzen sind fest eingebaut:

  * GRAD 1 als Standard. In log-Skala ist das ein exponentieller Verlauf und
    damit die Fortsetzung erster Ordnung der perspektivischen Verkuerzung. Ein
    Polynom 2. Grades trifft die Stuetzstellen besser, kann aber ausserhalb
    davon weglaufen - genau dort, wo niemand es nachpruefen kann.
  * Die fertige Karte wird auf das Doppelte bzw. die Haelfte des gemessenen
    Bereichs begrenzt. Eine Extrapolation, die um mehr als das danebenliegt,
    ist keine Messung mehr; die Begrenzung macht daraus einen sichtbaren
    Plateaubereich statt einer stillen Fehlmessung.

Das Kontrollbild zeigt weiss umrandet, wo gemessen wurde. Nur dort ist der
Massstab belegt - das ist beim Aufkleben des Bretts zu beruecksichtigen.
"""
import os
import numpy as np
import cv2


GRAD_STANDARD = 1
KAPPUNG = 2.0        # Faktor, um den die Karte den Messbereich verlassen darf
FELD_MM_STANDARD = 10.0


def vorgabe(formatname="a4", feld_mm=FELD_MM_STANDARD):
    """Eckenzahl und Feldgroesse des MITGELIEFERTEN Musterbogens.

    Wird von der Oberflaeche als Vorgabe benutzt. Die Zahlen kommen aus
    demselben Code, der die Druckvorlage erzeugt - so koennen Bogen und
    Eingabefeld nicht auseinanderlaufen."""
    import sys as _sys, os as _os
    _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
    from schachbrett_drucken import blatt_masse
    ecken, _, _, _ = blatt_masse(feld_mm, formatname)
    return ecken, feld_mm


def ecken_raten(grau, kandidaten=None, suchbereich=None):
    """Eckenzahl aus dem Bild bestimmen -> (ecken, stufe) oder (None, None).

    Die verwechselte Eckenzahl ist der klassische Fehlschlag am Messtag: Ein
    Brett mit 6x9 FELDERN hat 5x8 innere ECKEN, und wer sich vertut, bekommt
    nur 'nicht gefunden' ohne Hinweis worauf. Hier wird deshalb der Reihe nach
    probiert, beginnend mit den mitgelieferten Musterboegen.

    Bewusst eine kurze Liste statt einer vollstaendigen Suche: Jeder Versuch
    kostet auf einem grossen Bild eine knappe Sekunde, und in Wirklichkeit
    kommt genau eine Handvoll Groessen vor."""
    if kandidaten is None:
        kandidaten = []
        for fmt in ("a4", "a3"):
            try:
                kandidaten.append(vorgabe(fmt)[0])
            except Exception:
                pass
        kandidaten += [(16, 25), (26, 37), (5, 8), (7, 10), (9, 6), (6, 9), (4, 7)]
    gesehen, reihe = set(), []
    for e in kandidaten:                     # Doppelte raus, Reihenfolge halten
        if tuple(e) not in gesehen:
            gesehen.add(tuple(e)); reihe.append(tuple(e))
    for e in reihe:
        ec, stufe, _ = _ein_brett(grau, e, suchbereich)
        if ec is not None:
            return e, stufe
    return None, None


# ------------------------------------------------------------------- Erkennung
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
    """Subpixelgenaue innere Schachbrettecken -> (ecken, verwendete_stufe),
    sonst (None, None).

    Die Kontrastaufbereitung ist nicht optional: im dunklen unteren
    Panelbereich (mittlere Helligkeit ~52 statt ~128) findet der Detektor auf
    dem Rohbild NICHTS, mit CLAHE dagegen alle Ecken. Genau dort waechst das Eis
    zuerst - ohne diese Kette waere der Massstab ausgerechnet im wichtigsten
    Bereich nicht bestimmbar. Die Eckenlage aendert sich dabei nicht,
    Kontrastspreizung ist rein radiometrisch."""
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


# ------------------------------------------------------------------- Rechnung
def vierecksflaeche(p):
    """Flaeche eines Vierecks aus 4 Punkten (Gauss/Schnuersenkelformel)."""
    x, y = p[:, 0], p[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def skalenproben(ec, ecken, feld_mm):
    """Je Schachbrettfeld eine Stuetzstelle -> (punkte, skala, kant_x, kant_y,
    metrik).

    skala:  mm^2/px^2, die FLAECHENskala.
    metrik: je Feld (gxx, gxy, gyy) des lokalen metrischen Tensors. Damit ist
            die Laenge eines Pixelvektors v in mm  sqrt(v^T G v).

    WARUM DIE METRIK GEBRAUCHT WIRD: Aus der Flaechenskala allein laesst sich
    keine Laenge berechnen. sqrt(Flaechenskala) ist das geometrische Mittel der
    beiden Hauptrichtungen; blickt die Kamera schraeg auf die Flaeche, sind
    diese Richtungen verschieden stark verkuerzt, und die Laenge in EINER
    bestimmten Richtung weicht davon ab - um bis zu der Anisotropie, die der
    Bericht ausweist. Fuer die Flaechenmessung ist das ohne Belang, fuer die
    Laserlinie nicht: Die Eisdicke ist eine Laenge entlang der Normalen.

    Aufgebaut wird die Metrik aus den beiden Kantenrichtungen jedes Feldes: Sie
    entsprechen bekannten feld_mm in der Ebene, also ist die lineare Abbildung
    mm -> px bekannt. G = (A^-1)^T A^-1 aus ihrer Inversen.

    G haengt NICHT davon ab, wie das Brett gedreht aufgeklebt war: Eine Drehung
    des Bretts ersetzt A durch A*R, und (A*R)^-1^T (A*R)^-1 = G. Nur deshalb
    duerfen Aufnahmen mit unterschiedlich orientiertem Brett in einen
    gemeinsamen Fit."""
    spalten, zeilen = ecken
    gitter = ec.reshape(zeilen, spalten, 2)
    punkte, skala, kant_x, kant_y, metrik = [], [], [], [], []
    for i in range(zeilen - 1):
        for j in range(spalten - 1):
            quad = np.array([gitter[i, j], gitter[i, j+1],
                             gitter[i+1, j+1], gitter[i+1, j]])
            flaeche_px = vierecksflaeche(quad)
            if flaeche_px < 1e-6:
                continue
            # Kantenrichtungen aus dem Mittel der jeweils gegenueberliegenden
            # Kanten - das trifft die Mitte des Feldes besser als eine Ecke.
            e1 = 0.5 * ((gitter[i, j+1] - gitter[i, j])
                        + (gitter[i+1, j+1] - gitter[i+1, j]))
            e2 = 0.5 * ((gitter[i+1, j] - gitter[i, j])
                        + (gitter[i+1, j+1] - gitter[i, j+1]))
            A = np.column_stack([e1, e2]) / feld_mm      # mm -> px
            if abs(np.linalg.det(A)) < 1e-9:
                continue
            Ai = np.linalg.inv(A)                        # px -> mm
            G = Ai.T @ Ai
            punkte.append(quad.mean(axis=0))
            skala.append(feld_mm**2 / flaeche_px)
            kant_x.append(np.linalg.norm(e1) / feld_mm)
            kant_y.append(np.linalg.norm(e2) / feld_mm)
            metrik.append((G[0, 0], G[0, 1], G[1, 1]))
    return (np.array(punkte), np.array(skala), np.array(kant_x),
            np.array(kant_y), np.array(metrik))


def _vandermonde(x, y, grad):
    spalten = [x**i * y**j for i in range(grad + 1) for j in range(grad + 1 - i)]
    return np.stack(spalten, axis=-1)


def fit_karte(punkte, skala, shape, grad=GRAD_STANDARD):
    """2D-Polynomfit auf log(Skala) -> (Koeffizienten, relativer Restfehler %).
    Der Restfehler sagt, wie gut die Karte die Messpunkte trifft, und gehoert
    in jede Ergebnisangabe."""
    H, W = shape
    x = punkte[:, 0] / W * 2 - 1        # auf [-1,1] normieren, sonst schlecht konditioniert
    y = punkte[:, 1] / H * 2 - 1
    A = _vandermonde(x, y, grad)
    koef, *_ = np.linalg.lstsq(A, np.log(skala), rcond=None)
    rest = np.exp(A @ koef) / skala - 1
    return koef, float(np.sqrt(np.mean(rest**2)) * 100)


def fit_metrik(punkte, metrik, shape, grad=GRAD_STANDARD):
    """Die drei Felder des metrischen Tensors ueber das Bild fortsetzen.

    Gefittet werden log(gxx), log(gyy) und die Korrelation
    rho = gxy / sqrt(gxx*gyy). Diese Zerlegung ist kein Schoenheitsfehler,
    sondern haelt das Ergebnis gueltig: gxx und gyy sind positiv (Logarithmus),
    und mit |rho| < 1 ist die zurueckgerechnete Matrix garantiert positiv
    definit. Ein Polynom direkt auf gxx, gxy, gyy koennte im fortgesetzten
    Bereich eine Matrix liefern, die gar keine Metrik mehr ist - und damit
    negative Laengen unter der Wurzel."""
    gxx, gxy, gyy = metrik[:, 0], metrik[:, 1], metrik[:, 2]
    rho = gxy / np.sqrt(np.maximum(gxx * gyy, 1e-30))
    H, W = shape
    x = punkte[:, 0] / W * 2 - 1
    y = punkte[:, 1] / H * 2 - 1
    A = _vandermonde(x, y, grad)
    koef = []
    for ziel in (np.log(np.maximum(gxx, 1e-30)), np.log(np.maximum(gyy, 1e-30)), rho):
        k, *_ = np.linalg.lstsq(A, ziel, rcond=None)
        koef.append(k)
    return np.stack(koef)          # (3, n_koeffizienten)


def metrik_bauen(koef, shape, grad=GRAD_STANDARD):
    """Volle Metrikkarte -> (gxx, gxy, gyy), je in Bildgroesse."""
    H, W = shape
    yy, xx = np.mgrid[0:H, 0:W]
    A = _vandermonde((xx / W * 2 - 1).ravel(), (yy / H * 2 - 1).ravel(), grad)
    gxx = np.exp(A @ koef[0]).reshape(H, W)
    gyy = np.exp(A @ koef[1]).reshape(H, W)
    rho = np.clip((A @ koef[2]).reshape(H, W), -0.995, 0.995)
    gxy = rho * np.sqrt(gxx * gyy)
    return (gxx.astype(np.float32), gxy.astype(np.float32), gyy.astype(np.float32))


def karte_bauen(koef, shape, grad=GRAD_STANDARD, grenzen=None):
    """Volle Massstabskarte mm^2/px^2 je Bildpixel.

    grenzen=(min,max) der gemessenen Skala begrenzt die Extrapolation auf das
    KAPPUNG-fache des belegten Bereichs - siehe Modulkopf."""
    H, W = shape
    yy, xx = np.mgrid[0:H, 0:W]
    x = xx / W * 2 - 1
    y = yy / H * 2 - 1
    A = _vandermonde(x.ravel(), y.ravel(), grad)
    karte = np.exp(A @ koef).reshape(H, W)
    if grenzen is not None:
        lo, hi = grenzen
        karte = np.clip(karte, lo / KAPPUNG, hi * KAPPUNG)
    return karte.astype(np.float32)


# ------------------------------------------------------------------ Kontrolle
def kontrollbild(bild, ec_liste, ecken, punkte, karte, kopfzeilen, pfad=None):
    """Ein Bild, das in Sekunden die drei Fragen beantwortet: Wurde erkannt?
    Ist der Fit gut? Und - am wichtigsten - WO ist die Karte durch Messungen
    gestuetzt und wo nur fortgesetzt?

    Bei mehreren Kalibrierbildern werden ALLE Eckengitter eingezeichnet, auch
    wenn nur eines der Bilder den Hintergrund liefert. Erst dadurch ist zu
    sehen, wie weit die Brettpositionen zusammen das Bildfeld abdecken - und
    genau darum geht es beim Kalibrieren mit mehreren Aufnahmen."""
    H, W = karte.shape
    vis = bild.copy() if bild.ndim == 3 else cv2.cvtColor(bild, cv2.COLOR_GRAY2BGR)

    gestuetzt = np.zeros((H, W), np.uint8)
    huelle = cv2.convexHull(punkte.astype(np.float32)).astype(np.int32)
    cv2.fillPoly(gestuetzt, [huelle], 1)

    norm = (karte - karte.min()) / max(1e-12, karte.max() - karte.min())
    hm = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS)
    vis = cv2.addWeighted(vis, 0.55, hm, 0.45, 0)
    aussen = gestuetzt == 0
    vis[aussen] = (vis[aussen] * 0.45 + 40).astype(np.uint8)   # fortgesetzt: blasser
    cv2.polylines(vis, [huelle], True, (255, 255, 255), max(2, W // 400))

    if isinstance(ec_liste, np.ndarray):
        ec_liste = [ec_liste]
    for ec in ec_liste:
        cv2.drawChessboardCorners(vis, ecken,
                                  ec.reshape(-1, 1, 2).astype(np.float32), True)
    for p in punkte:
        cv2.circle(vis, (int(p[0]), int(p[1])), max(2, W // 700), (255, 255, 255), -1)

    # Kennzahlen oben links auf abgedunkeltem Grund
    sk = W / 1600.0
    x0, y0, zh = int(30 * sk), int(30 * sk), int(38 * sk)
    kh, kb = zh * len(kopfzeilen) + int(20 * sk), int(1000 * sk)
    kb = min(kb, W - 2 * x0)
    kasten = vis[y0:y0 + kh, x0:x0 + kb]
    vis[y0:y0 + kh, x0:x0 + kb] = (kasten * 0.25).astype(np.uint8)
    for i, (text, gut) in enumerate(kopfzeilen):
        farbe = ((120, 255, 140) if gut is True
                 else (120, 160, 255) if gut is False else (255, 255, 255))
        cv2.putText(vis, text, (x0 + int(12 * sk), y0 + int(34 * sk) + i * zh),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8 * sk, farbe, max(1, int(2 * sk)),
                    cv2.LINE_AA)

    if pfad:
        breit = min(1600, W)
        cv2.imwrite(pfad, cv2.resize(vis, (breit, int(H * breit / W)),
                                     interpolation=cv2.INTER_AREA))
    return vis


def kreuzprobe(alle_p, alle_s, shape, grad):
    """Ein Brett weglassen, auf den uebrigen fitten, das weggelassene vorhersagen.

    Das misst genau das, worauf es ankommt: wie gut die Karte dort stimmt, wo
    nicht gemessen wurde. Der Restfehler des Fits kann das nicht - er wird mit
    steigendem Grad immer kleiner, auch wenn die Fortsetzung schlechter wird.
    -> mittlerer Betragsfehler in Prozent, oder None bei zu wenigen Brettern."""
    n = len(alle_p)
    if n < 3:
        return None
    H, W = shape
    fehler = []
    for i in range(n):
        p = np.concatenate([alle_p[j] for j in range(n) if j != i])
        s = np.concatenate([alle_s[j] for j in range(n) if j != i])
        try:
            koef, _ = fit_karte(p, s, shape, grad)
        except np.linalg.LinAlgError:
            return None
        A = _vandermonde(alle_p[i][:, 0] / W * 2 - 1,
                         alle_p[i][:, 1] / H * 2 - 1, grad)
        fehler.append(np.mean(np.abs(np.exp(A @ koef) / alle_s[i] - 1)))
    return float(np.mean(fehler) * 100)


def _grad_waehlen(alle_p, alle_s, shape):
    """Grad 1 oder 2? Entschieden wird an der Kreuzprobe, nicht nach Gefuehl.

    Mit einem einzigen Brett ist die Frage nicht entscheidbar - dann Grad 1,
    weil er sich ausserhalb der Stuetzstellen am gutmuetigsten verhaelt."""
    if len(alle_p) < 3:
        return 1, "ein bis zwei Bretter - Grad 1 ist ausserhalb gutmuetiger"
    k1 = kreuzprobe(alle_p, alle_s, shape, 1)
    k2 = kreuzprobe(alle_p, alle_s, shape, 2)
    if k1 is None or k2 is None or not np.isfinite([k1, k2]).all():
        return 1, "Kreuzprobe nicht auswertbar - Grad 1"
    if k2 < k1:
        return 2, f"Kreuzprobe: Grad 2 besser ({k2:.2f}% gegen {k1:.2f}%)"
    return 1, f"Kreuzprobe: Grad 1 besser ({k1:.2f}% gegen {k2:.2f}%)"


def _ein_brett(grau, ecken, suchbereich):
    """Ein Bild absuchen -> (ecken_im_vollbild, stufe, fehlertext)."""
    if suchbereich:
        sx0, sy0, sx1, sy1 = [int(v) for v in suchbereich]
        sx0, sy0 = max(0, sx0), max(0, sy0)
        sx1, sy1 = min(grau.shape[1], sx1), min(grau.shape[0], sy1)
        if sx1 - sx0 < 40 or sy1 - sy0 < 40:
            return None, None, "Suchbereich zu klein"
        ec, stufe = ecken_finden(grau[sy0:sy1, sx0:sx1], ecken)
        if ec is not None:
            ec = ec + np.float32([sx0, sy0])      # zurueck ins Vollbild
    else:
        ec, stufe = ecken_finden(grau, ecken)
    if ec is None:
        return None, None, (
            f"kein Schachbrett mit {ecken[0]}x{ecken[1]} INNEREN Ecken gefunden "
            f"({'im Suchbereich' if suchbereich else 'im Bild'})")
    return ec, stufe, None


# ------------------------------------------------------------------ Hauptweg
def kalibrieren(bilder, ecken, feld_mm, grad=GRAD_STANDARD, npz_pfad=None,
                png_pfad=None, name="", suchbereich=None, anzeige=0):
    """Ein ODER MEHRERE Schachbrettbilder -> eine Massstabskarte.

    bilder: ein Graubild, oder eine Liste von
            {"name": str, "grau": ndarray, "suchbereich": (x0,y0,x1,y1)|None}

    WARUM MEHRERE BILDER: Ein einzelnes Brett stuetzt nur den Bildbereich, den
    es bedeckt - typisch wenige Prozent der Bildflaeche. Ueberall sonst setzt
    die Karte den Verlauf nur fort. Wird dasselbe Brett nacheinander an mehrere
    Stellen geklebt und jedes Mal aufgenommen, gehen ALLE Stuetzstellen in
    EINEN gemeinsamen Fit ein; die Karte ist dann dort belegt, wo spaeter auch
    gemessen wird. Das ist der eigentliche Weg zu einer ortsaufgeloesten
    Pixel-zu-Millimeter-Umrechnung.

    Die Bilder muessen dieselbe Groesse haben und von derselben, unbewegten
    Kamera stammen - sonst beschreiben ihre Stuetzstellen verschiedene
    Abbildungen und der gemeinsame Fit ist sinnlos. Abweichende Bildgroessen
    werden deshalb verworfen und im Bericht ausgewiesen.

    suchbereich: nur fuer den Einbild-Aufruf; bei einer Liste steht er je Bild
    im jeweiligen Eintrag.

    Gibt einen Bericht zurueck; bei Misserfolg steht in bericht['fehler'], was
    zu tun ist. Wirft nicht - am Messtag soll eine misslungene Erkennung eine
    Meldung sein, kein Absturz."""
    ecken = (int(ecken[0]), int(ecken[1]))
    feld_mm = float(feld_mm)
    if isinstance(bilder, np.ndarray):
        bilder = [dict(name=name or "Bild", grau=bilder, suchbereich=suchbereich)]
    if not bilder:
        return dict(ok=False, fehler="kein Kalibrierbild geladen")

    shape = tuple(bilder[0]["grau"].shape)
    alle_p, alle_s, alle_kx, alle_ky, alle_g, ec_liste = [], [], [], [], [], []
    protokoll = []
    for b in bilder:
        kurz = b.get("name") or "Bild"
        grau = b["grau"]
        if tuple(grau.shape) != shape:
            protokoll.append(dict(name=kurz, ok=False,
                                  fehler=f"andere Bildgroesse {grau.shape[1]}x{grau.shape[0]}"))
            continue
        ec, stufe, fehler = _ein_brett(grau, ecken, b.get("suchbereich"))
        if ec is None:
            protokoll.append(dict(name=kurz, ok=False, fehler=fehler))
            continue
        p, s, kx, ky, g = skalenproben(ec, ecken, feld_mm)
        if len(s) < 4:
            protokoll.append(dict(name=kurz, ok=False, fehler="zu wenige Felder"))
            continue
        alle_p.append(p); alle_s.append(s); alle_kx.append(kx)
        alle_ky.append(ky); alle_g.append(g)
        ec_liste.append(ec)
        protokoll.append(dict(name=kurz, ok=True, ecken=int(len(ec)),
                              felder=int(len(s)), stufe=stufe))

    if not alle_p:
        grund = protokoll[0]["fehler"] if protokoll else "kein Bild verwertbar"
        return dict(ok=False, bilder=protokoll, fehler=(
            f"Kein Bild verwertbar ({grund}). Eckenzahl nachzaehlen - ein Brett "
            f"mit 6x9 Feldern hat 5x8 INNERE Ecken. Sonst Suchbereich aufziehen "
            f"oder Massstab von Hand messen."))

    punkte = np.concatenate(alle_p)
    skala = np.concatenate(alle_s)
    kx = np.concatenate(alle_kx)
    ky = np.concatenate(alle_ky)
    metrik = np.concatenate(alle_g)

    if str(grad).lower() in ("auto", "automatisch", "0"):
        grad, grad_grund = _grad_waehlen(alle_p, alle_s, shape)
    else:
        grad = int(grad)
        grad_grund = f"fest eingestellt auf Grad {grad}"
    kreuz = kreuzprobe(alle_p, alle_s, shape, grad)

    koef, rest = fit_karte(punkte, skala, shape, grad)
    karte = karte_bauen(koef, shape, grad, (float(skala.min()), float(skala.max())))
    metrik_koef = fit_metrik(punkte, metrik, shape, grad)

    px_pro_mm = 1.0 / np.sqrt(karte)
    anisotropie = 100 * abs(np.median(kx) - np.median(ky)) / max(1e-9, np.median(kx))
    spanne = 100 * (karte.max() / karte.min() - 1)
    gestuetzt = 100 * cv2.contourArea(
        cv2.convexHull(punkte.astype(np.float32))) / (shape[0] * shape[1])

    bericht = dict(
        ok=True, fehler=None, art="schachbrett", name=name,
        stufe=protokoll[0].get("stufe", ""),
        bilder=protokoll, bilder_ok=len(ec_liste), bilder_gesamt=len(bilder),
        ecken_gefunden=int(sum(len(e) for e in ec_liste)), felder=int(len(skala)),
        feld_mm=feld_mm, grad=int(grad), grad_grund=grad_grund,
        kreuzprobe_prozent=None if kreuz is None else round(kreuz, 3),
        restfehler_prozent=round(rest, 3),
        px_pro_mm_median=round(float(np.median(px_pro_mm)), 3),
        px_pro_mm_min=round(float(px_pro_mm.min()), 3),
        px_pro_mm_max=round(float(px_pro_mm.max()), 3),
        schwankung_prozent=round(float(spanne), 1),
        anisotropie_prozent=round(float(anisotropie), 1),
        gestuetzt_prozent=round(float(gestuetzt), 1),
        shape=[int(shape[0]), int(shape[1])],
    )

    if npz_pfad:
        os.makedirs(os.path.dirname(npz_pfad) or ".", exist_ok=True)
        np.savez(npz_pfad, koeffizienten=koef, grad=grad, shape=np.array(shape),
                 punkte=punkte, skala=skala, feld_mm=feld_mm,
                 ecken=np.array(ecken), restfehler_prozent=rest,
                 grenzen=np.array([skala.min(), skala.max()]),
                 bilder=np.array(len(ec_liste)),
                 metrik_koeffizienten=metrik_koef)
        bericht["npz"] = npz_pfad

    kopf = [(f"{name or 'Kalibrierung'}   {len(ec_liste)} von {len(bilder)} "
             f"Bild(ern), {len(skala)} Felder", None)]
    for e in protokoll:
        kopf.append((f"  {e['name']}: "
                     + (f"{e['ecken']} Ecken ueber {e['stufe']}" if e["ok"]
                        else f"NICHT ERKANNT - {e['fehler']}"), e["ok"]))
    kopf += [
        (f"Polynomgrad {grad} - {grad_grund}", None),
        (f"Restfehler des Fits: {rest:.2f}%" + ("  OK" if rest < 3 else "  ZU HOCH - pruefen"),
         rest < 3),
    ]
    if kreuz is not None:
        kopf.append((f"Kreuzprobe (weggelassenes Brett): {kreuz:.2f}% - "
                     f"so gut trifft die Karte, wo nicht gemessen wurde",
                     kreuz < 3))
    kopf += [
        (f"Massstab: {np.median(px_pro_mm):.2f} px/mm im Median "
         f"({px_pro_mm.min():.2f} bis {px_pro_mm.max():.2f})", None),
        (f"Flaechenmassstab schwankt um {spanne:.0f}% ueber das Bild", None),
        (f"Anisotropie {anisotropie:.0f}% - Laengen ueber die Metrik, "
         f"nicht ueber die Wurzel der Flaeche", None),
        (f"gemessen gestuetzt: {gestuetzt:.0f}% der Bildflaeche (weiss umrandet)",
         gestuetzt > 25),
        ("ausserhalb wird nur fortgesetzt - dort ist der Massstab nicht belegt", None),
    ]
    # Zeichenbausteine mitgeben: Die Oberflaeche kann das Overlay damit auf ein
    # ANDERES der geladenen Bilder legen, ohne die Erkennung zu wiederholen.
    bericht["_zeichnung"] = dict(ec_liste=ec_liste, ecken=ecken, punkte=punkte,
                                 karte=karte, kopfzeilen=kopf)
    hintergrund = bilder[min(anzeige, len(bilder) - 1)]["grau"]
    if tuple(hintergrund.shape) != shape:
        hintergrund = bilder[0]["grau"]
    bericht["bild"] = kontrollbild(hintergrund, ec_liste, ecken, punkte, karte,
                                   kopf, png_pfad)
    if png_pfad:
        bericht["png"] = png_pfad
    return bericht


# ------------------------------------------------------- Massstab von Hand
def _konstante_karte(skala, shape, punkte, art, name, npz_pfad, png_pfad,
                     grau, zusatz):
    """Gemeinsamer Abschluss der beiden Handmessungen: eine ORTSUNABHAENGIGE
    Skala als Karte vom Grad 0 speichern.

    Damit passt die Handmessung in dieselbe Maschinerie wie das Schachbrett -
    die Auswertung muss nicht wissen, woher die Karte kommt. Der Preis steht im
    Bericht: eine konstante Skala unterschlaegt, dass ein Pixel am gekruemmten
    Rand mehr Flaeche abdeckt als eines in der Bildmitte. Beim Schachbrett
    schwankt der Flaechenmassstab ueber das Bild typisch um Zehnerprozente."""
    skala = float(skala)
    koef = np.array([np.log(skala)])
    karte = karte_bauen(koef, shape, 0, (skala, skala))
    ppm = float(1.0 / np.sqrt(skala))
    kopf_extra = zusatz.pop("_kopf", [])
    if npz_pfad:
        os.makedirs(os.path.dirname(npz_pfad) or ".", exist_ok=True)
        np.savez(npz_pfad, koeffizienten=koef, grad=0, shape=np.array(shape),
                 punkte=np.array(punkte, np.float32), skala=np.array([skala]),
                 feld_mm=0.0, ecken=np.array([0, 0]), restfehler_prozent=0.0,
                 grenzen=np.array([skala, skala]), art=art)
    b = dict(ok=True, fehler=None, art=art, name=name, stufe="von Hand",
             ecken_gefunden=0, felder=0, feld_mm=0.0, grad=0,
             restfehler_prozent=0.0, px_pro_mm_median=round(ppm, 3),
             px_pro_mm_min=round(ppm, 3), px_pro_mm_max=round(ppm, 3),
             schwankung_prozent=0.0, anisotropie_prozent=0.0,
             gestuetzt_prozent=0.0, shape=[int(shape[0]), int(shape[1])])
    b.update(zusatz)
    if npz_pfad:
        b["npz"] = npz_pfad
    kopf = [(f"{name or 'Massstab'} von Hand", None)] + kopf_extra + [
        (f"Massstab: {ppm:.3f} px/mm, ortsunabhaengig", None),
        ("konstante Skala - Perspektive und Kruemmung NICHT erfasst", False),
    ]
    b["bild"] = kontrollbild_hand(grau, punkte, art, karte, kopf, png_pfad)
    if png_pfad:
        b["png"] = png_pfad
    return b


def manuell_strecke(p0, p1, laenge_mm, shape, grau, npz_pfad=None,
                    png_pfad=None, name=""):
    """Massstab aus einer von Hand gezogenen Strecke bekannter Laenge.

    Fuer die Laserlinie die passende Handmessung: dort ist die gesuchte Groesse
    eine LAENGE quer zur Linie, keine Flaeche. Zwei Punkte mit bekanntem
    Abstand - Panelbreite, Markierung, Klebebandkante - genuegen."""
    laenge_mm = float(laenge_mm)
    if laenge_mm <= 0:
        return dict(ok=False, fehler="reale Laenge muss groesser als null sein")
    laenge_px = float(np.hypot(p1[0] - p0[0], p1[1] - p0[1]))
    if laenge_px < 5:
        return dict(ok=False, fehler="Strecke zu kurz - im Zoom genauer ziehen")
    ppm = laenge_px / laenge_mm
    return _konstante_karte(
        1.0 / ppm**2, shape, [p0, p1], "strecke", name, npz_pfad, png_pfad, grau,
        dict(laenge_px=round(laenge_px, 2), laenge_mm=laenge_mm,
             _kopf=[(f"Strecke {laenge_px:.1f} px = {laenge_mm:.2f} mm", None)]))


def manuell_rechteck(rechteck, breite_mm, hoehe_mm, shape, grau, npz_pfad=None,
                     png_pfad=None, name=""):
    """Massstab aus einem von Hand gezogenen Rechteck bekannter Groesse.

    Fuer die Flaechenansicht die passende Handmessung: dort ist die gesuchte
    Groesse eine FLAECHE. Gerechnet wird direkt mm^2 je px^2 und nicht ueber
    eine Kantenlaenge - sonst muesste man annehmen, dass beide Richtungen
    denselben Massstab haben, und genau das ist bei Schraegblick nicht der
    Fall. Wie schief es steht, zeigt die ausgewiesene Anisotropie."""
    breite_mm, hoehe_mm = float(breite_mm), float(hoehe_mm)
    if breite_mm <= 0 or hoehe_mm <= 0:
        return dict(ok=False, fehler="Breite und Hoehe muessen groesser als null sein")
    x0, y0, x1, y1 = [float(v) for v in rechteck]
    b_px, h_px = x1 - x0, y1 - y0
    if b_px < 5 or h_px < 5:
        return dict(ok=False, fehler="Rechteck zu klein - im Zoom genauer ziehen")
    skala = (breite_mm * hoehe_mm) / (b_px * h_px)
    kx, ky = b_px / breite_mm, h_px / hoehe_mm
    aniso = 100 * abs(kx - ky) / max(1e-9, kx)
    return _konstante_karte(
        skala, shape, [(x0, y0), (x1, y0), (x1, y1), (x0, y1)], "rechteck",
        name, npz_pfad, png_pfad, grau,
        dict(anisotropie_prozent=round(float(aniso), 1),
             breite_mm=breite_mm, hoehe_mm=hoehe_mm,
             _kopf=[(f"Rechteck {b_px:.0f}x{h_px:.0f} px = "
                     f"{breite_mm:.1f}x{hoehe_mm:.1f} mm", None),
                    (f"Anisotropie x gegen y: {aniso:.1f}%"
                     + ("  - achsparallel ziehen!" if aniso > 10 else ""),
                     aniso <= 10)]))


def kontrollbild_hand(grau, punkte, art, karte, kopfzeilen, pfad=None):
    """Kontrollbild der Handmessung: das Bild mit der gezogenen Form."""
    H, W = karte.shape
    vis = grau.copy() if grau.ndim == 3 else cv2.cvtColor(grau, cv2.COLOR_GRAY2BGR)
    dick = max(2, W // 500)
    p = np.array(punkte, np.int32)
    if art == "strecke":
        cv2.line(vis, tuple(p[0]), tuple(p[1]), (60, 210, 255), dick, cv2.LINE_AA)
        for q in p:
            cv2.circle(vis, tuple(q), dick * 3, (60, 210, 255), dick)
    else:
        cv2.polylines(vis, [p], True, (60, 210, 255), dick, cv2.LINE_AA)

    sk = W / 1600.0
    x0, y0, zh = int(30 * sk), int(30 * sk), int(38 * sk)
    kh = zh * len(kopfzeilen) + int(20 * sk)
    kb = min(int(1000 * sk), W - 2 * x0)
    kasten = vis[y0:y0 + kh, x0:x0 + kb]
    vis[y0:y0 + kh, x0:x0 + kb] = (kasten * 0.25).astype(np.uint8)
    for i, (text, gut) in enumerate(kopfzeilen):
        farbe = ((120, 255, 140) if gut is True
                 else (120, 160, 255) if gut is False else (255, 255, 255))
        cv2.putText(vis, text, (x0 + int(12 * sk), y0 + int(34 * sk) + i * zh),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8 * sk, farbe,
                    max(1, int(2 * sk)), cv2.LINE_AA)
    if pfad:
        os.makedirs(os.path.dirname(pfad) or ".", exist_ok=True)
        breit = min(1600, W)
        cv2.imwrite(pfad, cv2.resize(vis, (breit, int(H * breit / W)),
                                     interpolation=cv2.INTER_AREA))
    return vis


def gegenprobe(karte, punkte, art, soll_mm=None, soll_mm2=None):
    """Was sagt eine vorhandene Karte ueber dieselbe Form? -> Abweichung in %.

    Damit wird die Handmessung zur unabhaengigen KONTROLLE der automatischen
    Kalibrierung, statt sie nur zu ersetzen. Eine Abweichung von mehreren
    Prozent heisst: eine der beiden Angaben stimmt nicht - meistens die
    eingetragene Feldgroesse des Bretts, die am Ausdruck nachzumessen ist."""
    if karte is None:
        return None
    p = np.array(punkte, float)
    if art == "strecke":
        # Laenge in mm entlang der Strecke: in Pixelschritten aufsummiert, jeder
        # Schritt mit der lokalen Skala gewichtet.
        ges_px = float(np.hypot(*(p[1] - p[0])))
        n = max(2, int(ges_px))
        t = np.linspace(0, 1, n)
        xs = np.clip((p[0, 0] + t * (p[1, 0] - p[0, 0])).astype(int), 0, karte.shape[1] - 1)
        ys = np.clip((p[0, 1] + t * (p[1, 1] - p[0, 1])).astype(int), 0, karte.shape[0] - 1)
        mm = float(np.sum(np.sqrt(karte[ys, xs])) * (ges_px / n))
        return 100 * (mm / soll_mm - 1) if soll_mm else None
    x0, y0 = int(min(p[:, 0])), int(min(p[:, 1]))
    x1, y1 = int(max(p[:, 0])), int(max(p[:, 1]))
    mm2 = float(karte[y0:y1, x0:x1].sum())
    return 100 * (mm2 / soll_mm2 - 1) if soll_mm2 else None


def zusammenfassung(b):
    """Eine Zeile fuer die Statuszeile."""
    if not b.get("ok"):
        return b.get("fehler", "Kalibrierung fehlgeschlagen")
    if b.get("art") == "strecke":
        return (f"Strecke von Hand | {b['px_pro_mm_median']:.3f} px/mm | "
                f"{b['laenge_px']:.0f} px = {b['laenge_mm']:.2f} mm | ortsunabhaengig")
    if b.get("art") == "rechteck":
        return (f"Rechteck von Hand | {b['px_pro_mm_median']:.3f} px/mm | "
                f"Anisotropie {b['anisotropie_prozent']:.1f}% | ortsunabhaengig")
    warnung = ""
    if b["restfehler_prozent"] >= 3:
        warnung = "  ACHTUNG Restfehler hoch"
    elif b["gestuetzt_prozent"] <= 25:
        warnung = "  ACHTUNG wenig gestuetzt - weitere Brettpositionen aufnehmen"
    bilder = (f"{b.get('bilder_ok', 1)}/{b.get('bilder_gesamt', 1)} Bilder | "
              if b.get("bilder_gesamt", 1) > 1 else "")
    return (f"{bilder}{b['felder']} Felder | {b['px_pro_mm_median']:.2f} px/mm | "
            f"Restfehler {b['restfehler_prozent']:.2f}% | "
            f"gestuetzt {b['gestuetzt_prozent']:.0f}%{warnung}")


# ------------------------------------------------------------------ Selbsttest
def selbsttest(ecken=(5, 8), feld_mm=10.0, feld_px=90):
    """Synthetisches Brett mit BEKANNTEM Massstab durch die Pipeline schicken.
    Damit laesst sich vor dem Messtag pruefen, ob die Kette stimmt - am Messtag
    selbst ist dafuer keine Zeit. Gibt (ok, zeilen) zurueck."""
    W, H = 1600, 1200
    brett = np.full((H, W), 255, np.uint8)
    x0, y0 = 300, 200
    for i in range(ecken[1] + 1):
        for j in range(ecken[0] + 1):
            if (i + j) % 2 == 0:
                cv2.rectangle(brett, (x0 + j*feld_px, y0 + i*feld_px),
                              (x0 + (j+1)*feld_px - 1, y0 + (i+1)*feld_px - 1), 0, -1)
    quelle = np.float32([[0, 0], [W, 0], [W, H], [0, H]])
    ziel = np.float32([[80, 40], [W-40, 150], [W-120, H-60], [140, H-120]])
    verzerrt = cv2.warpPerspective(brett, cv2.getPerspectiveTransform(quelle, ziel),
                                   (W, H), borderValue=255)

    b = kalibrieren(verzerrt, ecken, feld_mm, name="Selbsttest")
    if not b["ok"]:
        return False, [b["fehler"]]

    # Kernpruefung: rekonstruierte reale Brettflaeche gegen den Sollwert
    ec, _ = ecken_finden(verzerrt, ecken)
    punkte, skala, _, _, _ = skalenproben(ec, ecken, feld_mm)
    koef, _ = fit_karte(punkte, skala, verzerrt.shape)
    karte = karte_bauen(koef, verzerrt.shape,
                        grenzen=(float(skala.min()), float(skala.max())))
    soll = (ecken[0] - 1) * (ecken[1] - 1) * feld_mm**2
    maske = np.zeros(verzerrt.shape, np.uint8)
    cv2.fillPoly(maske, [cv2.convexHull(ec.astype(np.float32)).astype(np.int32)], 1)
    ist = float(karte[maske > 0].sum())
    abw = 100 * (ist / soll - 1)
    ok = abs(abw) < 2 and b["restfehler_prozent"] < 2
    return ok, [
        f"{b['ecken_gefunden']} Ecken ueber '{b['stufe']}', {b['felder']} Felder",
        f"Restfehler des Fits: {b['restfehler_prozent']:.2f}%",
        f"Brettflaeche: Soll {soll:.1f} mm2, ueber die Karte {ist:.1f} mm2 "
        f"-> Abweichung {abw:+.2f}%",
    ]


if __name__ == "__main__":
    ok, zeilen = selbsttest()
    for z in zeilen:
        print(" ", z)
    print("ERGEBNIS:", "OK" if ok else "PRUEFEN")
