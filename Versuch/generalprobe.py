# -*- coding: utf-8 -*-
"""
generalprobe.py - Der komplette Messtag-Ablauf in einem Durchlauf, ohne Kanal.

Zweck: Einzeln getestete Teile koennen zusammen trotzdem scheitern. Diese
Probe durchlaeuft die ganze Kette in derselben Reihenfolge wie am Messtag und
benutzt dabei den ECHTEN Code der Live-Programme - nicht Nachbauten. Was hier
laeuft, laeuft vor Ort.

Ablauf (entspricht dem Versuchsplan):
  1  Schachbrettvorlage erzeugen
  2  Kalibrierbilder simulieren (Muster perspektivisch auf ein echtes Panelbild)
  3  Kalibrierpipeline -> Massstabskarte, Restfehler pruefen
  4  Messbereich setzen (erzeugt, was roi_werkzeug.py speichert)
  5  Kamera simulieren: Frames laufen einzeln in den Aufnahmeordner
  6  Live-Auswertung Flaeche: Referenz sammeln, dann messen, mm2 ueber die Karte
  7  Live-Auswertung Laser: Geometrie aus Referenzframes, dann Dicke
  8  Bericht

Aufruf:
  python generalprobe.py                  alles
  python generalprobe.py flaeche          nur der Flaechen-PC
  python generalprobe.py laser            nur der Laser-PC
"""
import os, sys, glob, shutil, time, threading, re
import numpy as np
import cv2

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
ARBEIT = os.path.join(BASE, "_generalprobe")

# Quelldaten dieser Probe. Am Messtag kommen die Bilder von der Kamera; hier
# nehmen wir Aufnahmen der bisherigen Messreihe.
REPO = os.path.dirname(BASE)
QUELLE_FLAECHE = os.path.join(REPO, "eisflaeche", "frames", "left")
QUELLE_LASER = os.path.join(REPO, "frame differencing", "serie_260402-174444", "frames")

ergebnisse = []


def schritt(nr, titel):
    print(f"\n{'='*66}\n{nr}  {titel}\n{'='*66}")


def melde(name, ok, text=""):
    ergebnisse.append((name, ok))
    print(f"   {'OK   ' if ok else 'FEHLER'}  {name}{('  -  ' + text) if text else ''}")
    return ok


def frame_nr(p):
    m = re.search(r"(\d+)", os.path.basename(p))
    return int(m.group(1)) if m else 0


# ─────────────────────────────────────────────────────────── 1 + 2
def kalibrierbilder_erzeugen(ziel, grundbild):
    """Musterbogen perspektivisch auf ein echtes Panelbild legen.

    Beide Positionen entstehen aus EINER Homographie - also aus einer festen
    Kamera, die schraeg auf eine ebene Flaeche blickt. Zwei willkuerlich ins
    Bild gesetzte Vierecke waeren geometrisch widerspruechlich, und der Fit
    muesste daran scheitern."""
    import schachbrett_drucken as sd
    pfad, ecken, (bb, bh), _ = sd.blatt_bauen(10.0, "a4")

    try:
        import pypdfium2 as pdfium
        seite = pdfium.PdfDocument(pfad)[0].render(scale=200 / 72).to_numpy()
        muster = cv2.cvtColor(seite, cv2.COLOR_RGB2GRAY) if seite.ndim == 3 else seite
    except ImportError:
        # Ohne PDF-Renderer das Muster direkt als Bild aufbauen - die Probe
        # soll nicht an einem Zusatzpaket scheitern.
        f = 40
        muster = np.full(((ecken[1] + 1) * f, (ecken[0] + 1) * f), 255, np.uint8)
        for i in range(ecken[1] + 1):
            for j in range(ecken[0] + 1):
                if (i + j) % 2 == 0:
                    muster[i*f:(i+1)*f, j*f:(j+1)*f] = 20
        muster = cv2.copyMakeBorder(muster, 60, 60, 60, 60, cv2.BORDER_CONSTANT, value=255)

    grund = cv2.imread(grundbild, cv2.IMREAD_GRAYSCALE)
    H, W = grund.shape
    panel_mm = (420.0, 190.0)
    bild_quad = np.float32([[140, 210], [W - 160, 90], [W - 280, 1520], [280, 1420]])
    Hom = cv2.getPerspectiveTransform(
        np.float32([[0, 0], [panel_mm[0], 0], [panel_mm[0], panel_mm[1]], [0, panel_mm[1]]]),
        bild_quad)

    def nach_bild(p_mm):
        p = np.hstack([np.float32(p_mm), np.ones((len(p_mm), 1), np.float32)])
        q = (Hom @ p.T).T
        return q[:, :2] / q[:, 2:3]

    os.makedirs(ziel, exist_ok=True)
    for alt in glob.glob(os.path.join(ziel, "*.png")):
        os.remove(alt)

    rng = np.random.default_rng(0)
    mh, mw = muster.shape
    beleuchtung = cv2.GaussianBlur(grund.astype(np.float32), (0, 0), 60)
    papier = np.clip(beleuchtung * 1.45, 12, 255)

    for i, (x0, y0) in enumerate(((40.0, 15.0), (215.0, 80.0)), start=1):
        ecken_mm = [(x0, y0), (x0 + 170, y0), (x0 + 170, y0 + 90), (x0, y0 + 90)]
        quad = nach_bild(ecken_mm)
        M = cv2.getPerspectiveTransform(
            np.float32([[0, 0], [mw, 0], [mw, mh], [0, mh]]), np.float32(quad))
        warp = cv2.warpPerspective(muster.astype(np.float32) / 255.0, M, (W, H),
                                   flags=cv2.INTER_AREA, borderValue=-1)
        drin = warp >= 0
        aus = grund.astype(np.float32).copy()
        aus[drin] = papier[drin] * warp[drin]
        aus[drin] += rng.normal(0, 2.0, int(drin.sum()))
        aus = cv2.GaussianBlur(np.clip(aus, 0, 255), (3, 3), 0.7)
        cv2.imwrite(os.path.join(ziel, f"flaeche_unten_{i:02d}.png"),
                    aus.astype(np.uint8))
    return ecken, Hom


def kalibrierung_pruefen(bilder_ordner, ecken):
    import kalibrierung_schachbrett as ks
    ks.BILDER = bilder_ordner
    ks.OUT = os.path.join(ARBEIT, "kalibrierung_ergebnis")
    ks.BRETT["flaeche_unten"] = dict(ecken=list(ecken), feld_mm=10.0)
    ks.STANDARD = dict(ecken=list(ecken), feld_mm=10.0)
    ks.main()
    npz = os.path.join(ks.OUT, "flaeche_unten_kalibrierung.npz")
    if not os.path.exists(npz):
        return melde("Kalibrierung", False, "keine Ergebnisdatei"), None
    d = np.load(npz)
    rest = float(d["restfehler_prozent"])
    ok = melde("Kalibrierung", rest < 3.0, f"Restfehler {rest:.2f}% (Grenze 3%)")
    return ok, npz


# ─────────────────────────────────────────────────────────── 4
def messbereich_setzen(grundbild, ziel_npz):
    """Erzeugt exakt das, was roi_werkzeug.py speichert. Die Oberflaeche selbst
    ist Handarbeit; geprueft wird hier das Format und dass die Live-Auswertung
    damit umgehen kann."""
    g = cv2.imread(grundbild, cv2.IMREAD_GRAYSCALE)
    H, W = g.shape
    x0, y0, x1, y1 = int(W * 0.05), int(H * 0.05), int(W * 0.95), int(H * 0.50)
    m = np.zeros((H, W), np.uint8)
    m[y0:y1, x0:x1] = 1
    np.savez(ziel_npz, maske=m, bbox=np.array([x0, y0, x1, y1]),
             flaeche_px=int(m.sum()), quelle=os.path.basename(grundbild))
    return melde("Messbereich", int(m.sum()) > 0,
                 f"{int(m.sum()):,} px, Zuschnitt {x1-x0}x{y1-y0}")


# ─────────────────────────────────────────────────────────── 5
def kamera_starten(quelle, ziel, anzahl, pause, endung=".tiff", ref=0):
    """Schreibt Frames einzeln in den Ordner - wie die Kamerasoftware.

    ref: so viele Frames vom Serienanfang (die eisfreie Referenz), der Rest
    ueber die ganze Serie verteilt. Sonst pruefte die Probe nur die ersten
    Sekunden, in denen noch kaum Eis liegt - sie wuerde bestehen, ohne dass
    die Messung jemals einen nennenswerten Wert zeigen musste."""
    alle = sorted(glob.glob(os.path.join(quelle, "*" + endung)), key=frame_nr)
    if ref and len(alle) > anzahl:
        spaet = np.linspace(len(alle) * 0.15, len(alle) - 1, anzahl - ref).astype(int)
        dateien = alle[:ref] + [alle[i] for i in spaet]
    else:
        dateien = alle[:anzahl]
    os.makedirs(ziel, exist_ok=True)

    def lauf():
        for i, q in enumerate(dateien, 1):
            vorl = os.path.join(ziel, f".teil_{i:05d}")
            shutil.copyfile(q, vorl)
            os.replace(vorl, os.path.join(ziel, f"frame_{i:05d}{endung}"))
            time.sleep(pause)

    t = threading.Thread(target=lauf, daemon=True)
    t.start()
    return t, len(dateien)


# ─────────────────────────────────────────────────────────── 6
def probe_flaeche(kalib_npz):
    schritt("6", "Live-Auswertung Flaeche (echter Code aus live_flaeche.py)")
    import live_flaeche as lf
    from gemeinsam.ordnerwache import Ordnerwache, bild_lesen
    from gemeinsam.massstab import Massstab

    aufnahme = os.path.join(ARBEIT, "aufnahme_flaeche")
    shutil.rmtree(aufnahme, ignore_errors=True)
    os.makedirs(aufnahme)

    REF = 4
    aus = lf.Auswerter(os.path.join(BASE, "modelle", "flaeche.pt"),
                       os.path.join(ARBEIT, "messbereich.npz"),
                       0.5, "auto", referenz_anzahl=REF)
    print(f"   Modell: {aus.kanaele} Kanal/Kanaele, {aus.geraet_text}")
    ms = Massstab(kalib_npz)
    print(f"   Massstab: {ms.quelle}")

    t, anzahl = kamera_starten(QUELLE_FLAECHE, aufnahme, 10, 0.6, ref=REF)
    wache = Ordnerwache(aufnahme, ab_bestand=False)

    gesehen, gemessen, werte = 0, 0, []
    ref_fertig_bei = None
    t0 = time.time()
    while time.time() - t0 < 60 and gesehen < anzahl:
        for pfad in wache.neue():
            gesehen += 1
            img = bild_lesen(pfad)
            if img is None:
                melde("Bild lesbar", False, os.path.basename(pfad)); continue
            if not aus.referenz_bereit:
                if aus.referenz_sammeln(img):
                    ref_fertig_bei = gesehen
                    print(f"   Referenz steht nach {gesehen} Frames")
                continue
            eis, crop = aus.auswerten(img)
            bez = aus.bezugsflaeche_px or eis.size
            anteil = 100.0 * float(eis.sum()) / bez
            versatz = (aus.bbox[0], aus.bbox[1]) if aus.bbox else (0, 0)
            mm2 = ms.flaeche_mm2(eis, versatz)
            gemessen += 1
            werte.append((anteil, mm2))
        time.sleep(0.2)
    t.join(timeout=5)

    melde("Frames erkannt", gesehen == anzahl, f"{gesehen} von {anzahl}")
    melde("Referenz gesammelt", ref_fertig_bei == REF,
          f"nach {ref_fertig_bei} Frames (erwartet {REF})")
    melde("Auswertung gelaufen", gemessen > 0, f"{gemessen} Frames gemessen")
    if werte:
        a = [w[0] for w in werte]
        print(f"   Bedeckungsgrad {min(a):.2f} bis {max(a):.2f}%")
        mm = [w[1] for w in werte if w[1] is not None]
        melde("Flaeche in mm2", bool(mm),
              f"{min(mm):.0f} bis {max(mm):.0f} mm2" if mm else "keine Kalibrierung wirksam")
        melde("Messung spricht an", max(a) - min(a) > 5.0,
              f"Spanne {max(a)-min(a):.1f} Prozentpunkte ueber die Serie (erwartet > 5)")


# ─────────────────────────────────────────────────────────── 7
def probe_laser():
    schritt("7", "Live-Auswertung Laser (echter Code aus live_laser.py)")
    if not os.path.isdir(QUELLE_LASER):
        melde("Laser-Testdaten", False, f"nicht gefunden: {QUELLE_LASER}"); return
    import live_laser as ll
    from gemeinsam.ordnerwache import Ordnerwache, bild_lesen

    aufnahme = os.path.join(ARBEIT, "aufnahme_laser")
    shutil.rmtree(aufnahme, ignore_errors=True)
    os.makedirs(aufnahme)

    aus = ll.Auswerter(os.path.join(BASE, "modelle", "laser.pt"), 0.5, "auto", 100)
    print(f"   Modell geladen, {aus.geraet_text}")

    REF = 3
    t, anzahl = kamera_starten(QUELLE_LASER, aufnahme, 8, 0.6, ".png", ref=REF)
    wache = Ordnerwache(aufnahme, ab_bestand=False)
    ref_masken, gesehen, gemessen, dicken = [], 0, 0, []
    d0 = None
    t0 = time.time()
    while time.time() - t0 < 90 and gesehen < anzahl:
        for pfad in wache.neue():
            gesehen += 1
            img = bild_lesen(pfad)
            if img is None:
                continue
            m = aus.maske(img)
            if aus.geo is None:
                ref_masken.append(m)
                if len(ref_masken) >= REF:
                    mittel = (np.mean(ref_masken, axis=0) > 127).astype(np.uint8) * 255
                    aus.geo = ll.geometrie_ableiten(mittel)
                    print(f"   Geometrie aus {REF} Referenzframes: "
                          f"{len(aus.geo['x'])} Stuetzstellen")
                continue
            d = aus.versatz(m, -25, 60)
            if d0 is None:
                d0 = d.copy()
            dicke = ll.median_glatt(d - d0, 9)
            gemessen += 1
            gueltig = np.isfinite(dicke)
            if gueltig.any():
                dicken.append(float(np.nanmax(dicke)))
        time.sleep(0.2)
    t.join(timeout=5)

    melde("Frames erkannt", gesehen == anzahl, f"{gesehen} von {anzahl}")
    melde("Geometrie abgeleitet", aus.geo is not None,
          f"{len(aus.geo['x'])} Stuetzstellen" if aus.geo is not None else "fehlgeschlagen")
    melde("Dicke berechnet", gemessen > 0 and bool(dicken),
          f"{gemessen} Frames, max. Versatz {max(dicken):.1f} px" if dicken else "keine Werte")
    # Nicht nur "laeuft", sondern "misst": ueber die Serie muss ein
    # nennenswerter Versatz entstehen, sonst zeigt die Kette nichts an.
    melde("Messung spricht an", bool(dicken) and max(dicken) > 2.0,
          f"groesster Versatz {max(dicken):.1f} px (erwartet > 2)" if dicken else "keine Werte")


# ─────────────────────────────────────────────────────────── Ablauf
def main():
    was = sys.argv[1].lower() if len(sys.argv) > 1 else "alles"
    shutil.rmtree(ARBEIT, ignore_errors=True)
    os.makedirs(ARBEIT)

    grund = sorted(glob.glob(os.path.join(QUELLE_FLAECHE, "*.tiff")), key=frame_nr)
    if not grund:
        print(f"Keine Flaechen-Frames in {QUELLE_FLAECHE}"); return
    grundbild = grund[0]

    print(f"Generalprobe - Arbeitsordner {ARBEIT}")
    print(f"Testdaten: {os.path.basename(grundbild)} und {len(grund)} Frames")

    kalib_npz = None
    if was in ("alles", "flaeche"):
        schritt("1+2", "Schachbrettvorlage und Kalibrierbilder erzeugen")
        bilder = os.path.join(ARBEIT, "kalibrierung_bilder")
        ecken, _ = kalibrierbilder_erzeugen(bilder, grundbild)
        melde("Kalibrierbilder", len(glob.glob(os.path.join(bilder, "*.png"))) == 2,
              f"2 Positionen, Brett {ecken[0]}x{ecken[1]} innere Ecken")

        schritt("3", "Kalibrierpipeline")
        ok, kalib_npz = kalibrierung_pruefen(bilder, ecken)

        schritt("4", "Messbereich setzen")
        messbereich_setzen(grundbild, os.path.join(ARBEIT, "messbereich.npz"))

        schritt("5+6", "Kamera simulieren und Flaeche live auswerten")
        probe_flaeche(kalib_npz)

    if was in ("alles", "laser"):
        probe_laser()

    schritt("8", "Bericht")
    fehler = [n for n, ok in ergebnisse if not ok]
    for name, ok in ergebnisse:
        print(f"   {'OK    ' if ok else 'FEHLER'}  {name}")
    print(f"\n{len(ergebnisse)-len(fehler)} von {len(ergebnisse)} Pruefungen bestanden")
    print("ERGEBNIS:", "bereit fuer den Messtag" if not fehler else f"OFFEN: {fehler}")


if __name__ == "__main__":
    main()
