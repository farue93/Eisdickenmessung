# -*- coding: utf-8 -*-
"""
generalprobe.py - Der komplette Messtag-Ablauf in einem Durchlauf, ohne Kanal.

Zweck: Einzeln getestete Teile koennen zusammen trotzdem scheitern. Diese Probe
durchlaeuft die ganze Kette in derselben Reihenfolge wie am Messtag und benutzt
dabei den ECHTEN Code der Messprogramme - nicht Nachbauten. Was hier laeuft,
laeuft vor Ort.

Ablauf (entspricht dem Versuchsplan):
  1  Schachbrettvorlage erzeugen
  2  EIN Kalibrierbild simulieren (Muster perspektivisch auf ein echtes Panelbild)
  3  Kalibrierung aus einem Bild -> Massstabskarte
  4  Wahrer Massstabsfehler: Karte gegen die bekannte Homographie
  5  Messbereich: Vorschlag und Speicherung (gemeinsam/messbereich.py)
  6  Ordnerwache: Unterordner finden, halbfertige Dateien, Live-Modus
  7  Kamera simulieren: Frames laufen einzeln in einen Unterordner
  8  Flaechenmessung: Referenz sammeln, dann messen, mm2 ueber die Karte
  9  Lasermessung: Geometrie aus Referenzframes, dann Dicke
 10  Bericht

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
    print(f"\n{'='*68}\n{nr}  {titel}\n{'='*68}")


def melde(name, ok, text=""):
    ergebnisse.append((name, ok))
    print(f"   {'OK   ' if ok else 'FEHLER'}  {name}{('  -  ' + text) if text else ''}")
    return ok


def frame_nr(p):
    m = re.search(r"(\d+)", os.path.basename(p))
    return int(m.group(1)) if m else 0


# ─────────────────────────────────────────────────────────── 1 + 2
def kalibrierbild_erzeugen(ziel_png, grundbild, ecken=(16, 10), feld_mm=10.0,
                           feld_px=40, rand_felder=2):
    """EIN Musterbogen perspektivisch auf ein echtes Panelbild legen.

    Das Muster wird hier synthetisch aufgebaut statt aus dem PDF gerendert, und
    zwar aus einem Grund: Die zurueckgegebene Homographie ist die WAHRHEIT, an
    der Schritt 4 den rekonstruierten Massstab misst. Dafuer muss die reale
    Groesse des Musters exakt bekannt sein. Ein gerendertes PDF-Blatt bringt
    seinen eigenen Seitenrand mit; wird dessen Ausdehnung beim Aufkleben auch
    nur um einen Faktor falsch angesetzt, prueft der Test hinterher gegen eine
    falsche Wahrheit - und meldet einen Fehler, den die Kalibrierung gar nicht
    gemacht hat. Genau das ist beim ersten Durchlauf passiert.

    Das Drucken selbst wird getrennt geprueft (schachbrett_pruefen)."""
    felder_x, felder_y = ecken[0] + 1, ecken[1] + 1
    mw = (felder_x + 2 * rand_felder) * feld_px
    mh = (felder_y + 2 * rand_felder) * feld_px
    muster = np.full((mh, mw), 255, np.uint8)
    for i in range(felder_y):
        for j in range(felder_x):
            if (i + j) % 2 == 0:
                y, x = (i + rand_felder) * feld_px, (j + rand_felder) * feld_px
                muster[y:y+feld_px, x:x+feld_px] = 20
    breite_mm = mw / feld_px * feld_mm      # reale Ausdehnung des ganzen Bogens
    hoehe_mm = mh / feld_px * feld_mm

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

    rng = np.random.default_rng(0)
    beleuchtung = cv2.GaussianBlur(grund.astype(np.float32), (0, 0), 60)
    papier = np.clip(beleuchtung * 1.45, 12, 255)

    # Mehrere Brettpositionen - so wie das Brett am Messtag nacheinander an
    # verschiedene Stellen geklebt wird.
    frei_x = max(0.0, panel_mm[0] - breite_mm)
    frei_y = max(0.0, panel_mm[1] - hoehe_mm)
    stellen = [(frei_x * fx, frei_y * fy)
               for fx, fy in ((0.02, 0.05), (0.98, 0.05), (0.5, 0.95), (0.02, 0.95))]
    os.makedirs(os.path.dirname(ziel_png), exist_ok=True)
    pfade = []
    for i, (x0, y0) in enumerate(stellen, start=1):
        quad = nach_bild([(x0, y0), (x0 + breite_mm, y0),
                          (x0 + breite_mm, y0 + hoehe_mm), (x0, y0 + hoehe_mm)])
        M = cv2.getPerspectiveTransform(
            np.float32([[0, 0], [mw, 0], [mw, mh], [0, mh]]), np.float32(quad))
        warp = cv2.warpPerspective(muster.astype(np.float32) / 255.0, M, (W, H),
                                   flags=cv2.INTER_AREA, borderValue=-1)
        drin = warp >= 0
        aus = grund.astype(np.float32).copy()
        aus[drin] = papier[drin] * warp[drin]
        aus[drin] += rng.normal(0, 2.0, int(drin.sum()))
        aus = cv2.GaussianBlur(np.clip(aus, 0, 255), (3, 3), 0.7)
        p = ziel_png.replace(".png", f"_{i:02d}.png")
        cv2.imwrite(p, aus.astype(np.uint8))
        pfade.append(p)
    return ecken, Hom, pfade


def schachbrett_pruefen():
    """Die Druckvorlage entsteht am Vorabend, nicht am Messtag - hier wird nur
    geprueft, dass sie sich erzeugen laesst und die Eckenparitaet stimmt."""
    import schachbrett_drucken as sd
    try:
        pfad, ecken, _, _ = sd.blatt_bauen(10.0, "a4")
    except Exception as e:
        return melde("Schachbrettvorlage", False, str(e))
    # gerade x ungerade: sonst ist das Brett um 180 Grad mehrdeutig
    paritaet = ecken[0] % 2 == 0 and ecken[1] % 2 == 1
    return melde("Schachbrettvorlage", os.path.exists(pfad) and paritaet,
                 f"{os.path.basename(pfad)}, {ecken[0]}x{ecken[1]} innere Ecken "
                 f"({'gerade x ungerade' if paritaet else 'PARITAET FALSCH'})")


def wahre_karte(Hom, shape, schritt_px=16):
    """Tatsaechliche Flaechenskala mm^2/px^2 aus der bekannten Homographie.

    Numerisch ueber die Jacobi-Determinante: ein Pixelquadrat wird nach mm
    zurueckgerechnet, seine Flaeche ist die gesuchte Skala. Bewusst numerisch
    statt analytisch - eine Formel, die man nur einmal hinschreibt, kann
    genauso falsch sein wie der Code, den sie pruefen soll."""
    H, W = shape
    G = np.linalg.inv(Hom)
    ys, xs = np.mgrid[0:H:schritt_px, 0:W:schritt_px]
    p0 = np.stack([xs.ravel(), ys.ravel()], axis=-1).astype(np.float32)
    p1 = p0 + [1, 0]
    p2 = p0 + [0, 1]
    def nach_mm(p):
        return cv2.perspectiveTransform(p.reshape(-1, 1, 2), G).reshape(-1, 2)
    a, b, c = nach_mm(p0), nach_mm(p1), nach_mm(p2)
    u, v = b - a, c - a
    flaeche = np.abs(u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0])
    return flaeche.reshape(xs.shape), (ys, xs)


# ─────────────────────────────────────────────────────────── 3 + 4
def kalibrierung_pruefen(pfade, ecken, Hom):
    """Ein Brett gegen vier, jeweils gegen die BEKANNTE Wahrheit gemessen."""
    from gemeinsam import kalibrierung as kal
    from gemeinsam.massstab import Massstab

    bilder = [dict(name=os.path.basename(p),
                   grau=cv2.imread(p, cv2.IMREAD_GRAYSCALE), suchbereich=None)
              for p in pfade]
    shape = bilder[0]["grau"].shape
    wahr, (ys, xs) = wahre_karte(Hom, shape)

    def bewerten(npz):
        ms = Massstab(npz)
        fehler = 100 * np.abs(ms.karte[ys, xs] / wahr - 1)
        d = np.load(npz)
        huelle = cv2.convexHull(d["punkte"].astype(np.float32)).astype(np.int32)
        drin = np.zeros(xs.shape, bool)
        for i in range(xs.shape[0]):
            for j in range(xs.shape[1]):
                drin[i, j] = cv2.pointPolygonTest(
                    huelle, (float(xs[i, j]), float(ys[i, j])), False) >= 0
        innen = float(fehler[drin].mean()) if drin.any() else float("nan")
        aussen = float(fehler[~drin].mean()) if (~drin).any() else float("nan")
        return ms, innen, aussen

    ergebnis = {}
    for anzahl in (1, len(bilder)):
        npz = os.path.join(ARBEIT, "kalibrierung", f"probe_{anzahl}.npz")
        b = kal.kalibrieren(bilder[:anzahl], ecken, 10.0, grad="automatisch",
                            npz_pfad=npz,
                            png_pfad=os.path.join(ARBEIT, "kalibrierung",
                                                  f"probe_{anzahl}_kontrolle.png"),
                            name=f"{anzahl} Brett(er)")
        if not b["ok"]:
            melde(f"Kalibrierung aus {anzahl} Bild(ern)", False, b["fehler"])
            continue
        ms, innen, aussen = bewerten(npz)
        ergebnis[anzahl] = (npz, b, ms, innen, aussen)
        print(f"   {anzahl} Brett(er): {kal.zusammenfassung(b)}")
        print(f"      Grad {b['grad']} ({b['grad_grund']})")
        print(f"      gegen die Wahrheit: auf dem Brett {innen:.2f}%, "
              f"ausserhalb {aussen:.2f}%")

    if len(ergebnis) < 2:
        return None
    (npz1, b1, _, in1, aus1) = ergebnis[1]
    (npz4, b4, ms4, in4, aus4) = ergebnis[len(bilder)]

    melde("Kalibrierung aus einem Bild", b1["restfehler_prozent"] < 3.0,
          f"Restfehler {b1['restfehler_prozent']:.2f}%, "
          f"{b1['gestuetzt_prozent']:.0f}% gestuetzt, ausserhalb {aus1:.2f}%")
    melde("mehr Brettpositionen -> mehr belegte Bildflaeche",
          b4["gestuetzt_prozent"] > b1["gestuetzt_prozent"],
          f"{b1['gestuetzt_prozent']:.0f}% -> {b4['gestuetzt_prozent']:.0f}%")
    melde("mehr Brettpositionen -> besser ausserhalb", aus4 < aus1,
          f"{aus1:.2f}% -> {aus4:.2f}% mittlerer Betragsfehler")
    melde("Massstab auf dem Brett", in4 < 3.0, f"{in4:.2f}% (Grenze 3%)")

    # ---- Laengen: Metrik gegen isotrope Naeherung
    proben = []
    for xm in np.linspace(40, 380, 10):
        for ym in np.linspace(25, 165, 5):
            for dx, dy in ((0.0, -5.0), (5.0, 0.0)):
                p_mm = np.float32([[xm, ym], [xm + dx, ym + dy], [1, 1]])
                p = np.hstack([p_mm[:2], np.ones((2, 1), np.float32)])
                q = (Hom @ p.T).T
                q = q[:, :2] / q[:, 2:3]
                proben.append((q[0], q[1]))
    p0 = np.array([a for a, _ in proben])
    p1 = np.array([b for _, b in proben])
    mit = 100 * np.abs(ms4.laenge_mm(p0, p1) / 5.0 - 1)
    ohne_ms = Massstab(npz4); ohne_ms.metrik = None
    ohne = 100 * np.abs(ohne_ms.laenge_mm(p0, p1) / 5.0 - 1)
    global_ppm = np.hypot(p1[:, 0]-p0[:, 0], p1[:, 1]-p0[:, 1]) / ms4.px_pro_mm()
    med = 100 * np.abs(global_ppm / 5.0 - 1)
    print(f"   5-mm-Strecken an {len(proben)} Stellen, Fehler gegen die Wahrheit:")
    print(f"      Metrik (richtungsabhaengig)  Mittel {mit.mean():5.2f}% / max {mit.max():5.2f}%")
    print(f"      isotrop sqrt(Flaeche)        Mittel {ohne.mean():5.2f}% / max {ohne.max():5.2f}%")
    print(f"      ein globaler px/mm-Wert      Mittel {med.mean():5.2f}% / max {med.max():5.2f}%")
    melde("Laenge ueber die Metrik trifft", mit.max() < 3.0,
          f"max {mit.max():.2f}% (Grenze 3%)")
    melde("Metrik schlaegt die isotrope Naeherung", mit.max() < ohne.max(),
          f"max {mit.max():.2f}% statt {ohne.max():.2f}%")
    return npz4


# ─────────────────────────────────────────────────────────── 5
def messbereich_setzen(grundbild, ziel_npz):
    """Vorschlag rechnen, Ausschluss anwenden, speichern - der Weg, den der
    Reiter 'Messbereich' geht. Das Anpassen von Hand ist Handarbeit; geprueft
    wird, dass Vorschlag, Format und Auswertung zusammenpassen."""
    from gemeinsam import messbereich as mbr
    g = cv2.imread(grundbild, cv2.IMREAD_GRAYSCALE)
    H, W = g.shape
    v = mbr.vorschlagen(g)
    melde("Messbereich vorgeschlagen", v is not None,
          f"{v[2]-v[0]}x{v[3]-v[1]} ab ({v[0]},{v[1]}), "
          f"{100*(v[2]-v[0])*(v[3]-v[1])/(W*H):.0f}% des Bildes" if v else "kein Vorschlag")
    if v is None:
        v = (int(W * 0.05), int(H * 0.05), int(W * 0.95), int(H * 0.50))
    # Wie beim Anpassen von Hand: Rechteck etwas verkleinern, eine Stoerstelle
    # ausschliessen - genau das, was der Reiter mit Anfassern und Pinsel tut.
    v = (v[0] + 40, v[1] + 40, v[2] - 40, v[3] - 40)
    aus = np.zeros((H, W), np.uint8)
    cv2.circle(aus, (v[0] + 300, v[1] + 300), 150, 1, -1)
    b = mbr.speichern(ziel_npz, g, v, aus, quelle=os.path.basename(grundbild))
    melde("Messbereich gespeichert", b["ok"] and b["ausgeschlossen"] > 0,
          mbr.bericht_text(b) if b["ok"] else b["fehler"])
    return b["ok"]


# ─────────────────────────────────────────────────────────── 6
def wache_pruefen():
    """Die drei Eigenschaften, auf die sich am Messtag alles stuetzt."""
    from gemeinsam.ordnerwache import Ordnerwache, AUTO
    wurzel = os.path.join(ARBEIT, "wachetest")
    shutil.rmtree(wurzel, ignore_errors=True)
    lauf = os.path.join(wurzel, "lauf_01")
    os.makedirs(lauf)

    # (a) halbfertige Datei
    w = Ordnerwache(wurzel, AUTO, ab_bestand=True)
    ziel = os.path.join(lauf, "bild_0001.png")
    with open(ziel, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n" + b"\x00" * 5000)
    sofort = len(w.neue())
    cv2.imwrite(ziel, (np.random.rand(80, 120) * 255).astype(np.uint8))
    w.neue()
    danach = len(w.neue())
    melde("halbfertige Datei erkannt", sofort == 0 and danach == 1,
          f"waehrend Schreiben {sofort} (erwartet 0), danach {danach} (erwartet 1)")

    # (b) Unterordner gefunden
    melde("Unterordner gefunden", w.aktiv_kurz == "lauf_01",
          f"aktiv: {w.aktiv_kurz} (erwartet lauf_01)")

    # (c) Live-Modus laesst Frames aus
    for i in range(2, 7):
        cv2.imwrite(os.path.join(lauf, f"bild_{i:04d}.png"),
                    (np.random.rand(80, 120) * 255).astype(np.uint8))
    w.neue()                       # Groessen merken
    neuestes = w.neueste()
    melde("Live-Modus nimmt das neueste", neuestes is not None
          and os.path.basename(neuestes) == "bild_0006.png" and w.uebersprungen == 4,
          f"{os.path.basename(neuestes) if neuestes else '-'}, "
          f"{w.uebersprungen} uebersprungen (erwartet bild_0006.png / 4)")
    shutil.rmtree(wurzel, ignore_errors=True)


# ─────────────────────────────────────────────────────────── 7
def kamera_starten(quelle, wurzel, anzahl, pause, endung=".tiff", ref=0,
                   unterordner="lauf_01"):
    """Schreibt Frames einzeln in einen UNTERORDNER - wie die Kamerasoftware.

    ref: so viele Frames vom Serienanfang (die eisfreie Referenz), der Rest
    ueber die ganze Serie verteilt. Sonst pruefte die Probe nur die ersten
    Sekunden, in denen noch kaum Eis liegt - sie wuerde bestehen, ohne dass die
    Messung jemals einen nennenswerten Wert zeigen musste."""
    alle = sorted(glob.glob(os.path.join(quelle, "*" + endung)), key=frame_nr)
    if ref and len(alle) > anzahl:
        spaet = np.linspace(len(alle) * 0.15, len(alle) - 1, anzahl - ref).astype(int)
        dateien = alle[:ref] + [alle[i] for i in spaet]
    else:
        dateien = alle[:anzahl]
    ziel = os.path.join(wurzel, unterordner)
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


# ─────────────────────────────────────────────────────────── 8
def probe_flaeche(kalib_npz):
    schritt("7+8", "Kamera simulieren und Flaeche messen (Code aus messung_flaeche.py)")
    import messung_flaeche as mf
    from gemeinsam.ordnerwache import Ordnerwache, bild_lesen, AUTO
    from gemeinsam.massstab import Massstab

    aufnahme = os.path.join(ARBEIT, "aufnahme_flaeche")
    shutil.rmtree(aufnahme, ignore_errors=True)
    os.makedirs(aufnahme)

    REF = 4
    aus = mf.Auswerter(os.path.join(BASE, "modelle", "flaeche.pt"), 0.5, "auto")
    px = aus.panel_laden(os.path.join(ARBEIT, "messbereich.npz"))["px"]
    print(f"   Modell: {aus.kanaele} Kanal/Kanaele, {aus.geraet_text}")
    print(f"   Messbereich: {px:,} px".replace(",", "."))
    ms = Massstab(kalib_npz)
    print(f"   Massstab: {ms.quelle}")
    aus.referenz_start(REF)

    t, anzahl = kamera_starten(QUELLE_FLAECHE, aufnahme, 10, 0.6, ref=REF)
    wache = Ordnerwache(aufnahme, AUTO, ab_bestand=False)

    gesehen, gemessen, werte = 0, 0, []
    ref_fertig_bei = None
    t0 = time.time()
    while time.time() - t0 < 90 and gesehen < anzahl:
        for pfad in wache.neue():
            gesehen += 1
            img = bild_lesen(pfad)
            if img is None:
                melde("Bild lesbar", False, os.path.basename(pfad)); continue
            if not aus.referenz_bereit:
                fertig, _ = aus.referenz_sammeln(img)
                if fertig:
                    ref_fertig_bei = gesehen
                    print(f"   Referenz steht nach {gesehen} Frames")
                continue
            eis, crop = aus.auswerten(img)
            bez = aus.bezugsflaeche_px or eis.size
            anteil = 100.0 * float(eis.sum()) / bez
            versatz = (aus.bbox[0], aus.bbox[1]) if aus.bbox else (0, 0)
            werte.append((anteil, ms.flaeche_mm2(eis, versatz)))
            gemessen += 1
        time.sleep(0.2)
    t.join(timeout=5)

    melde("Frames erkannt", gesehen == anzahl, f"{gesehen} von {anzahl}")
    melde("Unterordner benutzt", wache.aktiv_kurz == "lauf_01",
          f"gelesen aus {wache.aktiv_kurz}")
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


# ─────────────────────────────────────────────────────────── 9
def probe_laser():
    schritt("9", "Lasermessung (Code aus messung_laser.py)")
    if not os.path.isdir(QUELLE_LASER):
        melde("Laser-Testdaten", False, f"nicht gefunden: {QUELLE_LASER}"); return
    import messung_laser as ml
    from gemeinsam.ordnerwache import Ordnerwache, bild_lesen, AUTO

    aufnahme = os.path.join(ARBEIT, "aufnahme_laser")
    shutil.rmtree(aufnahme, ignore_errors=True)
    os.makedirs(aufnahme)

    aus = ml.Auswerter(os.path.join(BASE, "modelle", "laser.pt"), 0.5, "auto", 100)
    print(f"   Modell geladen, {aus.geraet_text}")

    REF = 3
    t, anzahl = kamera_starten(QUELLE_LASER, aufnahme, 8, 0.6, ".png", ref=REF)
    wache = Ordnerwache(aufnahme, AUTO, ab_bestand=False)
    ref_masken, gesehen, gemessen, dicken = [], 0, 0, []
    t0 = time.time()
    while time.time() - t0 < 120 and gesehen < anzahl:
        for pfad in wache.neue():
            gesehen += 1
            img = bild_lesen(pfad)
            if img is None:
                continue
            m = aus.maske(img)
            if not aus.referenz_bereit:
                ref_masken.append(m)
                if len(ref_masken) >= REF:
                    mittel = (np.mean(ref_masken, axis=0) > 127).astype(np.uint8) * 255
                    aus.geo = ml.geometrie_ableiten(mittel)
                    aus.d0 = aus.versatz(mittel, -25, 60)
                    print(f"   Geometrie aus {REF} Referenzframes: "
                          f"{len(aus.geo['x'])} Stuetzstellen")
                continue
            dicke = ml.median_glatt(aus.versatz(m, -25, 60) - aus.d0, 9)
            gemessen += 1
            if np.isfinite(dicke).any():
                dicken.append(float(np.nanmax(dicke)))
        time.sleep(0.2)
    t.join(timeout=5)

    melde("Frames erkannt", gesehen == anzahl, f"{gesehen} von {anzahl}")
    melde("Unterordner benutzt", wache.aktiv_kurz == "lauf_01",
          f"gelesen aus {wache.aktiv_kurz}")
    melde("Geometrie abgeleitet", aus.geo is not None,
          f"{len(aus.geo['x'])} Stuetzstellen" if aus.geo is not None else "fehlgeschlagen")
    melde("Dicke berechnet", gemessen > 0 and bool(dicken),
          f"{gemessen} Frames, max. Versatz {max(dicken):.1f} px" if dicken else "keine Werte")
    # Nicht nur "laeuft", sondern "misst": ueber die Serie muss ein nennenswerter
    # Versatz entstehen, sonst zeigt die Kette nichts an.
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

    schritt("6", "Ordnerwache")
    wache_pruefen()

    kalib_npz = None
    if was in ("alles", "flaeche"):
        schritt("1+2", "Schachbrettvorlage und Kalibrierbilder erzeugen")
        schachbrett_pruefen()
        bild = os.path.join(ARBEIT, "kalibrierung", "kalibrierbild.png")
        ecken, Hom, pfade = kalibrierbild_erzeugen(bild, grundbild)
        melde("Kalibrierbilder", len(pfade) == 4,
              f"{len(pfade)} Brettpositionen, {ecken[0]}x{ecken[1]} innere Ecken")

        schritt("3+4", "Kalibrierung und Vergleich mit der Wahrheit")
        kalib_npz = kalibrierung_pruefen(pfade, ecken, Hom)

        schritt("5", "Messbereich setzen")
        messbereich_setzen(grundbild, os.path.join(ARBEIT, "messbereich.npz"))

        probe_flaeche(kalib_npz)

    if was in ("alles", "laser"):
        probe_laser()

    schritt("10", "Bericht")
    fehler = [n for n, ok in ergebnisse if not ok]
    for name, ok in ergebnisse:
        print(f"   {'OK    ' if ok else 'FEHLER'}  {name}")
    print(f"\n{len(ergebnisse)-len(fehler)} von {len(ergebnisse)} Pruefungen bestanden")
    print("ERGEBNIS:", "bereit fuer den Messtag" if not fehler else f"OFFEN: {fehler}")


if __name__ == "__main__":
    main()
