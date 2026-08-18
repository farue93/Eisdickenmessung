# -*- coding: utf-8 -*-
"""
trockentest.py - Prueft die Live-Auswertung OHNE Kamera und ohne Kanal.

Zweck: Am Messtag darf nichts mehr grundsaetzlich fehlschlagen. Dieser Test
kopiert vorhandene Frames einzeln und mit Verzoegerung in einen Testordner und
simuliert damit genau das, was die Kamerasoftware spaeter tut - inklusive des
kritischen Falls, dass eine Datei noch im Schreiben ist.

Aufruf:
  python trockentest.py module      Module und Abhaengigkeiten pruefen
  python trockentest.py wache       Ordnerwache gegen halbfertige Dateien pruefen
  python trockentest.py speisen ORDNER [ANZAHL] [PAUSE_S]
                                    Frames in ORDNER einspielen (Kamera-Ersatz)
"""
import os, sys, time, glob, shutil, threading

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)


def module():
    print("Abhaengigkeiten:")
    fehlt = []
    for name in ("numpy", "cv2", "torch", "PIL", "tkinter"):
        try:
            m = __import__(name)
            print(f"  {name:10s} OK  {getattr(m, '__version__', '')}")
        except Exception as e:
            print(f"  {name:10s} FEHLT  ({e.__class__.__name__})")
            fehlt.append(name)

    print("\nEigene Module:")
    for name in ("gemeinsam.konfig", "gemeinsam.ordnerwache", "gemeinsam.massstab",
                 "gemeinsam.kalibrierung", "gemeinsam.kalibrier_tab",
                 "gemeinsam.geraet", "gemeinsam.gui"):
        try:
            __import__(name); print(f"  {name:22s} OK")
        except Exception as e:
            print(f"  {name:22s} FEHLER: {e}"); fehlt.append(name)

    print("\nRechenwerk:")
    try:
        from gemeinsam.geraet import geraet_waehlen
        _, text = geraet_waehlen("auto")
        print(f"  {text}")
        if text.startswith("CPU"):
            print("  -> ohne GPU wird das U-Net deutlich langsamer;")
            print("     in der Oberflaeche 'jedes N-te Bild' erhoehen")
    except Exception as e:
        print(f"  nicht bestimmbar: {e}")

    print("\nMitgelieferte Netzdefinitionen:")
    for name in ("netze.unet_flaeche", "netze.unet_laser"):
        try:
            __import__(name); print(f"  {name:22s} OK")
        except Exception as e:
            print(f"  {name:22s} FEHLER: {e}"); fehlt.append(name)

    print("\nModellgewichte (muessen im Ordner modelle/ liegen):")
    for beschreibung, datei in (("Flaeche (U-Net)", "flaeche.pt"),
                                ("Laser (U-Net)", "laser.pt")):
        pfad = os.path.join(BASE, "modelle", datei)
        da = os.path.exists(pfad)
        groesse = f"{os.path.getsize(pfad)/1024**2:.0f} MB" if da else ""
        print(f"  {beschreibung:18s} {'gefunden' if da else 'FEHLT'}  {groesse}")
        if not da:
            fehlt.append(f"modelle/{datei}")

    print("\nProgramme fuer den Messtag:")
    for beschreibung, datei in (("Lasermessung", "messung_laser.py"),
                                ("Flaechenmessung", "messung_flaeche.py"),
                                ("Messbereich setzen", "roi_werkzeug.py"),
                                ("Druckvorlage", "schachbrett_drucken.py")):
        da = os.path.exists(os.path.join(BASE, datei))
        print(f"  {beschreibung:18s} {'gefunden' if da else 'FEHLT'}")
        if not da:
            fehlt.append(datei)

    print("\nKalibrierkette (Selbsttest mit synthetischem Brett):")
    try:
        from gemeinsam.kalibrierung import selbsttest
        ok, zeilen = selbsttest()
        for z in zeilen:
            print(f"  {z}")
        print(f"  -> {'OK' if ok else 'PRUEFEN'}")
        if not ok:
            fehlt.append("Kalibrierung (Selbsttest)")
    except Exception as e:
        print(f"  FEHLER: {e}")
        fehlt.append("Kalibrierung (Selbsttest)")

    print("\nMessbereich:")
    vorhanden = [f for f in os.listdir(BASE) if f.startswith("messbereich") and f.endswith(".npz")]
    if vorhanden:
        print(f"  vorhanden: {', '.join(vorhanden)}")
        print("  PRUEFEN, ob er zur AKTUELLEN Kameraposition passt.")
        print("  Ein Messbereich aus einem frueheren Aufbau ist wertlos.")
    else:
        print("  noch nicht gesetzt - am Messtag in der Flaechen-Oberflaeche")
        print("  ueber 'Messbereich festlegen ...' anlegen (kein Fehler: gehoert vor Ort)")

    print("\nErgebnis:", "alles bereit" if not fehlt else f"FEHLT: {fehlt}")


def wache():
    """Die drei Eigenschaften, auf die sich am Messtag alles stuetzt:
    halbfertige Dateien werden nicht ausgeliefert, der Unterordner der Kamera
    wird gefunden, und im Live-Modus wird das neueste Bild genommen."""
    from gemeinsam.ordnerwache import Ordnerwache, AUTO
    import numpy as np, cv2

    def bild(pfad):
        cv2.imwrite(pfad, (np.random.rand(80, 120) * 255).astype(np.uint8))

    wurzel = os.path.join(BASE, "_trockentest_wache")
    shutil.rmtree(wurzel, ignore_errors=True)
    lauf = os.path.join(wurzel, "lauf_01")
    os.makedirs(lauf)
    w = Ordnerwache(wurzel, AUTO, ab_bestand=True)

    ziel = os.path.join(lauf, "bild_0001.png")
    with open(ziel, "wb") as f:                     # angefangen, noch nicht fertig
        f.write(b"\x89PNG\r\n\x1a\n" + b"\x00" * 5000)
    sofort = w.neue()
    print(f"  waehrend des Schreibens ausgeliefert: {len(sofort)}  (erwartet 0)")

    bild(ziel)
    w.neue()                                        # Groesse merken
    time.sleep(0.05)
    danach = w.neue()
    print(f"  nach Abschluss ausgeliefert:         {len(danach)}  (erwartet 1)")

    nochmal = w.neue()
    print(f"  erneut ausgeliefert:                 {len(nochmal)}  (erwartet 0)")

    ordner_ok = w.aktiv_kurz == "lauf_01"
    print(f"  ueberwachter Unterordner:            {w.aktiv_kurz}  (erwartet lauf_01)")

    for i in range(2, 7):
        bild(os.path.join(lauf, f"bild_{i:04d}.png"))
    w.neue()                                        # Groessen merken
    neu = w.neueste()
    live_ok = (neu is not None and os.path.basename(neu) == "bild_0006.png"
               and w.uebersprungen == 4)
    print(f"  Live-Modus liefert:                  "
          f"{os.path.basename(neu) if neu else '-'}, "
          f"{w.uebersprungen} uebersprungen  (erwartet bild_0006.png / 4)")

    shutil.rmtree(wurzel, ignore_errors=True)
    alles = (not sofort and len(danach) == 1 and not nochmal and ordner_ok and live_ok)
    print("  Ergebnis:", "OK" if alles else "FEHLER")


def speisen(wurzel, anzahl=40, pause=1.0):
    """Kamera-Ersatz: kopiert vorhandene Frames einzeln in einen UNTERORDNER
    des Zielordners - genau so, wie die Kamerasoftware je Lauf einen anlegt.
    In der Oberflaeche wird WURZEL gewaehlt, nicht der Unterordner."""
    ziel = os.path.join(wurzel, time.strftime("lauf_%H%M%S"))
    quelle = os.environ.get("TESTFRAMES") or os.path.join(
        os.path.dirname(BASE), "eisflaeche", "frames", "left")
    dateien = sorted(glob.glob(os.path.join(quelle, "*.tif*")))
    if not dateien:
        print(f"Keine Quellframes in {quelle}")
        print(r"Anderen Ordner angeben:  set TESTFRAMES=D:\pfad\zu\frames")
        return
    import re
    def nr(p):
        m = re.search(r"(\d+)", os.path.basename(p))
        return int(m.group(1)) if m else 0
    dateien.sort(key=nr)
    dateien = dateien[:anzahl]
    os.makedirs(ziel, exist_ok=True)
    print(f"Speise {len(dateien)} Frames nach {ziel}, alle {pause}s")
    print(f"In der Oberflaeche als Aufnahmeordner waehlen:  {wurzel}")
    print("(der Unterordner wird automatisch gefunden), dann 'Kamera verbinden'.")
    for i, q in enumerate(dateien, 1):
        # erst unter Hilfsnamen kopieren, dann umbenennen: so taucht die Datei
        # nur vollstaendig auf - wie es eine gut gebaute Kamerasoftware macht
        vorlaeufig = os.path.join(ziel, f".teil_{i:05d}")
        endgueltig = os.path.join(ziel, f"frame_{i:05d}.tiff")
        shutil.copyfile(q, vorlaeufig)
        os.replace(vorlaeufig, endgueltig)
        print(f"  {i}/{len(dateien)}  {os.path.basename(endgueltig)}", flush=True)
        time.sleep(pause)
    print("fertig")


if __name__ == "__main__":
    befehl = sys.argv[1] if len(sys.argv) > 1 else "module"
    if befehl == "module":
        module()
    elif befehl == "wache":
        wache()
    elif befehl == "speisen":
        if len(sys.argv) < 3:
            print("Aufruf: python trockentest.py speisen ORDNER [ANZAHL] [PAUSE_S]")
        else:
            speisen(sys.argv[2],
                    int(sys.argv[3]) if len(sys.argv) > 3 else 40,
                    float(sys.argv[4]) if len(sys.argv) > 4 else 1.0)
    else:
        print(__doc__)
