# Eisdickenmessung – Vorverarbeitung + Frame Differencing + Canny

Bildbasierte Messung des Eisansatzes an der Slat-Vorderkante im
Enteisungswindkanal. Eine Laserlinie auf dem Profil wird detektiert; die
senkrechte Verschiebung dieser Linie über der Bogenlänge ist die Eisdicke.

Dieser Stand enthält die **Vorverarbeitung** (Laserlinien-Detektion + ROI) sowie
zwei Messmethoden (**Frame Differencing**, **Canny**), inkl. der interaktiven
HTML-Ansicht (Film + Dickenkurve).

## Schnellstart

```
1. Repo ziehen
2. Je Messreihe einen Unterordner mit den *.tif in  input/  ablegen
   (erstes *.tif = eisfreier Frame 0)
3. python run.py
```

`run.py` verarbeitet **alle** Serien in `input/` (Vorverarbeitung -> Frame
Differencing -> Canny) und schreibt `ergebnis.html` mit Links auf alle Viewer.

## Struktur

```
run.py                  Alle Serien in input/ auswerten
input/                  ← hier die Bilderserien ablegen (ein Unterordner je Serie)
pre processing/Bild 0/
    laser_pipeline.py   Laserlinie im Referenzframe (Frame 0) detektieren + fitten
    crop_roi.py         gerundetes ROI-Band um die Laserlinie
frame differencing/
    serie_eis.py        Eisdicke je Frame über die Serie + HTML-Viewer
    frame_diff.py       Differenzbild-Variante (D = I_N - I_0) für ein Zielframe
canny ice detection/
    canny_eis.py        Eisdicke über Canny-Kanten + HTML-Viewer
```

Die Skripte finden einander über relative Pfade (die Ordner liegen als
Schwesterordner nebeneinander).

## Manuell (einzelne Schritte)

Eine Serie = ein Ordner voller `*.tif`. Das **erste** `*.tif` ist Frame 0 und
muss der **eisfreie** Referenzframe sein.

```bash
# 1) Vorverarbeitung auf Frame 0 (einmal je Serie):
python "pre processing/Bild 0/laser_pipeline.py" "PFAD/frame0.tif" --out "pre processing/Bild 0/output"
python "pre processing/Bild 0/crop_roi.py" --out "pre processing/Bild 0/output" --raw "PFAD/ZUR/SERIE" --stem <frame0-stamm>

# 2) Messmethoden (Serienordner als Argument, oder ohne = erste Serie in input/):
python "frame differencing/serie_eis.py"  "PFAD/ZUR/SERIE"
python "canny ice detection/canny_eis.py" "PFAD/ZUR/SERIE"
```

Jede Methode schreibt einen Ordner `serie_<suffix>/` mit den gecroppten Frames,
`data.json` und **`viewer.html`**. Die `viewer.html` im Browser öffnen: Film
synchron zur Eisdickenkurve über der Bogenlänge, umschaltbar px/mm, mit
optionaler Ausreißerglättung.

Ohne Argument laufen die Skripte auf einer lokal hinterlegten Beispielreihe
(Pfad oben im jeweiligen Skript, per Argument überschreibbar).

## Voraussetzungen

Python 3 mit `numpy`, `opencv-python`, `scipy`, `scikit-image`, `matplotlib`,
`tifffile`.

## Methodik (kurz)

Gemeinsames Messgerüst: Referenzpunkte der Laserlinie, Bogenlänge *s* und
Außennormalen. Pro Frame wird entlang derselben Normalen gemessen; Eisdicke =
Versatz gegenüber Frame 0. Die Methoden sind unabhängig (Intensitätsrücken,
Differenzbild-Dipol, Gradientenkante) – Übereinstimmung ihrer Kurven ist die
zentrale Validierung. Kalibrierung px→mm über die 1:1-CAD-Geometrie der Slat.
