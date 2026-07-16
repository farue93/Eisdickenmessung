# Eisdickenmessung – automatisierte Bildauswertung

Bildbasierte Messung des Eisansatzes an der Slat-Vorderkante im
Enteisungswindkanal (TU Braunschweig, IMA). Eine Laserlinie auf dem Profil wird
detektiert; die **senkrechte Verschiebung** dieser Linie über der Bogenlänge ist
die Eisdicke. Fünf Detektionsmethoden werden implementiert und verglichen.

## Struktur

```
run.py                     Alle Serien in input/ auswerten (Klassik + HED)
input/                     ← hier die Bilderserien ablegen (ein Unterordner je Serie)

preprocessing/             Vorverarbeitung (einmal je Serie)
    laser_pipeline.py      Laserlinie in Frame 0 detektieren (Steger) + fitten
    crop_roi.py            ROI-Band um die Laserlinie -> Crop

methoden/                  eine Methode je Unterordner
    frame_differencing/    Methode A: Laserlinienversatz (serie_eis.py, frame_diff.py)
    canny/                 Methode B: Canny-Kante (canny_eis.py)
    hed/                   Methode E: KI-Kante HED (hed_eis.py + hed.caffemodel)
    unet/                  Methode D: U-Net Segmentierung (modell/train/apply)
    sam/                   Methode C: SAM zero-shot Baseline (sam_baseline.py)

gemeinsam/                 methodenübergreifend
    pfade.py               relative Repo-/Serien-Auflösung (alle Skripte nutzen sie)
    label_tool.py / .bat   Handlabeln der Laserlinie (für U-Net)
    setup_labels.py        Frames auswählen + Best-Punkt-Vorbelegung
    mittellinie.py         aus dicken Handlabels die 1px-Mittellinie (U-Net-Ziel)
    vergleich4.py          EIN zoombarer Viewer mit allen Methoden
```

Alle Skripte finden einander **relativ** über `gemeinsam/pfade.py` (sucht die
Repo-Wurzel = Ordner mit `methoden/` + `preprocessing/`). Nichts ist fest verdrahtet.

## Schnellstart (Klassik + HED, automatisch)

```
1. Repo ziehen
2. Je Messreihe einen Unterordner mit *.tif in  input/  ablegen
   (erstes *.tif = eisfreier Frame 0)
3. python run.py
```

`run.py` läuft je Serie: Vorverarbeitung → Frame-Differencing → Canny → HED und
schreibt `ergebnis.html` mit Links auf alle Viewer.

## KI-Segmentierung (U-Net, mit Handlabels)

Das U-Net wird auf wenigen von Hand gelabelten Frames **selbst trainiert**:

```bash
python gemeinsam/setup_labels.py     # Frames wählen + Best-Punkte vorbelegen
gemeinsam/label.bat                  # Laserlinie nachziehen (Pinsel, Zoom)
python gemeinsam/mittellinie.py      # dicke Labels -> 1px-Mittellinie (Ziel)
python methoden/unet/train.py        # -> unet.pt
python methoden/unet/apply_unet.py   # -> methoden/unet/data.json
python methoden/sam/sam_baseline.py  # SAM-Baseline (zero-shot, ohne Labels)
python gemeinsam/vergleich4.py       # -> gemeinsam/vergleich/viewer.html (alle Methoden)
```

## Methoden (Überblick)

| Methode | Ordner | Typ |
|---|---|---|
| Frame-Differencing | `methoden/frame_differencing` | intensitätsbasiert, klassisch |
| Canny | `methoden/canny` | klassische Kantendetektion |
| HED | `methoden/hed` | **KI-Kantendetektion** (vortrainiertes CNN) |
| U-Net | `methoden/unet` | **KI-Segmentierung**, selbst trainiert |
| SAM | `methoden/sam` | Foundation-Modell, zero-shot (Baseline) |

Gemeinsames Messgerüst: Referenzpunkte der Laserlinie, Bogenlänge *s* und
Außennormalen; pro Frame wird entlang derselben Normalen gemessen, Eisdicke =
Versatz gegenüber Frame 0. Kalibrierung px→mm über die 1:1-CAD-Geometrie der Slat.

## Voraussetzungen

Python 3 mit `numpy`, `opencv-python`, `scipy`, `scikit-image`, `matplotlib`,
`tifffile`, `torch` (U-Net), `ultralytics` (SAM). Daten/Gewichte werden nicht
versioniert (siehe `.gitignore`); `hed.caffemodel` liegt bei, `mobile_sam.pt`
lädt ultralytics beim ersten Lauf.
