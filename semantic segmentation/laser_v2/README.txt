laser_v2 - Laserlinien-Pipeline (SAM-Baseline vs. eigenes U-Net)
================================================================
Push-tauglich: alle Skripte nutzen RELATIVE Pfade ueber pfade.py
(keine festen G:/-Pfade). Gehoert als  semantic segmentation/laser_v2/
in das Repo 'Eisdickenmessung'.

VORAUSSETZUNG
-------------
Die Basis-Pipeline muss einmal gelaufen sein (erzeugt Geometrie + gecroppte
Frames, die laser_v2 braucht):
  1. Bilderserie in  input/  ablegen
  2.  python run.py         (laser_pipeline -> serie_eis -> canny)
pfade.py findet danach automatisch:
  - Repo-Wurzel (Ordner mit 'frame differencing' + 'pre processing')
  - die Serie (frame differencing/serie_*/), Frame-0-Stamm, Laser-Geometrie
  - fd/cn-Ergebnis-JSONs fuer den Vergleich

STRUKTUR
--------
laser_v2/
|  pfade.py                   zentrale, relative Pfad-Aufloesung
|  setup_labels.py            5 Frames + Best-Punkt-Vorlabel
|  label_tool.py / label.bat  Label-Tool (Pinsel, blaue Helligkeits-Vorschau, Zoom)
|  frames/ masks/ split.json  Labeldaten (lokal, .gitignore)
|  vergleich4.py              4-Methoden-Vergleichs-Viewer
|  doku_sam_vergleich.tex     Overleaf-Zusammenfassung
|  .gitignore                 Daten/Gewichte/Ergebnisse ausschliessen
|
+- sam/    << NUR SAM >>       sam_baseline.py + mobile_sam.pt (.gitignore)
+- unet/   << NUR U-Net >>     modell/daten_lader/train/apply_unet

ABLAUF
------
1) Labeln:   label.bat                     (5 Frames)
2) SAM:      python sam/sam_baseline.py     (zero-shot, braucht keine Labels)
3) U-Net:    python unet/train.py           -> unet/unet.pt
             python unet/apply_unet.py      -> unet/data.json
4) Vergleich:python vergleich4.py           -> vergleich/viewer.html

PUSHEN
------
Nur Code wird versioniert (.gitignore schliesst frames/masks/*.pt/data.json/
Overlays aus). Eigene Handlabels bewusst mitnehmen:  git add -f masks/
