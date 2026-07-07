# input – hier die Bilderserien ablegen

Lege je Messreihe **einen Unterordner** mit den `*.tif`-Frames ab, z. B.:

```
input/
    reihe_260402-174444/
        ..._image0000000.tif   ← Frame 0 (MUSS eisfrei sein: Referenz/Nulllinie)
        ..._image0000001.tif
        ...
    reihe_260407-152857/
        ...
```

Regeln:
- **Ein Unterordner = eine Serie.**
- Das **erste `*.tif`** (alphabetisch) ist **Frame 0** und muss **eisfrei** sein.

Dann im Repo-Wurzelordner:

```
python run.py
```

Die abgelegten Serien-Bilder werden nicht mit eingecheckt
(per `.gitignore` ausgeschlossen) – nur dieser Ordner bleibt erhalten.
