# Live-Auswertung Messkampagne

**Dieser Ordner ist vollständig — einfach ganz kopieren (32 MB).** Modelle,
Netzdefinitionen und alle Werkzeuge liegen darin; es wird nichts aus dem
übrigen Projekt gebraucht. Getestet an einer Kopie außerhalb des Repos.

| Programm | Start | Ergebnis |
|---|---|---|
| Laser-PC | `start_laser.bat` | Eisdicke in mm über die Bogenlänge |
| Flächen-PC | `start_flaeche.bat` | Bedeckungsgrad in % und Eisfläche in mm² |
| Messbereich setzen | `start_roi.bat` | `messbereich.npz` — **vor Ort nötig** |
| Kalibrierung | `start_kalibrierung.bat` | Maßstabskarte je Ansicht |

Alle Einstellungen werden in der Oberfläche gesetzt und automatisch gesichert —
nach einem Neustart ist alles wieder da. Im Code muss nichts geändert werden.

### Was auf den Versuchs-PCs installiert sein muss

`python`, dazu `numpy`, `opencv-python`, `torch`, `pillow`. Ohne NVIDIA-GPU
läuft alles, nur langsamer — die Programme erkennen das selbst und zeigen es an.

---

## Vor dem Messtag: einmal durchtesten

```
python trockentest.py module     Abhängigkeiten, GPU, Modelle, Werkzeuge
python trockentest.py wache      Erkennung halbfertiger Dateien
```

Danach ein Durchlauf ohne Kamera. In einem Fenster:

```
set TESTFRAMES=D:\pfad\zu\alten\frames
python trockentest.py speisen C:\temp\testaufnahme 40 1.0
```

kopiert 40 vorhandene Frames im Sekundentakt in den Ordner — genau so, wie es
die Kamerasoftware später tut. Im anderen Fenster `start_flaeche.bat` starten,
diesen Ordner als Aufnahmeordner wählen und Start drücken.

**Wenn das läuft, läuft es am Messtag auch.**

---

## Der Messbereich muss vor Ort neu gesetzt werden

> Die Panelmaske der bisherigen Auswertung gehört zu der **damaligen
> Kameraposition**. Sobald die Kameras neu ausgerichtet sind, weist sie den
> falschen Bildbereich als Bezugsfläche aus. `start_roi.bat` erzeugt in zwei
> Minuten einen neuen: Rechteck aufziehen, optional Störbereiche ausschließen,
> speichern. Die Datei `messbereich.npz` wird dann in der Live-Oberfläche unter
> „Panelmaske" eingetragen.

Bewusst ohne SAM — kein Zusatzpaket, kein 40-MB-Modell, kein Fehlschlag im
ungünstigen Moment. Die physikalisch belastbare Größe ist ohnehin die Fläche in
mm² aus der Kalibrierung; der Prozentwert bezieht sich dann auf das Rechteck
und ist als solcher zu protokollieren.

---

## Einstellungen

### Beide Programme

| Feld | Bedeutung |
|---|---|
| Aufnahmeordner | Wohin die Kamerasoftware schreibt. Wird überwacht. |
| Ergebnisordner | Wohin Messwerte und Overlays geschrieben werden. |
| U-Net (.pt) | Trainiertes Modell. |
| Kalibrierung (.npz) | Aus `kalibrierung_schachbrett.py`. **Ohne diese Datei läuft alles weiter — aber nur in Pixel und Prozent, nicht in mm.** Die Statuszeile sagt es an. |
| jedes N-te Bild | Bei knapper Rechenzeit hochsetzen. |
| Rechenwerk | `auto` nimmt die GPU, wenn vorhanden. `cpu` erzwingt CPU. |
| Overlays mitspeichern | Ein JPEG je ausgewertetem Bild. Zum Abschalten, wenn die Platte knapp wird. |

### Nur Fläche

| Feld | Bedeutung |
|---|---|
| Panelmaske (.npz) | Legt Zuschnitt und Bezugsfläche fest (SAM-Maske). Bestimmt, worauf sich die Prozentangabe bezieht. |
| Schwelle | 0…1 auf die Eis-Wahrscheinlichkeit. Höher = strenger. |
| Referenzframes | Anzahl eisfreier Bilder am Anfang, aus denen der saubere Ausgangszustand bestimmt wird. **Wird beim 2-Kanal-Modell zwingend gebraucht** — vorher wertet das Programm nichts aus. |

> **Das Flächenmodell hat zwei Eingangskanäle:** das Bild und die Abweichung
> vom eisfreien Ausgangszustand. Der zweite Kanal beantwortet die Frage, die
> das Bild allein nicht beantworten kann — *ist diese Struktur neu?* Ohne ihn
> meldete das Netz auf dem nachweislich eisfreien Startbild im unteren
> Panelband 22 % Eis, weil dort ein Rückstandsband liegt, das wie feines
> Eisgefüge aussieht.
>
> Die Kanalzahl wird aus der Modelldatei gelesen. Ein 1-Kanal-Modell läuft
> weiterhin ohne Referenz.

### Nur Laser

| Feld | Bedeutung |
|---|---|
| Geometrie (.npz) | **Leer lassen.** Dann wird die Geometrie aus den eisfreien Referenzframes des laufenden Versuchs abgeleitet — am Messtag ist der Aufbau neu, eine mitgebrachte Geometrie wäre ungültig. |
| Referenzframes | Anzahl eisfreier Bilder am Anfang, aus denen Nulllage und Geometrie entstehen. |
| Suchbereich von/bis | Wie weit entlang der Normalen nach der Linie gesucht wird (px). |
| Glättung | Medianfilter über die Bogenlänge (px). |
| min. Linienfläche | Kleinere Fragmente gelten als Rauschen (px). |
| px/mm ersatzweise | Wird **nur** benutzt, wenn keine Kalibrierdatei geladen ist. |

---

## Ablauf je Versuchslauf

1. Aufnahmeordner in der Oberfläche prüfen, **Start** drücken
2. Kamera starten, Kanal auf Zielbedingung
3. **Beide PCs:** Die Statuszeile zählt „Referenz 1/10 … — noch NICHT sprühen".
   Erst wenn dort **„Referenz steht"** erscheint, sprühen.
   - Laser-PC: vorher gesprüht heißt, Eis landet in der Nulllage und wird nie
     als Eis gemessen.
   - Flächen-PC: vorher gesprüht heißt, das Eis wird Teil des „sauberen"
     Referenzzustands und fehlt danach in jeder Messung.
4. Sprühen. Kurve und Overlay laufen mit.
5. Nach dem Lauf **Stopp** drücken.

Bei einem neuen Lauf an **beiden** PCs die Referenz neu setzen (Laser: Knopf
„Referenz neu setzen"; Fläche: Stopp und erneut Start) — sonst wird gegen den
Zustand des vorherigen Laufs gerechnet.

> **Bekannte Einschränkung nach der Enteisung.** Die Referenz beschreibt die
> **trockene** Oberfläche vom Anfang des Laufs. Ist die Oberfläche am Ende
> sauber, aber **nass**, weicht sie davon ab und die Anzeige liegt zu hoch —
> auf unserer Testserie 6,5 % auf einem nachweislich eisfreien Bild. Für die
> Live-Überwachung ist das unkritisch, für die Endauswertung nicht: Dort wird
> die Referenz aus eisfreien Frames von **Anfang und Ende** gebildet, was den
> Fehler auf 0,0 % an beiden Enden bringt. Deshalb Punkt 15 des
> Versuchsablaufs — nach der Enteisung noch 10 s weiter aufnehmen.

---

## Ausgabe

Im Ergebnisordner:

- `messwerte.json` — fortlaufend geschrieben, überlebt einen Absturz
- `overlay_00001.jpg …` — je ausgewertetes Bild
- `geometrie.npz` (nur Laser) — die abgeleitete Geometrie, für die Offline-Auswertung

Die **Rohbilder bleiben unangetastet.** Die Live-Auswertung liest nur; sie
verschiebt, löscht und überschreibt nichts.

---

## Warum nur ein Verfahren live

Live läuft ausschließlich das U-Net, kein Methodenvergleich. Die Live-Ansicht
beantwortet am Kanal genau eine Frage: **läuft der Versuch sauber?** Mehrere
Verfahren kosten Rechenzeit und Bedienaufwand, ohne dass davon eine
Entscheidung abhängt.

Der Methodenvergleich gehört in die Offline-Auswertung, wo beliebig oft neu
gerechnet werden kann — und da die Rohbilder vollständig erhalten bleiben, geht
dadurch nichts verloren.

---

## Wenn etwas nicht funktioniert

| Symptom | Ursache und Abhilfe |
|---|---|
| „Aufnahmeordner existiert nicht" | Pfad falsch oder Netzlaufwerk nicht verbunden |
| Keine Bilder trotz laufender Kamera | Anderes Dateiformat? Unterstützt sind tif, tiff, png, jpg, bmp |
| „nicht lesbar: …" | Datei war noch im Schreiben; wird automatisch wiederholt. Häufung deutet auf ein langsames Netzlaufwerk — lokal aufnehmen |
| Anzeige hängt hinterher | „jedes N-te Bild" hochsetzen; Statuszeile zeigt ms/Bild |
| „keine Kalibrierung" in der Statuszeile | Kalibrierdatei nicht gewählt oder unlesbar → Ausgabe nur in Pixel/Prozent |
| Laser: „zu wenige Linienpunkte" | Schwelle zu hoch, Laser nicht im Bild, oder falsches Modell |
| Fläche: alles oder nichts erkannt | Falsche Panelmaske, oder Beleuchtung weicht stark vom Training ab → Nachtraining nötig |

---

## Generalprobe: der ganze Messtag in einem Durchlauf

```
python generalprobe.py            beide PCs
python generalprobe.py flaeche    nur Flächen-PC
python generalprobe.py laser      nur Laser-PC
```

Durchläuft die komplette Kette in der Reihenfolge des Versuchsplans —
Schachbrettvorlage, Kalibrierbilder, Kalibrierpipeline, Messbereich,
simulierte Kamera, Live-Auswertung — und benutzt dabei den **echten Code** der
Live-Programme, keine Nachbauten. Zwölf Prüfungen, jede mit klarem OK oder
FEHLER.

Zwei Prüfungen heißen „Messung spricht an". Sie sind der eigentliche Punkt:
Ein Test, der nur zeigt, dass etwas *läuft*, würde auch bestehen, wenn die
Auswertung durchgehend null meldet. Deshalb laufen die Testframes über die
ganze Serie statt nur über den Anfang, und es wird geprüft, dass Bedeckungsgrad
und Laserversatz tatsächlich ansteigen.
