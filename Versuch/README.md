# Messkampagne — Laser- und Flächenmessung

**Dieser Ordner ist vollständig — einfach ganz kopieren.** Modelle,
Netzdefinitionen und alle Werkzeuge liegen darin; es wird nichts aus dem
übrigen Projekt gebraucht. Getestet an einer Kopie außerhalb des Repos.

| Versuchs-PC | Start | Ergebnis |
|---|---|---|
| Laser-PC | **`start_laser.bat`** | Eisdicke in mm über die Bogenlänge |
| Flächen-PC | **`start_flaeche.bat`** | Bedeckungsgrad in % und Eisfläche in mm², Kamera **unten und oben** |

Doppelklick auf die `.bat` — sonst nichts. **Kalibrierung, Anleitung und
Referenz stecken in diesen beiden Programmen**; es gibt kein zweites Skript
mehr, das vorher laufen müsste.

### Was sonst noch im Ordner liegt

| Datei / Ordner | Wofür |
|---|---|
| `start_laser.bat` → `messung_laser.py` | **Messoberfläche Laser-PC** |
| `start_flaeche.bat` → `messung_flaeche.py` | **Messoberfläche Flächen-PC** |
| `start_trockentest.bat` → `trockentest.py` | **Vorabprüfung ohne Kamera** — am Versuchs-PC doppelklicken, das Fenster bleibt offen |
| `generalprobe.py` | ganzer Ablauf in einem Durchlauf, vor dem Messtag (Konsole) |
| `schachbrett_drucken.py` | Druckvorlagen neu erzeugen, z. B. mit anderer Feldgröße (Konsole) |
| `README.md` | dieser Text — wird im Reiter *Anleitung* angezeigt |
| `muster_A4_10mm_16x25.pdf`, `muster_A3_10mm_26x37.pdf` | Druckvorlagen |
| `modelle/` | `laser.pt`, `flaeche.pt` — die trainierten Netze |
| `netze/` | Netzdefinitionen, damit der Ordner ohne das übrige Projekt läuft |
| `gemeinsam/` | Bausteine beider Programme (Ordnerwache, Kalibrierung, Maßstab, Oberfläche) |
| `ergebnisse/` | entsteht beim ersten Lauf; je Messung ein Unterordner mit Zeitstempel |
| `einstellungen_*.json` | entsteht automatisch; alles, was in der Oberfläche eingestellt wurde |

### Was auf den Versuchs-PCs installiert sein muss

`python`, dazu `numpy`, `opencv-python`, `torch`, `pillow`. Ohne NVIDIA-GPU
läuft alles, nur langsamer — die Programme erkennen das selbst und zeigen es an.

---

## Die Reiter

**Laser-PC:** `Kalibrierung` · `Messung` · `Anleitung`
**Flächen-PC:** je Kamera `Kalibrierung` · `Messbereich` · `Messung`, dazu `Anleitung`

Der Reiter **Anleitung** zeigt genau diesen Text — mit Inhaltsverzeichnis und
Suche, direkt im Programm. Am Kanal muss also nichts nachgeschlagen werden.

Die Kalibrierung ist ein **eigener Reiter und gehört vor die Messung** — dann
steht das Brett noch, es ist Zeit, und ein misslungener Versuch kostet nichts.
Der Messreiter braucht danach nur noch zu wissen, *welche* Kalibrierung gilt.

Links stehen die Einstellungen in nummerierten Abschnitten, rechts laufen Bild,
Kurve und **Zustandsanzeige**. Die Zustandsanzeige sagt für jeden Schritt, ob er
erledigt ist. Solange dort ein roter Punkt steht, fehlt etwas.

| Reihenfolge | Wo | Was |
|---|---|---|
| 1 | Kalibrierreiter | Schachbrettbilder laden (mehrere Positionen!) → läuft automatisch |
| 2 | Kalibrierreiter | prüfen, ggf. von Hand nachhelfen, **Übernehmen** |
| 3 | Messbereichsreiter | Bild wählen → Vorschlag → anpassen → **Übernehmen** (nur Fläche) |
| 4 | Messreiter | **Kamera verbinden** |
| 5 | Messreiter | **Eisfrei-Referenz aufnehmen** — **vor** dem Sprühen |
| 6 | Messreiter | **Messung starten**, dann sprühen |

Alle Einstellungen werden gesichert — nach einem Neustart, auch nach einem
Absturz, ist alles wieder da. Im Code muss nichts geändert werden.

---

## Aufnahmeordner und Unterordner

Gewählt wird der **Aufnahmeordner der Kamera**, nicht der Unterordner. Die
Kamerasoftware legt je Lauf einen Unterordner darin an; die Auswertung findet
den zuletzt beschriebenen von selbst und wechselt mit, sobald ein neuer Lauf
beginnt. Welcher gerade gelesen wird, steht in der Zustandszeile „Kamera".

Wer einen bestimmten Unterordner festhalten will, wählt ihn im Feld
*Unterordner* aus (↻ lädt die Liste neu).

### Bildauswahl

| Einstellung | Wirkung |
|---|---|
| **immer das neueste (Frames auslassen)** | Live-Betrieb. Rechnet die Auswertung langsamer als die Kamera aufnimmt, werden Bilder übersprungen. Die Anzeige zeigt dann immer den **aktuellen** Zustand statt hinterherzulaufen. Wie viele Bilder ausgelassen wurden, steht in der Statuszeile. |
| **jedes Bild** | lückenlos. Nur sinnvoll, wenn die Auswertung schneller ist als die Aufnahme — sonst wächst der Rückstand. |
| **jedes N-te Bild** | fester Takt, N daneben einstellbar. |

Für die Live-Überwachung am Kanal ist **immer das neueste** richtig. Die
vollständige Auswertung passiert ohnehin offline auf den Rohbildern, die
unangetastet bleiben.

---

## Der Kalibrierreiter

Ergebnis ist eine **Maßstabskarte**: für jeden Bildpunkt der lokale
Umrechnungsfaktor Pixel → Millimeter. Ein einzelner px/mm-Wert genügt nicht,
weil die Kamera schräg auf eine gekrümmte Fläche blickt.

### 1 · Schachbrettbilder laden — mehrere

Über **Bilder hinzufügen …** aus der Dateiauswahl (Mehrfachauswahl möglich) —
Bilder mit aufgeklebtem Schachbrett, ohne Eis. Die Kalibrierung läuft nach jeder
Änderung der Liste von selbst; einzelne Bilder lassen sich jederzeit wieder
entfernen.

> **Mehrere Brettpositionen sind der Normalfall, nicht die Ausnahme.**
> Ein einzelnes Brett stützt nur den Bildbereich, den es bedeckt — in der
> Generalprobe 5 % der Bildfläche. Überall sonst setzt die Karte den Verlauf
> nur fort. Dasselbe Brett nacheinander an mehrere Stellen kleben und jedes Mal
> aufnehmen; alle Stützstellen gehen in **einen** gemeinsamen Fit.
>
> Gemessen an der bekannten Testgeometrie: **ein** Brett → 0,87 % Fehler
> außerhalb des Bretts, **vier** Bretter → 0,16 %.

Einzutragen sind vorher:

* **Feldgröße in mm** — vorbelegt mit **10 mm**, dem Sollwert des
  mitgelieferten Bogens. Zu ersetzen durch den **nachgemessenen** Wert: 10
  Felder am Stück mit dem Messschieber messen, durch 10 teilen. Drucker
  skalieren fast immer ein wenig; bleibt der Sollwert stehen, ist die gesamte
  mm-Skala um denselben Faktor falsch, und zwar unauffällig.
* **innere Ecken** — vorbelegt mit **16 × 25**, den Werten des mitgelieferten
  A4-Bogens (A3: 26 × 37). Die Vorgabe kommt aus demselben Code, der die
  Druckvorlage erzeugt — Bogen und Eingabefeld können nicht auseinanderlaufen.
  Bei einem fremden Brett gilt: ein Brett mit 6×9 **Feldern** hat 5×8 innere
  **Ecken**. Der Knopf **Eckenzahl aus Bild bestimmen** nimmt einem das
  Nachzählen ab; misslingt die Kalibrierung, versucht das Programm es von
  selbst einmal und rechnet mit dem Fund weiter.
* **Polynomgrad** — auf `automatisch` lassen. Ab drei Brettern entscheidet eine
  **Kreuzprobe**: ein Brett weglassen, auf den übrigen fitten, das weggelassene
  vorhersagen. Das misst, wie gut die Karte dort stimmt, wo *nicht* gemessen
  wurde — was der Restfehler des Fits nicht kann, denn der wird mit steigendem
  Grad immer kleiner, auch wenn die Fortsetzung schlechter wird.

Das ausgewählte Bild wird mit **Overlay** angezeigt: die Eckengitter **aller**
Bilder, die Maßstabskarte als Farbe, und weiß umrandet der Bereich, in dem
tatsächlich gemessen wurde. Das Mausrad zoomt, die rechte Taste schiebt. Der
Haken *Overlay anzeigen* schaltet zwischen Overlay und Rohbild um.

### 2 · Suchbereich, wenn die Automatik danebengreift

Liegt außer dem Brett noch etwas Schachbrettartiges im Bild — eine
Gitterstruktur, eine Spiegelung des Bretts im nassen Panel —, kann der Detektor
das Falsche finden und **trotzdem Erfolg melden**. Mit **Suchbereich für dieses
Bild** wird ein Rechteck um das richtige Brett gelegt; danach *Automatisch
kalibrieren*.

Der Suchbereich gehört **je Bild** — das Brett liegt in jeder Aufnahme
woanders. In der Liste ist er mit `[Suchbereich]` markiert.

### 3 · Maßstab von Hand

Zwei Fälle: Es ist gar kein Brett im Bild, oder das automatische Ergebnis soll
unabhängig geprüft werden. Das Werkzeug ist je Strang ein anderes, weil die
Messgröße eine andere ist:

| | Werkzeug | Eingabe |
|---|---|---|
| **Laser** | Strecke ziehen | reale Länge in mm |
| **Fläche** | Rechteck ziehen | reale Breite × Höhe in mm |

Beim Laser wird eine **Länge** gemessen (Verschiebung quer zur Linie), bei der
Fläche eine **Fläche**. Ein Rechteck für den Laser wäre Scheingenauigkeit, eine
Strecke für die Fläche eine unbelegte Annahme über die zweite Richtung.

> **Die Handmessung ist vor allem eine Gegenprobe.** Liegt bereits eine
> automatische Kalibrierung vor, zeigt der Reiter, um wie viel Prozent beide
> auseinanderliegen. Mehr als 3 % heißt fast immer: die eingetragene Feldgröße
> stimmt nicht mit dem Ausdruck überein. Im Test schlägt eine um 2 mm falsch
> eingetragene Feldgröße mit +20 % durch — genau dafür ist die Gegenprobe da.
>
> Als *Maßstab* ist die Handmessung der schwächere Weg: sie liefert eine
> **ortsunabhängige** Skala und unterschlägt damit, dass ein Pixel am
> gekrümmten Rand mehr Fläche abdeckt als eines in der Bildmitte. Beim
> Schachbrett schwankt der Flächenmaßstab über das Bild typisch um
> Zehnerprozente. Deshalb im Zweifel das Brett.

### 4 · Übernehmen

Der Radioknopf wählt, was gilt — Karte aus den Bildern oder Handmessung.
**Für die Messung übernehmen** trägt die Datei im zugehörigen Messreiter ein.

> **Wo das Brett klebt, ist der Maßstab belegt — sonst nicht.**
> Außerhalb setzt die Karte den Verlauf nur fort. Das Kontrollbild zeigt den
> gestützten Bereich weiß umrandet; alles Blassere ist fortgesetzt. Deshalb das
> Brett **dorthin kleben, wo später gemessen wird** — und lieber ein paar
> Positionen mehr aufnehmen.

Ohne Kalibrierung läuft alles weiter, dann aber nur in Pixel und Prozent. Die
Zustandsanzeige sagt es deutlich an.

---

## Warum die Umrechnung richtungsabhängig ist

Die Karte enthält nicht nur die Flächenskala, sondern den lokalen **metrischen
Tensor** — drei Zahlen je Bildpunkt statt einer. Der Grund ist messtechnisch,
nicht mathematisch:

Blickt die Kamera schräg auf die Fläche, sind die beiden Hauptrichtungen
verschieden stark verkürzt. `sqrt(Flächenskala)` ist nur ihr geometrisches
Mittel. Eine **Fläche** ist damit richtig bestimmt, eine **Länge in einer
bestimmten Richtung** aber nicht — und die Eisdicke ist genau das: eine Länge
entlang der Normalen zur Laserlinie.

Gemessen an der Testgeometrie der Generalprobe, 5-mm-Strecken an 100 Stellen:

| Verfahren | mittlerer Fehler | schlimmster Fall |
|---|---:|---:|
| **metrischer Tensor** (jetzt) | **0,02 %** | **0,09 %** |
| isotrop, `sqrt(Fläche)` | 11,42 % | 17,63 % |
| ein globaler px/mm-Wert | 12,58 % | 39,40 % |

Der Laser rechnet deshalb **an jeder Stützstelle einzeln** um, entlang der
tatsächlichen Verschiebungsrichtung. Die Statuszeile weist die Spanne der
lokalen Maßstäbe aus (`Karte 3.12-4.48 px/mm`) oder `Ersatzwert`, wenn keine
Kalibrierung geladen ist.

Handmessungen (Strecke, Rechteck) liefern keine Metrik — dort wird isotrop
genähert. Auch das steht im Bericht.

---

## Der Messbereichsreiter (nur Fläche)

> Die Panelmaske der bisherigen Auswertung gehört zur **damaligen
> Kameraposition**. Sobald die Kameras neu ausgerichtet sind, weist sie den
> falschen Bildbereich als Bezugsfläche aus.

Derselbe Dreischritt wie bei der Kalibrierung:

**1 · Bild wählen** — über **Bild wählen …** ein **eisfreies** Bild dieser
Kamera aus der Dateiauswahl.

**2 · Vorschlag** — läuft sofort danach. Der Vorschlag ist die größte
zusammenhängende helle Fläche: das Panel hebt sich vom dunklen Kanalhintergrund
ab. Auf der Testreihe trifft er das von Hand geprüfte Ergebnis mit IoU 0,87 —
gleicher oberer Rand, etwas schmaler. Er ist ausdrücklich zum Anpassen gedacht,
kein Orakel.

**3 · Anpassen** — direkt im Bild:

| Ziehen | Wirkung |
|---|---|
| an einer **Ecke** | Größe ändern |
| **innerhalb** des Rechtecks | ganzes Rechteck verschieben |
| **außerhalb** | neues Rechteck aufziehen |
| Mausrad / rechte Taste | zoomen / schieben |

Mit dem **Pinsel** lassen sich Störstellen innerhalb des Rechtecks
ausschließen — Halterungen, Reflexe, Kanalwand — und mit *freigeben* wieder
zurücknehmen. „letzten Strich zurück" macht einen ganzen Pinselzug rückgängig.

**Für die Messung übernehmen** speichert und trägt die Datei im Messreiter ein.
Daneben entsteht ein Kontrollbild.

Bewusst ohne SAM — kein Zusatzpaket, kein 40-MB-Modell, kein Fehlschlag im
ungünstigen Moment. Die physikalisch belastbare Größe ist ohnehin die Fläche in
mm² aus der Kalibrierung; der Prozentwert bezieht sich auf dieses Rechteck und
ist als solcher zu protokollieren.

**Jede Kamera braucht ihren eigenen Messbereich.** Wird er geändert, verwirft
das Programm die Eisfrei-Referenz — sie gehörte zum alten Zuschnitt und wäre
sonst stillschweigend falsch.

### Welcher Zuschnitt wirklich gilt

Die Zustandszeile **Messbereich** zeigt ihn vollständig:

```
messbereich.npz | Zuschnitt 850x500 ab (150,100) | Bezug 425.000 px
```

Also: welche Datei, wie groß das Rechteck ist, wo es im Vollbild sitzt und wie
viele Pixel die Bezugsfläche der Prozentangabe hat. Ohne diese Angaben ist ein
Prozentwert später nicht mehr zu deuten. Dieselben Angaben stehen live im
Messbereichsreiter, während du das Rechteck ziehst.

**Rand abtragen (px)** schrumpft den Bereich ringsum, Standard **0** — gemessen
wird, was gezogen wurde. Sinnvoll nur, wenn die Panelkante im Bild unscharf ist
und mitgemessen würde.

> Bis jetzt waren hier fest **25 px** eingebaut (eine Erosion mit 51×51), aus
> der Zeit der SAM-Panelmaske mit ihrem ausgefransten Rand. Bei einem von Hand
> gezogenen Rechteck verschwanden dadurch 25 px ringsum, ohne dass es irgendwo
> stand: aus 425 000 px Bezugsfläche wurden 360 000 — **15 % weniger**, und
> jeder Bedeckungsgrad damit rund 18 % zu hoch.

### Laser: der Ausschnitt um die Linie

Auch der Laser schneidet zu — für die gespeicherten Bilder. Die Zustandszeile
**Bildausschnitt** zeigt ihn, sobald die Geometrie steht:

```
1455x1179 ab (289,254) | Rand 40 px um die Linie
```

Das Rechteck ergibt sich aus den Stützstellen der Laserlinie plus dem Feld
**Ausschnittsrand**. Es betrifft **nur** die Dateien in `crops/` — gemessen
wird immer auf dem vollen Bild.

---

## Eisfrei-Referenz — der eine Punkt ohne Reparatur

Beide Verfahren messen eine **Differenz** zu einem eisfreien Zustand. Wird
gesprüht, bevor die Referenz steht, geht der Lauf verloren, und zwar unbemerkt:

* **Laser:** Das Eis landet in der Nulllage und wird nie als Eis gemessen.
* **Fläche:** Das Eis wird Teil des „sauberen" Referenzzustands und fehlt danach
  in jeder Messung.

Beide Male sieht das Ergebnis plausibel aus und ist falsch. Deshalb: **erst
sprühen, wenn in der Zustandsanzeige „Eisfrei-Referenz — steht" erscheint.**

> **Das Flächenmodell hat zwei Eingangskanäle:** das Bild und die Abweichung vom
> eisfreien Ausgangszustand. Der zweite Kanal beantwortet die Frage, die das
> Bild allein nicht beantworten kann — *ist diese Struktur neu?* Ohne ihn
> meldete das Netz auf dem nachweislich eisfreien Startbild im unteren
> Panelband 22 % Eis, weil dort ein Rückstandsband liegt, das wie feines
> Eisgefüge aussieht. Ein 1-Kanal-Modell läuft weiterhin ohne Referenz; die
> Kanalzahl wird aus der Modelldatei gelesen, nicht eingestellt.

Bei einem **neuen Lauf** an beiden PCs die Referenz neu aufnehmen — sonst wird
gegen den Zustand des vorherigen Laufs gerechnet.

> **Bekannte Einschränkung nach der Enteisung.** Die Referenz beschreibt die
> **trockene** Oberfläche vom Anfang des Laufs. Ist die Oberfläche am Ende
> sauber, aber **nass**, weicht sie davon ab und die Anzeige liegt zu hoch — auf
> unserer Testserie 6,5 % auf einem nachweislich eisfreien Bild. Für die
> Live-Überwachung unkritisch, für die Endauswertung nicht: Dort wird die
> Referenz aus eisfreien Frames von **Anfang und Ende** gebildet, was den Fehler
> auf 0,0 % an beiden Enden bringt. Deshalb nach der Enteisung noch 10 s
> weiter aufnehmen.

---

## Einstellungen im Einzelnen

### Kalibrierreiter

| Feld | Bedeutung |
|---|---|
| Bildliste | Alle geladenen Schachbrettbilder. Änderbar; nach jeder Änderung wird neu gerechnet. |
| Feldgröße / innere Ecken | Vorbelegt aus dem mitgelieferten A4-Bogen (16 × 25, 10 mm). Feldgröße **nachmessen**. |
| Polynomgrad | `automatisch` lassen — ab drei Brettern entscheidet die Kreuzprobe. |
| reale Länge (Laser) | Länge der von Hand gezogenen Strecke in mm. |
| reale Größe (Fläche) | Breite × Höhe des von Hand gezogenen Rechtecks in mm. |

### Messreiter — beide Programme

| Feld | Bedeutung |
|---|---|
| Aufnahmeordner | Wurzelordner der Kamera. Unterordner werden selbst gefunden. |
| Ergebnisordner | Je Messung entsteht darin ein Unterordner mit Zeitstempel. |
| Kalibrierung (.npz) | Kommt aus dem Kalibrierreiter; kann auch von Hand gewählt werden. |
| Rand abtragen (nur Fläche) | Pixel, die ringsum vom Messbereich abgezogen werden. Standard 0. |
| Ausschnittsrand (nur Laser) | Rand um die Laserlinie für die gespeicherten Ausschnitte. Standard 40 px. |
| Referenzframes | Anzahl eisfreier Bilder für den Nullzustand. |
| Bildauswahl / N | siehe oben. |
| Overlays mitspeichern | Ein JPEG je ausgewertetem Bild. Zum Abschalten, wenn die Platte knapp wird. |
| **gecroppte Bilder mitspeichern** | Der Bildausschnitt, den die Messung benutzt hat, verlustfrei als PNG in `crops/`. Fläche: der Messbereich. Laser: der Bereich um die Linie. Für die Offline-Auswertung, die damit ohne erneutes Zuschneiden aufsetzen kann. |
| Schwelle | 0…1 auf die Wahrscheinlichkeit. Höher = strenger. Wirkt sofort, auch während die Messung läuft. |
| Rechenwerk | `auto` nimmt die GPU, wenn vorhanden. `cpu` erzwingt CPU. |
| U-Net (.pt) | Ist vorbelegt; nur ändern, wenn ein neues Modell mitgebracht wird. |

### Nur Laser

| Feld | Bedeutung |
|---|---|
| Suchbereich von/bis | Wie weit entlang der Normalen nach der Linie gesucht wird (px). Größer heißt auch: mehr ferne Reflexe im Suchfenster. |
| Glättung | Medianfilter über die Bogenlänge (px). |
| min. Linienfläche | Kleinere Fragmente gelten als Rauschen (px). |
| px/mm ersatzweise | Wird **nur** benutzt, wenn keine Kalibrierung geladen ist — dann ortsunabhängig und ohne Richtungskorrektur. Die Statuszeile weist es als `Ersatzwert` aus. |

Die **Geometrie** (Stützstellen und Normalen entlang der Linie) wird aus den
eisfreien Referenzframes des laufenden Versuchs abgeleitet und im Ergebnisordner
mitgeschrieben. Eine mitgebrachte Geometrie wäre ungültig, sobald Kamera oder
Laser neu ausgerichtet sind.

---

## Vor dem Messtag: einmal durchtesten

**`start_trockentest.bat`** doppelklicken — prüft Abhängigkeiten, GPU, Modelle,
Kalibrierkette und Ordnerwache und lässt das Fenster offen stehen. Einzeln geht
auch:

```
python trockentest.py module     Abhängigkeiten, GPU, Modelle, Kalibrierkette
python trockentest.py wache      halbfertige Dateien, Unterordner, Live-Modus
python generalprobe.py           der ganze Ablauf in einem Durchlauf
```

Danach ein Durchlauf ohne Kamera. In einem Fenster:

```
set TESTFRAMES=D:\pfad\zu\alten\frames
python trockentest.py speisen C:\temp\testaufnahme 40 1.0
```

kopiert 40 vorhandene Frames im Sekundentakt in einen **Unterordner** von
`C:\temp\testaufnahme` — genau so, wie es die Kamerasoftware tut. Im anderen
Fenster `start_flaeche.bat` starten, `C:\temp\testaufnahme` als Aufnahmeordner
wählen und die fünf Schritte durchgehen.

**Wenn das läuft, läuft es am Messtag auch.**

### Generalprobe

```
python generalprobe.py            beide PCs
python generalprobe.py flaeche    nur Flächen-PC
python generalprobe.py laser      nur Laser-PC
```

Durchläuft die komplette Kette in der Reihenfolge des Versuchsplans und benutzt
dabei den **echten Code** der Messprogramme, keine Nachbauten. Zwanzig
Prüfungen, jede mit klarem OK oder FEHLER.

Zwei davon heißen „Messung spricht an". Sie sind der eigentliche Punkt: Ein
Test, der nur zeigt, dass etwas *läuft*, würde auch bestehen, wenn die
Auswertung durchgehend null meldet. Deshalb laufen die Testframes über die ganze
Serie statt nur über den Anfang.

Eine weitere vergleicht den rekonstruierten Maßstab gegen die **bekannte**
Homographie des Testbildes — sie beantwortet, was die Ein-Bild-Kalibrierung
tatsächlich kostet, statt es zu behaupten.

---

## Ausgabe

Je Messung ein Unterordner `JJJJMMTT_HHMMSS_<Kameraordner>` im Ergebnisordner:

- `messwerte.json` — fortlaufend geschrieben, überlebt einen Absturz
- `overlay_00001.jpg …` — je ausgewertetes Bild
- `crops/crop_00001.png …` — falls eingeschaltet: der ausgewertete Bildausschnitt
- `geometrie.npz` (nur Laser) — die abgeleitete Geometrie für die Offline-Auswertung

Im Ergebnisordner selbst liegt zusätzlich `kalibrierung/` mit Maßstabsdatei und
Kontrollbild — die gilt für alle Läufe, bis neu kalibriert wird.

Die **Rohbilder bleiben unangetastet.** Die Auswertung liest nur; sie
verschiebt, löscht und überschreibt nichts im Aufnahmeordner.

---

## Warum nur ein Verfahren live

Live läuft ausschließlich das U-Net, kein Methodenvergleich. Die Live-Ansicht
beantwortet am Kanal genau eine Frage: **läuft der Versuch sauber?** Mehrere
Verfahren kosten Rechenzeit und Bedienaufwand, ohne dass davon eine Entscheidung
abhängt.

Der Methodenvergleich gehört in die Offline-Auswertung, wo beliebig oft neu
gerechnet werden kann — und da die Rohbilder vollständig erhalten bleiben, geht
dadurch nichts verloren.

---

## Wenn etwas nicht funktioniert

| Symptom | Ursache und Abhilfe |
|---|---|
| „Aufnahmeordner existiert nicht" | Pfad falsch oder Netzlaufwerk nicht verbunden |
| Keine Bilder trotz laufender Kamera | Anderes Dateiformat? Unterstützt sind tif, tiff, png, jpg, bmp. Sonst ↻ neben *Unterordner* drücken |
| Liest den falschen Unterordner | Im Feld *Unterordner* fest auswählen statt „automatisch" |
| „nicht lesbar: …" | Datei war noch im Schreiben; wird automatisch wiederholt. Häufung deutet auf ein langsames Netzlaufwerk — lokal aufnehmen |
| Viele „übersprungen" in der Statuszeile | Auswertung langsamer als die Aufnahme. Im Live-Modus unkritisch; sonst „jedes N-te Bild" wählen |
| „Kein Bild verwertbar" | **Eckenzahl aus Bild bestimmen** drücken. Sonst: Brett ganz im Bild? Reflexe? Suchbereich aufziehen oder Maßstab von Hand messen |
| „NICHT erkannt: …" nach dem Kalibrieren | Einzelne Bilder gingen nicht ein. Für diese Suchbereich setzen oder sie entfernen |
| Restfehler > 3 % | Brett verrutscht oder Feldgröße falsch eingetragen |
| Gegenprobe > 3 % | Feldgröße am Ausdruck nachmessen — das ist fast immer die Ursache |
| Erkennung sitzt auf dem falschen Muster | Suchbereich um das richtige Brett aufziehen |
| „wenig gestützt" | Weitere Brettpositionen aufnehmen, dorthin, wo gemessen wird |
| „andere Bildgroesse" | Bild stammt von einer anderen Kamera oder Auflösung — gehört nicht in diese Liste |
| Messung stoppt von selbst | Messbereich wurde geändert → Referenz ist ungültig, neu aufnehmen |
| Laser: „zu wenige Linienpunkte" | Schwelle zu hoch, Laser nicht im Bild, oder falsches Modell |
| Fläche: alles oder nichts erkannt | Falscher Messbereich, oder Beleuchtung weicht stark vom Training ab → Nachtraining nötig |
| Vorschlag sitzt daneben | Ecken ziehen oder außerhalb ein neues Rechteck aufziehen — der Vorschlag ist nur der erste Wurf |
| Rechteck lässt sich nicht mehr anfassen | Nach dem Pinseln erst **Rechteck bearbeiten** drücken |
