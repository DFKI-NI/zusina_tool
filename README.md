# ZuSiNa Tool

Im Forschungsprojekt ZuSiNa (Besserer Zugang und Sichtbarkeit von Nachhaltigkeitsinformationen im Online-Handel durch KI) wurde ein Textklassifikationstool entwickelt, das Texte in Nachhaltigkeitsklassen einordnet. Es soll Verbraucher:innen dabei helfen sich anhand glaubwürdiger Informationen über die Nachhaltigkeit im Textilsektor zu bestimmten Marken zu informieren. Mehr Informationen dazu können im [zweiten Teil des Online-Guides](https://www.zusina-guide.de/glaubwuerdige-nachhaltigkeitsinformation/) des Projekts gelesen werden. Die Bilder zeigen die Applikation mit Beispieldaten.

![image](https://github.com/DFKI-NI/zusina_tool/assets/56087728/d1fbb43e-d9fd-440e-bacb-d48f58d1d3f4)

Auf der ersten Seite kann ein User optional eigene Textdateien im PDF-Format hochladen und sich danach dazu entscheiden, ob nur die eigenen Daten oder die eigenen Daten und die bestehenden Daten im Hintergrund für die Analyse genutzt werden sollen. Wenn keine eigenen Daten hochgeladen wurden, werden nur die Daten im Hintergrund genutzt.

![image](https://github.com/DFKI-NI/zusina_tool/assets/56087728/7735efef-806f-45d4-812d-d614b899e98b)

Die zweite Seite zeigt das Ergebnis der Analyse. Das Tortendiagramm zeigt die Verteilung der Textabschnitte über die Nachhaltigkeitsklassen. Ein User kann die Tortenstücke anklicken, um die Textabschnitte in dieser Klasse zu sehen. Die Textabschnitte stammen aus den Input Dateien, die in Textabschnitten von drei Sätzen aufgeteilt wurden. Danach wurden die Abschnitte mit dem trainierten Modell klassifiziert und nur angezeigt, wenn sie eine Wahrscheinlichkeitsmindestgrenze überschreiten. Zu jedem Textabschnitt wird auch die Quelle angezeigt als Hyperlink. Zusätzlich kann noch nach Stichworten gesucht werden, z.B. Firmennamen. Dann werden nur Textabschnitte angezeigt, die das Stichwort enthalten und in der Klasse sind. Die Stichworte werden dann im Text gelb markiert.

## Installation
Das GitHub Repository muss heruntergeladen werden und die Pakete in der `requirements.txt` Datei müssen installiert werden.

## Umsetzung
### Die Web-Applikation starten
Um die Applikation zu starten, muss dieses Command ausgeführt werden:

`python Application/app.py`

Dadurch wird eine Flask-Applikation gestartet. Die kann z.B. auch auf einem Server laufen und von mehreren Usern gleichzeitig genutzt werden. Die Applikation kümmert sich auch um das User-Management.

### Ein Modell Trainieren
1. Ein Datensatz mit aktuellen, glaubwürdigen Texten über die Nachhaltigkeit im Textilsektor erstellen. Die Datei `Model_training/Literaturliste.csv` enthält eine Literaturliste entsprechender Dateien vom Stand von 2023.
2. Das Preprocessing Skript mit dem Pfad zu den Dateien ausführen: `python Model_training/preprocessing.py`
3. Die Textabschnitte labeln (entweder mit den vorgegebenen Nachhaltigkeitsklassen in `Model_training/Nachhaltigkeitsklassen.csv` oder mit selbstdefinierten Klassen).
4. Das Modell mit den gelabelten Daten trainieren:  `python Model_training/train_model.py`
5. Die Pfade in der Applikation zum trainierten Modell und dem Datensatz verändern, sodass es in der Applikation genutzt wird.

## Lizenz
Das Projekt ist unter der BSD-3 Lizenz veröffentlicht.

## Wartung
Diese Software ist ein Produkt des abgeschlossenenen Forschungsprojekts ZuSiNa und wird nicht maintained. Wenn Sie diese Anwendung selber umsetzen möchten, stehen wir Ihnen gerne zur Verfügung (Daphne Theodorakopoulos, daphne.theodorakopoulos@dfki.de).


## Förderung
Die Dateien in diesem Repository wurden im Rahmen des Forschungsprojekts ZuSiNa, das vom Bundesministerium für Umwelt, Naturschutz, nukleare Sicherheit und Verbraucherschutz (BMUV) gefördert wird (FKZ: 67KI21009A) entwickelt. Weitere Informationen sind auf der [Projektseite](https://www.zusina-projekt.de/) und im [Online-Guide](https://www.zusina-guide.de/) zu finden. 






