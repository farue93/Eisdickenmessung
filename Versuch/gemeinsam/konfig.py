# -*- coding: utf-8 -*-
"""
konfig.py - Einstellungen laden und speichern (JSON neben dem Skript).

Absicht: Am Messtag soll nichts im Code geaendert werden muessen. Alle
Einstellungen kommen aus der Oberflaeche und werden sofort gesichert, damit
sie nach einem Neustart noch da sind - auch nach einem Absturz.
"""
import json
import os


class Konfig:
    def __init__(self, pfad, standard):
        self.pfad = pfad
        self.werte = dict(standard)
        self.laden()

    def laden(self):
        if os.path.exists(self.pfad):
            try:
                gespeichert = json.load(open(self.pfad, encoding="utf-8"))
                # Nur bekannte Schluessel uebernehmen: so bleibt eine alte
                # Datei nach einer Programmerweiterung benutzbar, statt zu
                # ueberschreiben oder zu scheitern.
                for k, v in gespeichert.items():
                    if k in self.werte:
                        self.werte[k] = v
            except Exception as e:
                print(f"Einstellungen nicht lesbar ({e}) - Standardwerte werden benutzt")

    def speichern(self):
        try:
            os.makedirs(os.path.dirname(self.pfad) or ".", exist_ok=True)
            json.dump(self.werte, open(self.pfad, "w", encoding="utf-8"),
                      indent=1, ensure_ascii=False)
        except Exception as e:
            print(f"Einstellungen konnten nicht gespeichert werden: {e}")

    def __getitem__(self, k):
        return self.werte[k]

    def __setitem__(self, k, v):
        self.werte[k] = v
