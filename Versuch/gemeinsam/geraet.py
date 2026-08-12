# -*- coding: utf-8 -*-
"""
geraet.py - GPU oder CPU erkennen und die Folgen sichtbar machen.

Die beiden Versuchs-PCs sind unterschiedlich ausgestattet. Statt das im Code
festzulegen, wird beim Start gemessen, wie lange ein Bild tatsaechlich
braucht - daraus ergibt sich, ob jedes Bild ausgewertet werden kann oder nur
jedes N-te. Die Oberflaeche zeigt das an, damit am Kanal niemand raet.
"""
import time
import numpy as np


def geraet_waehlen(bevorzugt="auto"):
    """-> (torch.device, Klartextbeschreibung)"""
    import torch
    if bevorzugt == "cpu" or not torch.cuda.is_available():
        grund = "erzwungen" if bevorzugt == "cpu" else "keine CUDA-GPU gefunden"
        return torch.device("cpu"), f"CPU ({grund})"
    name = torch.cuda.get_device_name(0)
    return torch.device("cuda"), f"GPU: {name}"


def tempo_messen(funktion, testbild, laeufe=3):
    """Mittlere Dauer je Bild in Sekunden. Der erste Lauf wird verworfen,
    weil CUDA dabei noch initialisiert und Speicher belegt."""
    funktion(testbild)
    t0 = time.time()
    for _ in range(laeufe):
        funktion(testbild)
    return (time.time() - t0) / laeufe


def empfehlung(sekunden_je_bild, aufnahme_fps):
    """Wie viele Bilder koennen mitgerechnet werden?
    -> (jedes_n_te, Klartext)"""
    if sekunden_je_bild <= 0:
        return 1, ""
    schaffbar = 1.0 / sekunden_je_bild
    if aufnahme_fps <= 0 or schaffbar >= aufnahme_fps:
        return 1, f"{sekunden_je_bild*1000:.0f} ms je Bild - jedes Bild wird ausgewertet"
    n = int(np.ceil(aufnahme_fps / schaffbar))
    return n, (f"{sekunden_je_bild*1000:.0f} ms je Bild, Aufnahme {aufnahme_fps:.0f} B/s "
               f"-> nur jedes {n}. Bild live (Rest wird nach dem Lauf gerechnet)")
