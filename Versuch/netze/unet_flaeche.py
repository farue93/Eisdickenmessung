# -*- coding: utf-8 -*-
"""
unet_modell.py - U-Net-Architektur, Augmentierung und gekachelte Inferenz.

Die Augmentierung ist hier der eigentliche Trick gegen das Dunkelproblem:
statt Helligkeitsunabhaengigkeit von Hand zu konstruieren (Flat-Field,
Variationskoeffizient - beides half nur begrenzt), zeigen wir dem Netz
dieselbe Struktur in vielen Helligkeiten und Kontrasten. Es muss die
Invarianz dann selbst lernen, weil Helligkeit schlicht nicht mehr mit dem
Label korreliert.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def block(ein, aus):
    return nn.Sequential(
        nn.Conv2d(ein, aus, 3, padding=1, bias=False), nn.BatchNorm2d(aus), nn.ReLU(inplace=True),
        nn.Conv2d(aus, aus, 3, padding=1, bias=False), nn.BatchNorm2d(aus), nn.ReLU(inplace=True),
    )


class UNet(nn.Module):
    """4 Ebenen, Basisbreite 32. Das Sichtfeld reicht dadurch ueber ~140 px -
    genug, um die ANORDNUNG des Eisgefueges zu sehen statt nur die lokale
    Strukturmenge wie unsere bisherigen 15-61px-Fenstermerkmale.

    kanaele=1: nur das CLAHE-Bild.
    kanaele=2: zusaetzlich die Abweichung vom eisfreien Referenzzustand.
      Grund: Am unteren Panelrand liegt ein Rueckstandsband, das im Einzelbild
      wie feines Eisgefuege aussieht - auf dem nachweislich eisfreien Frame 0
      meldet das 1-Kanal-Netz dort 22% Eis, auf Frame 789 an derselben Stelle
      nur 0,11%. Das Aussehen allein reicht also nicht; erst der Vergleich mit
      dem sauberen Ausgangszustand trennt 'strukturiert' von 'neu hinzugekommen'."""

    def __init__(self, basis=32, kanaele=1):
        super().__init__()
        b = basis
        self.kanaele = kanaele
        self.e1, self.e2, self.e3, self.e4 = block(kanaele, b), block(b, b*2), block(b*2, b*4), block(b*4, b*8)
        self.mitte = block(b*8, b*16)
        self.u4 = nn.ConvTranspose2d(b*16, b*8, 2, 2); self.d4 = block(b*16, b*8)
        self.u3 = nn.ConvTranspose2d(b*8, b*4, 2, 2);  self.d3 = block(b*8, b*4)
        self.u2 = nn.ConvTranspose2d(b*4, b*2, 2, 2);  self.d2 = block(b*4, b*2)
        self.u1 = nn.ConvTranspose2d(b*2, b, 2, 2);    self.d1 = block(b*2, b)
        self.aus = nn.Conv2d(b, 1, 1)

    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(F.max_pool2d(e1, 2))
        e3 = self.e3(F.max_pool2d(e2, 2))
        e4 = self.e4(F.max_pool2d(e3, 2))
        m = self.mitte(F.max_pool2d(e4, 2))
        d = self.d4(torch.cat([self.u4(m), e4], 1))
        d = self.d3(torch.cat([self.u3(d), e3], 1))
        d = self.d2(torch.cat([self.u2(d), e2], 1))
        d = self.d1(torch.cat([self.u1(d), e1], 1))
        return self.aus(d)


def augmentieren(bild, ziel, gewicht, rng, abweichung=None):
    """bild/ziel/gewicht: 2D float32. abweichung optional als 2. Kanal.
    Gibt (eingang, ziel, gewicht) zurueck; eingang ist (K, H, W)."""
    def spiegeln(*felder):
        return [f[:, ::-1] for f in felder]

    if rng.random() < 0.5:
        bild, ziel, gewicht = spiegeln(bild, ziel, gewicht)
        if abweichung is not None:
            abweichung = abweichung[:, ::-1]
    if rng.random() < 0.5:
        bild, ziel, gewicht = bild[::-1], ziel[::-1], gewicht[::-1]
        if abweichung is not None:
            abweichung = abweichung[::-1]

    # Helligkeit/Kontrast/Gamma kraeftig variieren -> Helligkeit traegt keine
    # Information mehr ueber das Label, das Netz muss auf Struktur ausweichen.
    bild = bild.copy()
    bild = np.clip(bild * rng.uniform(0.5, 1.6) + rng.uniform(-0.25, 0.25), 0, 1)
    bild = np.clip(bild ** rng.uniform(0.6, 1.7), 0, 1)
    if rng.random() < 0.5:
        bild = np.clip(bild + rng.normal(0, rng.uniform(0.005, 0.04), bild.shape), 0, 1)

    kanaele = [bild]
    if abweichung is not None:
        # BEWUSST andere Augmentierung als beim Bild: Die Abweichung ist eine
        # physikalische Groesse ("wie stark weicht es vom sauberen Zustand ab").
        # Dieselbe Helligkeitsverschiebung wie beim Bild wuerde ihre Bedeutung
        # zerstoeren. Nur milde Skalierung und Rauschen, damit das Netz sich
        # nicht auf exakte Werte verlaesst.
        a = np.clip(abweichung.copy() * rng.uniform(0.85, 1.2)
                    + rng.normal(0, 0.02, abweichung.shape), 0, 1)
        kanaele.append(a)

    eingang = np.ascontiguousarray(np.stack(kanaele), np.float32)
    return (eingang,
            np.ascontiguousarray(ziel, np.float32),
            np.ascontiguousarray(gewicht, np.float32))


# VERWORFEN - Test-Time-Augmentation (Mittel ueber die vier Spiegelungen):
# gemessen am Endmodell brachte sie auf den eisfreien Frames nur 0,86->0,81%,
# 1,08->0,98% und 0,02->0,00%, auf Frame 438 gar nichts. Das liegt im Rauschen
# und ist zudem geschoent, weil diese Frames im Training waren. Bezahlt haette
# man es mit 4,6s statt 0,6s je Frame (Serie 61 statt 30 min). Bewusst nicht
# implementiert - Geschwindigkeit ist hier mehr wert.


@torch.no_grad()
def vorhersage_gekachelt(netz, bild, geraet, kachel=512, rand=64, batch=4):
    """Bild gekachelt durchs Netz, mit Ueberlappung. Der Rand jeder Kachel wird
    verworfen, damit an den Kachelgrenzen keine Naehte entstehen.
    bild: (H, W) oder (K, H, W) float32 0..1."""
    netz.eval()
    if bild.ndim == 2:
        bild = bild[None]
    K, H, W = bild.shape
    schritt = kachel - 2 * rand
    summe = np.zeros((H, W), np.float32)
    zaehler = np.zeros((H, W), np.float32)

    stapel, orte = [], []

    def leeren():
        if not stapel:
            return
        x = torch.from_numpy(np.stack(stapel)).to(geraet)      # (N, K, kachel, kachel)
        with torch.autocast(geraet.type, torch.float16, enabled=(geraet.type == "cuda")):
            p = torch.sigmoid(netz(x)).float().cpu().numpy()[:, 0]
        for (y0, x0), pk in zip(orte, p):
            h = min(kachel, H - y0); w = min(kachel, W - x0)
            summe[y0:y0+h, x0:x0+w] += pk[:h, :w]
            zaehler[y0:y0+h, x0:x0+w] += 1
        stapel.clear(); orte.clear()

    for y0 in range(0, H, schritt):
        for x0 in range(0, W, schritt):
            y0 = min(y0, max(0, H - kachel)); x0 = min(x0, max(0, W - kachel))
            k = np.zeros((K, kachel, kachel), np.float32)
            h = min(kachel, H - y0); w = min(kachel, W - x0)
            k[:, :h, :w] = bild[:, y0:y0+h, x0:x0+w]
            stapel.append(k); orte.append((y0, x0))
            if len(stapel) >= batch:
                leeren()
    leeren()
    return summe / np.maximum(zaehler, 1)
