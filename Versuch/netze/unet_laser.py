# -*- coding: utf-8 -*-
"""
modell.py - U-Net fuer die Laserlinien-Segmentierung (identisch zu laserlinie_seg).
Eingang: 1 Kanal (Graustufe)   Ausgang: 1 Logit je Pixel.
Voll-faltend (jede durch 8 teilbare Groesse).
"""
import torch
import torch.nn as nn


def block(cin, cout):
    """Zwei 3x3-Faltungen, je BatchNorm + ReLU."""
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class UNet(nn.Module):
    def __init__(self, ch=(16, 32, 64, 128)):
        super().__init__()
        c1, c2, c3, cb = ch
        self.pool = nn.MaxPool2d(2)
        self.enc1 = block(1,  c1); self.enc2 = block(c1, c2); self.enc3 = block(c2, c3)
        self.bottleneck = block(c3, cb)
        self.up3 = nn.ConvTranspose2d(cb, c3, 2, stride=2); self.dec3 = block(2 * c3, c3)
        self.up2 = nn.ConvTranspose2d(c3, c2, 2, stride=2); self.dec2 = block(2 * c2, c2)
        self.up1 = nn.ConvTranspose2d(c2, c1, 2, stride=2); self.dec1 = block(2 * c1, c1)
        self.out = nn.Conv2d(c1, 1, 1)

    def forward(self, x):
        e1 = self.enc1(x); e2 = self.enc2(self.pool(e1)); e3 = self.enc3(self.pool(e2))
        b = self.bottleneck(self.pool(e3))
        d3 = self.dec3(torch.cat([self.up3(b),  e3], 1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], 1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], 1))
        return self.out(d1)
