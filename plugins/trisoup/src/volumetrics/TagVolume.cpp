/*
 * TagVolume.cpp
 *
 * Copyright (C) 2010 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */
#include "trisoup/volumetrics/TagVolume.h"
#include "stdafx.h"

using namespace megamol;
using namespace megamol::trisoup;
using namespace megamol::trisoup::volumetrics;

#include <memory.h>

TagVolume::TagVolume(unsigned int xRes, unsigned int yRes, unsigned int zRes) {
    this->xRes = xRes;
    this->yRes = yRes;
    this->zRes = zRes;
    this->volSize = (xRes * yRes * zRes + 7) / 8;
    this->volume = new unsigned char[volSize];
    this->Reset();
}

void TagVolume::Tag(unsigned int x, unsigned int y, unsigned int z) {
    unsigned int pos = z * this->xRes * this->yRes + y * xRes + x;
    unsigned int cellPos = pos / 8;
    unsigned char bitPos = 1 << (pos % 8);
    this->volume[cellPos] |= bitPos;
}

bool TagVolume::IsTagged(unsigned int x, unsigned int y, unsigned int z) {
    unsigned int pos = z * this->xRes * this->yRes + y * xRes + x;
    unsigned int cellPos = pos / 8;
    unsigned char bitPos = 1 << (pos % 8);
    return (this->volume[cellPos] & bitPos) > 0;
}

void TagVolume::UnTag(unsigned int x, unsigned int y, unsigned int z) {
    unsigned int pos = z * this->xRes * this->yRes + y * xRes + x;
    unsigned int cellPos = pos / 8;
    unsigned char bitPos = 1 << (pos % 8);
    this->volume[cellPos] &= ~bitPos;
}

void TagVolume::Reset() {
    memset(this->volume, 0, volSize);
}

TagVolume::~TagVolume(void) {
    delete[] volume;
}
