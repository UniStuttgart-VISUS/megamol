/*
 * CallBinaryVolumeData.cpp
 *
 * Copyright (C) 2010 by Sebastian Grottel
 * Copyright (C) 2010 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */
#include "trisoup/CallBinaryVolumeData.h"

using namespace megamol;
using namespace megamol::trisoup;


/*
 * CallBinaryVolumeData::CallBinaryVolumeData
 */
CallBinaryVolumeData::CallBinaryVolumeData(void)
        : core::AbstractGetData3DCall()
        , sizeX(0)
        , sizeY(0)
        , sizeZ(0)
        , voxSizeX(1.0f)
        , voxSizeY(1.0f)
        , voxSizeZ(1.0f)
        , volume(NULL) {
    // intentionally empty
}


/*
 * CallBinaryVolumeData::~CallBinaryVolumeData
 */
CallBinaryVolumeData::~CallBinaryVolumeData(void) {
    this->volume = NULL; // DO NOT DELETE
}


/*
 * CallBinaryVolumeData::GetSafeVoxel
 */
bool CallBinaryVolumeData::GetSafeVoxel(unsigned int x, unsigned int y, unsigned int z) const {
    return ((x >= this->sizeX) || (y >= this->sizeY) || (z >= this->sizeZ) || (this->volume == NULL))
               ? false
               : this->volume[x + (y + z * this->sizeY) * this->sizeX];
}
