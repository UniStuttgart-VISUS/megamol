/*
 * CallSpheres.cpp
 *
 * Copyright (C) 2016 by Karsten Schatz
 * Copyright (C) 2016 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */
#include "CallPNGPics.h"

using namespace megamol;
using namespace megamol::molsurfmapcluster;

/*
 * CallPNGPics::CallForGetData
 */
const unsigned int CallPNGPics::CallForGetData = 0;

/*
 * CallPNGPics::CallForGetExtent
 */
const unsigned int CallPNGPics::CallForGetExtent = 1;

/*
 * CallPNGPics::CallPNGPics
 */
CallPNGPics::CallPNGPics(void) : numberofpictures(0), pngpictures(nullptr) {}

/*
 * CallPNGPics::~CallPNGPics
 */
CallPNGPics::~CallPNGPics(void) {
    numberofpictures = 0;
    pngpictures = nullptr;
}

/*
 * CallPNGPics::Count
 */
SIZE_T CallPNGPics::Count(void) const {
    return this->numberofpictures;
}

/*
 * CallPNGPics::getPNGPictures
 */
PNGPicLoader::PNGPIC* CallPNGPics::getPNGPictures(void) const {
    return this->pngpictures;
}

/*
 * CallPNGPics::SetData
 */
void CallPNGPics::SetData(SIZE_T countofpictures, PNGPicLoader::PNGPIC* pictures) {
    this->numberofpictures = countofpictures;
    this->pngpictures = pictures;
}

/*
 * CallPNGPics::operator=
 */
CallPNGPics& CallPNGPics::operator=(const CallPNGPics& rhs) {
    AbstractGetData3DCall::operator=(rhs);
    this->numberofpictures = rhs.numberofpictures;
    this->pngpictures = rhs.pngpictures;
    return *this;
}
