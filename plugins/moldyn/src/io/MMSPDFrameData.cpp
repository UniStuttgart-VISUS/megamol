/*
 * MMSPDFrameData.cpp
 *
 * Copyright (C) 2011 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#include "io/MMSPDFrameData.h"

using namespace megamol::moldyn::io;


/****************************************************************************/

/*
 * MMSPDFrameData::Particles::Particles
 */
MMSPDFrameData::Particles::Particles() : count(0), data(), fieldMap(NULL) {
    // Intentionally empty
}


/*
 * MMSPDFrameData::Particles::~Particles
 */
MMSPDFrameData::Particles::~Particles() {
    delete[] this->fieldMap;
    this->fieldMap = NULL;
}


/*
 * MMSPDFrameData::Particles::operator==
 */
bool MMSPDFrameData::Particles::operator==(const MMSPDFrameData::Particles& rhs) {
    return (this->count == rhs.count) && (&this->data == &rhs.data) // sufficient
           && (this->fieldMap == rhs.fieldMap);                     // sufficient
}


/****************************************************************************/

/*
 * MMSPDFrameData::MMSPDFrameData
 */
MMSPDFrameData::MMSPDFrameData() : data(), idxRec() {
    // Intentionally empty
}


/*
 * MMSPDFrameData::~MMSPDFrameData
 */
MMSPDFrameData::~MMSPDFrameData() {
    // Intentionally empty
}
