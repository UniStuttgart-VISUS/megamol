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
MMSPDFrameData::Particles::Particles(void) : count(0), data(), fieldMap(NULL) {
    // Intentionally empty
}


/*
 * MMSPDFrameData::Particles::~Particles
 */
MMSPDFrameData::Particles::~Particles(void) {
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
MMSPDFrameData::MMSPDFrameData(void) : data(), idxRec() {
    // Intentionally empty
}


/*
 * MMSPDFrameData::~MMSPDFrameData
 */
MMSPDFrameData::~MMSPDFrameData(void) {
    // Intentionally empty
}
