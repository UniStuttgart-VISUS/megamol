/*
 * MMSPDFrameData.cpp
 *
 * Copyright (C) 2011 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "MMSPDFrameData.h"

using namespace megamol::core;


/****************************************************************************/

/*
 * moldyn::MMSPDFrameData::Particles::Particles
 */
moldyn::MMSPDFrameData::Particles::Particles(void) : count(0), data(), fieldMap(NULL) {
    // Intentionally empty
}


/*
 * moldyn::MMSPDFrameData::Particles::~Particles
 */
moldyn::MMSPDFrameData::Particles::~Particles(void) {
    delete[] this->fieldMap;
    this->fieldMap = NULL;
}


/*
 * moldyn::MMSPDFrameData::Particles::operator==
 */
bool moldyn::MMSPDFrameData::Particles::operator==(const moldyn::MMSPDFrameData::Particles& rhs) {
    return (this->count == rhs.count)
        && (&this->data == &rhs.data) // sufficient
        && (this->fieldMap == rhs.fieldMap); // sufficient
}


/****************************************************************************/

/*
 * moldyn::MMSPDFrameData::MMSPDFrameData
 */
moldyn::MMSPDFrameData::MMSPDFrameData(void) : data(), idxRec() {
    // Intentionally empty
}


/*
 * moldyn::MMSPDFrameData::~MMSPDFrameData
 */
moldyn::MMSPDFrameData::~MMSPDFrameData(void) {
    // Intentionally empty
}
