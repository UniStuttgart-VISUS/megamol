/*
 * LinesDataCall.cpp
 *
 * Copyright (C) 2010 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "LinesDataCall.h"

using namespace megamol::core;


/*
 * misc::LinesDataCall::Lines::Lines
 */
misc::LinesDataCall::Lines::Lines(void) : colArray(NULL), colHasAlpha(false),
        count(0), globCol(192, 192, 192, 255), idxArray(NULL),
        useFloatCol(false), vertArray(NULL) {
    // Intentionally empty
}


/*
 * misc::LinesDataCall::Lines::~Lines
 */
misc::LinesDataCall::Lines::~Lines(void) {
    this->colArray = NULL; // DO NOT DELETE
    this->count = 0; // Paranoia
    this->idxArray = NULL; // DO NOT DELETE
    this->vertArray = NULL; // DO NOT DELETE
}


/*
 * misc::LinesDataCall::LinesDataCall
 */
misc::LinesDataCall::LinesDataCall(void)
        : AbstractGetData3DCall(), count(0), lines(NULL) {
    // Intentionally empty
}


/*
 * misc::LinesDataCall::~LinesDataCall
 */
misc::LinesDataCall::~LinesDataCall(void) {
    this->Unlock();
    this->count = 0;
    this->lines = NULL;
}


/*
 * misc::LinesDataCall::SetData
 */
void misc::LinesDataCall::SetData(unsigned int count,
        const misc::LinesDataCall::Lines *lines) {
    this->count = (lines == NULL) ? 0 : count;
    this->lines = lines;
}


/*
 * misc::LinesDataCall::operator=
 */
misc::LinesDataCall& misc::LinesDataCall::operator=(
        const misc::LinesDataCall& rhs) {
    AbstractGetData3DCall::operator =(rhs);
    this->count = rhs.count;
    this->lines = rhs.lines;
    return *this;
}
