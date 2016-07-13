/*
 * LinesDataCall.cpp
 *
 * Copyright (C) 2010 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "TriSoup/LinesDataCall.h"

using namespace megamol;
using namespace megamol::core;

/****************************************************************************/


/*
 * trisoup::LinesDataCall::Lines::Lines
 */
trisoup::LinesDataCall::Lines::Lines(void) : colDT(CDT_NONE), count(0),
        globCol(192, 192, 192, 255), idxDT(DT_NONE), vrtDT(DT_NONE) {
    this->col.dataByte = NULL;
    this->idx.dataByte = NULL;
    this->vrt.dataFloat = NULL;
}


/*
 * trisoup::LinesDataCall::Lines::~Lines
 */
trisoup::LinesDataCall::Lines::~Lines(void) {
    this->count = 0; // Paranoia
    this->col.dataByte = NULL; // DO NOT DELETE
    this->idx.dataByte = NULL; // DO NOT DELETE
    this->vrt.dataFloat = NULL; // DO NOT DELETE
}

/****************************************************************************/


/*
 * trisoup::LinesDataCall::LinesDataCall
 */
trisoup::LinesDataCall::LinesDataCall(void)
        : AbstractGetData3DCall(), count(0), lines(NULL) {
    // Intentionally empty
}


/*
 * trisoup::LinesDataCall::~LinesDataCall
 */
trisoup::LinesDataCall::~LinesDataCall(void) {
    this->Unlock();
    this->count = 0;
    this->lines = NULL;
}


/*
 * trisoup::LinesDataCall::SetData
 */
void trisoup::LinesDataCall::SetData(unsigned int count,
        const trisoup::LinesDataCall::Lines *lines) {
    this->count = (lines == NULL) ? 0 : count;
    this->lines = lines;
}


/*
 * trisoup::LinesDataCall::operator=
 */
trisoup::LinesDataCall& trisoup::LinesDataCall::operator=(
        const trisoup::LinesDataCall& rhs) {
    AbstractGetData3DCall::operator =(rhs);
    this->count = rhs.count;
    this->lines = rhs.lines;
    return *this;
}
