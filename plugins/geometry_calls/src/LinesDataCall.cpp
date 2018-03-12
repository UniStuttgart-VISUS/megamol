/*
 * LinesDataCall.cpp
 *
 * Copyright (C) 2010-2018 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "geometry_calls/LinesDataCall.h"

using namespace megamol;
using namespace megamol::core;

/****************************************************************************/


/*
 * LinesDataCall::Lines::Lines
 */
megamol::geocalls::LinesDataCall::Lines::Lines(void) : colDT(CDT_NONE), count(0),
globCol(192, 192, 192, 255), idxDT(DT_NONE), vrtDT(DT_NONE), id(0), colPtr{nullptr}, vertPtr{nullptr}, idxPtr{nullptr} {
    this->col.dataByte = NULL;
    this->idx.dataByte = NULL;
    this->vrt.dataFloat = NULL;
}

/*
* LinesDataCall::Lines::Lines
*/
megamol::geocalls::LinesDataCall::Lines::Lines(Lines const& rhs)
    : colDT{rhs.colDT}
    , count{rhs.count}
    , globCol{rhs.globCol}
    , idxDT{rhs.idxDT}
    , vrtDT{rhs.vrtDT}
    , id{rhs.id}
    , col{rhs.col}
    , idx{rhs.idx}
    , vrt{rhs.vrt}
    , colPtr{rhs.colPtr}
    , idxPtr{rhs.idxPtr}
    , vertPtr{rhs.vertPtr} {
    this->colorAccessor = rhs.colorAccessor->Clone();
    this->vertexAccessor = rhs.vertexAccessor->Clone();
    this->indexAccessor = rhs.indexAccessor->Clone();
}

/*
 * LinesDataCall::Lines::~Lines
 */
megamol::geocalls::LinesDataCall::Lines::~Lines(void) {
    this->count = 0; // Paranoia
    this->col.dataByte = NULL; // DO NOT DELETE
    this->idx.dataByte = NULL; // DO NOT DELETE
    this->vrt.dataFloat = NULL; // DO NOT DELETE
}

/****************************************************************************/


/*
 * LinesDataCall::LinesDataCall
 */
megamol::geocalls::LinesDataCall::LinesDataCall(void)
        : AbstractGetData3DCall(), count(0), lines(NULL), time(0.0f) {
    // Intentionally empty
}


/*
 * LinesDataCall::~LinesDataCall
 */
megamol::geocalls::LinesDataCall::~LinesDataCall(void) {
    this->Unlock();
    this->count = 0;
    this->lines = NULL;
}


/*
 * LinesDataCall::SetData
 */
void megamol::geocalls::LinesDataCall::SetData(unsigned int count,
        const megamol::geocalls::LinesDataCall::Lines *lines,
        const float time) {
    this->count = (lines == NULL) ? 0 : count;
    this->lines = lines;
    this->time = time;
}


/*
 * LinesDataCall::operator=
 */
megamol::geocalls::LinesDataCall& megamol::geocalls::LinesDataCall::operator=(
        const megamol::geocalls::LinesDataCall& rhs) {
    AbstractGetData3DCall::operator =(rhs);
    this->count = rhs.count;
    this->lines = rhs.lines;
    this->time = rhs.time;
    return *this;
}
