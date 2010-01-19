/*
 * BezierDataCall.cpp
 *
 * Copyright (C) 2009 by Universitaet Stuttgart (VISUS).
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "BezierDataCall.h"

using namespace megamol::core;


/*
 * misc::BezierDataCall::BezierDataCall
 */
misc::BezierDataCall::BezierDataCall(void)
        : AbstractGetData3DCall(), count(0), curves(NULL) {
    // Intentionally empty
}


/*
 * misc::BezierDataCall::~BezierDataCall
 */
misc::BezierDataCall::~BezierDataCall(void) {
    this->Unlock();
    this->count = 0;
    this->curves = NULL;
}


/*
 * misc::BezierDataCall::SetData
 */
void misc::BezierDataCall::SetData(unsigned int count,
        const vislib::math::BezierCurve<BezierPoint, 3> *curves) {
    this->count = (curves == NULL) ? 0 : count;
    this->curves = curves;
}

/*
 * misc::BezierDataCall::operator=
 */
misc::BezierDataCall& misc::BezierDataCall::operator=(
        const misc::BezierDataCall& rhs) {
    AbstractGetData3DCall::operator =(rhs);
    this->count = rhs.count;
    this->curves = rhs.curves;
    return *this;
}
