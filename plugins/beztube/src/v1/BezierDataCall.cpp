/*
 * BezierDataCall.cpp
 *
 * Copyright (C) 2009 by Universitaet Stuttgart (VISUS).
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "v1/BezierDataCall.h"

using namespace megamol::beztube;


/*
 * v1::BezierDataCall::BezierDataCall
 */
v1::BezierDataCall::BezierDataCall(void)
        : core::AbstractGetData3DCall(), count(0), curves(NULL) {
    // Intentionally empty
}


/*
 * v1::BezierDataCall::~BezierDataCall
 */
v1::BezierDataCall::~BezierDataCall(void) {
    this->Unlock();
    this->count = 0;
    this->curves = NULL;
}


/*
 * v1::BezierDataCall::SetData
 */
void v1::BezierDataCall::SetData(unsigned int count,
        const vislib::math::BezierCurve<BezierPoint, 3> *curves) {
    this->count = (curves == NULL) ? 0 : count;
    this->curves = curves;
}

/*
 * v1::BezierDataCall::operator=
 */
v1::BezierDataCall& v1::BezierDataCall::operator=(
        const v1::BezierDataCall& rhs) {
    AbstractGetData3DCall::operator =(rhs);
    this->count = rhs.count;
    this->curves = rhs.curves;
    return *this;
}
