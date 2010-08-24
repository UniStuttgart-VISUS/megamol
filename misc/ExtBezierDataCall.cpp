/*
 * ExtBezierDataCall.cpp
 *
 * Copyright (C) 2010 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "ExtBezierDataCall.h"

using namespace megamol::core;


/*
 * misc::ExtBezierDataCall::ExtBezierDataCall
 */
misc::ExtBezierDataCall::ExtBezierDataCall(void)
        : AbstractGetData3DCall(), count(0), curves(NULL) {
    // Intentionally empty
}


/*
 * misc::ExtBezierDataCall::~ExtBezierDataCall
 */
misc::ExtBezierDataCall::~ExtBezierDataCall(void) {
    this->Unlock();
    this->count = 0;
    this->curves = NULL;
}


/*
 * misc::ExtBezierDataCall::SetData
 */
void misc::ExtBezierDataCall::SetData(unsigned int count,
        const vislib::math::BezierCurve<Point, 3> *curves) {
    this->count = (curves == NULL) ? 0 : count;
    this->curves = curves;
}

/*
 * misc::ExtBezierDataCall::operator=
 */
misc::ExtBezierDataCall& misc::ExtBezierDataCall::operator=(
        const misc::ExtBezierDataCall& rhs) {
    AbstractGetData3DCall::operator =(rhs);
    this->count = rhs.count;
    this->curves = rhs.curves;
    return *this;
}
