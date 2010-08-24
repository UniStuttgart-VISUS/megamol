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
        : AbstractGetData3DCall(), cntEllip(0), cntRect(0),
        ellipCurves(NULL), rectCurves(NULL) {
    // Intentionally empty
}


/*
 * misc::ExtBezierDataCall::~ExtBezierDataCall
 */
misc::ExtBezierDataCall::~ExtBezierDataCall(void) {
    this->Unlock();
    this->cntEllip = 0;
    this->cntRect = 0;
    this->ellipCurves = NULL;
    this->rectCurves = NULL;
}


/*
 * misc::ExtBezierDataCall::SetData
 */
void misc::ExtBezierDataCall::SetData(unsigned int cntEllip, unsigned int cntRect,
                const vislib::math::BezierCurve<Point, 3> *ellipCurves,
                const vislib::math::BezierCurve<Point, 3> *rectCurves) {
    this->cntEllip = (ellipCurves == NULL) ? 0 : cntEllip;
    this->cntRect = (rectCurves == NULL) ? 0 : cntRect;
    this->ellipCurves = ellipCurves;
    this->rectCurves = rectCurves;
}

/*
 * misc::ExtBezierDataCall::operator=
 */
misc::ExtBezierDataCall& misc::ExtBezierDataCall::operator=(
        const misc::ExtBezierDataCall& rhs) {
    AbstractGetData3DCall::operator =(rhs);
    this->cntEllip = rhs.cntEllip;
    this->cntRect = rhs.cntRect;
    this->ellipCurves = rhs.ellipCurves;
    this->rectCurves = rhs.rectCurves;
    return *this;
}
