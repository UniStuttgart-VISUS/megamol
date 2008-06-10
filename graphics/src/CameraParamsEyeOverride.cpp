/*
 * CameraParamsEyeOverride.cpp
 *
 * Copyright (C) 2006 - 2007 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 * Copyright (C) 2007, Sebastian Grottel. All rights reserved.
 */

#include "vislib/CameraParamsEyeOverride.h"
#include "vislib/assert.h"


/*
 * vislib::graphics::CameraParamsEyeOverride::CameraParamsEyeOverride
 */
vislib::graphics::CameraParamsEyeOverride::CameraParamsEyeOverride(void)
        : CameraParamsOverride(), eye(LEFT_EYE) {
}


/*
 * vislib::graphics::CameraParamsEyeOverride::CameraParamsEyeOverride
 */
vislib::graphics::CameraParamsEyeOverride::CameraParamsEyeOverride(
        const vislib::SmartPtr<vislib::graphics::CameraParameters>& params)
        : CameraParamsOverride(params), eye(params->Eye()) {
    this->indicateValueChange();
}


/*
 *  vislib::graphics::CameraParamsEyeOverride::~CameraParamsEyeOverride
 */
vislib::graphics::CameraParamsEyeOverride::~CameraParamsEyeOverride(void) {
}


/*
 * vislib::graphics::CameraParamsEyeOverride::Eye
 */
vislib::graphics::CameraParameters::StereoEye 
vislib::graphics::CameraParamsEyeOverride::Eye(void) const {
    return this->eye;
}


/*
 * vislib::graphics::CameraParamsEyeOverride::SetEye
 */
void vislib::graphics::CameraParamsEyeOverride::SetEye(
        vislib::graphics::CameraParameters::StereoEye eye) {
    this->eye = eye; // no need to adjust anything else
    this->indicateValueChange();
}


/*
 *  vislib::graphics::CameraParamsEyeOverride::operator=
 */
vislib::graphics::CameraParamsEyeOverride& 
vislib::graphics::CameraParamsEyeOverride::operator=(
        const vislib::graphics::CameraParamsEyeOverride& rhs) {
    CameraParamsOverride::operator=(rhs);
    this->eye = rhs.eye;
    this->indicateValueChange();
    return *this;
}


/*
 *  vislib::graphics::CameraParamsEyeOverride::operator==
 */
bool vislib::graphics::CameraParamsEyeOverride::operator==(
        const vislib::graphics::CameraParamsEyeOverride& rhs) const {
    return (CameraParamsOverride::operator==(rhs)
        && (this->eye == rhs.eye));
}


/*
 *  vislib::graphics::CameraParamsEyeOverride::preBaseSet
 */
void vislib::graphics::CameraParamsEyeOverride::preBaseSet(
        const SmartPtr<CameraParameters>& params) {
    // intentionally empty
}


/*
 *  vislib::graphics::CameraParamsEyeOverride::resetOverride
 */
void vislib::graphics::CameraParamsEyeOverride::resetOverride(void) {
    ASSERT(!this->paramsBase().IsNull());
    this->eye = this->paramsBase()->Eye();
    this->indicateValueChange();
}
