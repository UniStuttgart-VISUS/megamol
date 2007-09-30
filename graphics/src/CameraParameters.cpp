/*
 * CameraParameters.cpp
 *
 * Copyright (C) 2006 - 2007 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */


#include "vislib/CameraParameters.h"
#include "vislib/mathtypes.h"


/*
 * vislib::graphics::CameraParameters::CameraParameters
 */
vislib::graphics::CameraParameters::CameraParameters(void) {
}


/* 
 * vislib::graphics::CameraParameters::CameraParameters 
 */
vislib::graphics::CameraParameters::CameraParameters(
        const vislib::graphics::CameraParameters& rhs) {
    *this = rhs;
}


/*
 * vislib::graphics::CameraParameters::~CameraParameters
 */
vislib::graphics::CameraParameters::~CameraParameters(void) {
}


/*
 * vislib::graphics::CameraParameters::operator=
 */
vislib::graphics::CameraParameters& 
vislib::graphics::CameraParameters::operator=(
        const vislib::graphics::CameraParameters& rhs) {
    // Intentionally empty
    return *this;
}


/*
 * vislib::graphics::CameraParameters::operator==
 */
bool vislib::graphics::CameraParameters::operator==(
        const vislib::graphics::CameraParameters& rhs) const {
    return (this == &rhs);
}
