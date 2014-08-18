/*
 * CameraParamsProjectionOverride.cpp
 *
 * Copyright (C) 2006 - 2012 by Visualisierungsinstitut Universitaet Stuttgart.
 * Alle Rechte vorbehalten.
 */

#include "vislib/CameraParamsProjectionOverride.h"

#include "vislib/StackTrace.h"


/*
 * ...::CameraParamsProjectionOverride::CameraParamsProjectionOverride
 */
vislib::graphics::CameraParamsProjectionOverride
        ::CameraParamsProjectionOverride(void) 
        : Super(), projection(STEREO_PARALLEL) {
    VLSTACKTRACE(
        "CameraParamsProjectionOverride::CameraParamsProjectionOverride", 
        __FILE__, __LINE__);
}


/*
 * ...::CameraParamsProjectionOverride::CameraParamsProjectionOverride
 */
vislib::graphics::CameraParamsProjectionOverride
        ::CameraParamsProjectionOverride(
        const SmartPtr<CameraParameters>& params)
        : Super(params), projection(params->Projection()) {
    VLSTACKTRACE(
        "CameraParamsProjectionOverride::CameraParamsProjectionOverride", 
        __FILE__, __LINE__);
}


/*
 * ....::CameraParamsProjectionOverride::~CameraParamsProjectionOverride
 */
vislib::graphics::CameraParamsProjectionOverride
        ::~CameraParamsProjectionOverride(void) {
    VLSTACKTRACE(
        "CameraParamsProjectionOverride::~CameraParamsProjectionOverride", 
        __FILE__, __LINE__);
}


/*
 * vislib::graphics::CameraParamsProjectionOverride::Projection
 */
vislib::graphics::CameraParameters::ProjectionType 
vislib::graphics::CameraParamsProjectionOverride::Projection(void) const {
    VLSTACKTRACE("CameraParamsProjectionOverride::Projection", 
        __FILE__, __LINE__);
    return this->projection;
}


/*
 * vislib::graphics::CameraParamsProjectionOverride::SetProjection
 */
void vislib::graphics::CameraParamsProjectionOverride::SetProjection(
        ProjectionType projectionType) {
    VLSTACKTRACE("CameraParamsProjectionOverride::SetProjection", 
        __FILE__, __LINE__);
    this->projection = projection;
    this->indicateValueChange();
}


/*
 *  vislib::graphics::CameraParamsProjectionOverride::operator=
 */
vislib::graphics::CameraParamsProjectionOverride& 
vislib::graphics::CameraParamsProjectionOverride::operator=(
        const CameraParamsProjectionOverride& rhs) {
    VLSTACKTRACE("CameraParamsProjectionOverride::operator =", 
        __FILE__, __LINE__);
    Super::operator=(rhs);
    this->projection = rhs.projection;
    this->indicateValueChange();
    return *this;
}


/*
 *  vislib::graphics::CameraParamsProjectionOverride::operator ==
 */
bool vislib::graphics::CameraParamsProjectionOverride::operator ==(
        const CameraParamsProjectionOverride& rhs) const {
    VLSTACKTRACE("CameraParamsProjectionOverride::operator ==", 
        __FILE__, __LINE__);
    // TODO: Does this generally make sense? It should never be true if the 
    // override is actually used as the parent class must return false in this
    // case.
    return (Super::operator==(rhs) && (this->projection == rhs.projection));
}


/*
 *  vislib::graphics::CameraParamsProjectionOverride::preBaseSet
 */
void vislib::graphics::CameraParamsProjectionOverride::preBaseSet(
        const SmartPtr<CameraParameters>& params) {
    VLSTACKTRACE("CameraParamsProjectionOverride::preBaseSet", 
        __FILE__, __LINE__);
    // intentionally empty
}


/*
 *  vislib::graphics::CameraParamsProjectionOverride::resetOverride
 */
void vislib::graphics::CameraParamsProjectionOverride::resetOverride(void) {
    VLSTACKTRACE("CameraParamsProjectionOverride::resetOverride", 
        __FILE__, __LINE__);
    ASSERT(!this->paramsBase().IsNull());
    this->projection = this->paramsBase()->Projection();
    this->indicateValueChange();
}