/*
 * CameraParamsProjectionOverride.cpp
 *
 * Copyright (C) 2006 - 2012 by Visualisierungsinstitut Universitaet Stuttgart.
 * Alle Rechte vorbehalten.
 */

#include "vislib/graphics/CameraParamsProjectionOverride.h"



/*
 * ...::CameraParamsProjectionOverride::CameraParamsProjectionOverride
 */
vislib::graphics::CameraParamsProjectionOverride
        ::CameraParamsProjectionOverride(void) 
        : Super(), projection(STEREO_PARALLEL) {
}


/*
 * ...::CameraParamsProjectionOverride::CameraParamsProjectionOverride
 */
vislib::graphics::CameraParamsProjectionOverride
        ::CameraParamsProjectionOverride(
        const SmartPtr<CameraParameters>& params)
        : Super(params), projection(params->Projection()) {
}


/*
 * ....::CameraParamsProjectionOverride::~CameraParamsProjectionOverride
 */
vislib::graphics::CameraParamsProjectionOverride
        ::~CameraParamsProjectionOverride(void) {
}


/*
 * vislib::graphics::CameraParamsProjectionOverride::Projection
 */
vislib::graphics::CameraParameters::ProjectionType 
vislib::graphics::CameraParamsProjectionOverride::Projection(void) const {
    return this->projection;
}


/*
 * vislib::graphics::CameraParamsProjectionOverride::SetProjection
 */
void vislib::graphics::CameraParamsProjectionOverride::SetProjection(
        ProjectionType projectionType) {
    this->projection = projection;
    this->indicateValueChange();
}


/*
 *  vislib::graphics::CameraParamsProjectionOverride::operator=
 */
vislib::graphics::CameraParamsProjectionOverride& 
vislib::graphics::CameraParamsProjectionOverride::operator=(
        const CameraParamsProjectionOverride& rhs) {
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
    // intentionally empty
}


/*
 *  vislib::graphics::CameraParamsProjectionOverride::resetOverride
 */
void vislib::graphics::CameraParamsProjectionOverride::resetOverride(void) {
    ASSERT(!this->paramsBase().IsNull());
    this->projection = this->paramsBase()->Projection();
    this->indicateValueChange();
}