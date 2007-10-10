/*
 * Camera.cpp
 *
 * Copyright (C) 2006 - 2007 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */


#include "vislib/Camera.h"
#include "vislib/assert.h"
#include "vislib/CameraParamsStore.h"


/*
 * vislib::graphics::Camera::Camera
 */
vislib::graphics::Camera::Camera(void) : syncNumber(), 
        parameters(new CameraParamsStore()) {
    this->syncNumber = this->parameters->SyncNumber() - 1; // force update
}


/*
 * vislib::graphics::Camera::Camera
 */
vislib::graphics::Camera::Camera(
        const vislib::SmartPtr<vislib::graphics::CameraParameters>& params) 
        : syncNumber(params->SyncNumber() - 1), parameters(params) {
    ASSERT(!this->parameters.IsNull());
}


/*
 * vislib::graphics::Camera::Camera
 */
vislib::graphics::Camera::Camera(const vislib::graphics::Camera &rhs) 
        : parameters() {
    *this = rhs;
}


/*
 * vislib::graphics::Camera::~Camera
 */
vislib::graphics::Camera::~Camera(void) {
}


/*
 * vislib::graphics::CameraParameters::Parameters
 */
vislib::SmartPtr<vislib::graphics::CameraParameters>& 
vislib::graphics::Camera::Parameters(void) {
    return this->parameters;
}

/*
 * vislib::graphics::CameraParameters::Parameters
 */
const vislib::SmartPtr<vislib::graphics::CameraParameters>&
vislib::graphics::Camera::Parameters(void) const {
    return this->parameters;
}


/*
 * vislib::graphics::Camera::SetParameters
 */
void vislib::graphics::Camera::SetParameters(const 
        vislib::SmartPtr<vislib::graphics::CameraParameters>& params) {
    this->parameters = params;
    this->syncNumber = this->parameters->SyncNumber() - 1; // force update
}


/*
 * vislib::graphics::Camera::Camera
 */
vislib::graphics::Camera& vislib::graphics::Camera::operator=(
        const vislib::graphics::Camera &rhs) {
    this->parameters = rhs.parameters;
    this->syncNumber = this->parameters->SyncNumber() - 1; // force update
    return *this;
}


/*
 * vislib::graphics::Camera::Camera
 */
bool vislib::graphics::Camera::operator==(
        const vislib::graphics::Camera &rhs) const {
    return ((this->parameters == rhs.parameters));
}
