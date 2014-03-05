/*
 * AbstractCameraController.cpp
 *
 * Copyright (C) 2006 - 2007 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#include <cstddef> // for NULL
#include "vislib/AbstractCameraController.h"
#include "the/assert.h"


/*
 * vislib::graphics::AbstractCameraController::AbstractCameraController
 */
vislib::graphics::AbstractCameraController::AbstractCameraController(
        const SmartPtr<CameraParameters>& cameraParams) 
        : cameraParams(cameraParams) {
}


/*
 * vislib::graphics::AbstractCameraController::~AbstractCameraController
 */
vislib::graphics::AbstractCameraController::~AbstractCameraController(void) {
    // intentionally empty
}


/*
 * vislib::graphics::AbstractCameraController::CameraParams
 */
vislib::SmartPtr<vislib::graphics::CameraParameters>& 
vislib::graphics::AbstractCameraController::CameraParams(void) {
    THE_ASSERT(this->IsCameraParamsValid());
    return this->cameraParams;
}


/*
 * vislib::graphics::AbstractCameraController::CameraParams
 */
const vislib::SmartPtr<vislib::graphics::CameraParameters>& 
vislib::graphics::AbstractCameraController::CameraParams(void) const {
    THE_ASSERT(this->IsCameraParamsValid());
    return this->cameraParams;
}


/*
 * vislib::graphics::AbstractCameraController::
 */
void vislib::graphics::AbstractCameraController::SetCameraParams(
        const SmartPtr<CameraParameters>& cameraParams) {
    this->cameraParams = cameraParams;
}
