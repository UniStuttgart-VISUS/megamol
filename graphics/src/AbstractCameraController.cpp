/*
 * AbstractCameraController.cpp
 *
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 */


#include "vislib/AbstractCameraController.h"
#include "vislib/memutils.h"


/*
 * vislib::graphics::AbstractCameraController::AbstractCameraController
 */
vislib::graphics::AbstractCameraController::AbstractCameraController(void) 
        : camera(NULL) {
}


/*
 * vislib::graphics::AbstractCameraController::~AbstractCameraController
 */
vislib::graphics::AbstractCameraController::~AbstractCameraController(void) {
    // Do not delete camera !
}


/*
 * vislib::graphics::AbstractCameraController::SetCamera
 */
void vislib::graphics::AbstractCameraController::SetCamera(vislib::graphics::Camera *camera) {
    this->camera = camera;
}
