/*
 * AbstractBeholderController.cpp
 *
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 */


#include "vislib/AbstractBeholderController.h"
#include "vislib/memutils.h"


/*
 * vislib::graphics::AbstractBeholderController::AbstractBeholderController
 */
vislib::graphics::AbstractBeholderController::AbstractBeholderController(void) 
        : beholder(NULL) {
}


/*
 * vislib::graphics::AbstractBeholderController::~AbstractBeholderController
 */
vislib::graphics::AbstractBeholderController::~AbstractBeholderController(void) {
    // Do not detele this->beholder
}


/*
 * vislib::graphics::AbstractBeholderController::SetBeholder
 */
void vislib::graphics::AbstractBeholderController::SetBeholder(Beholder *beholder) {
    this->beholder = beholder;
}
