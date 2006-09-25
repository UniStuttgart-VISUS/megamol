/*
 * Camera.cpp
 *
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 */


#include "vislib/Camera.h"


/*
 * vislib::graphics::Camera::Camera
 */
vislib::graphics::Camera::Camera(void) : holder(NULL) {
}


/**
 * vislib::graphics::Camera::~Camera
 */
vislib::graphics::Camera::~Camera(void) {
    delete this->holder;
}




// TODO: This is Debug Stuff: DELETE ME
#include "vislib/Beholder.h"
void DoStuffTestingCamera() {
    vislib::graphics::Beholder<int> intBeholder;
    vislib::graphics::Beholder<float> floatBeholder;
    vislib::graphics::Beholder<float> *fpB;
    vislib::graphics::Camera testCam;
    testCam.SetBeholder(&intBeholder);
    try {
        fpB = testCam.GetBeholder<float>();
    } catch(...) {
    }
    testCam.SetBeholder(&floatBeholder);
    try {
        fpB = testCam.GetBeholder<float>();
    } catch(...) {
    }
}

