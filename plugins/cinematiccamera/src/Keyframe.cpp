/**
 * Keyframe.cpp
 */

#include "stdafx.h"

#include "Keyframe.h"

using namespace megamol::cinematiccamera;


/*
* Keyframe::Keyframe
*/
Keyframe::Keyframe() {

    this->camera = vislib::graphics::Camera();
    this->time   = 0.0f;
}
Keyframe::Keyframe(vislib::graphics::Camera c, float t) {

	this->camera = c;
    this->time   = t;
}


/*
* Keyframe::~Keyframe
*/
Keyframe::~Keyframe() {

}

