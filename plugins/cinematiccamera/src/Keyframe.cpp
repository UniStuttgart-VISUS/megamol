/**
 * Keyframe.cpp
 */

#include "stdafx.h"
#include "Keyframe.h"

using namespace megamol::cinematiccamera;

Keyframe::Keyframe() {
}

Keyframe::Keyframe(int ID) {
	this->ID = ID;
}

Keyframe::Keyframe(vislib::graphics::Camera camera, float time, int ID) {

	this->camera = camera;
	if (time >= 0 && time <= 1)	this->time = time;
	else this->time = 1.0f;
	this->ID = ID;

}


Keyframe::~Keyframe()
{
}

