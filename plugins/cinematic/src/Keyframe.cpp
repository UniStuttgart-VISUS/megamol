/**
 * Keyframe.cpp
*
* Copyright (C) 2017 by VISUS (Universitaet Stuttgart).
* Alle Rechte vorbehalten.
*/

#include "stdafx.h"

#include "Keyframe.h"

using namespace megamol::cinematic;

Keyframe::Keyframe() :
    animTime(0.0f),
    simTime(0.0f),
    camera()
{
    this->camera.position      = vislib::math::Point<float, 3>(1.0f, 0.0f, 0.0f);
    this->camera.lookat        = vislib::math::Point<float, 3>(0.0f, 0.0f, 0.0f);
    this->camera.up            = vislib::math::Vector<float, 3>(0.0f, 1.0f, 0.0f);
    this->camera.apertureangle = 30.0f;
}


Keyframe::Keyframe(float at, float st, vislib::math::Point<float, 3> pos, vislib::math::Vector<float, 3> up, vislib::math::Point<float, 3> lookat, float aperture) : 
    animTime(at),
    simTime(st),
    camera() 
{
    this->camera.position       = pos;
    this->camera.lookat         = lookat;
    this->camera.up             = up;
    this->camera.apertureangle  = aperture;
}


Keyframe::~Keyframe() {

    // nothing to do here ...
}


void Keyframe::Serialise(vislib::Serialiser& serialiser) {

    serialiser.Serialise((float)this->animTime, "AnimationTime");
    serialiser.Serialise((float)this->simTime, "SimulationTime");
    serialiser.Serialise((float)this->camera.apertureangle, "ApertureAngle");
    serialiser.Serialise((float)this->camera.position.X(), "PositionX");
    serialiser.Serialise((float)this->camera.position.Y(), "PositionY");
    serialiser.Serialise((float)this->camera.position.Z(), "PositionZ");
    serialiser.Serialise((float)this->camera.lookat.X(), "LookAtX");
    serialiser.Serialise((float)this->camera.lookat.Y(), "LookAtY");
    serialiser.Serialise((float)this->camera.lookat.Z(), "LookAtZ");
    serialiser.Serialise((float)this->camera.up.X(), "UpX");
    serialiser.Serialise((float)this->camera.up.Y(), "UpY");
    serialiser.Serialise((float)this->camera.up.Z(), "UpZ");
}


void Keyframe::Deserialise(vislib::Serialiser& serialiser) {

    float f0, f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11;
    serialiser.Deserialise(f0, "AnimationTime");
    serialiser.Deserialise(f1, "SimulationTime");
    serialiser.Deserialise(f2, "ApertureAngle");
    serialiser.Deserialise(f3, "PositionX");
    serialiser.Deserialise(f4, "PositionY");
    serialiser.Deserialise(f5, "PositionZ");
    serialiser.Deserialise(f6, "LookAtX");
    serialiser.Deserialise(f7, "LookAtY");
    serialiser.Deserialise(f8, "LookAtZ");
    serialiser.Deserialise(f9, "UpX");
    serialiser.Deserialise(f10, "UpY");
    serialiser.Deserialise(f11, "UpZ");
    this->animTime = f0;
    this->simTime = f1;
    this->camera.apertureangle = f2;
    this->camera.position = vislib::math::Point<float, 3>(f3, f4, f5);
    this->camera.lookat = vislib::math::Point<float, 3>(f6, f7, f8);
    this->camera.up = vislib::math::Vector<float, 3>(f9, f10, f11);
}
