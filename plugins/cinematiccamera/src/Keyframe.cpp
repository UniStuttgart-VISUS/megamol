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
    this->animTime              = 0.0f;
    this->camera.position       = vislib::math::Point<float, 3>(1.0f, 0.0f, 0.0f);
    this->camera.lookat         = vislib::math::Point<float, 3>(0.0f, 0.0f, 0.0f);
    this->camera.up             = vislib::math::Vector<float, 3>(0.0f, 1.0f, 0.0f);
    this->camera.apertureangle  = 30.0f;
}

Keyframe::Keyframe(float at) {
    this->animTime              = at;
    this->camera.position       = vislib::math::Point<float, 3>(1.0f, 0.0f, 0.0f);
    this->camera.lookat         = vislib::math::Point<float, 3>(0.0f, 0.0f, 0.0f);
    this->camera.up             = vislib::math::Vector<float, 3>(0.0f, 1.0f, 0.0f);
    this->camera.apertureangle  = 30.0f;
}

Keyframe::Keyframe(float at, vislib::math::Point<float, 3> pos, vislib::math::Vector<float, 3> up, vislib::math::Point<float, 3> lookat, float aperture) {
    this->animTime              = at;
    this->camera.position       = pos;
    this->camera.lookat         = lookat;
    this->camera.up             = up;
    this->camera.apertureangle  = aperture;
}

/*
* Keyframe::~Keyframe
*/
Keyframe::~Keyframe() {
    // intentionally empty
}


/*
* Keyframe::serialise
*/
void Keyframe::serialise(vislib::Serialiser& serialiser) {
    serialiser.Serialise((float)this->animTime, "AnimTime");
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


/*
* Keyframe::deserialise
*/
void Keyframe::deserialise(vislib::Serialiser& serialiser) {
    float f0, f1, f2, f3, f4, f5, f6, f7, f8, f9, f10;
    serialiser.Deserialise(f0, "AnimTime");
    serialiser.Deserialise(f1, "ApertureAngle");
    serialiser.Deserialise(f2, "PositionX");
    serialiser.Deserialise(f3, "PositionY");
    serialiser.Deserialise(f4, "PositionZ");
    serialiser.Deserialise(f5, "LookAtX");
    serialiser.Deserialise(f6, "LookAtY");
    serialiser.Deserialise(f7, "LookAtZ");
    serialiser.Deserialise(f8, "UpX");
    serialiser.Deserialise(f9, "UpY");
    serialiser.Deserialise(f10, "UpZ");
    this->animTime = f0;
    this->camera.apertureangle = f1;
    this->camera.position = vislib::math::Point<float, 3>(f2, f3, f4);
    this->camera.lookat = vislib::math::Point<float, 3>(f5, f6, f7);
    this->camera.up = vislib::math::Vector<float, 3>(f8, f9, f10);

}