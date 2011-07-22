/*
 * TimeControl.cpp
 *
 * Copyright (C) 2011 by VISUS (Universitaet Stuttgart). 
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "TimeControl.h"
#include "vislib/assert.h"
#include "param/BoolParam.h"
#include "param/ButtonParam.h"
#include "param/FloatParam.h"
#include "vislib/mathfunctions.h"

using namespace megamol::core;


/*
 * view::TimeControl::TimeControl
 */
view::TimeControl::TimeControl(void) :
        animPlaySlot("anim::play", "Bool parameter to play/stop the animation"),
        animSpeedSlot("anim::speed", "Float parameter of animation speed in time frames per second"),
        animTimeSlot("anim::time", "The slot holding the current time to display"),
        toggleAnimPlaySlot("anim::togglePlay", "Button to toggle animation"),
        animSpeedUpSlot("anim::SpeedUp", "Speeds up the animation"),
        animSpeedDownSlot("anim::SpeedDown", "Slows down the animation"),
        frames(1), isInSitu(false), instOffset(0.0) {

    this->animPlaySlot << new param::BoolParam(false);

    this->animSpeedSlot << new param::FloatParam(4.0f, 0.01f, 100.0f);

    this->animTimeSlot << new param::FloatParam(0.0f, 0.0f);

    this->animSpeedUpSlot << new param::ButtonParam('m');
    this->animSpeedUpSlot.SetUpdateCallback(this, &TimeControl::onAnimSpeedStep);

    this->animSpeedDownSlot << new param::ButtonParam('n');
    this->animSpeedDownSlot.SetUpdateCallback(this, &TimeControl::onAnimSpeedStep);

    this->toggleAnimPlaySlot << new param::ButtonParam(' ');
    this->toggleAnimPlaySlot.SetUpdateCallback(this, &TimeControl::onAnimToggleButton);

}


/*
 * view::TimeControl::~TimeControl
 */
view::TimeControl::~TimeControl(void) {
    // intentionally empty
}


/*
 * view::TimeControl::onAnimToggleButton
 */
float view::TimeControl::Time(double instTime) const {

    if (this->animPlaySlot.IsDirty() || this->animSpeedSlot.IsDirty() || this->animTimeSlot.IsDirty()) {
        this->animPlaySlot.ResetDirty();
        this->animSpeedSlot.ResetDirty();

        this->instOffset = instTime - 
            (this->animTimeSlot.Param<param::FloatParam>()->Value()
            / this->animSpeedSlot.Param<param::FloatParam>()->Value());

    }

    if (this->isInSitu) {
        return static_cast<float>(this->frames);

    } else if (this->animPlaySlot.Param<param::BoolParam>()->Value()) {
        float f = static_cast<float>(
            (instTime - this->instOffset) * this->animSpeedSlot.Param<param::FloatParam>()->Value());
        unsigned int rt = static_cast<unsigned int>(f / static_cast<float>(this->frames));
        f -= static_cast<float>(rt * this->frames);
        this->animTimeSlot.Param<param::FloatParam>()->SetValue(f, false);

    }

    this->animTimeSlot.ResetDirty();
    return this->animTimeSlot.Param<param::FloatParam>()->Value();
}


/*
 * view::TimeControl::onAnimToggleButton
 */
void view::TimeControl::SetTimeExtend(unsigned int frames, bool isInSitu) {
    this->frames = frames;
    if (this->frames <= 0) this->frames = 1;
    this->isInSitu = isInSitu;
}


/*
 * view::TimeControl::onAnimToggleButton
 */
bool view::TimeControl::onAnimToggleButton(param::ParamSlot& p) {
    ASSERT(&p == &this->toggleAnimPlaySlot);
    param::BoolParam *bp = this->animPlaySlot.Param<param::BoolParam>();
    bp->SetValue(!bp->Value());
    return true;
}


/*
 * view::TimeControl::onAnimSpeedStep
 */
bool view::TimeControl::onAnimSpeedStep(param::ParamSlot& p) {
    float spd = this->animSpeedSlot.Param<param::FloatParam>()->Value();
    float ospd = spd;
    if (&p == &this->animSpeedUpSlot) {
        if (spd >= 1.0f && spd < 100.0f) {
            spd += 0.25f;
        } else {
            spd += 0.01f;
            if (spd > 0.999999f) spd = 1.0f;
        }

    } else if (&p == &this->animSpeedDownSlot) {
        if (spd > 1.0f) {
            spd -= 0.25f;
        } else {
            if (spd > 0.01f) spd -= 0.01f;
        }

    }
    if (!vislib::math::IsEqual(ospd, spd)) {
        this->animSpeedSlot.Param<param::FloatParam>()->SetValue(spd);
    }
    return true;
}
