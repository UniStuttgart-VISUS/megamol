/*
 * TimeControl.cpp
 *
 * Copyright (C) 2011 by VISUS (Universitaet Stuttgart).
 * Alle Rechte vorbehalten.
 */

#include "mmcore/view/TimeControl.h"
#include "mmcore/param/BoolParam.h"
#include "mmcore/param/ButtonParam.h"
#include "mmcore/param/FloatParam.h"
#include "mmcore/view/CallTimeControl.h"
#include "stdafx.h"
#include "vislib/assert.h"
#include "vislib/math/mathfunctions.h"

using namespace megamol::core;


/*
 * view::TimeControl::TimeControl
 */
view::TimeControl::TimeControl(void)
        : animPlaySlot("anim::play", "Bool parameter to play/stop the animation")
        , animSpeedSlot("anim::speed", "Float parameter of animation speed in time frames per second")
        , animTimeSlot("anim::time", "The slot holding the current time to display")
        , toggleAnimPlaySlot("anim::togglePlay", "Button to toggle animation")
        , animSpeedUpSlot("anim::SpeedUp", "Speeds up the animation")
        , animSpeedDownSlot("anim::SpeedDown", "Slows down the animation")
        , slaveSlot("timecontrolslave", "Slot used if this time control is slave")
        , masterSlot("timecontrolmaster", "Slot used if this time control is master")
        , frames(1)
        , isInSitu(false)
        , instOffset(0.0) {

    this->animPlaySlot << new param::BoolParam(false);

    this->animSpeedSlot << new param::FloatParam(4.0f, 0.01f, 500.0f);

    this->animTimeSlot << new param::FloatParam(0.0f, 0.0f);

    this->animSpeedUpSlot << new param::ButtonParam(view::Key::KEY_M);
    this->animSpeedUpSlot.SetUpdateCallback(this, &TimeControl::onAnimSpeedStep);

    this->animSpeedDownSlot << new param::ButtonParam(view::Key::KEY_N);
    this->animSpeedDownSlot.SetUpdateCallback(this, &TimeControl::onAnimSpeedStep);

    this->toggleAnimPlaySlot << new param::ButtonParam(view::Key::KEY_SPACE);
    this->toggleAnimPlaySlot.SetUpdateCallback(this, &TimeControl::onAnimToggleButton);

    this->slaveSlot.SetCompatibleCall<CallTimeControlDescription>();

    this->masterSlot.SetCallback(
        CallTimeControl::ClassName(), CallTimeControl::FunctionName(0), this, &TimeControl::masterCallback);
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
    TimeControl* master = this->getMaster();

    if (master != NULL) {
        return master->Time(instTime);

    } else {
        if (this->animPlaySlot.IsDirty() || this->animSpeedSlot.IsDirty() || this->animTimeSlot.IsDirty()) {
            this->animPlaySlot.ResetDirty();
            this->animSpeedSlot.ResetDirty();

            this->instOffset = instTime - (this->animTimeSlot.Param<param::FloatParam>()->Value() /
                                              this->animSpeedSlot.Param<param::FloatParam>()->Value());
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

    return 0.0f;
}


/*
 * view::TimeControl::onAnimToggleButton
 */
void view::TimeControl::SetTimeExtend(unsigned int frames, bool isInSitu) {
    TimeControl* master = this->getMaster();

    if (master != NULL) {
        master->SetTimeExtend(frames, isInSitu);
    } else {
        this->frames = frames;
        if (this->frames <= 0)
            this->frames = 1;
        this->isInSitu = isInSitu;
    }
}


/*
 * view::TimeControl::onAnimToggleButton
 */
bool view::TimeControl::onAnimToggleButton(param::ParamSlot& p) {
    ASSERT(&p == &this->toggleAnimPlaySlot);

    TimeControl* master = this->getMaster();
    param::BoolParam* bp = NULL;

    if (master != NULL) {
        bp = master->animPlaySlot.Param<param::BoolParam>();
    } else {
        bp = this->animPlaySlot.Param<param::BoolParam>();
    }

    bp->SetValue(!bp->Value());
    return true;
}


/*
 * view::TimeControl::onAnimSpeedStep
 */
bool view::TimeControl::onAnimSpeedStep(param::ParamSlot& p) {

    TimeControl* master = this->getMaster();
    param::FloatParam* fp = NULL;

    if (master != NULL) {
        fp = master->animSpeedSlot.Param<param::FloatParam>();
    } else {
        fp = this->animSpeedSlot.Param<param::FloatParam>();
    }

    float spd = fp->Value();
    float ospd = spd;
    if (&p == &this->animSpeedUpSlot) {
        if (spd >= 1.0f && spd < 100.0f) {
            spd += 0.25f;
        } else {
            spd += 0.01f;
            if (spd > 0.999999f)
                spd = 1.0f;
        }

    } else if (&p == &this->animSpeedDownSlot) {
        if (spd > 1.0f) {
            spd -= 0.25f;
        } else {
            if (spd > 0.01f)
                spd -= 0.01f;
        }
    }

    if (!vislib::math::IsEqual(ospd, spd)) {
        fp->SetValue(spd);
    }
    return true;
}


/*
 * view::TimeControl::masterCallback
 */
bool view::TimeControl::masterCallback(Call& c) {
    CallTimeControl* ctc = dynamic_cast<CallTimeControl*>(&c);
    if (ctc == NULL)
        return false;
    ctc->SetMaster(this);
    return true;
}


/*
 * view::TimeControl::getMaster
 */
view::TimeControl* view::TimeControl::getMaster(void) const {
    CallTimeControl* ctc = this->slaveSlot.CallAs<CallTimeControl>();
    if (ctc == NULL)
        return NULL;
    if (!(*ctc)(0))
        return NULL;
    if (ctc->Master() == this)
        return NULL;
    return ctc->Master();
}
