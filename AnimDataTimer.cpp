/*
 * AnimDataTimer.cpp
 *
 * Copyright (C) 2008 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "AnimDataTimer.h"
#include "param/BoolParam.h"
#include "param/FloatParam.h"
#include "vislib/IllegalStateException.h"
#include <cmath>

using namespace megamol;
using namespace megamol::core;


/*
 * protein::AnimDataTimer::AnimDataTimer
 */
protein::AnimDataTimer::AnimDataTimer(void) : m_source(NULL), m_runAnim(false), 
        m_animTimer(true), m_animTime(0.0f), 
		m_animationParam("animation", "Animation Timer Active"), 
        m_animSpeedParam("animSpeed", "Animation Timer Speed") 
{
    this->m_animationParam.SetParameter(new param::BoolParam(this->m_runAnim));
    this->m_animSpeedParam.SetParameter(new param::FloatParam(4.0f, 0.01f, 100.0f));
}


/*
 * protein::AnimDataTimer::~AnimDataTimer
 */
protein::AnimDataTimer::~AnimDataTimer(void) {
    this->m_source = NULL; // DO NOT DELETE
}


/*
 * protein::AnimDataTimer::Start
 */
void protein::AnimDataTimer::Start(void) {
    if (this->m_runAnim) return;
    if (this->m_source == NULL) {
        throw vislib::IllegalStateException(
            "Source must be set before animation timer can be started", 
            __FILE__, __LINE__);
    }
    this->m_animTimer.SetMark();
    this->m_runAnim = true;
    this->m_animationParam.Param<param::BoolParam>()->SetValue(true);
}


/*
 * protein::AnimDataTimer::Stop
 */
void protein::AnimDataTimer::Stop(void) {
    if (!this->m_runAnim) return;
    this->m_runAnim = false;
    this->m_animationParam.Param<param::BoolParam>()->SetValue(false);
}


/*
 * protein::AnimDataTimer::Time
 */
float protein::AnimDataTimer::Time(void) const {

    if (bool(this->m_animationParam.Param<param::BoolParam>()->Value()) != this->m_runAnim) 
	{
        this->m_runAnim = !this->m_runAnim;
        if (this->m_runAnim) {
            if ((this->m_source != NULL) && (this->m_source->FrameCount() > 1)) {
                // animation starts
                this->m_animTimer.SetMark();
            } else {
				this->m_runAnim = false;
                this->m_animationParam.Param<param::BoolParam>()->SetValue(false);
            }
        } else {
            this->m_animTime = floor(this->m_animTime + 0.5f);
        }
    }

    if (this->m_source == NULL) {
		this->m_runAnim = false;
        this->m_animationParam.Param<param::BoolParam>()->SetValue(false);
    }

    if (this->m_runAnim) {
        this->m_animTime = fmodf(float(this->m_animSpeedParam.Param<param::FloatParam>()->Value()) * 0.001f * 
			               float( vislib::sys::PerformanceCounter::ToMillis(this->m_animTimer.Difference())) , 
                           float(this->m_source->FrameCount()) - 0.9999f);
        if (this->m_animTime > float(this->m_source->FrameCount() - 1)) {
            this->m_animTime = float(this->m_source->FrameCount() - 1);
        }

        //static unsigned int mydummy = 100000000;
        //if (mydummy != static_cast<unsigned int>(10.0f * this->animTime)) {
        //    printf("Animation: %f\n", this->animTime);
        //    mydummy = static_cast<unsigned int>(10.0f * this->animTime);
        //}
    }
    return this->m_animTime;
}
