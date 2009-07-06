/*
 * FpsCounter.cpp
 *
 * Copyright (C) 2006 - 2007 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#include "vislib/FpsCounter.h"
#include <climits>
#include <cfloat>
#include "vislib/assert.h"
#include "vislib/mathfunctions.h"
#include "vislib/memutils.h"
#include "vislib/IllegalParamException.h"
#include "vislib/IllegalStateException.h"
#include "vislib/PerformanceCounter.h"
#include "vislib/UnsupportedOperationException.h"


/*
 * vislib::graphics::FpsCounter::FpsCounter
 */
vislib::graphics::FpsCounter::FpsCounter(unsigned int bufLength) 
        : now(0), timeValues(NULL), timeValuesCount(0), timeValuesPos(0), 
        wholeBufferValid(false), frameRunning(false), avrMillis(FLT_MAX), 
        fpsValuesValid(false) {
    this->SetBufferLength(bufLength);
}


/*
 * vislib::graphics::FpsCounter::~FpsCounter
 */
vislib::graphics::FpsCounter::~FpsCounter(void) {
    ARY_SAFE_DELETE(this->timeValues);
    this->timeValuesCount = 0; // paranoia
}


/*
 * vislib::graphics::FpsCounter::FrameBegin
 */
void vislib::graphics::FpsCounter::FrameBegin(void) {
    if (this->frameRunning) {
        throw IllegalStateException("Must call \"FrameEnd\" first.", 
            __FILE__, __LINE__);
    }

    double newNow = vislib::sys::PerformanceCounter::QueryMillis();
    double diff = (newNow > this->now) ? (newNow - this->now) : 0.0;
    this->timeValues[this->timeValuesPos].before = diff;
    this->now = newNow;
    this->frameRunning = true;
    this->fpsValuesValid = false;
}


/*
 * vislib::graphics::FpsCounter::FrameEnd
 */
void vislib::graphics::FpsCounter::FrameEnd(void) {
    if (!this->frameRunning) {
        throw IllegalStateException("Must call \"FrameBegin\" first.", 
            __FILE__, __LINE__);
    }

    double newNow = vislib::sys::PerformanceCounter::QueryMillis();
    double diff = (newNow > this->now) ? (newNow - this->now) : 0.0;
    this->timeValues[this->timeValuesPos++].frame = diff;
    if (this->timeValuesPos >= this->timeValuesCount) {
        this->timeValuesPos = 0;
        this->wholeBufferValid = true;
    }
    this->now = newNow;
    this->frameRunning = false;
    this->fpsValuesValid = false;
}


/*
 * vislib::graphics::FpsCounter::LastFrameTime
 */
double vislib::graphics::FpsCounter::LastFrameTime(void) const {
    unsigned int idx = this->timeValuesPos;
    if (idx > 0) {
        idx--;
    } else if (this->wholeBufferValid) {
        idx = this->timeValuesCount - 1;
    } else {
        return 0.0;
    }
    return this->timeValues[idx].before + this->timeValues[idx].frame;
}


/*
 * vislib::graphics::FpsCounter::Reset
 */
void vislib::graphics::FpsCounter::Reset(void) {
    this->now = vislib::sys::PerformanceCounter::QueryMillis();
    this->timeValuesPos = 0;
    this->wholeBufferValid = false;
    this->frameRunning = false;
}


/*
 * vislib::graphics::FpsCounter::SetBufferLength
 */
void vislib::graphics::FpsCounter::SetBufferLength(unsigned int bufLength) {
    ASSERT(bufLength > 0);
    ARY_SAFE_DELETE(this->timeValues);
    this->timeValuesCount = bufLength;
    this->timeValues = new TimeValues[this->timeValuesCount];
    this->Reset();
}


/*
 * vislib::graphics::FpsCounter::FpsCounter
 */
vislib::graphics::FpsCounter::FpsCounter(const FpsCounter& rhs) {
    throw vislib::UnsupportedOperationException("Copy Ctor", __FILE__, 
        __LINE__);
}


/*
 * vislib::graphics::FpsCounter::operator =
 */
vislib::graphics::FpsCounter& vislib::graphics::FpsCounter::operator =(
        const vislib::graphics::FpsCounter& rhs) {
    if (&rhs != this) {
        throw vislib::IllegalParamException("rhs", __FILE__, __LINE__);
    }
    return *this;
}


/*
 * vislib::graphics::FpsCounter::evaluate
 */
void vislib::graphics::FpsCounter::evaluate(void) const {
    unsigned int count = (this->wholeBufferValid ? this->timeValuesCount : 
        vislib::math::Max<int>(this->timeValuesPos - 1, 0));

    if (count == 0) {
        this->avrFPS = this->minFPS = this->maxFPS = 0.0f;
        this->fpsValuesValid = true;
        return;
    }

    unsigned int avrCount = count;
    double time;
    double allTime = 0.0;
    double maxTime = 0.0;
    double minTime = FLT_MAX;

    /** summarise over the whole measurement buffer */
    for (unsigned int i = 0; i < count; i++) {
        time = this->timeValues[i].before + this->timeValues[i].frame;

        if (allTime < this->avrMillis) {
            allTime += time;
            if (allTime >= this->avrMillis) {
                avrCount = i + 1;
            }
        }

        if (maxTime < time) maxTime = time;
        if (minTime > time) minTime = time;
    }

    /** average fps */
    if (allTime > 0) {
        this->avrFPS = static_cast<float>(static_cast<double>(avrCount) 
            * 1000.0 / allTime);
    } else {
        this->avrFPS = 0.0f;
    }

    /** maximum fps */
    if (minTime > 0) {
        this->maxFPS = static_cast<float>(1000.0 / minTime);
    } else {
        this->maxFPS = 0.0f;
    }

    /** minimum fps */
    if (maxTime > 0) {
        this->minFPS = static_cast<float>(1000.0 / maxTime);
    } else {
        this->minFPS = 0.0f;
    }
    
    this->fpsValuesValid = true;
}
