/*
 * FpsCounter.cpp
 *
 * Copyright (C) 2006 - 2007 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#include "vislib/FpsCounter.h"
#include <climits>
#include "vislib/assert.h"
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
        wholeBufferValid(false), frameRunning(false), fpsValuesValid(false) {
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
        throw IllegalStateException("Must call \"FrameEnd\" first.", __FILE__, __LINE__);
    }

    UINT64 newNow = vislib::sys::PerformanceCounter::Query();
    this->timeValues[this->timeValuesPos].before = static_cast<unsigned int>(newNow - this->now);
    this->now = newNow;
    this->frameRunning = true;
    this->fpsValuesValid = false;
}


/*
 * vislib::graphics::FpsCounter::FrameEnd
 */
void vislib::graphics::FpsCounter::FrameEnd(void) {
    if (!this->frameRunning) {
        throw IllegalStateException("Must call \"FrameBegin\" first.", __FILE__, __LINE__);
    }

    UINT64 newNow = vislib::sys::PerformanceCounter::Query();
    this->timeValues[this->timeValuesPos++].frame = static_cast<unsigned int>(newNow - this->now);
    if (this->timeValuesPos >= this->timeValuesCount) {
        this->timeValuesPos = 0;
        this->wholeBufferValid = true;
    }
    this->now = newNow;
    this->frameRunning = false;
    this->fpsValuesValid = false;
}


/*
 * vislib::graphics::FpsCounter::Reset
 */
void vislib::graphics::FpsCounter::Reset(void) {
    this->now = vislib::sys::PerformanceCounter::Query();
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
    throw vislib::UnsupportedOperationException("Copy Ctor", __FILE__, __LINE__);
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
void vislib::graphics::FpsCounter::evaluate(void) {
    unsigned int count = (this->wholeBufferValid ? this->timeValuesCount : this->timeValuesPos);
    unsigned int time;
    unsigned int allTime = 0;
    unsigned int maxTime = 0;
    unsigned int minTime = UINT_MAX;

    /** summarise over the whole measurement buffer */
    for (unsigned int i = 0; i < count; i++) {
        time = this->timeValues[i].before + this->timeValues[i].frame;
        allTime += time;
        if (maxTime < time) maxTime = time;
        if (minTime > time) minTime = time;
    }

    /** average fps */
    if (allTime > 0) {
        this->avrFPS = static_cast<float>(count) * 1000.0f / static_cast<float>(allTime);
    } else {
        this->avrFPS = 0.0f;
    }

    /** maximum fps */
    if (minTime > 0) {
        this->maxFPS = 1000.0f / static_cast<float>(minTime);
    } else {
        this->maxFPS = 0.0f;
    }

    /** minimum fps */
    if (maxTime > 0) {
        this->minFPS = 1000.0f / static_cast<float>(maxTime);
    } else {
        this->minFPS = 0.0f;
    }
    
    this->fpsValuesValid = true;
}
