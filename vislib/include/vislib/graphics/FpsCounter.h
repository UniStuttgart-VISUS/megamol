/*
 * FpsCounter.h
 *
 * Copyright (C) 2006 - 2007 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#pragma once
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */


#include "vislib/types.h"


namespace vislib::graphics {


/**
 * Utility class implementing a frames per second counter. Simply mark the
 * beginning and the end of each frame using "FrameBegin" and "FrameEnd".
 */
class FpsCounter {

public:
    /**
     * Ctor.
     *
     * @param bufLength The length of the buffer storing the measurements.
     *                  Must not be zero.
     */
    FpsCounter(unsigned int bufLength = 100);

    /** Dtor. */
    ~FpsCounter();

    /**
     * Answers the averaged fps over the whole measurement buffer.
     *
     * @return The averaged fps.
     */
    inline float FPS() const {
        if (!this->fpsValuesValid) {
            this->evaluate();
        }
        return this->avrFPS;
    }


    /**
     * Marks the beginning of a frame. Should be called immediately before
     * the first rendering command (e.g. glClear) of one frame is called.
     *
     * @throws IllegalStateException if FrameEnd was not called since the
     *                               last time FrameBegin was called.
     */
    void FrameBegin();

    /**
     * Marks the end of a frame. Should be called immediately after the
     * last rendering command (e.g. *SwapBuffers) of one frame is called.
     *
     * @throws IllegalStateException if no FrameBegin was called since the
     *                               last time FrameEnd was called or since
     *                               this object was created.
     */
    void FrameEnd();

    /**
     * Answer the time in milliseconds the last frame needed to be
     * rendered.
     *
     * @return The time required to renderer the last frame.
     */
    double LastFrameTime() const;

    /**
     * Answer the maximum time in milliseconds to be used when calculating
     * the averaged fps.
     *
     * @return The maximum time used when calculating the averaged fps.
     */
    inline double MaxAveragingTime() const {
        return this->avrMillis;
    }

    /**
     * Answers the maximum fps of the whole measurement buffer.
     *
     * @return The maximum fps.
     */
    inline float MaxFPS() const {
        if (!this->fpsValuesValid) {
            this->evaluate();
        }
        return this->maxFPS;
    }

    /**
     * Answers the minimum fps of the whole measurement buffer.
     *
     * @return The minimum fps.
     */
    inline float MinFPS() const {
        if (!this->fpsValuesValid) {
            this->evaluate();
        }
        return this->minFPS;
    }

    /**
     * Resets the FpsCounter. If called between "FrameBegin2 and "FrameEnd"
     * the behaviour is the same as if "FrameEnd" were called implicitly.
     */
    void Reset();

    /**
     * Sets the buffer length to a new value. This also resets the counter
     * similar to calling "Reset".
     *
     * @param bufLength The new length of the buffer. Must not be zero.
     */
    void SetBufferLength(unsigned int bufLength);

    /**
     * Sets the maximum time in milliseconds to be used when calculating
     * the averaged fps.
     *
     * @param millis The maximum time used when calculating the averaged
     *               fps.
     */
    inline void SetMaxAveragingTime(double millis) {
        this->avrMillis = millis;
    }

private:
    /**
     * Forbidden copy ctor.
     *
     * @param rhs The object to be cloned.
     *
     * @throws UnsupportedOperationException Always.
     */
    FpsCounter(const FpsCounter& rhs);

    /**
     * Forbidden assignment.
     *
     * @param rhs The right hand side operand.
     *
     * @throws IllegalParamException if &rhs != this.
     */
    FpsCounter& operator=(const FpsCounter& rhs);

    /**
     * Performs lazy evaluation of the fps values
     */
    void evaluate() const;

    /** marks the reset time */
    double now;

    /** type of internal data structure storing the measurements */
    typedef struct TimeValues_t {
        double before; /* time from last end to begin */
        double frame;  /* time from begin to end */
    } TimeValues;

    /** internal data structure storing the measurements */
    TimeValues* timeValues;

    /** buffer length */
    unsigned int timeValuesCount;

    /** buffer position of the element to be overwritten next */
    unsigned int timeValuesPos;

    /** flag indicating that the buffer is completly valid */
    bool wholeBufferValid;

    /** flag indicating that "FrameBegin" was called */
    bool frameRunning;

    /**
     * maximum time in milliseconds to be used when calculating the
     * averaged fps.
     */
    double avrMillis;

    /** flag indicating if lazy evaluation of the fps values is needed. */
    mutable bool fpsValuesValid;

    /** the average fps */
    mutable float avrFPS;

    /** the maximum fps */
    mutable float maxFPS;

    /** the minimum fps */
    mutable float minFPS;
};

} // namespace vislib::graphics

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
