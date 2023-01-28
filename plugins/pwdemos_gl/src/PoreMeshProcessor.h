/*
 * PoreMeshProcessor.h
 *
 * Copyright (C) 2018 by VISUS (Universitaet Stuttgart).
 * Alle Rechte vorbehalten.
 */

#pragma once

//#include "ArxelBuffer.h"
#include "BufferMTPConnection.h"
#include "LoopBuffer.h"
//#include "vislib/sys/Event.h"
//#include "vislib/sys/File.h"
#include "vislib/math/Point.h"
#include "vislib/sys/Runnable.h"
//#include "vislib/Array.h"
#include "vislib/math/Vector.h"

namespace megamol::demos_gl {

/**
 * Processor for pore-mesh collection/generation/whatever
 */
class PoreMeshProcessor : public vislib::sys::Runnable {
public:
    typedef struct _sliceloops_t {
        float* data;
        unsigned int cnt; // number of vertices
        struct _sliceloops_t* next;
    } SliceLoops;

    /**
     * Ctor
     */
    PoreMeshProcessor();

    /**
     * Dtor
     */
    ~PoreMeshProcessor() override;

    /**
     * Perform the work of a thread.
     *
     * @param userData A pointer to user data that are passed to the thread,
     *                 if it started.
     *
     * @return The application dependent return code of the thread. This
     *         must not be STILL_ACTIVE (259).
     */
    DWORD Run(void* userData) override;

    /**
     * The Runnable should abort its work as soon as possible. This method
     * should never block! If the Runnable will really stop its work at the
     * next possible possition, return true. Return false (the default
     * implementation), if the Runnable is not able to interrupt its work.
     * Note, that the return value of this method is only a hint to prevent
     * deadlocks and that a thread might be forcefully terminated anyway.
     *
     * @return true to acknowledge that the Runnable will finish as soon
     *         as possible, false if termination is not possible.
     */
    bool Terminate() override;

    /**
     * Sets the input buffer pool
     *
     * @param inBuffers The new input buffer pool
     */
    inline void SetInputBuffers(BufferMTPConnection<LoopBuffer>& inputBuffers) {
        this->inputBuffers = &inputBuffers;
    }

    /**
     * Sets the geometry information for transforming arxel space
     * coordinates into object space
     *
     * @param origin The minimum position in object space
     * @param x The x axis of arxel space in object space
     * @param y The y axis of arxel space in object space
     * @param z The z axis of arxel space in object space
     */
    inline void SetGeometryInformation(const vislib::math::Point<float, 3>& origin,
        const vislib::math::Vector<float, 3>& x, const vislib::math::Vector<float, 3>& y,
        const vislib::math::Vector<float, 3>& z) {
        this->origin = origin;
        this->axes[0] = x;
        this->axes[1] = y;
        this->axes[2] = z;
    }

    ///**
    // * Sets the output buffer pool
    // *
    // * @param outBuffers The new output buffer pool
    // */
    //inline void SetOutputBuffers(BufferMTPConnection<LoopBuffer>& outputBuffers) {
    //    this->outputBuffers = &outputBuffers;
    //}

    SliceLoops* debugoutschlupp;

private:
    /**
     * Performs the magic on a buffer
     *
     * @param inBuffer The buffer to work on
     //* @param outBuffer The output buffer
     */
    void workOnBuffer(LoopBuffer& inBuffer /*, LoopBuffer& outBuffer*/);

    /** The input buffer pool */
    BufferMTPConnection<LoopBuffer>* inputBuffers;

    /** The incoming slice number */
    unsigned int sliceNum;

    /** The origin in data space */
    vislib::math::Point<float, 3> origin;

    /** The main axes of arxel space in data space */
    vislib::math::Vector<float, 3> axes[3];
};

} // namespace megamol::demos_gl
