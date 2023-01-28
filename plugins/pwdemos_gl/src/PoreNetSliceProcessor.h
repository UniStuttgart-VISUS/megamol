/*
 * PoreNetSliceProcessor.h
 *
 * Copyright (C) 2018 by VISUS (Universitaet Stuttgart).
 * Alle Rechte vorbehalten.
 */

#pragma once

#include "ArxelBuffer.h"
#include "BufferMTPConnection.h"
#include "LoopBuffer.h"
#include "vislib/Array.h"
#include "vislib/math/Point.h"
#include "vislib/math/Vector.h"
#include "vislib/sys/Event.h"
#include "vislib/sys/File.h"
#include "vislib/sys/Runnable.h"

namespace megamol::demos_gl {

/**
 * Processor for pore-net-extraction slices
 */
class PoreNetSliceProcessor : public vislib::sys::Runnable {
public:
    /**
     * Ctor
     */
    PoreNetSliceProcessor();

    /**
     * Dtor
     */
    ~PoreNetSliceProcessor() override;

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
    inline void SetInputBuffers(BufferMTPConnection<ArxelBuffer>& inputBuffers) {
        this->inputBuffers = &inputBuffers;
    }

    /**
     * Sets the output buffer pool
     *
     * @param outBuffers The new output buffer pool
     */
    inline void SetOutputBuffers(BufferMTPConnection<LoopBuffer>& outputBuffers) {
        this->outputBuffers = &outputBuffers;
    }

private:
    /** Direction we are walking around the image */
    enum cellDirection { Up = 0, Right = 1, Down = 2, Left = 3 };

    /** Vector initialized to the offsets corresponding to cellDirection */
    vislib::math::Vector<int, 2> dirOffset[4];

    /**
     * Vector initialized to the offsets corresponding to the edge flags.
     * 1 = top, 2 = right, 4 = bottom, 8 = left.
     */
    vislib::math::Vector<int, 2> edgeOffset[9];

    /** Data structure for holding temporary data about black-to-color borders */
    struct BorderElement {
        vislib::math::Point<int, 2> pos;
        ArxelBuffer::ArxelType val;
        ArxelBuffer::ArxelType edges;

        bool operator==(const BorderElement& rhs) const {
            return this->pos == rhs.pos && this->val == rhs.val && this->edges == rhs.edges;
        }
    };

    /**
     * Adds a point to the current point strip. The offset points from pos (the black pixel)
     * to the filled Arxel that generated the point. Two possibilities exist:
     * 1) We are at the slice border. In this case, pos is used, as otherwise the pixel would wrap
     * 2) Otherwise. Pos + offset is used.
     *
     * @param buffer the Arxelbuffer of the slice
     * @param strip  the strip to append this point to
     * @param pos    the position of the black pixel
     * @param offset the direction of the value pixel
     */
    inline void addToStrip(ArxelBuffer& buffer, vislib::Array<vislib::math::Point<int, 2>>& strip,
        vislib::math::Point<int, 2>& pos, vislib::math::Vector<int, 2>& offset) {
        if (buffer.Get(pos + offset) == 2) {
            strip.Add(pos);
        } else {
            strip.Add(pos + offset);
        }
    }

    /**
     * Collects an edge strip starting from hole (x,y) via Grottel mixed with Yan Liu et al.
     * Warning: edgeStore is destroyed, as visited edges are consumed. Thus you can call this algorithm
     * several times on adjacent pixels in buffer without incurring the cost of collecting
     * the edges several times.
     *
     * @param vertices  the array holding the edge strips. If an unconsumed edge is
     *                  found, all strips (=multiple point pairs) i.e. one per face are added.
     * @param buffer    the buffer of Arxels that is worked upon.
     * @param edgeStore the buffer containing the edge flags. THIS IS WRITTEN TO!
     * @param x         the starting point x coordinate
     * @param y         the starting point y coordinate
     * @param inDir     the direction the virtual cursor is facing when entering (x,y)
     */
    void collectEdge(
        LoopBuffer& outBuffer, ArxelBuffer& buffer, ArxelBuffer& edgeStore, int x, int y, cellDirection inDir);

    /**
     * Answer whether (x, y) is adjacent to a zero pixel.
     *
     * @param buffer buffer containing all the pixels
     * @param x      the x coordinate
     * @param y      the y coordinate
     *
     * @return true if any of the 8 neighbors is zero.
     */
    bool isEdgePixel(ArxelBuffer& buffer, int x, int y);

    /**
     * Performs the magic on a buffer
     *
     * @param buffer The buffer to work on
     * @param outBuffer The output buffer
     */
    void workOnBuffer(ArxelBuffer& buffer, LoopBuffer& outBuffer);

    /** The input buffer pool */
    BufferMTPConnection<ArxelBuffer>* inputBuffers;

    /** The output buffer pool */
    BufferMTPConnection<LoopBuffer>* outputBuffers;
};

} // namespace megamol::demos_gl
