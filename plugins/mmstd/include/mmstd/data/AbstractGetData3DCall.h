/**
 * MegaMol
 * Copyright (c) 2009, MegaMol Dev Team
 * All rights reserved.
 */

#pragma once

#include "mmcore/BoundingBoxes.h"
#include "mmcore/factories/CallAutoDescription.h"
#include "mmstd/data/AbstractGetDataCall.h"

namespace megamol::core {

/**
 * Abstract base class for calls for data
 */
class AbstractGetData3DCall : public AbstractGetDataCall {
public:
    /** Ctor. */
    AbstractGetData3DCall();

    /** Dtor. */
    virtual ~AbstractGetData3DCall();

    /**
     * Answers the bounding box of the data set
     *
     * @return The bounding box of the data set
     */
    inline BoundingBoxes& AccessBoundingBoxes() {
        return this->bboxs;
    }

    /**
     * Answer the frame count.
     *
     * @return The frame count.
     */
    inline unsigned int FrameCount() const {
        return this->frameCnt;
    }

    /**
     * Answer the frameID
     *
     * @return the frameID
     */
    inline unsigned int FrameID() const {
        return this->frameID;
    }

    /**
     * Answers the bounding box of the data set
     *
     * @return The bounding box of the data set
     */
    inline const BoundingBoxes& GetBoundingBoxes() const {
        return this->bboxs;
    }

    /**
     * Answer if the frame is forced. If 'true' a data provider must
     * return exactly the requested frame, no matter how long the loading
     * takes. If 'false' a data provider should return as fast as possible
     * returning the closest match to the requested data (updating the
     * frameID attribute).
     *
     * @return The flag if the frame is forced
     */
    inline bool IsFrameForced() const {
        return this->forceFrame;
    }

    /**
     * Sets the extents of the data.
     * Called modules uses this method to output their data.
     *
     * @param frameCnt The number of frames in the trajectory.
     * @param minX The minimum x coordinate
     * @param minY The minimum y coordinate
     * @param minZ The minimum z coordinate
     * @param maxX The maximum x coordinate
     * @param maxY The maximum y coordinate
     * @param maxZ The maximum z coordinate
     */
    inline void SetExtent(
        unsigned int frameCnt, float minX, float minY, float minZ, float maxX, float maxY, float maxZ) {
        this->frameCnt = frameCnt;
        this->bboxs.Clear();
        this->bboxs.SetObjectSpaceBBox(minX, minY, minZ, maxX, maxY, maxZ);
        this->bboxs.SetObjectSpaceClipBox(minX, minY, minZ, maxX, maxY, maxZ);
    }

    /**
     * Sets the extents of the data.
     * Called modules uses this method to output their data.
     *
     * @param frameCnt The number of frames in the trajectory.
     * @param bboxs The bounding boxes of the data
     */
    inline void SetExtent(unsigned int frameCnt, const BoundingBoxes& bboxs) {
        this->frameCnt = frameCnt;
        this->bboxs = bboxs;
    }

    /**
     * Sets the number of time frames of the data
     *
     * @param frameCnt The number of frames in the trajectory.
     */
    inline void SetFrameCount(unsigned int frameCnt) {
        this->frameCnt = frameCnt;
    }

    /**
     * Sets the frameID to request data for.
     *
     * @param frameID The frameID to request data for.
     * @param force Flag whether or not to force the frame data. If the
     *              data is not forces, a source module may return data
     *              from any other frame (the best data currently
     *              available). If the data is forced, the call might not
     *              return until the data is loaded.
     */
    inline void SetFrameID(unsigned int frameID, bool force = false) {
        this->forceFrame = force;
        this->frameID = frameID;
    }

    /**
     * Assignment operator.
     * Makes a deep copy of all members. While for data these are only
     * pointers, the pointer to the unlocker object is also copied.
     *
     * @param rhs The right hand side operand
     *
     * @return A reference to this
     */
    AbstractGetData3DCall& operator=(const AbstractGetData3DCall& rhs);

protected:
    /**
     * Answer the number of functions used for this call.
     *
     * @return The number of functions used for this call.
     */
    static unsigned int FunctionCount() {
        return 2;
    }

    /**
     * Answer the name of the function used for this call.
     *
     * @param idx The index of the function to return it's name.
     *
     * @return The name of the requested function.
     */
    static const char* FunctionName(unsigned int idx) {
        switch (idx) {
        case 0:
            return "GetData";
        case 1:
            return "GetExtent";
        }
        return NULL;
    }

private:
    /** Flag whether or not to force the frame data */
    bool forceFrame;

    /** The number of frames in the trajectory */
    unsigned int frameCnt;

    /** The requested/stored frameID */
    unsigned int frameID;

    /** the coordinate extents */
    BoundingBoxes bboxs;
};

} // namespace megamol::core
