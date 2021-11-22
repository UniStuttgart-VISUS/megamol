/*
 * BrickStatsCall.h
 *
 * Author: Guido Reina
 * Copyright (C) 2016 by Universitaet Stuttgart (VISUS).
 * All rights reserved.
 */

#pragma once

#include "mmcore/AbstractGetData3DCall.h"
#include "mmcore/factories/CallAutoDescription.h"
#include "vislib/Array.h"
#include "vislib/math/Cuboid.h"

namespace megamol {
namespace moldyn {

/**
 * Call transporting information about particle/point
 * dataset bricks.
 */
class BrickStatsCall : public megamol::core::AbstractGetData3DCall {
public:
    typedef float BrickStatsType;

    class BrickInfo {
    public:
        /** check whether this is contained at least partially inside otherBrick */
        bool CheckContainment(const vislib::math::Cuboid<BrickStatsType>& otherBrick);

        inline BrickInfo() : bounds(), mean(), stddev() {}

        inline BrickInfo(UINT64 offset, UINT64 length, const BrickStatsType& left, const BrickStatsType& bottom,
            const BrickStatsType& back, const BrickStatsType& right, const BrickStatsType& top,
            const BrickStatsType& front, const BrickStatsType& meanX, const BrickStatsType& meanY,
            const BrickStatsType& meanZ, const BrickStatsType& stdDevX, const BrickStatsType& stdDevY,
            const BrickStatsType& stdDevZ)
                : offset(offset)
                , length(length) {

            this->bounds.Set(left, bottom, back, right, top, front);
            this->mean.Set(meanX, meanY, meanZ);
            this->stddev.Set(stdDevX, stdDevY, stdDevZ);
        }

        inline bool operator==(const BrickInfo& rhs) const {
            return (this == &rhs) ||
                   ((this->bounds == rhs.bounds) && (this->mean == rhs.mean) && (this->stddev == rhs.stddev));
        }

    private:
        UINT64 offset, length;
        vislib::math::Cuboid<BrickStatsType> bounds;
        vislib::math::Point<BrickStatsType, 3> mean;
        vislib::math::Point<BrickStatsType, 3> stddev;
    };

    // NOT INLINE! return the size of BrickStatsType at the time of compiling THIS MODULE! Assert against this. always!
    static const SIZE_T GetTypeSize();

    /**
     * Answer the name of the objects of this description.
     *
     * @return The name of the objects of this description.
     */
    static const char* ClassName(void) {
        return "BrickStatsCall";
    }

    /**
     * Gets a human readable description of the module.
     *
     * @return A human readable description of the module.
     */
    static const char* Description(void) {
        return "Call to get information about bricks of a particle/point dataset";
    }

    /**
     * Answer the number of functions used for this call.
     *
     * @return The number of functions used for this call.
     */
    static unsigned int FunctionCount(void) {
        return AbstractGetData3DCall::FunctionCount();
    }

    /**
     * Answer the name of the function used for this call.
     *
     * @param idx The index of the function to return its name.
     *
     * @return The name of the requested function.
     */
    static const char* FunctionName(unsigned int idx) {
        return AbstractGetData3DCall::FunctionName(idx);
    }

    inline const vislib::Array<BrickInfo>* GetBricks() const {
        return bricks;
    }

    inline void SetBricks(vislib::Array<BrickInfo>* bricks) {
        this->bricks = bricks;
    }

    //inline vislib::Array<int> *GetSelectionPointer(void) const {
    //    return this->selection;
    //}

    //inline void SetSelectionPointer(vislib::Array<int> *selection) {
    //    this->selection = selection;
    //}

    BrickStatsCall(void);
    virtual ~BrickStatsCall(void);

private:
    vislib::Array<BrickInfo>* bricks;
};

/** Description class typedef */
typedef megamol::core::factories::CallAutoDescription<BrickStatsCall> BrickStatsCallDescription;

} /* end namespace moldyn */
} /* end namespace megamol */
