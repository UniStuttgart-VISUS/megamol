/*
 * BuckyBall.h
 *
 * Copyright (C) 2011-2019 by Universitaet Stuttgart (VISUS).
 * All rights reserved.
 */

#pragma once

#include "geometry_calls/VolumetricDataCall.h"
#include "mmcore/CalleeSlot.h"
#include "mmcore/Module.h"

#include <array>
#include <vector>

namespace megamol {
namespace volume {

/**
 * Class generation buck ball informations
 */
class BuckyBall : public core::Module {
public:
    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char* ClassName(void) {
        return "BuckyBall";
    }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static const char* Description(void) {
        return "Generates a dataset of an icosahedron inside a sphere.";
    }

    /**
     * Answers whether this module is available on the current system.
     *
     * @return 'true' if the module is available, 'false' otherwise.
     */
    static bool IsAvailable(void) {
        return true;
    }

    /** Ctor */
    BuckyBall(void);

    /** Dtor */
    ~BuckyBall(void) override;

protected:
    /**
     * Implementation of 'Create'.
     *
     * @return 'true' on success, 'false' otherwise.
     */
    bool create(void) override;

    /**
     * Implementation of 'Release'.
     */
    void release(void) override;

private:
    /**
     * Gets the data or extent from the source.
     *
     * @param caller The calling call.
     *
     * @return 'true' on success, 'false' on failure.
     */
    bool getDataCallback(core::Call& caller);
    bool getExtentCallback(core::Call& caller);
    bool getDummyCallback(core::Call& caller);

    /** The slot for requesting data */
    core::CalleeSlot getDataSlot;

    /** The distance volume */
    std::vector<float> volume;

    /** Metadata */
    const std::array<float, 3> resolution;
    const std::array<float, 3> sliceDists;
    const double minValue;
    const double maxValue;

    geocalls::VolumetricMetadata_t metaData;
};

} /* end namespace volume */
} /* end namespace megamol */
