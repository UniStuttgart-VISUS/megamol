/*
 * ScaledBoundingBoxes.h
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCORE_SCALEDBOUNDINGBOXES_INCLUDED
#define MEGAMOLCORE_SCALEDBOUNDINGBOXES_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif

#include "mmcore/BoundingBoxes.h"

#include <vector>

namespace megamol {
namespace core {
namespace utility {

    /**
     * Scale a bounding box with the given scale factor
     *
     * @param scale Scale factor
     * @param bb Bounding box
     *
     * @return Scaled bounding box
     */
    BoundingBoxes MEGAMOLCORE_API scaleBoundingBoxes(float scale, const BoundingBoxes& bb);

    /**
     * Combine multiple bounding boxes and scale the result with the given scale factor
     *
     * @param scale Scale factor
     * @param bbs Bounding boxes
     *
     * @return Combined and scaled bounding box
     */
    BoundingBoxes MEGAMOLCORE_API combineAndScaleBoundingBoxes(float scale, const std::vector<BoundingBoxes>& bbs);

    /**
     * Scale a bounding box with the MegaMol-internal magic scale factor
     *
     * @param bb Bounding box
     *
     * @return Scaled bounding box
     */
    BoundingBoxes MEGAMOLCORE_API magicScaleBoundingBoxes(const BoundingBoxes& bb);

    /**
     * Combine multiple bounding boxes and scale the result
     * with the MegaMol-internal magic scale factor
     *
     * @param bbs Bounding boxes
     *
     * @return Combined and scaled bounding box
     */
    BoundingBoxes MEGAMOLCORE_API combineAndMagicScaleBoundingBoxes(const std::vector<BoundingBoxes>& bbs);

    /**
     * RAII class for applying "MegaMol magic scaling" and eventually reversing it
     */
    class MEGAMOLCORE_API glMagicScale {
    public:
        glMagicScale();
        ~glMagicScale();

        void apply(const BoundingBoxes& bb);

        glMagicScale(const glMagicScale&) = delete;
        glMagicScale(glMagicScale&&) noexcept = delete;
        glMagicScale& operator=(const glMagicScale&) = delete;
        glMagicScale& operator=(glMagicScale&&) noexcept = delete;

    private:
        float scale;
    };

}
}
}

#endif