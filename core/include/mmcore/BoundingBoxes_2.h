/*
 * BoundingBoxes_2.h
 *
 * Copyright (C) 2018-2019 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCORE_BOUNDINGBOXES_2_H_INCLUDED
#define MEGAMOLCORE_BOUNDINGBOXES_2_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "mmcore/api/MegaMolCore.std.h"
#include "vislib/math/Cuboid.h"
#include "mmcore/BoundingBoxes.h"

namespace megamol {
namespace core {

    /**
     * New variant of the Bounding Box class. The determination between object space and world space
     * has been removed. Therefore, the scale value is now neither necessary nor wanted.
     *
     */
    class MEGAMOLCORE_API BoundingBoxes_2 {
    public:

        /**
         * Constructor
         */
        BoundingBoxes_2(void);
    
        /**
         * Copy Constructor
         *
         * @param src The object to clone from
         */
        BoundingBoxes_2(const BoundingBoxes_2& src);

        /**
         * Destructor
         */
        virtual ~BoundingBoxes_2(void);
    
        /**
         * Clears all members to invalid bounding boxes and a scale of zero.
         */
        inline void Clear(void) {
            this->clipBoxValid = false;
            this->boundingBoxValid = false;
        }

        /**
         * Safely answers the best match for a clipping box. It will use the
         * first valid box from this sequence: world space clipping box,
         * world space bounding box, object space clipping box, object space
         * bounding box. If all four boxes are invalid it will return a dummy
         * box (-1, -1, -1) ... (1, 1, 1).
         *
         * @return The best match for a clipping bounding box.
         */
        inline const vislib::math::Cuboid<float>& ClipBox(void) const {
            if (!this->clipBoxValid) {
                this->calcClipBox();
            }
            return this->clipBox;
        }

        /**
         * Answer whether any of the boxes is valid
         *
         * @return True if any of the boxes is valid
         */
        inline bool IsAnyValid(void) const {
            return this->clipBoxValid || this->boundingBoxValid;
        }

        /**
         * Answer whether or not the bounding box is valid.
         *
         * @return True if the bounding box is valid
         */
        inline bool IsBoundingBoxValid(void) const {
            return this->boundingBoxValid;
        }

        /**
         * Answer whether or not the clipping box is valid.
         *
         * @return True if the clipping box is valid
         */
        inline bool IsClipBoxValid(void) const {
            return this->clipBoxValid;
        }

        /**
         * Answer the bounding box
         *
         * @return The bounding box
         */
        inline const vislib::math::Cuboid<float>& BoundingBox(void) const {
            return this->boundingBox;
        }

        /**
         * Sets the bounding box.
         *
         * @param box The new bounding box
         */
        inline void SetBoundingBox(const vislib::math::Cuboid<float>& box) {
            this->boundingBox = box;
            this->boundingBoxValid = true;
            this->clipBoxValid = false;
        }

        /**
         * Sets the bounding box.
         *
         * @param left   The x-coordinate of the left/bottom/back point.
         * @param bottom The y-coordinate of the left/bottom/back point.
         * @param back   The z-coordinate of the left/bottom/back point.
         * @param right  The x-coordinate of the right/top/front point.
         * @param top    The y-coordinate of the right/top/front point.
         * @param front  The z-coordinate of the right/top/front point.
         */
        inline void SetBoundingBox(const float& left, const float& bottom,
                                       const float& back, const float& right, const float& top,
                                       const float& front) {
            this->boundingBox.Set(left, bottom, back, right, top, front);
            this->boundingBoxValid = true;
            this->clipBoxValid = false;
        }

        /**
         * Sets the object space clipping box.
         *
         * @param box The new object space clipping box
         */
        inline void SetClipBox(const vislib::math::Cuboid<float>& box) {
            this->clipBox = box;
            this->clipBoxValid = true;
        }

        /**
         * Sets the clipping box.
         *
         * @param left   The x-coordinate of the left/bottom/back point.
         * @param bottom The y-coordinate of the left/bottom/back point.
         * @param back   The z-coordinate of the left/bottom/back point.
         * @param right  The x-coordinate of the right/top/front point.
         * @param top    The y-coordinate of the right/top/front point.
         * @param front  The z-coordinate of the right/top/front point.
         */
        inline void SetClipBox(const float& left, const float& bottom,
                                          const float& back, const float& right, const float& top,
                                          const float& front) {
            this->clipBox.Set(left, bottom, back, right, top, front);
            this->clipBoxValid = true;
        }

        /**
         * Test for equality
         *
         * @param rhs The right hand side operand
         *
         * @return True if 'this' and 'rhs' are equal
         */
        bool operator==(const BoundingBoxes_2& rhs) const;

        /**
         * Assignment operator
         *
         * @param rhs The right hand side operand
         *
         * @return A reference to this.
         */
        BoundingBoxes_2& operator=(const BoundingBoxes_2& rhs);

        /**
         * Assignment operator for the old bounding box object to achieve backwards compatibility.
         *
         * @param rhs The right hand side operand
         *
         * @return A reference to this.
         */
        BoundingBoxes_2& operator=(const BoundingBoxes& rhs);
    private:

        /**
         * Calculate the composed clipping box.
         */
        void calcClipBox(void) const;
    
#ifdef _WIN32
#pragma warning (disable: 4251)
#endif
        /** The composed clipping box */
        mutable vislib::math::Cuboid<float> clipBox;

        /** The valid flag for the composed clipping box */
        mutable bool clipBoxValid;

        /** The bounding box */
        vislib::math::Cuboid<float> boundingBox;

        /** The valid flag for the bounding box */
        bool boundingBoxValid;
#ifdef _WIN32
#pragma warning (default: 4251)
#endif
    };
}
}


#endif /* MEGAMOLCORE_BOUNDINGBOXES_2_H_INCLUDED */