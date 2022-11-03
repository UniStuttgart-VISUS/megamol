/**
 * MegaMol
 * Copyright (c) 2009, MegaMol Dev Team
 * All rights reserved.
 */

#pragma once

#include "vislib/math/Cuboid.h"

namespace megamol::core {

/**
 * Class for storing and accessing bounding boxes. This class should be
 * used by modules storing data and by calls transporting data or
 * rendering information.
 *
 * The object space bounding box defines a reference bounding box for the
 * data in object coordinates. The object coordinate system scale gives
 * further information on how to interpret these values. This scale should
 * be based on meters and should be set to zero if an interpretation is
 * not supported.
 *
 * The world space bounding box defines a reference bounding box for the
 * data coordinates in Open GL world space. These are the coordinates
 * which are actually uploaded into the rendering pipeline. These
 * coordinates should always be closed to the unit-cube sizes to avoid
 * numeric problems with rendering.
 *
 * The world space clipping box defined the bounding box for the data in
 * Open GL world space. In contrast to the world space bounding box, all
 * visual representatives of the data must be inside this clipping box to
 * avoid visual artifacts.
 */
class BoundingBoxes {
public:
    /**
     * Ctor
     * All bounding boxes will be invalid and the object space scale will
     * be set to zero.
     */
    BoundingBoxes();

    /**
     * Copy Ctor
     *
     * @param src The object to clone from
     */
    BoundingBoxes(const BoundingBoxes& src);

    /**
     * Dtor
     */
    virtual ~BoundingBoxes();

    /**
     * Clears all members to invalid bounding boxes and an object space
     * scale of zero.
     */
    inline void Clear() {
        this->clipBoxValid = false;
        this->osScale = 0.0;
        this->osBBoxValid = false;
        this->osClipBoxValid = false;
        this->wsBBoxValid = false;
        this->wsClipBoxValid = false;
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
    inline const vislib::math::Cuboid<float>& ClipBox() const {
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
    inline bool IsAnyValid() const {
        return this->clipBoxValid || this->osBBoxValid || this->osClipBoxValid || this->wsBBoxValid ||
               this->wsClipBoxValid;
    }

    /**
     * Answer whether or not the object space bounding box is valid.
     *
     * @return True if the object space bounding box is valid
     */
    inline bool IsObjectSpaceBBoxValid() const {
        return this->osBBoxValid;
    }

    /**
     * Answer whether or not the object space clipping box is valid.
     *
     * @return True if the object space clipping box is valid
     */
    inline bool IsObjectSpaceClipBoxValid() const {
        return this->osClipBoxValid;
    }

    /**
     * Answer whether or not the world space bounding box is valid.
     *
     * @return True if the world space bounding box is valid
     */
    inline bool IsWorldSpaceBBoxValid() const {
        return this->wsBBoxValid;
    }

    /**
     * Answer whether or not the world space clipping box is valid.
     *
     * @return True if the world space clipping box is valid
     */
    inline bool IsWorldSpaceClipBoxValid() const {
        return this->wsClipBoxValid;
    }

    /**
     * Answer the object space bounding box
     *
     * @return The object space bounding box
     */
    inline const vislib::math::Cuboid<float>& ObjectSpaceBBox() const {
        return this->osBBox;
    }

    /**
     * Answer the object space clipping box
     *
     * @return The object space clipping box
     */
    inline const vislib::math::Cuboid<float>& ObjectSpaceClipBox() const {
        return this->osClipBox;
    }

    /**
     * Creates the world space bounding and clipping boxes by using scaled
     * copies of their object space counterparts.
     *
     * @param f The scaling factor used.
     */
    void MakeScaledWorld(float f);

    /**
     * Answer the object space scale value. This scale should be based on
     * meters.
     *
     * @return The object space scale value
     */
    inline double ObjectSpaceScale() const {
        return this->osScale;
    }

    /**
     * Sets the object space bounding box.
     *
     * @param box The new object space bounding box
     */
    inline void SetObjectSpaceBBox(const vislib::math::Cuboid<float>& box) {
        this->osBBox = box;
        this->osBBoxValid = true;
        this->clipBoxValid = false;
    }

    /**
     * Sets the object space bounding box.
     *
     * @param left   The x-coordinate of the left/bottom/back point.
     * @param bottom The y-coordinate of the left/bottom/back point.
     * @param back   The z-coordinate of the left/bottom/back point.
     * @param right  The x-coordinate of the right/top/front point.
     * @param top    The y-coordinate of the right/top/front point.
     * @param front  The z-coordinate of the right/top/front point.
     */
    inline void SetObjectSpaceBBox(const float& left, const float& bottom, const float& back, const float& right,
        const float& top, const float& front) {
        this->osBBox.Set(left, bottom, back, right, top, front);
        this->osBBoxValid = true;
        this->clipBoxValid = false;
    }

    /**
     * Sets the object space clipping box.
     *
     * @param box The new object space clipping box
     */
    inline void SetObjectSpaceClipBox(const vislib::math::Cuboid<float>& box) {
        this->osClipBox = box;
        this->osClipBoxValid = true;
        this->clipBoxValid = false;
    }

    /**
     * Sets the object space clipping box.
     *
     * @param left   The x-coordinate of the left/bottom/back point.
     * @param bottom The y-coordinate of the left/bottom/back point.
     * @param back   The z-coordinate of the left/bottom/back point.
     * @param right  The x-coordinate of the right/top/front point.
     * @param top    The y-coordinate of the right/top/front point.
     * @param front  The z-coordinate of the right/top/front point.
     */
    inline void SetObjectSpaceClipBox(const float& left, const float& bottom, const float& back, const float& right,
        const float& top, const float& front) {
        this->osClipBox.Set(left, bottom, back, right, top, front);
        this->osClipBoxValid = true;
        this->clipBoxValid = false;
    }

    /**
     * Sets the object space scale value. This scale should be based on
     * meters.
     *
     * @param scale The new object space scale value
     */
    inline void SetObjectSpaceScale(double scale) {
        this->osScale = scale;
    }

    /**
     * Sets the world space bounding box.
     *
     * @param box The new world space bounding box
     */
    inline void SetWorldSpaceBBox(const vislib::math::Cuboid<float>& box) {
        this->wsBBox = box;
        this->wsBBoxValid = true;
        this->clipBoxValid = false;
    }

    /**
     * Sets the world space bounding box.
     *
     * @param left   The x-coordinate of the left/bottom/back point.
     * @param bottom The y-coordinate of the left/bottom/back point.
     * @param back   The z-coordinate of the left/bottom/back point.
     * @param right  The x-coordinate of the right/top/front point.
     * @param top    The y-coordinate of the right/top/front point.
     * @param front  The z-coordinate of the right/top/front point.
     */
    inline void SetWorldSpaceBBox(const float& left, const float& bottom, const float& back, const float& right,
        const float& top, const float& front) {
        this->wsBBox.Set(left, bottom, back, right, top, front);
        this->wsBBoxValid = true;
        this->clipBoxValid = false;
    }

    /**
     * Sets the world space clipping box.
     *
     * @param box The new world space bounding box
     */
    inline void SetWorldSpaceClipBox(const vislib::math::Cuboid<float>& box) {
        this->wsClipBox = box;
        this->wsClipBoxValid = true;
        this->clipBoxValid = false;
    }

    /**
     * Sets the world space clipping box.
     *
     * @param left   The x-coordinate of the left/bottom/back point.
     * @param bottom The y-coordinate of the left/bottom/back point.
     * @param back   The z-coordinate of the left/bottom/back point.
     * @param right  The x-coordinate of the right/top/front point.
     * @param top    The y-coordinate of the right/top/front point.
     * @param front  The z-coordinate of the right/top/front point.
     */
    inline void SetWorldSpaceClipBox(const float& left, const float& bottom, const float& back, const float& right,
        const float& top, const float& front) {
        this->wsClipBox.Set(left, bottom, back, right, top, front);
        this->wsClipBoxValid = true;
        this->clipBoxValid = false;
    }

    /**
     * Answer the world space bounding box
     *
     * @return The world space bounding box
     */
    inline const vislib::math::Cuboid<float>& WorldSpaceBBox() const {
        return this->wsBBox;
    }

    /**
     * Answer the world space clipping box
     *
     * @return The world space clipping box
     */
    inline const vislib::math::Cuboid<float>& WorldSpaceClipBox() const {
        return this->wsClipBox;
    }

    /**
     * Test for equality
     *
     * @param rhs The right hand side operand
     *
     * @return True if 'this' and 'rhs' are equal
     */
    bool operator==(const BoundingBoxes& rhs) const;

    /**
     * Assignment operator
     *
     * @param rhs The right hand side operand
     *
     * @return A reference to this.
     */
    BoundingBoxes& operator=(const BoundingBoxes& rhs);

private:
    /**
     * Calculate the composed clipping box.
     */
    void calcClipBox() const;

    /** The composed clipping box */
    mutable vislib::math::Cuboid<float> clipBox;

    /** The valid flag for the composed clipping box */
    mutable bool clipBoxValid;

    /** The object space bounding box */
    vislib::math::Cuboid<float> osBBox;

    /** The valid flag for the object space bounding box */
    bool osBBoxValid;

    /** The object space clipping box */
    vislib::math::Cuboid<float> osClipBox;

    /** The valid flag for the object space clipping box */
    bool osClipBoxValid;

    /**
     * The scale value for the object space. This scale should be based
     * on meters.
     */
    double osScale;

    /** The world space bounding box */
    vislib::math::Cuboid<float> wsBBox;

    /** The valid flag for the world space bounding box */
    bool wsBBoxValid;

    /** The world space clipping box */
    vislib::math::Cuboid<float> wsClipBox;

    /** The valid flag for the world space clipping box */
    bool wsClipBoxValid;
};

} // namespace megamol::core
