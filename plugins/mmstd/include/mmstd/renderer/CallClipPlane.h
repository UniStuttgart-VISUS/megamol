/**
 * MegaMol
 * Copyright (c) 2009, MegaMol Dev Team
 * All rights reserved.
 */

#pragma once

#include "mmcore/Call.h"
#include "mmcore/factories/CallAutoDescription.h"
#include "vislib/math/Plane.h"
#include "vislib/math/Vector.h"

namespace megamol::core::view {

/**
 * Call transporting a clipping plane. Only data in the positive
 * halfspace should be visible.
 */
class CallClipPlane : public Call {
public:
    /**
     * Answer the name of the objects of this description.
     *
     * @return The name of the objects of this description.
     */
    static const char* ClassName() {
        return "CallClipPlane";
    }

    /**
     * Gets a human readable description of the module.
     *
     * @return A human readable description of the module.
     */
    static const char* Description() {
        return "Call for a clipping plane";
    }

    /**
     * Answer the number of functions used for this call.
     *
     * @return The number of functions used for this call.
     */
    static unsigned int FunctionCount() {
        return 1;
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
            return "GetPlane";
        default:
            return NULL;
        }
    }

    /** Ctor. */
    CallClipPlane();

    /** Dtor. */
    virtual ~CallClipPlane();

    /**
     * Calculates the coordinate system of the clipping plane
     *
     * @param outX the vector to receive the x axis
     * @param outY the vector to receive the y axis
     */
    inline void CalcPlaneSystem(vislib::math::Vector<float, 3>& outX, vislib::math::Vector<float, 3>& outY) const {
        vislib::math::Vector<float, 3> z;
        this->CalcPlaneSystem(outX, outY, z);
    }

    /**
     * Calculates the coordinate system of the clipping plane
     *
     * @param outX the vector to receive the x axis
     * @param outY the vector to receive the y axis
     * @param outZ the vector to receive the z axis (plane normal)
     */
    void CalcPlaneSystem(vislib::math::Vector<float, 3>& outX, vislib::math::Vector<float, 3>& outY,
        vislib::math::Vector<float, 3>& outZ) const;

    /**
     * Gets the colour of the clipping plane
     *
     * @return Pointer to 4 bytes holding the colour of the clipping plane
     */
    inline const unsigned char* GetColour() const {
        return this->col;
    }

    /**
     * Gets the clipping plane
     *
     * @return The clipping plane
     */
    inline const vislib::math::Plane<float>& GetPlane() const {
        return this->plane;
    }

    /**
     * Sets the colour of the clipping plane
     *
     * @param r The red colour component
     * @param g The green colour component
     * @param b The blue colour component
     * @param a The alpha component
     */
    inline void SetColour(unsigned char r, unsigned char g, unsigned char b, unsigned char a = 255) {
        this->col[0] = r;
        this->col[1] = g;
        this->col[2] = b;
        this->col[3] = a;
    }

    /**
     * Sets the clipping plane
     *
     * @param plane The new clipping plane
     */
    void SetPlane(const vislib::math::Plane<float>& plane);

private:
#ifdef _WIN32
#pragma warning(disable : 4251)
#endif /* _WIN32 */
    /** The clipping plane */
    vislib::math::Plane<float> plane;
#ifdef _WIN32
#pragma warning(default : 4251)
#endif /* _WIN32 */

    /** The colour of the plane */
    unsigned char col[4];
};


/** Description class typedef */
typedef factories::CallAutoDescription<CallClipPlane> CallClipPlaneDescription;

} // namespace megamol::core::view
