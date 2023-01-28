/*
 * ColourHSVf.h
 *
 * Copyright (C) 2015 by Sebastian Grottel, TU Dresden
 * Alle Rechte vorbehalten.
 */

#pragma once
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */

#include "vislib/graphics/ColourRGBAu8.h"
#include "vislib/math/mathfunctions.h"

namespace vislib::graphics {


/**
 * This class stores an HSV colour using floats
 */
class ColourHSVf {
public:
    /**
     * Ctor.
     * Sets all colour components to zero, resulting in black.
     */
    ColourHSVf();

    /**
     * Ctor
     *
     * @param h The hue colour component
     * @param s The saturation colour component
     * @param v The value colour component
     */
    ColourHSVf(float h, float s, float v);

    /**
     * Copy ctor
     *
     * @param src The object to clone from
     */
    ColourHSVf(const ColourHSVf& src);

    /**
     * Copy ctor
     *
     * @param src The object to clone from
     */
    ColourHSVf(const ColourRGBAu8& src);

    /** Dtor. */
    ~ColourHSVf();

    /**
     * Answer the Hue component
     *
     * @return The Hue component
     */
    inline float H() const {
        return this->comp[0];
    }

    /**
     * Answer the Saturation component
     *
     * @return The Saturation component
     */
    inline float S() const {
        return this->comp[1];
    }

    /**
     * Answer the Value component
     *
     * @return The Value component
     */
    inline float V() const {
        return this->comp[2];
    }

    /**
     * Sets the colour components
     *
     * @param h The hue colour component
     * @param s The saturation colour component
     * @param v The value colour component
     */
    inline void Set(float h, float s, float v) {
        SetH(h);
        SetS(s);
        SetV(v);
    }

    /**
     * Sets the hue component
     *
     * @param a The new value for the hue component
     */
    inline void SetH(float h) {
        h /= 360.0;
        h -= (long)h; // ]1 .. -1[
        h += 1.0f;
        h -= (long)h; // ]1 .. 0]
        this->comp[0] = h * 360.0f;
    }

    /**
     * Sets the saturation component
     *
     * @param a The new value for the saturation component
     */
    inline void SetS(float s) {
        this->comp[1] = vislib::math::Clamp(s, 0.0f, 1.0f);
    }

    /**
     * Sets the value component
     *
     * @param a The new value for the value component
     */
    inline void SetV(float v) {
        this->comp[2] = vislib::math::Clamp(v, 0.0f, 1.0f);
    }

    /**
     * Test for equality
     *
     * @param rhs The right hand side operand
     *
     * @return True if this and rhs are equal
     */
    bool operator==(const ColourHSVf& rhs) const;

    /**
     * Test for inequality
     *
     * @param rhs The right hand side operand
     *
     * @return False if this and rhs are equal
     */
    inline bool operator!=(const ColourHSVf& rhs) const {
        return !(*this == rhs);
    }

    /**
     * Assignment operator
     *
     * @param rhs The right hand side operand
     *
     * @return A reference to this
     */
    inline ColourHSVf& operator=(const ColourHSVf& rhs) {
        comp[0] = rhs.comp[0];
        comp[1] = rhs.comp[1];
        comp[2] = rhs.comp[2];
        return *this;
    }

    /** Cast operator to rgb color */
    operator ColourRGBAu8() const;

private:
    /** The colour components in HSV layout */
    float comp[3];
};

} // namespace vislib::graphics

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
