/*
 * ColourRGBAu8.h
 *
 * Copyright (C) 2006 - 2010 by Visualisierungsinstitut Universitaet Stuttgart.
 * Alle Rechte vorbehalten.
 */

#pragma once
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */


namespace vislib::graphics {


/**
 * This class stores an RGBA colour using 8-bit unsigned int (bytes)
 *
 * The class should be replaced by a proper instantiated template alias,
 * as soon as the colour template is available.
 * Template will allow for other colour spaces, colours without alpha
 * channel, different types (float), shallow colours and conversions.
 * ColourSpace, ColourSpaceLayout, and ColourChannelTraits
 * should do the job.
 */
class ColourRGBAu8 {
public:
    /**
     * Ctor.
     * Sets all colour components to zero and alpha to one (255),
     * resulting in black.
     */
    ColourRGBAu8();

    /**
     * Ctor
     *
     * @param r The red colour component
     * @param g The green colour component
     * @param b The blue colour component
     * @param a The alpha component
     */
    ColourRGBAu8(unsigned char r, unsigned char g, unsigned char b, unsigned char a);

    /**
     * Copy ctor
     *
     * @param src The object to clone from
     */
    ColourRGBAu8(const ColourRGBAu8& src);

    /** Dtor. */
    ~ColourRGBAu8();

    /**
     * Answer the alpha component
     *
     * @return The alpha component
     */
    inline unsigned char A() const {
        return this->comp[3];
    }

    /**
     * Answer the blue colour component
     *
     * @return The blue colour component
     */
    inline unsigned char B() const {
        return this->comp[2];
    }

    /**
     * Answer the green colour component
     *
     * @return The green colour component
     */
    inline unsigned char G() const {
        return this->comp[1];
    }

    /**
     * Interpolates between 'this' and 'rhs' linearly based on
     * '0 <= t <= 1'.
     *
     * @param rhs The second point to interpolate to (t=1)
     * @param t The interpolation value (0..1)
     *
     * @return The interpolation result
     */
    ColourRGBAu8 Interpolate(const ColourRGBAu8& rhs, float t) const;

    /**
     * Answer a pointer to all components
     *
     * @return A pointer to all components
     */
    inline const unsigned char* PeekComponents() const {
        return this->comp;
    }

    /**
     * Answer the red colour component
     *
     * @return The red colour component
     */
    inline unsigned char R() const {
        return this->comp[0];
    }

    /**
     * Sets the colour components
     *
     * @param r The red colour component
     * @param g The green colour component
     * @param b The blue colour component
     * @param a The alpha component
     */
    void Set(unsigned char r, unsigned char g, unsigned char b, unsigned char a);

    /**
     * Sets the alpha component
     *
     * @param a The new value for the alpha component
     */
    inline void SetA(unsigned char a) {
        this->comp[3] = a;
    }

    /**
     * Sets the blue colour component
     *
     * @param b The new value for the blue colour component
     */
    inline void SetB(unsigned char b) {
        this->comp[2] = b;
    }

    /**
     * Sets the green colour component
     *
     * @param g The new value for the green colour component
     */
    inline void SetG(unsigned char g) {
        this->comp[1] = g;
    }

    /**
     * Sets the red colour component
     *
     * @param r The new value for the red colour component
     */
    inline void SetR(unsigned char r) {
        this->comp[0] = r;
    }

    /**
     * Test for equality
     *
     * @param rhs The right hand side operand
     *
     * @return True if this and rhs are equal
     */
    bool operator==(const ColourRGBAu8& rhs) const;

    /**
     * Test for inequality
     *
     * @param rhs The right hand side operand
     *
     * @return False if this and rhs are equal
     */
    inline bool operator!=(const ColourRGBAu8& rhs) const {
        return !(*this == rhs);
    }

    /**
     * Assignment operator
     *
     * @param rhs The right hand side operand
     *
     * @return A reference to this
     */
    ColourRGBAu8& operator=(const ColourRGBAu8& rhs);

private:
    /** The colour components in RGBA layout */
    unsigned char comp[4];
};

} // namespace vislib::graphics

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
