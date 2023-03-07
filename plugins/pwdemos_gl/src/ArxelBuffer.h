/*
 * ArxelBuffer.h
 *
 * Copyright (C) 2018 by VISUS (Universitaet Stuttgart).
 * Alle Rechte vorbehalten.
 */

#pragma once

#include "vislib/Array.h"
#include "vislib/IllegalStateException.h"
#include "vislib/math/Point.h"
#include "vislib/math/mathfunctions.h"

namespace megamol::demos_gl {

/**
 * Class holding a flat area buffer
 */
class ArxelBuffer {
public:
    /** Data type for our AREA-Pixel-Thingys types */
    typedef unsigned char ArxelType;

    /** Struct of initialization values */
    typedef struct _initvalues_t {

        /** The buffer width */
        unsigned int width;

        /** The buffer height */
        unsigned int height;

    } InitValues;

    /**
     * The initialization function
     *
     * @param buffer The buffer to initialize
     * @param state Will be set to zero
     * @param ctxt The initialization values
     */
    static void Initialize(ArxelBuffer& buffer, int& state, const InitValues& ctxt);

    /** Ctor */
    ArxelBuffer();

    /** Dtor */
    ~ArxelBuffer();

    /**
     * Answer the buffer data
     *
     * @return The buffer data
     */
    inline ArxelType* Data() {
        return this->data;
    }

    /**
     * Answer the buffer data
     *
     * @return The buffer data
     */
    inline const ArxelType* Data() const {
        return this->data;
    }

    /**
     * Answer the height of the buffer
     *
     * @return The height of the buffer
     */
    inline unsigned int Height() const {
        return this->height;
    }

    /**
     * Answer the width of the buffer
     *
     * @return The width of the buffer
     */
    inline unsigned int Width() const {
        return this->width;
    }

    /**
     * Fill the buffer via something Active-Edge-Table-like. The polygon surrounding the
     * area to be filled is closed implicitly from the last to the first vertex.
     *
     * @param polygon an array of unique border corners, ordered CW or CCW.
     * @param val     the value that is used for filling
     * @param dryRun  whether the fill is performed, or just the pixels are counted
     *
     * @return the number of pixels filled (i.e. the area).
     */
    UINT64 Fill(const vislib::Array<vislib::math::Point<int, 2>>& polygon, const ArxelType& val, bool dryRun = false);

    /**
     * Gets a value from the buffer
     *
     * @param x The x coordinate
     * @param y The y coordinate
     *
     * @return The value
     */
    inline const ArxelType& Get(int x, int y) const {
#if defined(DEBUG) || defined(_DEBUG)
        if (this->data == NULL) {
            throw vislib::IllegalStateException("Buffer is empty", __FILE__, __LINE__);
        }
#endif /* DEBUG || _DEBUG */
        if ((x < 0) || (x >= static_cast<int>(this->width))) {
            return borderXVal;
        } else if ((y < 0) || (y >= static_cast<int>(this->height))) {
            return borderYVal;
        } else {
            return this->data[vislib::math::UMod<int>(x, this->width) +
                              vislib::math::UMod<int>(y, this->height) * this->width];
        }
    }

    /**
     * Gets a value from the buffer
     *
     * @param p Point (x,y) to read
     *
     * @return The value
     */
    inline const ArxelType& Get(vislib::math::Point<int, 2> p) const {
        return this->Get(p.X(), p.Y());
    }

    /**
     * Gets the value that is returned beyond defined data.
     *
     * @return the value
     */
    inline const ArxelType& GetBorderX() const {
        return this->borderXVal;
    }

    /**
     * Gets the value that is returned beyond defined data.
     *
     * @return the value
     */
    inline const ArxelType& GetBorderY() const {
        return this->borderYVal;
    }

    /**
     * Draws a line in the buffer using boring Bresenham.
     *
     * @param x1  the start x coordinate
     * @param y1  the start y coordinate
     * @param x2  the end x coordinate
     * @param y2  the end y coordinate
     * @param val the value to draw with
     * @param dryRun whether to actually perform the drawing operation
     *
     * @return the number of set pixels
     */
    inline unsigned int Line(int x1, int y1, int x2, int y2, const ArxelType& val, bool dryRun = false) {
        int dx = x2 - x1;
        int dy = y2 - y1;
        int sdx = vislib::math::Signum<int>(dx);
        int sdy = vislib::math::Signum<int>(dy);
        int pdx, pdy, es, el;
        unsigned int pixelCount = 0;
        dx = vislib::math::Abs<int>(dx);
        dy = vislib::math::Abs<int>(dy);

        if (dx > dy) {
            pdx = sdx;
            pdy = 0;
            es = dy;
            el = dx;
        } else {
            pdx = 0;
            pdy = sdy;
            es = dx;
            el = dy;
        }

        int x = x1, y = y1;
        if (!dryRun)
            Set(x, y, val);
        pixelCount++;
        int err = el / 2;
        for (int t = 0; t < el; t++) {
            err -= es;
            if (err < 0) {
                err += el;
                x += sdx;
                y += sdy;
            } else {
                x += pdx;
                y += pdy;
            }
            if (!dryRun)
                Set(x, y, val);
            pixelCount++;
        }
        return pixelCount;
    }

    /**
     * Sets a value
     *
     * @param x The x coordinate
     * @param y The y coordinate
     * @param val The new value
     */
    inline void Set(int x, int y, const ArxelType& val) {
#if defined(DEBUG) || defined(_DEBUG)
        if (this->data == NULL) {
            throw vislib::IllegalStateException("Buffer is empty", __FILE__, __LINE__);
        }
#endif /* DEBUG || _DEBUG */
        this->data[vislib::math::UMod<int>(x, this->width) + vislib::math::UMod<int>(y, this->height) * this->width] =
            val;
    }

    /**
     * Sets a value
     *
     * @param p Point (x,y) to read
     *
     * @return The value
     */
    inline void Set(vislib::math::Point<int, 2> p, const ArxelType& val) {
        this->Set(p.X(), p.Y(), val);
    }

    /**
     * Sets the value that is returned beyond defined data.
     *
     * @param valX the new value
     * @param valY the other new value
     */
    inline void SetBorders(const ArxelType& valX, const ArxelType& valY) {
        this->borderXVal = valX;
        this->borderYVal = valY;
    }

private:
    /** The width of the buffer */
    unsigned int width;

    /** The height of the buffer */
    unsigned int height;

    /** The buffer data */
    ArxelType* data;

    /** The value any access beyond the defined data yields. Defaults to 2. */
    ArxelType borderXVal;

    /** The value any access beyond the defined data yields. Defaults to 3. */
    ArxelType borderYVal;
};

} // namespace megamol::demos_gl
