/*
 * ExtBezierDataCall.h
 *
 * Copyright (C) 2010 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#ifndef BEZTUBE_EXTBEZIERDATACALL_H_INCLUDED
#define BEZTUBE_EXTBEZIERDATACALL_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#define _USE_MATH_DEFINES
#include <cmath>

#include "beztube/beztube.h"
#include "mmcore/AbstractGetData3DCall.h"
#include "mmcore/factories/CallAutoDescription.h"
#include "vislib/math/BezierCurve.h"
#include "vislib/graphics/ColourRGBAu8.h"
#include "vislib/forceinline.h"
#include "vislib/math/mathfunctions.h"
#include "vislib/math/Point.h"
#include "vislib/math/Quaternion.h"
#include "vislib/math/Vector.h"
#include "vislib/deprecated.h"


namespace megamol {
namespace beztube {
namespace ext {


    /**
     * Call for extended b�zier curve data.
     * This call transports an flat array of cubic b�zier curves.
     */
    class BEZTUBE_API ExtBezierDataCall : public core::AbstractGetData3DCall {
    public:

        /**
         * Class for control points of the bezier curve
         */
        class BEZTUBE_API Point {
        public:

            /**
             * Ctor.
             */
            Point(void) : pos(0.0f, 0.0f, 0.0f), y(0.0f, 1.0f, 0.0f),
                    radY(0.1f), radZ(0.1f), col(192, 192, 192, 255) {
                // intentionally empty
            }

            /**
             * Ctor.
             *
             * @param x The x coordinate of position
             * @param y The y coordinate of position
             * @param z The z coordinate of position
             * @param yx The x coordinate of the y vector
             * @param yy The y coordinate of the y vector
             * @param yz The z coordinate of the y vector
             * @param ry The radius in y direction
             * @param rz The radius in z direction
             * @param c The colour code
             */
            Point(float x, float y, float z, float yx, float yy, float yz,
                    float ry, float rz, unsigned char cr,
                    unsigned char cg, unsigned char cb) : pos(x, y, z),
                    y(yx, yy, yz), radY(ry), radZ(rz), col(cr, cg, cb, 255) {
                // intentionally empty
            }

            /**
             * Ctor.
             *
             * @param pos The position
             * @param y The y vector
             * @param ry The radius in y direction
             * @param rz The radius in z direction
             * @param c The colour
             */
            Point(const vislib::math::Point<float, 3> &pos,
                    const vislib::math::Vector<float, 3> &y,
                    float ry, float rz,
                    const vislib::graphics::ColourRGBAu8& c) : pos(pos),
                    y(y), radY(ry), radZ(rz), col(c) {
                // intentionally empty
            }

            /**
             * Copy ctor.
             *
             * @param src The object to clone from
             */
            Point(const Point& src) : pos(src.pos), y(src.y),
                    radY(src.radY), radZ(src.radZ), col(src.col) {
                // intentionally empty
            }

            /**
             * Dtor.
             */
            ~Point(void) {
                // intentionally empty
            }

            /**
             * Answer the position
             *
             * @return The position
             */
            inline const vislib::math::Point<float, 3>& GetPosition(void) const {
                return this->pos;
            }

            /**
             * Answer the orientation quaternion
             *
             * @return The orienatation quaternion
             */
            inline const vislib::math::Vector<float, 3>& GetY(void) const {
                return this->y;
            }

            /**
             * Answer the radius in y direction
             *
             * @return The radius in y direction
             */
            inline float GetRadiusY(void) const {
                return this->radY;
            }

            /**
             * Answer the radius in z direction
             *
             * @return The radius in z direction
             */
            inline float GetRadiusZ(void) const {
                return this->radZ;
            }

            /**
             * Answer the colour
             *
             * @return The colour
             */
            inline const vislib::graphics::ColourRGBAu8& GetColour(void) const {
                return this->col;
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
            Point Interpolate(const Point& rhs, float t) const {
                Point rv;

                rv.pos = this->pos.Interpolate(rhs.pos, t);

                //rv.y = this->y * (1.0f - t) + rhs.y * t; // could be slerp!
                if (this->y.IsParallel(rhs.y)) {
                    rv.y = this->y;
                    vislib::math::Vector<float, 3> a(rhs.y);
                    float l1 = rv.y.Normalise();
                    float l2 = a.Normalise();

                    if (rv.y == a) {
                        rv.y *= (l1 * (1.0f - t) + l2 * t);

                    } else { // rotate 180�
                        a.Set(1.0f, 0.0f, 0.0f);
                        if (rv.y.IsParallel(a)) {
                            a.Set(0.0f, 1.0f, 0.0);
                        }
                        a = rv.y.Cross(a);
                        a.Normalise();
                        vislib::math::Quaternion<float> rot(t * static_cast<float>(M_PI), a);
                        rv.y = rot * rv.y;
                        rv.y *= (l1 * (1.0f - t) + l2 * t);

                    }

                } else {
                    float l1 = this->y.Length();
                    float l2 = rhs.y.Length();
                    rv.y = this->y * (1.0f - t) + rhs.y * t;
                    rv.y.Normalise();
                    rv.y *= (l1 * (1.0f - t) + l2 * t);

                }

                rv.radY = this->radY * (1.0f - t) + rhs.radY * t;
                rv.radZ = this->radZ * (1.0f - t) + rhs.radZ * t;
                rv.col = this->col.Interpolate(rhs.col, t);

                return rv;
            }

            /**
             * Sets all attributes
             *
             * @param x The x coordinate of position
             * @param y The y coordinate of position
             * @param z The z coordinate of position
             * @param yx The x coordinate of the y vector
             * @param yy The y coordinate of the y vector
             * @param yz The z coordinate of the y vector
             * @param ry The radius in y direction
             * @param rz The radius in z direction
             * @param c The colour code
             */
            inline void Set(float x, float y, float z, float yx, float yy,
                    float yz, float ry, float rz, unsigned char cr,
                    unsigned char cg, unsigned char cb) {
                this->pos.Set(x, y, z);
                this->y.Set(yx, yy, yz);
                this->radY = ry;
                this->radZ = rz;
                this->col.Set(cr, cg, cb, 255);
            }

            /**
             * Sets all attributes
             *
             * @param pos The position
             * @param y The y Vector
             * @param ry The radius in y direction
             * @param rz The radius in z direction
             * @param c The colour
             */
            inline void Set(const vislib::math::Point<float, 3> &pos,
                    const vislib::math::Vector<float, 3> &y,
                    float ry, float rz,
                    const vislib::graphics::ColourRGBAu8& c) {
                this->pos = pos;
                this->y = y;
                this->radY = ry;
                this->radZ = rz;
                this->col = c;
            }

            /**
             * Sets the position
             *
             * @param pos The position
             */
            inline void SetPosition(vislib::math::Point<float, 3>& pos) {
                this->pos = pos;
            }

            /**
             * Sets the y vector
             *
             * @param ori The y vector
             */
            inline void SetY(vislib::math::Vector<float, 3>& y) {
                this->y = y;
            }

            /**
             * Sets the radius in y direction
             *
             * @param ry The radius in y direction
             */
            inline void SetRadiusY(float ry) {
                this->radY = ry;
            }

            /**
             * Sets the radius in z direction
             *
             * @param rz The radius in z direction
             */
            inline void SetRadiusZ(float rz) {
                this->radZ = rz;
            }

            /**
             * Sets the colour
             *
             * @param col The colour
             */
            inline void SetColour(const vislib::graphics::ColourRGBAu8& col) {
                this->col = col;
            }

            /**
             * Test for equality
             *
             * @param rhs The right hand side operand
             *
             * @return 'true' if 'this' and 'rhs' are equal, 'false' if not
             */
            bool operator==(const Point& rhs) const {
                return (this->pos == rhs.pos)
                    && (this->y == rhs.y)
                    && (this->radY == rhs.radY)
                    && (this->radZ == rhs.radZ)
                    && (this->col == rhs.col);
            }

            /**
             * Test for inequality
             *
             * @param rhs The right hand side operand
             *
             * @return 'true' if 'this' and 'rhs' are not equal
             */
            bool operator!=(const Point& rhs) const {
                return (this->pos != rhs.pos)
                    || (this->y != rhs.y)
                    || (this->radY != rhs.radY)
                    || (this->radZ != rhs.radZ)
                    || (this->col != rhs.col);
            }

            /**
             * Assignment operator
             *
             * @param rhs The right hand side operand
             *
             * @return Reference to 'this'
             */
            Point& operator=(const Point& rhs) {
                this->pos = rhs.pos;
                this->y = rhs.y;
                this->radY = rhs.radY;
                this->radZ = rhs.radZ;
                this->col = rhs.col;
                return *this;
            }

            /**
             * Calculates the difference vector of the positions of 'this' and
             * 'rhs'.
             *
             * @param rhs The right hand side operand
             *
             * @return The difference vector of the positions
             */
            vislib::math::Vector<float, 3>
            operator-(const Point& rhs) const {
                return this->pos - rhs.pos;
            }

        private:

#ifdef _WIN32
#pragma warning (disable: 4251)
#endif /* _WIN32 */
            /** The position */
            vislib::math::Point<float, 3> pos;

            /** The orientation */
            vislib::math::Vector<float, 3> y;

            /** The radii */
            float radY, radZ;

            /** The colour code */
            vislib::graphics::ColourRGBAu8 col;
#ifdef _WIN32
#pragma warning (default: 4251)
#endif /* _WIN32 */

        };

        /**
         * Answer the name of the objects of this description.
         *
         * @return The name of the objects of this description.
         */
        static const char *ClassName(void) {
            return "ExtBezierDataCall";
        }

        /**
         * Gets a human readable description of the module.
         *
         * @return A human readable description of the module.
         */
        static const char *Description(void) {
            return "Call to get extended bezier data";
        }

        /**
         * Answer the number of functions used for this call.
         *
         * @return The number of functions used for this call.
         */
        static unsigned int FunctionCount(void) {
            return core::AbstractGetData3DCall::FunctionCount();
        }

        /**
         * Answer the name of the function used for this call.
         *
         * @param idx The index of the function to return it's name.
         *
         * @return The name of the requested function.
         */
        static const char * FunctionName(unsigned int idx) {
            return core::AbstractGetData3DCall::FunctionName(idx);
        }

        /** Ctor. */
        ExtBezierDataCall(void);

        /** Dtor. */
        virtual ~ExtBezierDataCall(void);

        /**
         * Answer the number of b�zier curves.
         *
         * @return The number of b�zier curves
         */
        VLDEPRECATED
        VISLIB_FORCEINLINE unsigned int CountElliptic(void) const {
            return this->cntEllip;
        }

        /**
         * Answer the number of b�zier curves.
         *
         * @return The number of b�zier curves
         */
        VLDEPRECATED
        VISLIB_FORCEINLINE unsigned int CountRectangular(void) const {
            return this->cntRect;
        }

        /**
         * Answer the b�zier curves. Might be NULL! Do not delete the returned
         * memory.
         *
         * @return The b�zier curves
         */
        VLDEPRECATED
        VISLIB_FORCEINLINE const vislib::math::BezierCurve<Point, 3> *
        EllipticCurves(void) const {
            return this->ellipCurves;
        }

        /**
         * Answer the b�zier curves. Might be NULL! Do not delete the returned
         * memory.
         *
         * @return The b�zier curves
         */
        VLDEPRECATED
        VISLIB_FORCEINLINE const vislib::math::BezierCurve<Point, 3> *
        RectangularCurves(void) const {
            return this->rectCurves;
        }

        /**
         * Sets the data. The object will not take ownership of the memory
         * 'curves' points to. The caller is responsible for keeping the data
         * valid as long as it is used.
         *
         * @param cntEllip The number of b�zier curves stored in 'curves'
         *                 with elliptic profile
         * @param cntRect The number of b�zier curves stored in 'curves'
         *                with rectangular profile
         * @param ellipCurves Pointer to a flat array of b�zier curves
         *                    with elliptic profile
         * @param rectCurves Pointer to a flat array of b�zier curves
         *                   with rectangular profile
         */
        void SetData(unsigned int cntEllip, unsigned int cntRect,
                const vislib::math::BezierCurve<Point, 3> *ellipCurves,
                const vislib::math::BezierCurve<Point, 3> *rectCurves);

        /**
         * Assignment operator.
         * Makes a deep copy of all members. While for data these are only
         * pointers, the pointer to the unlocker object is also copied.
         *
         * @param rhs The right hand side operand
         *
         * @return A reference to this
         */
        ExtBezierDataCall& operator=(const ExtBezierDataCall& rhs);

    private:

        /** Number of curves with elliptic profile */
        unsigned int cntEllip;

        /** Number of curves with rectangular profile */
        unsigned int cntRect;

        /** Cubic b�zier curves with elliptic profile */
        const vislib::math::BezierCurve<Point, 3> *ellipCurves;

        /** Cubic b�zier curves with rectangular profile */
        const vislib::math::BezierCurve<Point, 3> *rectCurves;

    };

    /** Description class typedef */
    typedef core::factories::CallAutoDescription<ExtBezierDataCall> ExtBezierDataCallDescription;


} /* end namespace ext */
} /* end namespace beztube */
} /* end namespace megamol */

#endif /* BEZTUBE_EXTBEZIERDATACALL_H_INCLUDED */
