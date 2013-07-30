/*
 * BezierDataCall.h
 *
 * Copyright (C) 2009 by Universitaet Stuttgart (VISUS). 
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCORE_BEZIERDATACALL_H_INCLUDED
#define MEGAMOLCORE_BEZIERDATACALL_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "api/MegaMolCore.std.h"
#include "AbstractGetData3DCall.h"
#include "CallAutoDescription.h"
#include "vislib/BezierCurve.h"
#include "vislib/forceinline.h"
#include "vislib/mathfunctions.h"
#include "vislib/Point.h"
#include "vislib/Vector.h"

#include "vislib/deprecated.h"


namespace megamol {
namespace core {
namespace misc {


    /**
     * Call for bézier curve data.
     * This call transports an flat array of cubic bézier curves. The control
     * points of each curve define the position (x, y, z) and radius (r) value
     * for the curve.
     */
    class MEGAMOLCORE_API BezierDataCall : public AbstractGetData3DCall {
    public:

        /**
         * Class for control points of the bezier curve
         */
        class BezierPoint {
        public:

            /**
             * Ctor.
             */
            BezierPoint(void) : pos(0.0f, 0.0f, 0.0f), rad(0.1f), r(192),
                    g(192), b(192) {
                // intentionally empty
            }

            /**
             * Ctor.
             *
             * @param x The x coordinate
             * @param y The y coordinate
             * @param z The z coordinate
             * @param rad The radius
             * @param r The red colour component
             * @param g The red colour component
             * @param b The red colour component
             */
            BezierPoint(float x, float y, float z, float rad,
                    unsigned char r, unsigned char g, unsigned char b)
                    : pos(x, y, z), rad(rad), r(r), g(g), b(b) {
                // intentionally empty
            }

            /**
             * Copy ctor.
             *
             * @param src The object to clone from
             */
            BezierPoint(const BezierPoint& src) : pos(src.pos), rad(src.rad),
                    r(src.r), g(src.g), b(src.b) {
                // intentionally empty
            }

            /**
             * Dtor.
             */
            ~BezierPoint(void) {
                // intentionally empty
            }

            /**
             * Answer the blue colour component
             *
             * @return The blue colour component
             */
            unsigned char B(void) const {
                return this->b;
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
            BezierPoint Interpolate(const BezierPoint& rhs, float t) const {
                BezierPoint rv;
                float i = 1.0f - t;
                rv.pos = this->pos.Interpolate(rhs.pos, t);
                rv.rad = this->rad * i + rhs.rad * t;
                float fr = static_cast<float>(this->r) * i
                    + static_cast<float>(rhs.r) * t;
                float fg = static_cast<float>(this->g) * i
                    + static_cast<float>(rhs.g) * t;
                float fb = static_cast<float>(this->b) * i
                    + static_cast<float>(rhs.b) * t;
                if (fr < 0.0f) fr = 0.0f; else
                if (fr > 255.0f) fr = 255.0f;
                if (fg < 0.0f) fg = 0.0f; else
                if (fg > 255.0f) fg = 255.0f;
                if (fb < 0.0f) fb = 0.0f; else
                if (fb > 255.0f) fb = 255.0f;
                rv.r = static_cast<unsigned char>(fr);
                rv.g = static_cast<unsigned char>(fg);
                rv.b = static_cast<unsigned char>(fb);
                return rv;
            }

            /**
             * Answer the green colour component
             *
             * @return The green colour component
             */
            unsigned char G(void) const {
                return this->g;
            }

            /**
             * Answer the position
             *
             * @return The position
             */
            const vislib::math::Point<float, 3>& Position(void) const {
                return this->pos;
            }

            /**
             * Answer the red colour component
             *
             * @return The red colour component
             */
            unsigned char R(void) const {
                return this->r;
            }

            /**
             * Answer the radius
             *
             * @return The radius
             */
            float Radius(void) const {
                return this->rad;
            }

            /**
             * Sets all values
             *
             * @param x The x coordinate
             * @param y The y coordinate
             * @param z The z coordinate
             * @param rad The radius
             * @param r The red colour component
             * @param g The red colour component
             * @param b The red colour component
             */
            void Set(float x, float y, float z, float rad,
                    unsigned char r, unsigned char g, unsigned char b) {
                this->pos.Set(x, y, z);
                this->rad = rad;
                this->r = r;
                this->g = g;
                this->b = b;
            }

            /**
             * Sets all values
             *
             * @param pos The position
             * @param rad The radius
             * @param r The red colour component
             * @param g The red colour component
             * @param b The red colour component
             */
            void Set(const vislib::math::Point<float, 3> &pos, float rad,
                    unsigned char r, unsigned char g, unsigned char b) {
                this->pos = pos;
                this->rad = rad;
                this->r = r;
                this->g = g;
                this->b = b;
            }

            /**
             * Sets the blue colour component
             *
             * @param b The new value
             */
            void SetB(unsigned char b) {
                this->b = b;
            }

            /**
             * Sets the green colour component
             *
             * @param g The new value
             */
            void SetG(unsigned char g) {
                this->g = g;
            }

            /**
             * Sets the red colour component
             *
             * @param r The new value
             */
            void SetR(unsigned char r) {
                this->r = r;
            }

            /**
             * Sets the radius
             *
             * @param rad The new value
             */
            void SetRadius(float rad) {
                this->rad = rad;
            }

            /**
             * Sets the position
             *
             * @param pos The new value
             */
            void SetPosition(const vislib::math::Point<float, 3>& pos) {
                this->pos = pos;
            }

            /**
             * Sets the position
             *
             * @param x The new x coordinate
             * @param y The new y coordinate
             * @param z The new z coordinate
             */
            void SetPosition(float x, float y, float z) {
                this->pos.Set(x, y, z);
            }

            /**
             * Answer the x coordinate
             *
             * @return The x coordinate
             */
            float X(void) const {
                return this->pos.X();
            }

            /**
             * Answer the y coordinate
             *
             * @return The y coordinate
             */
            float Y(void) const {
                return this->pos.Y();
            }

            /**
             * Answer the z coordinate
             *
             * @return The z coordinate
             */
            float Z(void) const {
                return this->pos.Z();
            }

            /**
             * Test for equality
             *
             * @param rhs The right hand side operand
             *
             * @return 'true' if 'this' and 'rhs' are equal, 'false' if not
             */
            bool operator==(const BezierPoint& rhs) const {
                return (this->pos == rhs.pos)
                    && vislib::math::IsEqual(this->rad, rhs.rad)
                    && (this->r == rhs.r)
                    && (this->g == rhs.g)
                    && (this->b == rhs.b);
            }

            /**
             * Test for inequality
             *
             * @param rhs The right hand side operand
             *
             * @return 'true' if 'this' and 'rhs' are not equal
             */
            bool operator!=(const BezierPoint& rhs) const {
                return (this->pos != rhs.pos)
                    || !vislib::math::IsEqual(this->rad, rhs.rad)
                    || (this->r != rhs.r)
                    || (this->g != rhs.g)
                    || (this->b != rhs.b);
            }

            /**
             * Assignment operator
             *
             * @param rhs The right hand side operand
             *
             * @return Reference to 'this'
             */
            BezierPoint& operator=(const BezierPoint& rhs) {
                this->pos = rhs.pos;
                this->rad = rhs.rad;
                this->r = rhs.r;
                this->g = rhs.g;
                this->b = rhs.b;
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
            operator-(const BezierPoint& rhs) const {
                return this->pos - rhs.pos;
            }

        private:

            /** The position */
            vislib::math::Point<float, 3> pos;

            /** The radius */
            float rad;

            /** The colour */
            unsigned char r, g, b;

        };

        /**
         * Answer the name of the objects of this description.
         *
         * @return The name of the objects of this description.
         */
        static const char *ClassName(void) {
            return "BezierDataCall";
        }

        /**
         * Gets a human readable description of the module.
         *
         * @return A human readable description of the module.
         */
        static const char *Description(void) {
            return "Call to get bezier data";
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
         * @param idx The index of the function to return it's name.
         *
         * @return The name of the requested function.
         */
        static const char * FunctionName(unsigned int idx) {
            return AbstractGetData3DCall::FunctionName(idx);
        }

        /** Ctor. */
        BezierDataCall(void);

        /** Dtor. */
        virtual ~BezierDataCall(void);

        /**
         * Answer the number of bézier curves.
         *
         * @return The number of bézier curves
         */
        VLDEPRECATED
        VISLIB_FORCEINLINE unsigned int Count(void) const {
            return this->count;
        }

        /**
         * Answer the bézier curves. Might be NULL! Do not delete the returned
         * memory.
         *
         * @return The bézier curves
         */
        VLDEPRECATED
        VISLIB_FORCEINLINE const vislib::math::BezierCurve<BezierPoint, 3> *
        Curves(void) const {
            return this->curves;
        }

        /**
         * Sets the data. The object will not take ownership of the memory
         * 'curves' points to. The caller is responsible for keeping the data
         * valid as long as it is used.
         *
         * @param count The number of bézier curves stored in 'curves'
         * @param curves Pointer to a flat array of bézier curves.
         */
        VLDEPRECATED
        void SetData(unsigned int count,
                const vislib::math::BezierCurve<BezierPoint, 3> *curves);

        /**
         * Assignment operator.
         * Makes a deep copy of all members. While for data these are only
         * pointers, the pointer to the unlocker object is also copied.
         *
         * @param rhs The right hand side operand
         *
         * @return A reference to this
         */
        BezierDataCall& operator=(const BezierDataCall& rhs);

    private:

        /** Number of curves */
        unsigned int count;

        /** Cubic bézier curves */
        const vislib::math::BezierCurve<BezierPoint, 3> *curves;

    };

    /** Description class typedef */
    typedef CallAutoDescription<BezierDataCall> BezierDataCallDescription;


} /* end namespace misc */
} /* end namespace core */
} /* end namespace megamol */

#endif /* MEGAMOLCORE_BEZIERDATACALL_H_INCLUDED */
