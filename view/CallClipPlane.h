/*
 * CallClipPlane.h
 *
 * Copyright (C) 2009 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCORE_CALLCLIPPLANE_H_INCLUDED
#define MEGAMOLCORE_CALLCLIPPLANE_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "Call.h"
#include "CallAutoDescription.h"
#include "vislib/Plane.h"


namespace megamol {
namespace core {
namespace view {


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
        static const char *ClassName(void) {
            return "CallClipPlane";
        }

        /**
         * Gets a human readable description of the module.
         *
         * @return A human readable description of the module.
         */
        static const char *Description(void) {
            return "Call for a clipping plane";
        }

        /**
         * Answer the number of functions used for this call.
         *
         * @return The number of functions used for this call.
         */
        static unsigned int FunctionCount(void) {
            return 1;
        }

        /**
         * Answer the name of the function used for this call.
         *
         * @param idx The index of the function to return it's name.
         *
         * @return The name of the requested function.
         */
        static const char * FunctionName(unsigned int idx) {
            switch (idx) {
                case 0: return "GetPlane";
                default: return NULL;
            }
        }

        /** Ctor. */
        CallClipPlane(void);

        /** Dtor. */
        virtual ~CallClipPlane(void);

        /**
         * Gets the colour of the clipping plane
         *
         * @return Pointer to 3 bytes holding the colour of the clipping plane
         */
        inline const unsigned char * GetColour(void) const {
            return this->col;
        }

        /**
         * Gets the clipping plane
         *
         * @return The clipping plane
         */
        inline const vislib::math::Plane<float>& GetPlane(void) const {
            return this->plane;
        }

        /**
         * Sets the colour of the clipping plane
         *
         * @param r The red colour component
         * @param g The red colour component
         * @param b The red colour component
         */
        inline void SetColour(unsigned char r, unsigned char g, unsigned char b) {
            this->col[0] = r;
            this->col[1] = g;
            this->col[2] = b;
        }

        /**
         * Sets the clipping plane
         *
         * @param plane The new clipping plane
         */
        void SetPlane(const vislib::math::Plane<float>& plane);

    private:

        /** The clipping plane */
        vislib::math::Plane<float> plane;

        /** The colour of the plane */
        unsigned char col[3];

    };


    /** Description class typedef */
    typedef CallAutoDescription<CallClipPlane> CallClipPlaneDescription;


} /* end namespace view */
} /* end namespace core */
} /* end namespace megamol */

#endif /* MEGAMOLCORE_CALLCLIPPLANE_H_INCLUDED */
