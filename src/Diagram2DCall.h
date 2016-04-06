/*
 * Diagram2DCall.h
 *
 * Author: Michael Krone
 * Copyright (C) 2010 by Universitaet Stuttgart (VISUS).
 * All rights reserved.
 */


#ifndef MEGAMOL_PROTEIN_DIAGRAM2DCALL_H_INCLUDED
#define MEGAMOL_PROTEIN_DIAGRAM2DCALL_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "mmcore/Call.h"
#include "mmcore/CallAutoDescription.h"
#include "vislib/IllegalParamException.h"
#include "vislib/math/Cuboid.h"
#include "vislib/math/Vector.h"
#include <vector>
#include "vislib/graphics/gl/IncludeAllGL.h"

namespace megamol {
namespace protein_cuda {

    /**
     * Base class of rendering graph calls and data interfaces for volume data.
     */

    class Diagram2DCall : public megamol::core::Call {
    public:

        /**
         * Answer the name of the objects of this description.
         *
         * @return The name of the objects of this description.
         */
        static const char *ClassName(void) {
            return "Diagram2DCall";
        }

        /**
         * Gets a human readable description of the module.
         *
         * @return A human readable description of the module.
         */
        static const char *Description(void) {
            return "Call to get 2D diagram data";
        }

        /** Index of the 'GetData' function */
        static const unsigned int CallForGetData;

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
        static const char* FunctionName(unsigned int idx) {
            return "GetData";
        }

        /** Ctor. */
        Diagram2DCall(void);

        /** Dtor. */
        virtual ~Diagram2DCall(void);

        /**
         * Set the range of the diagram.
         * Note that the diagram will be reset when this function is called!
         *
         * @param x The range for the x axis
         * @param y The range for the y axis
         */
        inline void SetRange( float x, float y) {
            xRange = x;
            yRange = y;
        };

        /**
         * Get the range of values for the x axis.
         * 
         * @return The value range
         */
        inline float GetRangeX() { return xRange; };

        /**
         * Get the range of values for the y axis.
         * 
         * @return The value range
         */
        inline float GetRangeY() { return yRange; };

        /**
         * Set a value pair of the diagram.
         *
         * @param x The x value
         * @param y The y value
         */
        inline void SetValue( float x, float y) {
            xValue = x;
            yValue = y;
        };

        /**
         * Get the x value.
         * 
         * @return The value
         */
        inline float GetX() { return xValue; };

        /**
         * Get the y value.
         * 
         * @return The value
         */
        inline float GetY() { return yValue; };

        /**
         * Get the value pair.
         * 
         * @return The value pair
         */
        inline vislib::math::Vector<float, 2> GetValuePair() { return vislib::math::Vector<float, 2>( xValue, yValue); };

        /**
         * Sets the clear diagram flag.
         *
         * @param clear The clear parameter flag
         */
        void SetClearDiagramFlag( bool clear) { clearDiagram = clear; };

        /**
         * Returns the clear diagram flag.
         *
         * @return The clear flag
         */
         bool ClearDiagram() { return clearDiagram; };

        /**
         * Set marker flag.
         *
         * @param marker The marker flag
         */
        void SetMarkerFlag( bool marker) { markerFlag = marker; };

        /**
         * Returns the marker flag.
         *
         * @return The marker flag
         */
         bool Marker() { return markerFlag; };

        /**
         * Set the call time.
         *
         * @param ct The call time
         */
        void SetCallTime( float ct) { callTime = ct; };

        /**
         * Returns the call time.
         *
         * @return The call time
         */
         float CallTime() { return callTime; };

    private:
        /** the range of x values */
        float xRange;
        /** the range of y values */
        float yRange;

        /** the current value on the x axis */
        float xValue;
        /** the current value on the y axis */
        float yValue;

        /** the call time */
        float callTime;

        /** clear the diagram */
        bool clearDiagram;

        /** set marker */
        bool markerFlag;
    };

    /** Description class typedef */
    typedef megamol::core::CallAutoDescription<Diagram2DCall> Diagram2DCallDescription;


} /* end namespace protein_cuda */
} /* end namespace megamol */

#endif /* MEGAMOL_PROTEIN_DIAGRAM2DCALL_H_INCLUDED */
