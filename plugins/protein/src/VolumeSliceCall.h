/*
 * VolumeSliceCall.h
 *
 * Author: Michael Krone
 * Copyright (C) 2010 by Universitaet Stuttgart (VISUS).
 * All rights reserved.
 */


#ifndef MEGAMOL_PROTEIN_VOLUMESLICECALL_H_INCLUDED
#define MEGAMOL_PROTEIN_VOLUMESLICECALL_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "mmcore/Call.h"
#include "mmcore/factories/CallAutoDescription.h"
#include "vislib/IllegalParamException.h"
#include "vislib/math/Cuboid.h"
#include "vislib/math/Vector.h"
#include <vector>
#include "vislib/graphics/gl/IncludeAllGL.h"

namespace megamol {
namespace protein {

    /**
     * Base class of rendering graph calls and data interfaces for volume data.
     */

    class VolumeSliceCall : public megamol::core::Call {
    public:

        /**
         * Answer the name of the objects of this description.
         *
         * @return The name of the objects of this description.
         */
        static const char *ClassName(void) {
            return "VolumeSliceCall";
        }

        /**
         * Gets a human readable description of the module.
         *
         * @return A human readable description of the module.
         */
        static const char *Description(void) {
            return "Call to get volume slice and texture";
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
        VolumeSliceCall(void);

        /** Dtor. */
        virtual ~VolumeSliceCall(void);

        /**
         * Set the volume texture.
         *
         * @param tex The texture id
         */
        void setVolumeTex( GLuint tex) { volumeTex = tex; };

        /**
         * Get the volume texture.
         *
         * @return The texture id
         */
        GLuint getVolumeTex() { return volumeTex; };

        /**
         * Set the texture r coordinate.
         *
         * @param tex The texture id
         */
        void setTexRCoord( float r) { texRCoord = r; };

        /**
         * Get the texture r coordinate.
         *
         * @return The texture id
         */
        float getTexRCoord() { return texRCoord; };

        /**
         * Set the clip plane normal.
         *
         * @param n The clip plane normal
         */
        void setClipPlaneNormal( vislib::math::Vector<float, 3> n) { 
            clipPlaneNormal = n;
            clipPlaneNormal.Normalise();
        };

        /**
         * Get the clip plane normal.
         *
         * @return The clip plane normal
         */
        vislib::math::Vector<float, 3> getClipPlaneNormal() { return clipPlaneNormal; };

        /**
         * Set the bounding box dimensions.
         *
         * @param dim The bounding box dimensions
         */
        void setBBoxDimensions( vislib::math::Vector<float, 3> dim) { 
            bBoxDim = dim;
        };

        /**
         * Get the bounding box dimensions.
         *
         * @return The bounding box dimensions
         */
        vislib::math::Vector<float, 3> getBBoxDimensions() { return bBoxDim; };

        /**
         * Set the last mouse position relative to the volume.
         *
         * @param pos The last mouse position
         */
        void setMousePos( vislib::math::Vector<float, 3> pos) { mousePos = pos; };

        /**
         * Returns the last mouse position relative to the volume.
         *
         * @return The last mouse position
         */
        vislib::math::Vector<float, 3> getMousePos() { return mousePos; };

        /**
         * Set the last clicked mouse position relative to the volume.
         *
         * @param pos The clicked mouse position
         */
        void setClickPos( vislib::math::Vector<float, 3> pos) { clickPos = pos; };

        /**
         * Returns the last clicked mouse position relative to the volume.
         *
         * @return The clicked mouse position
         */
        vislib::math::Vector<float, 3> getClickPos() { return clickPos; };

        /**
         * Set the isovalue.
         *
         * @param iv The isovalue
         */
        void setIsovalue( float iv) { isoValue = iv; };

        /**
         * Get the isovalue.
         *
         * @return The isovalue
         */
        float getIsovalue() { return isoValue; };

    private:
        // volume texture
        GLuint volumeTex;

        // volume slice (r component of texture coordinate)
        float texRCoord;

        // isovalue for isosurface rendering
        float isoValue;

        // clip plane normal
        vislib::math::Vector<float, 3> clipPlaneNormal;

        // the bounding box dimensions
        vislib::math::Vector<float, 3> bBoxDim;
        
        // the mouse pos
        vislib::math::Vector<float, 3> mousePos;

        // the click position
        vislib::math::Vector<float, 3> clickPos;

    };

    /** Description class typedef */
    typedef megamol::core::factories::CallAutoDescription<VolumeSliceCall> VolumeSliceCallDescription;


} /* end namespace protein */
} /* end namespace megamol */

#endif /* MEGAMOL_PROTEIN_VOLUMESLICECALL_H_INCLUDED */
