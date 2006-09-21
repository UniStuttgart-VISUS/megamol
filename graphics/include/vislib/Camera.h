/*
 * Camera.h
 *
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 */

#ifndef VISLIB_CAMERA_H_INCLUDED
#define VISLIB_CAMERA_H_INCLUDED
#if (_MSC_VER > 1000)
#pragma once
#endif /* (_MSC_VER > 1000) */


#include "vislib/Beholder.h"
#include "vislib/mathtypes.h"


namespace vislib {
namespace graphics {

    /**
     * class modelling a 3d scene camera
     */
    template<class T > class Camera {
    public:

        /** This type is used for values in scene space */
        typedef float SceneSpaceValue;

        /** This type is used for values in image space */
        typedef unsigned int ImageSpaceValue;
        
        /** possible values for the stereo projection type */
        enum StereoProjectionType {
            PARALLEL_PROJECTION = 0,
            OFF_AXIS_PROJECTION,
            TOE_IN_PROJECTION
        };

        /*
         * TODO: Implement
         */

    private:

        /** 
         * aperture Angle of the camera
         * TODO: Think of the meaning of this value
         */
        math::AngleDeg apertureAngle;

        /** 
         * Pointer to the currently attached beholder 
         * TODO: Think of an nested template facade class holding this, so 
         *       Camera could be a normal (nontemplate) class.
         */
        Beholder<T> *beholder;

        /** distance of the far clipping plane */
        SceneSpaceValue farClip;

        /** focal distance for stereo images */
        SceneSpaceValue focalDistance;

        /** distance of the near clipping plane */
        SceneSpaceValue nearClip;

        /** eye disparity value for stereo images */
        SceneSpaceValue stereoDisparity;

        /** type of stereo projection */
        StereoProjectionType stereoProjectionType;

        /** Width of the virtual camera image */
        ImageSpaceValue virtualWidth;

        /** Height of the virtual camera image */
        ImageSpaceValue virtualHeight;
    };

} /* end namespace graphics */
} /* end namespace vislib */

#endif /* VISLIB_CAMERA_H_INCLUDED */
