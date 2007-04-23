/*
 * ZoomToRectangle.h
 *
 * Copyright (C) 2006 - 2007 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#ifndef VISLIB_ZOOMTORECTANGLE_H_INCLUDED
#define VISLIB_ZOOMTORECTANGLE_H_INCLUDED
#if (_MSC_VER > 1000)
#pragma once
#endif /* (_MSC_VER > 1000) */
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */


#include "vislib/graphicstypes.h"
#include "vislib/AbstractBeholderController.h"
#include "vislib/AbstractCameraController.h"
#include "vislib/Rectangle.h"


namespace vislib {
namespace graphics {


    /**
     * Utility class for zooming a view (consisting of one beholder and one 
     * camera) to zoom to a user defined rectangular area on the virtual camera
     * image. The zooming is performed by first, moving or rotating the 
     * beholder and the by changing the aperture angle of the camera or further
     * moving the beholder. The focal distance can be reduced to keep the focal
     * plane constant. Neither the clipping distances nor the stereo parameters
     * are changed. With cameras using orthographic projection, the beholder 
     * will always be moved and the size of the virtual camera image can be
     * changed (including the tile).
     */
    class ZoomToRectangle: public AbstractBeholderController, 
            public AbstractCameraController {

    public:

        /** possible values of zoom type */
        enum ZoomType {
            ZOOM_PAN_DOLLY, // rotate and move beholder (default)
            ZOOM_PAN_ZOOM, // rotate beholder and change camera aperture angle
            ZOOM_TRACK_DOLLY, // move beholder twice
            ZOOM_TRACK_ZOOM // move beholder and change camera aperture angle
        };

        /** Ctor. */
        ZoomToRectangle(void);

        /**
         * Copy ctor.
         *
         * @param rhs The right hand side operand.
         */
        ZoomToRectangle(const ZoomToRectangle& rhs);

        /** Dtor. */
        ~ZoomToRectangle(void);

        /**
         * Assignment operator
         *
         * @param rhs The right hand side operand.
         */
        ZoomToRectangle& operator=(const ZoomToRectangle& rhs);

        /**
         * Sets the zoom target rectangle in coordinates of the virtual camera 
         * image.
         *
         * @param rect The zoom target rectangle.
         */
        inline void SetZoomTargetRect(
                const vislib::math::Rectangle<ImageSpaceType> &rect) {
            this->targetRect = rect;
        }

        /**
         * Answers the zoom target rectangle in coordinates of the virtual 
         * camera image.
         *
         * @return The zoom target rectangle.
         */
        inline const vislib::math::Rectangle<ImageSpaceType>& 
                GetZoomTargetRect(void) const {
            return this->targetRect;
        }

        /** 
         * Sets the zoom type to perform. This type does not apply to zooming
         * cameras with orthographic projection.
         *
         * @param type The new zoom type.
         */
        inline void SetZoomType(ZoomType type) {
            this->type = type;
        }

        /**
         * Answers the zoom type. This type does not apply to zooming cameras 
         * with orthographic projection.
         *
         * @return The zoom type.
         */
        inline ZoomType GetZoomType(void) const {
            return this->type;
        }

        /**
         * Tells the object to keep the focal plane fixed, or not. When using
         * a fixed focal plane and dollying the beholder, the focal distance
         * is changed to keep the focal plane fix in the scene space. The focal
         * plane is fixed by default.
         *
         * @param fix Indicates if the focal plane should be fixed (true) or
         *            changing (false) when dollying the beholder.
         */
        inline void FixFocalPlane(bool fix = true) {
            this->fixFocus = fix;
        }

        /**
         * Answers whether the focal plane is fixed.
         *
         * @return True if the focal plane is fixed, false otherwise.
         */
        inline bool IsFocalPlaneFixed(void) const {
            return this->fixFocus;
        }

        /**
         * Tells the object to resize the virtual camera image of cameras
         * using orthographic projection. This value does only apply to cameras
         * with orthographic projection. The default value is false.
         *
         * @param canResize Indicates if the size of the virtual camera image
         *                  can be changed (true) or not (false).
         */
        inline void ChangeOrthoCameraVirtualSize(bool canResize = true) {
            this->resizeOrthoCams = canResize;
        }

        /**
         * Answers whether the size of the virtual camera image of cameras
         * using orthographic projection will be changed. This value does only
         * apply to cameras with orthographic projection.
         *
         * @return True if the virtual camera image could be resized, false
         *         if the virtual camera image size will not be changed.
         */
        inline bool CanChangeOrthoCameraVirtualSize(void) const {
            return this->resizeOrthoCams;
        }

        /**
         * Performs the zoom to the rectangular area of the virtual camera 
         * image in the specified way. The beholder and the camera will be
         * changed.
         *
         * @throws IllegalStateException if either no beholder or no camera is
         *         set, if the camera uses orthographic projection or if the 
         *         zoom target rectangle is empty.
         */
        void Zoom(void);

    private:

        /** Performs the zoom operation on an orthographic camera */
        inline void ZoomOrthographicCamera(void);

        /** Performs the zoom operation on a projective camera */
        inline void ZoomProjectiveCamera(void);

        /** the zoom target rectangle */
        vislib::math::Rectangle<ImageSpaceType> targetRect;

        /** the zoom type to perform */
        ZoomType type;

        /** flag indicating if the focal plane is fixed. */
        bool fixFocus;

        /** 
         * flag indicating if the virtual camera image size of cameras with 
         * orthographic projection can be changes.
         */
        bool resizeOrthoCams;

    };
    
} /* end namespace graphics */
} /* end namespace vislib */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_ZOOMTORECTANGLE_H_INCLUDED */

