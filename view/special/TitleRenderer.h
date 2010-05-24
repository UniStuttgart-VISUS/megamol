/*
 * TitleRenderer.h
 *
 * Copyright (C) 2010 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCORE_TITLERENDERER_H_INCLUDED
#define MEGAMOLCORE_TITLERENDERER_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "view/AbstractRenderingView.h"
#include "vislib/AbstractVISLogo.h"
#include "vislib/CameraOpenGL.h"
#include <GL/gl.h>
#include <GL/glu.h>


namespace megamol {
namespace core {
namespace view {
namespace special {


    /**
     * The title renderer
     */
    class TitleRenderer : public view::AbstractRenderingView::AbstractTitleRenderer {
    public:

        /** ctor */
        TitleRenderer();

        /** dtor */
        virtual ~TitleRenderer();

        /**
         * Create the renderer and allocates all resources
         *
         * @return True on success
         */
        virtual bool Create(void);

        /**
         * Renders the title scene
         *
         * @param tileX The view tile x coordinate
         * @param tileY The view tile y coordinate
         * @param tileW The view tile width
         * @param tileH The view tile height
         * @param virtW The virtual view width
         * @param virtH The virtual view height
         * @param stereo Flag if stereo rendering is to be performed
         * @param leftEye Flag if the stereo rendering is done for the left eye view
         * @param time The core time
         */
        virtual void Render(float tileX, float tileY, float tileW, float tileH,
            float virtW, float virtH, bool stereo, bool leftEye, double time);

        /**
         * Releases the renderer and all of its resources
         */
        virtual void Release(void);

    private:

        /**
         * Fallback implementation for the MegaMol icon (OpenGL 1.1 compatible)
         */
        class FallbackIcon : public vislib::graphics::AbstractVISLogo {
        public:

            /** Ctor. */
            FallbackIcon(void);

            /** Dtor. */
            virtual ~FallbackIcon(void);

            /**
             * Create all required resources for rendering a VIS logo.
             *
             * @throws Exception In case of an error.
             */
            virtual void Create(void);

            /**
             * Render the VIS logo. Create() must have been called before.
             *
             * @throws Exception In case of an error.
             */
            virtual void Draw(void);

            /**
             * Release all resources of the VIS logo.
             *
             * @throws Exception In case of an error.
             */
            virtual void Release(void);

        private:

            /** The GLU quadric state*/
            GLUquadric *quadric;

        };

        /**
         * Zooms the camera by adjusting the aperture angle to ensure the
         * visibility of the given point
         *
         * @param x The x coordinate
         * @param y The y coordinate
         * @param z The z coordinate
         */
        void zoomCamera(float x, float y, float z);

        /** The title text */
        vislib::graphics::AbstractVISLogo *title;

        /** The title text width */
        float titleWidth;

        /** The icon */
        vislib::graphics::AbstractVISLogo *icon;

        /** The camera */
        vislib::graphics::gl::CameraOpenGL camera;

    };


} /* end namespace special */
} /* end namespace view */
} /* end namespace core */
} /* end namespace megamol */

#endif /* MEGAMOLCORE_TITLERENDERER_H_INCLUDED */
