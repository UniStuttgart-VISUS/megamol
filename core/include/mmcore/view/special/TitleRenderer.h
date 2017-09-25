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

#include "mmcore/view/AbstractRenderingView.h"
#include "vislib/graphics/gl/IncludeAllGL.h"
#include "vislib/graphics/AbstractVISLogo.h"
#include "vislib/graphics/gl/CameraOpenGL.h"
#include "vislib/graphics/gl/GLSLShader.h"
#include <GL/glu.h>

//#define ICON_DEBUGGING


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
         * @param core The core
         */
        virtual void Render(float tileX, float tileY, float tileW, float tileH,
            float virtW, float virtH, bool stereo, bool leftEye, double instTime,
            class ::megamol::core::CoreInstance *core);

        /**
         * Releases the renderer and all of its resources
         */
        virtual void Release(void);

    private:

        class AbstractIcon : public vislib::graphics::AbstractVISLogo {
        public:

            /** Dtor */
            virtual ~AbstractIcon(void);

        protected:

            /**
             * Length of the dipole cylinder. When rendering, all parameters
             * must be normalised that this length is 2.0
             */
            static const float cylLen;

            /** Radius of the dipole cylinder */
            static const float cylRad;

            /** Radius of the positive off-centered mass */
            static const float s1Rad;

            /** Radius of the negative off-centered mass */
            static const float s2Rad;

            /** Distance between the off-centered masses */
            static const float sDist;

            /** Ctor */
            AbstractIcon(void);

        };

        /**
         * Fallback implementation for the MegaMol icon (OpenGL 1.1 compatible)
         */
        class FallbackIcon : public AbstractIcon {
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
         * GPU Raycasting implementation
         */
        class GPURaycastIcon : public AbstractIcon {
        public:

            /**
             * Ctor.
             *
             * @param core The core instance
             */
            GPURaycastIcon(class ::megamol::core::CoreInstance *core);

            /** Dtor. */
            virtual ~GPURaycastIcon(void);

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

            /**
             * Answer if the error flag is set
             *
             * @return The error flag
             */
            inline bool HasError(void) const {
                return this->error;
            }

        private:

            /** The error flag */
            bool error;

            /** The core instance */
            class ::megamol::core::CoreInstance *core;

            /** The shader */
            vislib::graphics::gl::GLSLShader *shader;

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

#ifdef ICON_DEBUGGING
        vislib::graphics::AbstractVISLogo *i2;
#endif /* ICON_DEBUGGING */

        /** The camera */
        vislib::graphics::gl::CameraOpenGL camera;

    };


} /* end namespace special */
} /* end namespace view */
} /* end namespace core */
} /* end namespace megamol */

#endif /* MEGAMOLCORE_TITLERENDERER_H_INCLUDED */
