/*
 * ToonRendererDeferred.h
 *
 * Copyright (C) 2011 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#ifndef MMPROTEINPLUGIN_TOONRENDERERDEFERRED_H_INCLUDED
#define MMPROTEINPLUGIN_TOONRENDERERDEFERRED_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "mmcore/view/AbstractRendererDeferred3D.h"
#include "vislib/graphics/gl/FramebufferObject.h"
#include "vislib/graphics/gl/GLSLShader.h"

namespace megamol {
namespace protein {

    /**
     * Render module which implements screen space rendering of contour lines
     */
    class ToonRendererDeferred : public megamol::core::view::AbstractRendererDeferred3D {
    public:

         /* Answer the name of this module.
          *
          * @return The name of this module.
          */
         static const char *ClassName(void) {
             return "ToonRendererDeferred";
         }

         /**
          * Answer a human readable description of this module.
          *
          * @return A human readable description of this module.
          */
         static const char *Description(void) {
             return "Offers screen space rendering of contour lines.";
         }

         /**
          * Answers whether this module is available on the current system.
          *
          * @return 'true' if the module is available, 'false' otherwise.
          */
         static bool IsAvailable(void) {
             if(!vislib::graphics::gl::GLSLShader::AreExtensionsAvailable())
                 return false;
             if(!vislib::graphics::gl::FramebufferObject::AreExtensionsAvailable())
                 return false;
             if(!isExtAvailable("GL_ARB_texture_rectangle"))
                 return false;
             return true;
         }

        /** Ctor. */
        ToonRendererDeferred(void);

        /** Dtor. */
        virtual ~ToonRendererDeferred(void);

    protected:

        /**
         * Implementation of 'create'.
         *
         * @return 'true' on success, 'false' otherwise.
         */
        bool create(void);

        /**
         * Implementation of 'release'.
         */
        void release(void);

        /**
         * The get extents callback. The module should set the members of
         * 'call' to tell the caller the extents of its data (bounding boxes
         * and times).
         *
         * @param call The calling call.
         *
         * @return The return value of the function.
         */
        virtual bool GetExtents(megamol::core::Call& call);

        /**
         * The render callback.
         *
         * @param call The calling call.
         *
         * @return The return value of the function.
         */
        virtual bool Render(megamol::core::Call& call);

    private:

        /**
         * Initialize the frame buffer object.
         *
         * @param width The width of the buffer.
         * @param height The height of the buffer.
         *
         * @return True if the fbo could be created.
         */
        bool createFBO(UINT width, UINT height);

        /**
         * Update parameters if necessary
         */
        bool updateParams();

        /**
         * Create a random texture for the vector rotation.
         */
        bool createRandomRotSampler();

        float getRandomFloat(float min, float max, unsigned int prec);
        bool createRandomKernel(UINT size);

        /** Threshold for fine lines */
        megamol::core::param::ParamSlot threshFineLinesParam;
        /** Threshold for coarse lines */
        megamol::core::param::ParamSlot threshCoarseLinesParam;
        /** SSAO params */
        megamol::core::param::ParamSlot ssaoParam;
        megamol::core::param::ParamSlot ssaoRadiusParam;
        /** Param to change local illumination */
        megamol::core::param::ParamSlot illuminationParam;
        /** Param to toggle coloring */
        megamol::core::param::ParamSlot colorParam;

#ifdef _WIN32
#pragma warning (disable: 4251)
#endif /* _WIN32 */

        /** The renderers frame buffer object */
        GLuint fbo;
        GLuint colorBuffer;
        GLuint normalBuffer;
        GLuint depthBuffer;
        GLuint gradientBuffer;
        GLuint ssaoBuffer;
        GLuint rotationSampler;
        GLuint randomKernel;
        int widthFBO;
        int heightFBO;

        /** The contour shader */
        vislib::graphics::gl::GLSLShader sobelShader;
        vislib::graphics::gl::GLSLShader ssaoShader;
        vislib::graphics::gl::GLSLShader toonShader;

#ifdef _WIN32
#pragma warning (default: 4251)
#endif /* _WIN32 */
    };


} // end namespace protein
} // end namespace megamol

#endif // MMPROTEINPLUGIN_TOONRENDERERDEFERRED_H_INCLUDED

