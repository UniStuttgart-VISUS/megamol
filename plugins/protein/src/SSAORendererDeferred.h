/*
 * SSAORendererDeferred.h
 *
 * Copyright (C) 2011 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#ifndef MMPROTEINPLUGIN_SSAORENDERERDEFERRED_H_INCLUDED
#define MMPROTEINPLUGIN_SSAORENDERERDEFERRED_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "mmcore/view/AbstractRendererDeferred3D.h"
#include "vislib/graphics/gl/FramebufferObject.h"
#include "vislib/graphics/gl/GLSLShader.h"

namespace megamol {
namespace protein {

    /**
     * Render module which implements screen space ambient occlusion
     */
    class SSAORendererDeferred : public megamol::core::view::AbstractRendererDeferred3D {
    public:

         /* Answer the name of this module.
          *
          * @return The name of this module.
          */
         static const char *ClassName(void) {
             return "SSAORendererDeferred";
         }

         /**
          * Answer a human readable description of this module.
          *
          * @return A human readable description of this module.
          */
         static const char *Description(void) {
             return "Renderer implementing screen space ambient occlusion using deferred shading.";
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
        SSAORendererDeferred(void);

        /** Dtor. */
        virtual ~SSAORendererDeferred(void);

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
         * Creates a 4x4 texture containing random rotation vectors.
         *
         * @return True if the texture could be created.
         */
        bool createRandomRotSampler();

        /**
         * Creates a random kernel sampling the positive hemisphere
         *
         * @param size The number of samples.
         *
         * @return True if the texture could be created.
         */
        bool createRandomKernel(UINT size);

        /*
         * Returns a random float within a specific range with a given precision
         *
         * @param min The minimum value
         * @param max The maximum value
         * @prec The precision
         *
         * @return The random float
         */
        float getRandomFloat(float min, float max, unsigned int prec);

        /**
         * Update parameters if necessary
         */
        bool updateParams();

        /** Parameter slot for the render mode */
        megamol::core::param::ParamSlot renderModeParam;

        /** Parameter slot for the render mode */
        megamol::core::param::ParamSlot aoRadiusParam;

        /** Parameter slot for the depth threshold */
        megamol::core::param::ParamSlot depthThresholdParam;

        /** Parameter slot for the number of ao samples */
        megamol::core::param::ParamSlot aoSamplesParam;

        /** Parameter slot for the scaling of ao factor */
        megamol::core::param::ParamSlot aoScaleParam;

        /** Parameter for the number of filter samples */
        megamol::core::param::ParamSlot nFilterSamplesParam;

        /** Parameter to adjust the mip map level of the depth texture while sampling */
        megamol::core::param::ParamSlot depthMipMapLvlParam;

        /** Parameter to toggle filtering of the ssao value */
        megamol::core::param::ParamSlot filterParam;

#ifdef _WIN32
#pragma warning (disable: 4251)
#endif /* _WIN32 */

        /** The renderers frame buffer object */
        GLuint deferredFBO;
        GLuint colorBuff;
        GLuint normalBuff;
        GLuint depthBuff;
        GLuint ssaoBuff;
        GLuint filterBuff;
        GLuint discBuff; // Discontinuity buffer for edge detection while filtering
        int widthFBO;
        int heightFBO;


        /** The SSAO shader */
        vislib::graphics::gl::GLSLShader ssaoShader;
        /** The deferred shader */
        vislib::graphics::gl::GLSLShader deferredShader;
        /** Filter */
        vislib::graphics::gl::GLSLShader filterShaderX;
        vislib::graphics::gl::GLSLShader filterShaderY;

#ifdef _WIN32
#pragma warning (default: 4251)
#endif /* _WIN32 */

        /** The rotation vector sampler */
        GLuint rotSampler;
        /** The kernel */
        GLuint randKernel;
    };


} // end namespace protein
} // end namespace megamol

#endif // MMPROTEINPLUGIN_SSAORENDERERDEFERRED_H_INCLUDED

