/*
 * ContourRendererDeferred.h
 *
 * Copyright (C) 2011 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#ifndef MMPROTEINPLUGIN_CONTOURRENDERERDEFERRED_H_INCLUDED
#define MMPROTEINPLUGIN_CONTOURRENDERERDEFERRED_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include <view/AbstractRendererDeferred3D.h>
#include <vislib/FramebufferObject.h>
#include <vislib/GLSLShader.h>

namespace megamol {
namespace protein {

    /**
     * Render module which implements screen space rendering of contour lines
     */
    class ContourRendererDeferred : public megamol::core::view::AbstractRendererDeferred3D {
    public:

         /* Answer the name of this module.
          *
          * @return The name of this module.
          */
         static const char *ClassName(void) {
             return "ContourRendererDeferred";
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
             if(!glh_extension_supported("GL_ARB_texture_rectangle"))
                 return false;
             return true;
         }

        /** Ctor. */
        ContourRendererDeferred(void);

        /** Dtor. */
        virtual ~ContourRendererDeferred(void);

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
         * The get capabilities callback. The module should set the members
         * of 'call' to tell the caller its capabilities.
         *
         * @param call The calling call.
         *
         * @return The return value of the function.
         */
        virtual bool GetCapabilities(megamol::core::Call& call);

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

        /** Parameter slot for the render mode */
        megamol::core::param::ParamSlot renderModeParam;

#ifdef _WIN32
#pragma warning (disable: 4251)
#endif /* _WIN32 */

        /** The renderers frame buffer object */
        GLuint fbo;
        GLuint colorBuff;
        GLuint normalBuff;
        GLuint depthBuff;
        int widthFBO;
        int heightFBO;

        /** The contour shader */
        vislib::graphics::gl::GLSLShader contourShader;

#ifdef _WIN32
#pragma warning (default: 4251)
#endif /* _WIN32 */
    };


} // end namespace protein
} // end namespace megamol

#endif // MMPROTEINPLUGIN_CONTOURRENDERERDEFERRED_H_INCLUDED

