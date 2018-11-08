/*
 * BlinnPhongRendererDeferred.h
 *
 * Copyright (C) 2011 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCORE_BLINNPHONGRENDERERDEFERRED_H_INCLUDED
#define MEGAMOLCORE_BLINNPHONGRENDERERDEFERRED_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "mmcore/view/AbstractRendererDeferred3D.h"
#include "vislib/graphics/gl/FramebufferObject.h"
#include "vislib/graphics/gl/GLSLShader.h"

namespace megamol {
namespace core {
namespace view {

    /**
     * Render module which implements the Blinn-Phong shading model using
     * deferred shading.
     */
    class MEGAMOLCORE_API BlinnPhongRendererDeferred : public AbstractRendererDeferred3D {
    public:

         /* Answer the name of this module.
          *
          * @return The name of this module.
          */
         static const char *ClassName(void) {
             return "BlinnPhongRendererDeferred";
         }

         /**
          * Answer a human readable description of this module.
          *
          * @return A human readable description of this module.
          */
         static const char *Description(void) {
             return "Renderer implementing the Blinn-Phong shading model using "
                 "deferred shading.";
         }

         /**
          * Answers whether this module is available on the current system.
          *
          * @return 'true' if the module is available, 'false' otherwise.
          */
         static bool IsAvailable(void) {
             return true;
         }

        /** Ctor. */
        BlinnPhongRendererDeferred(void);

        /** Dtor. */
        virtual ~BlinnPhongRendererDeferred(void);

        /** Possible rendering modes */
        enum renderMode {
             BLINN_PHONG,
             COLOR,
             NORMAL, 
             DEPTH
        };

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
        virtual bool GetExtents(Call& call);

        /**
         * The render callback.
         *
         * @param call The calling call.
         *
         * @return The return value of the function.
         */
        virtual bool Render(Call& call);

    private:

        /**
         * Initialize the frame buffer object.
         *
         * @param width The width of the buffer.
         * @param width The height of the buffer.
         *
         * @return True if the fbo could be created.
         */
        bool createFBO(UINT width, UINT height);

        /** Parameter slot for the render mode */
        megamol::core::param::ParamSlot renderModeParam;

#ifdef _WIN32
#pragma warning (disable: 4251)
#endif /* _WIN32 */

        /** The renderers frame buffer object */
        vislib::graphics::gl::FramebufferObject fbo;

        /** The bling phong shader */
        vislib::graphics::gl::GLSLShader blinnPhongShader;

#ifdef _WIN32
#pragma warning (default: 4251)
#endif /* _WIN32 */

    };

} /* end namespace view */
} /* end namespace core */
} /* end namespace megamol */

#endif /* MEGAMOLCORE_BLINNPHONGRENDERERDEFERRED_H_INCLUDED */
