/*
 * MipDepthSphereRenderer.h
 *
 * Copyright (C) 2009 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCORE_MIPDEPTHSPHERERENDERER_H_INCLUDED
#define MEGAMOLCORE_MIPDEPTHSPHERERENDERER_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "mmcore/view/Renderer3DModule.h"
#include "mmcore/Call.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/param/ParamSlot.h"
#include "vislib/graphics/gl/GLSLShader.h"
#include "vislib/graphics/gl/FramebufferObject.h"


namespace megamol {
namespace core {
namespace moldyn {

    /**
     * Renderer for sphere glyphs filtering based on a depth-max mipmap
     */
    class MipDepthSphereRenderer : public view::Renderer3DModule {
    public:

        /**
         * Answer the name of this module.
         *
         * @return The name of this module.
         */
        static const char *ClassName(void) {
            return "MipDepthSphereRenderer";
        }

        /**
         * Answer a human readable description of this module.
         *
         * @return A human readable description of this module.
         */
        static const char *Description(void) {
            return "Renderer for sphere glyphs filtering with a depth-min mipmap";
        }

        /**
         * Answers whether this module is available on the current system.
         *
         * @return 'true' if the module is available, 'false' otherwise.
         */
        static bool IsAvailable(void) {
            return vislib::graphics::gl::GLSLShader::AreExtensionsAvailable()
                && vislib::graphics::gl::FramebufferObject::AreExtensionsAvailable();
        }

        /**
         * Disallow usage in quickstarts
         *
         * @return false
         */
        static bool SupportQuickstart(void) {
            return false;
        }

        /** Ctor. */
        MipDepthSphereRenderer(void);

        /** Dtor. */
        virtual ~MipDepthSphereRenderer(void);

    protected:

        /**
         * Implementation of 'Create'.
         *
         * @return 'true' on success, 'false' otherwise.
         */
        virtual bool create(void);

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
         * Implementation of 'Release'.
         */
        virtual void release(void);

        /**
         * The render callback.
         *
         * @param call The calling call.
         *
         * @return The return value of the function.
         */
        virtual bool Render(Call& call);

    private:

        /** The sphere shader */
        vislib::graphics::gl::GLSLShader sphereShader;

        /** The shader to init the depth fbo*/
        vislib::graphics::gl::GLSLShader initDepthShader;

        /** The frame buffer object for the depth estimate */
        vislib::graphics::gl::FramebufferObject fbo;

        /** The textures used for storing/generating the max-mipmap depth */
        GLuint mipDepth, mipDepthTmp;

        /** The call for data */
        CallerSlot getDataSlot;

        /** The call for Transfer function */
        CallerSlot getTFSlot;

        /** The call for clipping plane */
        CallerSlot getClipPlaneSlot;

        /** A simple black-to-white transfer function texture as fallback */
        unsigned int greyTF;

    };

} /* end namespace moldyn */
} /* end namespace core */
} /* end namespace megamol */

#endif /* MEGAMOLCORE_MIPDEPTHSPHERERENDERER_H_INCLUDED */
