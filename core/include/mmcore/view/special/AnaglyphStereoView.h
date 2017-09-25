/*
 * AnaglyphStereoView.h
 *
 * Copyright (C) 2010 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCORE_ANAGLYPHSTEREOVIEW_H_INCLUDED
#define MEGAMOLCORE_ANAGLYPHSTEREOVIEW_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "mmcore/param/ParamSlot.h"
#include "mmcore/view/special/AbstractStereoView.h"
#include "vislib/graphics/gl/FramebufferObject.h"
#include "vislib/graphics/gl/GLSLShader.h"


namespace megamol {
namespace core {
namespace view {
namespace special {


    /**
     * Abstract base class of override rendering views
     */
    class AnaglyphStereoView : public AbstractStereoView {
    public:

        /**
         * Answer the name of this module.
         *
         * @return The name of this module.
         */
        static const char *ClassName(void) {
            return "AnaglyphStereoView";
        }

        /**
         * Answer a human readable description of this module.
         *
         * @return A human readable description of this module.
         */
        static const char *Description(void) {
            return "Override View Module for anaglyph stereo output";
        }

        /**
         * Answers whether this module is available on the current system.
         *
         * @return 'true' if the module is available, 'false' otherwise.
         */
        static bool IsAvailable(void) {
            return vislib::graphics::gl::FramebufferObject::AreExtensionsAvailable()
                && vislib::graphics::gl::GLSLShader::AreExtensionsAvailable()
                && isExtAvailable("GL_ARB_multitexture");
        }

        /** Ctor. */
        AnaglyphStereoView(void);

        /** Dtor. */
        virtual ~AnaglyphStereoView(void);

        /**
         * Resizes the AbstractView3D.
         *
         * @param width The new width.
         * @param height The new height.
         */
        virtual void Resize(unsigned int width, unsigned int height);

        /**
         * Renders this AbstractView3D in the currently active OpenGL context.
         */
        virtual void Render(const mmcRenderViewContext& context);

    protected:

        /**
         * Initializes the module directly after instanziation
         *
         * @return 'true' on success
         */
        virtual bool create(void);

        /**
         * Releases all resources of the module
         */
        virtual void release(void);

    private:

        /** The left frame buffer */
        vislib::graphics::gl::FramebufferObject leftBuffer;

        /** The right frame buffer */
        vislib::graphics::gl::FramebufferObject rightBuffer;

        /** The compositing shader */
        vislib::graphics::gl::GLSLShader shader;

        /** parameter specifying the left eye colour */
        param::ParamSlot leftColourSlot;

        /** parameter specifying the right eye colour */
        param::ParamSlot rightColourSlot;

        /** parameter to select available eye colour presets */
        param::ParamSlot colourPresetsSlot;

        /** The left eye colour */
        float leftColour[3];

        /** The right eye colour */
        float rightColour[3];

    };

} /* end namespace special */
} /* end namespace view */
} /* end namespace core */
} /* end namespace megamol */

#endif /* MEGAMOLCORE_ANAGLYPHSTEREOVIEW_H_INCLUDED */
