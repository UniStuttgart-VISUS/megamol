/*
 * ColStereoDisplay.h
 *
 * Copyright (C) 2009 by VISUS (Universitaet Stuttgart).
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCORE_COLSTEREODISPLAY_H_INCLUDED
#define MEGAMOLCORE_COLSTEREODISPLAY_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "mmcore/param/ParamSlot.h"
#include "mmcore/special/AbstractStereoDisplay.h"
#include "vislib/graphics/gl/FramebufferObject.h"
#include "vislib/graphics/gl/GLSLShader.h"


namespace megamol {
namespace core {
namespace special {

    /**
     * Special view module used for column based stereo displays
     */
    class ColStereoDisplay : public AbstractStereoDisplay {
    public:

        /**
         * Gets the name of this module.
         *
         * @return The name of this module.
         */
        static const char *ClassName(void) {
            return "ColStereoDisplay";
        }

        /**
         * Gets a human readable description of the module.
         *
         * @return A human readable description of the module.
         */
        static const char *Description(void) {
            return "Special view module used for column based stereo displays";
        }

        /**
         * Gets whether this module is available on the current system.
         *
         * @return 'true' if the module is available, 'false' otherwise.
         */
        static bool IsAvailable(void) {
            return vislib::graphics::gl::GLSLShader::AreExtensionsAvailable()
                && vislib::graphics::gl::FramebufferObject::AreExtensionsAvailable();
        }

        /** Ctor. */
        ColStereoDisplay(void);

        /** Dtor. */
        virtual ~ColStereoDisplay(void);

        /**
         * Renders this AbstractView3D in the currently active OpenGL context.
         */
        virtual void Render(void);

    private:

        /**
         * Implementation of 'Module::Create'.
         *
         * @return 'true' on success, 'false' otherwise.
         */
        virtual bool create(void);

        /**
         * Implementation of 'Module::Release'.
         */
        virtual void release(void);

        /** the intermediate fbo */
        vislib::graphics::gl::FramebufferObject fbo;

        /** the compositing shader */
        vislib::graphics::gl::GLSLShader compShader;

        /** flip the eye configuration for the columns */
        param::ParamSlot flipEyes;

    };


} /* end namespace special */
} /* end namespace core */
} /* end namespace megamol */

#endif /* MEGAMOLCORE_COLSTEREODISPLAY_H_INCLUDED */
