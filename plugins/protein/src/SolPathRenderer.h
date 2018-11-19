/*
 * SolPathRenderer.h
 *
 * Copyright (C) 2010 by VISUS (University of Stuttgart)
 * Alle Rechte vorbehalten.
 */
#ifndef MEGAMOL_PROTEIN_SOLPATHRENDERER_H_INCLUDED
#define MEGAMOL_PROTEIN_SOLPATHRENDERER_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "mmcore/view/Renderer3DModule.h"
#include "mmcore/param/ParamSlot.h"
#include "vislib/graphics/gl/IncludeAllGL.h"
#include "vislib/graphics/gl/GLSLShader.h"
#include "mmcore/Call.h"
#include "mmcore/CallerSlot.h"


namespace megamol {
namespace protein {

    /**
     * Renderer for solvent path raw data
     */
    class SolPathRenderer : public megamol::core::view::Renderer3DModule {
    public:

        /**
         * Answer the name of this module.
         *
         * @return The name of this module.
         */
        static const char *ClassName(void) {
            return "SolPathRenderer";
        }

        /**
         * Answer a human readable description of this module.
         *
         * @return A human readable description of this module.
         */
        static const char *Description(void) {
            return "Renderer for solvent path raw data.";
        }

        /**
         * Answers whether this module is available on the current system.
         *
         * @return 'true' if the module is available, 'false' otherwise.
         */
        static bool IsAvailable(void) {
            return vislib::graphics::gl::GLSLShader::AreExtensionsAvailable();
        }

        /** ctor */
        SolPathRenderer(void);

        /** dtor */
        virtual ~SolPathRenderer(void);

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
        virtual bool GetExtents(core::Call& call);

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
        virtual bool Render(core::Call& call);

    private:

        /** The slot to get the data */
        core::CallerSlot getdataslot;

        /** The shader for shading the path lines */
        vislib::graphics::gl::GLSLShader pathlineShader;

        /** The shader for shading the dots */
        vislib::graphics::gl::GLSLShader dotsShader;

    };

} /* end namespace protein */
} /* end namespace megamol */

#endif /*  MEGAMOL_PROTEIN_SOLPATHRENDERER_H_INCLUDED */
