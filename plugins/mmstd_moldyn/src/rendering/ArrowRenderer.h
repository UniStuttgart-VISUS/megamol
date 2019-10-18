/*
 * ArrowRenderer.h
 *
 * Copyright (C) 2009 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCORE_ARROWRENDERER_H_INCLUDED
#define MEGAMOLCORE_ARROWRENDERER_H_INCLUDED


#include "mmcore/CallerSlot.h"
#include "mmcore/param/ParamSlot.h"
#include "mmcore/moldyn/MultiParticleDataCall.h"
#include "mmcore/CoreInstance.h"
#include "mmcore/param/FloatParam.h"
#include "mmcore/view/CallClipPlane.h"
#include "mmcore/view/CallGetTransferFunction.h"
#include "mmcore/FlagCall.h"
#include "mmcore/view/CallRender3D_2.h"
#include "mmcore/view/Renderer3DModule_2.h"

#include "vislib/assert.h"
#include "vislib/graphics/gl/GLSLShader.h"
#include "vislib/graphics/gl/IncludeAllGL.h"


namespace megamol {
namespace stdplugin {
namespace moldyn {
namespace rendering {

    using namespace megamol::core;


    /**
     * Renderer for simple sphere glyphs
     */
    class ArrowRenderer : public view::Renderer3DModule_2 {
    public:

        /**
         * Answer the name of this module.
         *
         * @return The name of this module.
         */
        static const char *ClassName(void) {
            return "ArrowRenderer";
        }

        /**
         * Answer a human readable description of this module.
         *
         * @return A human readable description of this module.
         */
        static const char *Description(void) {
            return "Renderer for arrow glyphs.";
        }

        /**
         * Answers whether this module is available on the current system.
         *
         * @return 'true' if the module is available, 'false' otherwise.
         */
        static bool IsAvailable(void) {
            return vislib::graphics::gl::GLSLShader::AreExtensionsAvailable();
        }

        /** Ctor. */
        ArrowRenderer(void);

        /** Dtor. */
        virtual ~ArrowRenderer(void);

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
        virtual bool GetExtents(view::CallRender3D_2& call);

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
        virtual bool Render(view::CallRender3D_2& call);

    private:

        /** The call for data */
        CallerSlot getDataSlot;

        /** The call for Transfer function */
        CallerSlot getTFSlot;

        /** The call for selection flags */
        CallerSlot getFlagsSlot;

        /** The call for clipping plane */
        CallerSlot getClipPlaneSlot;

        /** The arrow shader */
        vislib::graphics::gl::GLSLShader arrowShader;

        /** A simple black-to-white transfer function texture as fallback */
        unsigned int greyTF;
        
        /** Scaling factor for arrow lengths */
        param::ParamSlot lengthScaleSlot;
        
        /** Length filter for arrow lengths */
        param::ParamSlot lengthFilterSlot;

    };

} /* end namespace rendering */
} /* end namespace moldyn */
} /* end namespace core */
} /* end namespace megamol */

#endif /* MEGAMOLCORE_ARROWRENDERER_H_INCLUDED */
