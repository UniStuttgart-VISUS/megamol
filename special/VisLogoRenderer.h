/*
 * VisLogoRenderer.h
 *
 * Copyright (C) 2008 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCORE_VISLOGORENDERER_H_INCLUDED
#define MEGAMOLCORE_VISLOGORENDERER_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "view/Renderer3DModule.h"
#include "Call.h"
#include "param/ParamSlot.h"
#include "vislib/OpenGLVISLogo.h"


namespace megamol {
namespace core {
namespace special {


    /**
     * Renderer for rendering the vis logo into the unit cube.
     */
    class VisLogoRenderer : public view::Renderer3DModule {
    public:

        /**
         * Answer the name of this module.
         *
         * @return The name of this module.
         */
        static const char *ClassName(void) {
            return "VisLogoRenderer";
        }

        /**
         * Answer a human readable description of this module.
         *
         * @return A human readable description of this module.
         */
        static const char *Description(void) {
            return "Renderer of the VIS logo.";
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
        VisLogoRenderer(void);

        /** Dtor. */
        virtual ~VisLogoRenderer(void);

    protected:

        /**
         * Implementation of 'Create'.
         *
         * @return 'true' on success, 'false' otherwise.
         */
        virtual bool create(void);

        /**
         * Implementation of 'Release'.
         */
        virtual void release(void);

        /**
         * The get capabilities callback. The module should set the members
         * of 'call' to tell the caller its capabilities.
         *
         * @param call The calling call.
         *
         * @return The return value of the function.
         */
        virtual bool GetCapabilities(Call& call);

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

        /** The vis logo */
        vislib::graphics::gl::OpenGLVISLogo visLogo;

        /** The scale param */
        param::ParamSlot scale;

    };


} /* end namespace special */
} /* end namespace core */
} /* end namespace megamol */

#endif /* MEGAMOLCORE_VISLOGORENDERER_H_INCLUDED */
