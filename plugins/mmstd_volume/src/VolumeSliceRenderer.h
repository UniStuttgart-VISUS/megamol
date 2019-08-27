/*
 * RenderVolumeSlice.h
 *
 * Copyright (C) 2012 by Universitaet Stuttgart (VISUS). 
 * Alle Rechte vorbehalten.
 */

#ifndef MMSTD_VOLUME_RENDERVOLUMESLICE_H_INCLUDED
#define MMSTD_VOLUME_RENDERVOLUMESLICE_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */


#include "mmcore/view/Renderer3DModule.h"
//#include "Call.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/param/ParamSlot.h"
//#include "vislib/GLSLShader.h"
//#include "Module.h"
//#include "CalleeSlot.h"
#include "mmcore/CallerSlot.h"
//#include "param/ParamSlot.h"
//#include "vislib/Cuboid.h"


namespace megamol {
namespace stdplugin {
namespace volume {

    /**
     * Renders one slice of a volume (slow)
     */
    class RenderVolumeSlice : public core::view::Renderer3DModule {
    public:


        /**
         * Answer the name of this module.
         *
         * @return The name of this module.
         */
        static const char *ClassName(void) {
            return "RenderVolumeSlice";
        }

        /**
         * Answer a human readable description of this module.
         *
         * @return A human readable description of this module.
         */
        static const char *Description(void) {
            return "Renders one slice of a volume (slow)";
        }

        /**
         * Answers whether this module is available on the current system.
         *
         * @return 'true' if the module is available, 'false' otherwise.
         */
        static bool IsAvailable(void) {
            return true;
        }

        /** ctor */
        RenderVolumeSlice(void);

        /** dtor */
        virtual ~RenderVolumeSlice(void);

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

        /** The call for data */
        core::CallerSlot getVolSlot;

        /** The call for Transfer function */
        core::CallerSlot getTFSlot;

        /** The call for clipping plane */
        core::CallerSlot getClipPlaneSlot;

        /** The volume attribute to show */
        core::param::ParamSlot attributeSlot;

        /** minimum value */
        core::param::ParamSlot lowValSlot;

        /** maximum value */
        core::param::ParamSlot highValSlot;

    };

} /* end namespace volume */
} /* end namespace stdplugin */
} /* end namespace megamol */

#endif /* MMSTD_VOLUME_RENDERVOLUMESLICE_H_INCLUDED */
