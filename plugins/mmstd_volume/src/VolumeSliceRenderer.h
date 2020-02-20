/*
 * VolumeSliceRenderer.h
 *
 * Copyright (C) 2012-2019 by Universitaet Stuttgart (VISUS).
 * All rights reserved.
 */

#ifndef MMSTD_VOLUME_RENDERVOLUMESLICE_H_INCLUDED
#define MMSTD_VOLUME_RENDERVOLUMESLICE_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "mmcore/CallerSlot.h"
#include "mmcore/view/CallRender3D_2.h"
#include "mmcore/view/Renderer3DModule_2.h"

#include "vislib/graphics/gl/GLSLComputeShader.h"
#include "vislib/graphics/gl/GLSLShader.h"

namespace megamol {
namespace stdplugin {
namespace volume {

    /**
     * Renders one slice of a volume (slow)
     */
    class VolumeSliceRenderer : public core::view::Renderer3DModule_2 {
    public:
        /**
         * Answer the name of this module.
         *
         * @return The name of this module.
         */
        static const char *ClassName(void) {
            return "VolumeSliceRenderer_2";
        }

        /**
         * Answer a human readable description of this module.
         *
         * @return A human readable description of this module.
         */
        static const char *Description(void) {
            return "Renders one slice of a volume";
        }

        /**
         * Answers whether this module is available on the current system.
         *
         * @return 'true' if the module is available, 'false' otherwise.
         */
        static bool IsAvailable(void) {
            return true;
        }

        VolumeSliceRenderer(void);
        virtual ~VolumeSliceRenderer(void);

    protected:
        /**
         * Implementation of 'Create'.
         *
         * @return 'true' on success, 'false' otherwise.
         */
        virtual bool create(void) override;

		/**
		* Implementation of 'Release'.
		*/
		virtual void release(void) override;

        /**
         * The get extents callback. The module should set the members of
         * 'call' to tell the caller the extents of its data (bounding boxes
         * and times).
         *
         * @param call The calling call.
         *
         * @return The return value of the function.
         */
        virtual bool GetExtents(core::view::CallRender3D_2& call) override;

        /**
         * The render callback.
         *
         * @param call The calling call.
         *
         * @return The return value of the function.
         */
        virtual bool Render(core::view::CallRender3D_2& call) override;

    private:
        /** The call for data */
        core::CallerSlot getVolSlot;

        /** The call for Transfer function */
        core::CallerSlot getTFSlot;

        /** The call for clipping plane */
        core::CallerSlot getClipPlaneSlot;

		/** Shader */
		vislib::graphics::gl::GLSLComputeShader compute_shader;
		vislib::graphics::gl::GLSLShader render_shader;
    };

} /* end namespace volume */
} /* end namespace stdplugin */
} /* end namespace megamol */

#endif /* MMSTD_VOLUME_RENDERVOLUMESLICE_H_INCLUDED */
