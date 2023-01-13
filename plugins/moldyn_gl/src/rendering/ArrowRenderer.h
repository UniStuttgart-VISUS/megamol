/*
 * ArrowRenderer.h
 *
 * Copyright (C) 2009 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCORE_ARROWRENDERER_H_INCLUDED
#define MEGAMOLCORE_ARROWRENDERER_H_INCLUDED


#include "OpenGL_Context.h"
#include "geometry_calls/MultiParticleDataCall.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/param/FloatParam.h"
#include "mmcore/param/ParamSlot.h"
#include "mmstd/renderer/CallClipPlane.h"
#include "mmstd_gl/flags/FlagCallsGL.h"
#include "mmstd_gl/renderer/CallGetTransferFunctionGL.h"
#include "mmstd_gl/renderer/CallRender3DGL.h"
#include "mmstd_gl/renderer/Renderer3DModuleGL.h"

#include "glowl/glowl.h"
#include "mmcore_gl/utility/ShaderFactory.h"

#include "vislib/assert.h"


namespace megamol {
namespace moldyn_gl {
namespace rendering {

using namespace megamol::core;


/**
 * Renderer for simple sphere glyphs
 */
class ArrowRenderer : public mmstd_gl::Renderer3DModuleGL {
public:
    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char* ClassName(void) {
        return "ArrowRenderer";
    }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static const char* Description(void) {
        return "Renderer for arrow glyphs.";
    }

    /**
     * Answers whether this module is available on the current system.
     *
     * @return 'true' if the module is available, 'false' otherwise.
     */
    static bool IsAvailable(void) {
        return true;
    }

#ifdef MEGAMOL_USE_PROFILING
    void requested_lifetime_resources(frontend_resources::ResourceRequest& req) override {
        ModuleGL::requested_lifetime_resources(req);
        req.require<frontend_resources::PerformanceManager>();
    }
#endif

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
    virtual bool GetExtents(mmstd_gl::CallRender3DGL& call);

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
    virtual bool Render(mmstd_gl::CallRender3DGL& call);

private:
#ifdef MEGAMOL_USE_PROFILING
    frontend_resources::PerformanceManager::handle_vector timers_;
    frontend_resources::PerformanceManager* perf_manager_ = nullptr;
#endif

    /** The call for data */
    CallerSlot get_data_slot_;

    /** The call for Transfer function */
    CallerSlot get_tf_slot_;

    /** The call for selection flags */
    CallerSlot get_flags_slot_;

    /** The call for clipping plane */
    CallerSlot get_clip_plane_slot_;

    /** The call for light sources */
    core::CallerSlot get_lights_slot_;

    /** The arrow shader */
    std::unique_ptr<glowl::GLSLProgram> arrow_pgrm_;

    /** A simple black-to-white transfer function texture as fallback */
    unsigned int grey_tf_;

    /** Scaling factor for arrow lengths */
    param::ParamSlot length_scale_slot_;

    /** Length filter for arrow lengths */
    param::ParamSlot length_filter_slot_;
};

} /* end namespace rendering */
} // namespace moldyn_gl
} /* end namespace megamol */

#endif /* MEGAMOLCORE_ARROWRENDERER_H_INCLUDED */
