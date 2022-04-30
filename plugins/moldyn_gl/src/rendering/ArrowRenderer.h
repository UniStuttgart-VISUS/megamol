/*
 * ArrowRenderer.h
 *
 * Copyright (C) 2009 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCORE_ARROWRENDERER_H_INCLUDED
#define MEGAMOLCORE_ARROWRENDERER_H_INCLUDED


#include "geometry_calls/MultiParticleDataCall.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/CoreInstance.h"
#include "mmcore/param/FloatParam.h"
#include "mmcore/param/ParamSlot.h"
#include "mmcore/view/CallClipPlane.h"
#include "mmcore_gl/flags/FlagCallsGL.h"
#include "mmcore_gl/view/CallGetTransferFunctionGL.h"
#include "mmcore_gl/view/CallRender3DGL.h"
#include "mmcore_gl/view/Renderer3DModuleGL.h"

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
class ArrowRenderer : public core_gl::view::Renderer3DModuleGL {
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

#ifdef PROFILING
    std::vector<std::string> requested_lifetime_resources() override {
        std::vector<std::string> resources = Module::requested_lifetime_resources();
        resources.emplace_back(frontend_resources::PerformanceManager_Req_Name);
        return resources;
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
    virtual bool GetExtents(core_gl::view::CallRender3DGL& call);

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
    virtual bool Render(core_gl::view::CallRender3DGL& call);

private:
#ifdef PROFILING
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
