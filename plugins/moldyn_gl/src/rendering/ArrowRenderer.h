/*
 * ArrowRenderer.h
 *
 * Copyright (C) 2009 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCORE_ARROWRENDERER_H_INCLUDED
#define MEGAMOLCORE_ARROWRENDERER_H_INCLUDED

#include "PerformanceManager.h"

#include "geometry_calls/MultiParticleDataCall.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/CoreInstance.h"
#include "mmcore/param/FloatParam.h"
#include "mmcore/param/ParamSlot.h"
#include "mmcore/view/CallClipPlane.h"
#include "mmcore_gl/FlagCallsGL.h"
#include "mmcore_gl/view/CallGetTransferFunctionGL.h"
#include "mmcore_gl/view/CallRender3DGL.h"
#include "mmcore_gl/view/Renderer3DModuleGL.h"

#include "vislib/assert.h"
#include "vislib_gl/graphics/gl/GLSLShader.h"
#include "vislib_gl/graphics/gl/IncludeAllGL.h"


namespace megamol {
namespace moldyn_gl {
namespace rendering {

using namespace megamol::core;


/**
 * Renderer for simple sphere glyphs
 */
class ArrowRenderer : public core_gl::view::Renderer3DModuleGL {
public:
#ifdef PROFILING
    std::vector<std::string> requested_lifetime_resources() override {
        std::vector<std::string> resources = ModuleGL::requested_lifetime_resources();
        resources.emplace_back(frontend_resources::PerformanceManager_Req_Name);
        return resources;
    }
#endif

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
    void loadData(geocalls::MultiParticleDataCall& in_data);

    /** The call for data */
    CallerSlot getDataSlot;

    /** The call for Transfer function */
    CallerSlot getTFSlot;

    /** The call for selection flags */
    CallerSlot getFlagsSlot;

    /** The call for clipping plane */
    CallerSlot getClipPlaneSlot;

    /** The call for light sources */
    core::CallerSlot getLightsSlot;

    /** The arrow shader */
    vislib_gl::graphics::gl::GLSLShader arrowShader;
    std::unique_ptr<glowl::GLSLProgram> arrowShader_;

    /** A simple black-to-white transfer function texture as fallback */
    unsigned int greyTF;

    /** Scaling factor for arrow lengths */
    param::ParamSlot lengthScaleSlot;

    /** Length filter for arrow lengths */
    param::ParamSlot lengthFilterSlot;

    std::vector<GLuint> data_buf_;

    uint64_t in_data_hash_ = std::numeric_limits<uint64_t>::max();

    int in_frame_id_ = -1;

#ifdef PROFILING
    frontend_resources::PerformanceManager::handle_vector timing_handles_;
#endif
};

} /* end namespace rendering */
} // namespace moldyn_gl
} /* end namespace megamol */

#endif /* MEGAMOLCORE_ARROWRENDERER_H_INCLUDED */
