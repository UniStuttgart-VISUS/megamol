/*
 * GlyphRenderer.h
 *
 * Copyright (C) 2019 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#ifndef MOLDYN_GLYPHRENDERER_H_INCLUDED
#define MOLDYN_GLYPHRENDERER_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "geometry_calls/EllipsoidalDataCall.h"
#include "mmcore/Call.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/param/ParamSlot.h"
#include "mmcore_gl/utility/SSBOBufferArray.h"
#include "mmstd_gl/renderer/Renderer3DModuleGL.h"

#include "glowl/glowl.h"
#include "mmcore_gl/utility/ShaderFactory.h"

namespace megamol {
namespace moldyn_gl {
namespace rendering {


/**
 * Renderer for ellipsoidal data
 */
class GlyphRenderer : public megamol::mmstd_gl::Renderer3DModuleGL {
public:
    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char* ClassName(void) {
        return "GlyphRenderer";
    }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static const char* Description(void) {
        return "Renderer for data with quaternion orientation and 3 radii";
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
    static void requested_lifetime_resources(frontend_resources::ResourceRequest& req) {
        ModuleGL::requested_lifetime_resources(req);
        req.require<frontend_resources::PerformanceManager>();
    }
#endif

    /** Ctor. */
    GlyphRenderer(void);

    /** Dtor. */
    virtual ~GlyphRenderer(void);

protected:
    /**
     * Implementation of 'Create'.
     *
     * @return 'true' on success, 'false' otherwise.
     */
    bool create(void) override;

    /**
     * The get extents callback. The module should set the members of
     * 'call' to tell the caller the extents of its data (bounding boxes
     * and times).
     *
     * @param call The calling call.
     *
     * @return The return value of the function.
     */
    bool GetExtents(mmstd_gl::CallRender3DGL& call) override;

    /**
     * Implementation of 'Release'.
     */
    void release(void) override;

    bool validateData(geocalls::EllipsoidalParticleDataCall* edc);

    /**
     * The Open GL Render callback.
     *
     * @param call The calling call.
     * @return The return value of the function.
     */
    bool Render(mmstd_gl::CallRender3DGL& call) override;

private:
#ifdef MEGAMOL_USE_PROFILING
    frontend_resources::PerformanceManager::handle_vector timers_;
    frontend_resources::PerformanceManager* perf_manager_ = nullptr;
#endif

    enum Glyph {
        BOX = 0,
        ELLIPSOID = 1,
        ARROW = 2,
        SUPERQUADRIC = 3,
        GIZMO_ARROWGLYPH = 4,
        GIZMO_LINE = 5,
    };

    enum GlyphOptions {
        USE_GLOBAL = 1 << 0,
        USE_TRANSFER_FUNCTION = 1 << 1,
        USE_FLAGS = 1 << 2,
        USE_CLIP = 1 << 3,
        USE_PER_AXIS = 1 << 4
    };

    /**The ellipsoid shader*/
    std::shared_ptr<glowl::GLSLProgram> box_prgm_;
    std::shared_ptr<glowl::GLSLProgram> ellipsoid_prgm_;
    std::shared_ptr<glowl::GLSLProgram> arrow_prgm_;
    std::shared_ptr<glowl::GLSLProgram> superquadric_prgm_;
    std::shared_ptr<glowl::GLSLProgram> gizmo_arrowglyph_prgm_;

    std::vector<core::utility::SSBOBufferArray> position_buffers_;
    std::vector<core::utility::SSBOBufferArray> radius_buffers_;
    std::vector<core::utility::SSBOBufferArray> direction_buffers_;
    std::vector<core::utility::SSBOBufferArray> color_buffers_;

    /** The slot to fetch the data */
    megamol::core::CallerSlot get_data_slot_;
    megamol::core::CallerSlot get_tf_slot_;
    megamol::core::CallerSlot get_clip_plane_slot_;
    megamol::core::CallerSlot read_flags_slot_;

    megamol::core::param::ParamSlot glyph_param_;
    megamol::core::param::ParamSlot scale_param_;
    megamol::core::param::ParamSlot radius_scale_param_;
    megamol::core::param::ParamSlot orientation_param_;
    megamol::core::param::ParamSlot length_filter_param_;
    megamol::core::param::ParamSlot color_interpolation_param_;
    megamol::core::param::ParamSlot min_radius_param_;
    megamol::core::param::ParamSlot color_mode_param_;
    megamol::core::param::ParamSlot superquadric_exponent_param_;
    megamol::core::param::ParamSlot gizmo_arrow_thickness_;

    SIZE_T last_hash_ = -1;
    uint32_t last_frame_id_ = -1;
    // TODO: glowl/Texture1D?
    GLuint grey_tf_;
};

} // namespace rendering
} // namespace moldyn_gl
} // namespace megamol

#endif /* MOLDYN_GLYPHRENDERER_H_INCLUDED */
