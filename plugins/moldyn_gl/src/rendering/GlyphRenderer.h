/*
 * GlyphRenderer.h
 *
 * Copyright (C) 2022 by VISUS (Universitaet Stuttgart)
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
#include "mmcore_gl/view/Renderer3DModuleGL.h"
#include "vislib_gl/graphics/gl/GLSLShader.h"
#include "vislib_gl/graphics/gl/IncludeAllGL.h"

namespace megamol {
namespace moldyn_gl {
namespace rendering {


/**
 * Renderer for ellipsoidal data
 */
class GlyphRenderer : public megamol::core_gl::view::Renderer3DModuleGL {
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

    bool makeShader(std::string vertex_name, std::string fragment_name, vislib_gl::graphics::gl::GLSLShader& shader);

    /**
     * The get extents callback. The module should set the members of
     * 'call' to tell the caller the extents of its data (bounding boxes
     * and times).
     *
     * @param call The calling call.
     *
     * @return The return value of the function.
     */
    bool GetExtents(core_gl::view::CallRender3DGL& call) override;

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
    bool Render(core_gl::view::CallRender3DGL& call) override;

private:
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
    vislib_gl::graphics::gl::GLSLShader m_box_shader;
    vislib_gl::graphics::gl::GLSLShader m_ellipsoid_shader;
    vislib_gl::graphics::gl::GLSLShader m_arrow_shader;
    vislib_gl::graphics::gl::GLSLShader m_superquadric_shader;
    vislib_gl::graphics::gl::GLSLShader m_gizmo_arrowglyph_shader;
    vislib_gl::graphics::gl::GLSLShader m_gizmo_line_shader;

    std::vector<core::utility::SSBOBufferArray> m_position_buffers;
    std::vector<core::utility::SSBOBufferArray> m_radius_buffers;
    std::vector<core::utility::SSBOBufferArray> m_direction_buffers;
    std::vector<core::utility::SSBOBufferArray> m_color_buffers;

    /** The slot to fetch the data */
    megamol::core::CallerSlot m_get_data_slot;
    megamol::core::CallerSlot m_get_tf_slot;
    megamol::core::CallerSlot m_get_clip_plane_slot;
    megamol::core::CallerSlot m_read_flags_slot;

    megamol::core::param::ParamSlot m_glyph_param;
    megamol::core::param::ParamSlot m_scale_param;
    megamol::core::param::ParamSlot m_radius_scale_param;
    megamol::core::param::ParamSlot m_orientation_param;
    megamol::core::param::ParamSlot m_length_filter_param;
    megamol::core::param::ParamSlot m_color_interpolation_param;
    megamol::core::param::ParamSlot m_min_radius_param;
    megamol::core::param::ParamSlot m_color_mode_param;
    megamol::core::param::ParamSlot m_superquadric_exponent_param;

    SIZE_T m_last_hash = -1;
    uint32_t m_last_frame_id = -1;
    GLuint m_grey_tf;
};

} // namespace rendering
} // namespace moldyn_gl
} // namespace megamol

#endif /* MOLDYN_GLYPHRENDERER_H_INCLUDED */
