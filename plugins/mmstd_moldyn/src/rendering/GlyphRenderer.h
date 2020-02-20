/*
 * GlyphRenderer.h
 *
 * Copyright (C) 2019 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#ifndef MOLDYN_GLYPHRENDERER_H_INCLUDED
#define MOLDYN_GLYPHRENDERER_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#    pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "mmcore/Call.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/param/ParamSlot.h"
#include "mmcore/view/AbstractCallRender3D.h"
#include "mmcore/view/Renderer3DModule_2.h"
#include "vislib/graphics/gl/GLSLShader.h"
#include "vislib/graphics/gl/IncludeAllGL.h"
#include "mmcore/utility/SSBOBufferArray.h"
#include "mmcore/moldyn/EllipsoidalDataCall.h"

namespace megamol {
namespace stdplugin {
namespace moldyn {
namespace rendering {


/**
 * Renderer for ellipsoidal data
 */
class GlyphRenderer : public megamol::core::view::Renderer3DModule_2 {
public:
    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char* ClassName(void) { return "GlyphRenderer"; }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static const char* Description(void) { return "Renderer for data with quaternion orientation and 3 radii"; }

    /**
     * Answers whether this module is available on the current system.
     *
     * @return 'true' if the module is available, 'false' otherwise.
     */
    static bool IsAvailable(void) { return vislib::graphics::gl::GLSLShader::AreExtensionsAvailable(); }

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

    bool makeShader(std::string vertexName, std::string fragmentName, vislib::graphics::gl::GLSLShader& shader);

    /**
     * The get extents callback. The module should set the members of
     * 'call' to tell the caller the extents of its data (bounding boxes
     * and times).
     *
     * @param call The calling call.
     *
     * @return The return value of the function.
     */
    bool GetExtents(core::view::CallRender3D_2& call) override;

    /**
     * Implementation of 'Release'.
     */
    void release(void) override;

    bool validateData(core::moldyn::EllipsoidalParticleDataCall *edc);

    /**
     * The Open GL Render callback.
     *
     * @param call The calling call.
     * @return The return value of the function.
     */
    bool Render(core::view::CallRender3D_2& call) override;

private:

    enum Glyph {
        BOX = 0,
        ELLIPSOID = 1,
        ARROW = 2,
        SUPERQUADRIC = 3,
    };

    enum glyph_options { USE_GLOBAL = 1 << 0, USE_TRANSFER_FUNCTION = 1 << 1, USE_FLAGS = 1 << 2, USE_CLIP = 1 << 3 };

    /**The ellipsoid shader*/
    vislib::graphics::gl::GLSLShader ellipsoidShader;
    vislib::graphics::gl::GLSLShader boxShader;

    std::vector<core::utility::SSBOBufferArray> position_buffers;
    std::vector<core::utility::SSBOBufferArray> radius_buffers;
    std::vector<core::utility::SSBOBufferArray> direction_buffers;
    std::vector<core::utility::SSBOBufferArray> color_buffers;
    
    /** The slot to fetch the data */
    megamol::core::CallerSlot getDataSlot;
    megamol::core::CallerSlot getTFSlot;
    megamol::core::CallerSlot getClipPlaneSlot;
    megamol::core::CallerSlot readFlagsSlot;

    megamol::core::param::ParamSlot glyphParam;
    megamol::core::param::ParamSlot scaleParam;
    megamol::core::param::ParamSlot colorInterpolationParam;

    SIZE_T lastHash = -1;
    uint32_t lastFrameID = -1;
    GLuint greyTF;
};

} // namespace rendering
} // namespace moldyn
} // namespace stdplugin
} // namespace megamol

#endif /* MOLDYN_GLYPHRENDERER_H_INCLUDED */
