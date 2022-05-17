/*
 * ProteinViewRenderer.h
 *
 * Copyright (C) 2021 by Karsten Schatz
 * Copyright (C) 2008-2021 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCORE_PROTEINVIEWRENDERER_H_INCLUDED
#define MEGAMOLCORE_PROTEINVIEWRENDERER_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "glowl/BufferObject.hpp"
#include "glowl/Texture2D.hpp"
#include "image_calls/Image2DCall.h"
#include "mmcore/Call.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/param/ParamSlot.h"
#include "mmcore_gl/utility/SDFFont.h"
#include "mmcore_gl/view/Renderer3DModuleGL.h"
#include "vislib/math/Cuboid.h"
#include "vislib/memutils.h"
#include "vislib_gl/graphics/gl/GLSLShader.h"
#include <glad/gl.h>

namespace megamol {
namespace molsurfmapcluster {


/**
 * Renderer for tri-mesh data
 */
class ProteinViewRenderer : public core_gl::view::Renderer3DModuleGL {
public:
    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char* ClassName(void) {
        return "ProteinViewRenderer";
    }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static const char* Description(void) {
        return "Renderer for tri-mesh data";
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
    ProteinViewRenderer(void);

    /** Dtor. */
    virtual ~ProteinViewRenderer(void);

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
    /** The slot to fetch the data */
    core::CallerSlot getDataSlot;

    /** The slot to fetch the image data */
    core::CallerSlot getImageDataSlot;

    /** Flag whether or not to show vertices */
    core::param::ParamSlot showVertices;

    /** Flag whether or not use lighting for the surface */
    core::param::ParamSlot lighting;

    /** The Triangle winding rule */
    core::param::ParamSlot windRule;

    /** The Triangle color */
    core::param::ParamSlot colorSlot;

    /**  The name slot */
    core::param::ParamSlot nameSlot;

    megamol::core::utility::SDFFont theFont;
    glm::ivec2 m_viewport;

    vislib_gl::graphics::gl::GLSLShader textureShader;
    std::unique_ptr<glowl::BufferObject> texBuffer;
    std::unique_ptr<glowl::Texture2D> texture = nullptr;
    GLuint texVa;

    SIZE_T lastHash = 0;
};


} // namespace molsurfmapcluster
} /* end namespace megamol */

#endif /* MEGAMOLCORE_TRISOUPRENDERER_H_INCLUDED */