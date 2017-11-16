/*
 * SimpleGeoSphereRenderer.h
 *
 * Copyright (C) 2012 by CGV (TU Dresden)
 * Copyright (C) 2017 by MegaMol Team (VISUS)
 *
 * Alle Rechte vorbehalten.
 */

#ifndef MMSTD_MOLDYN_SIMPLEGEOSPHERERENDERER_H_INCLUDED
#define MMSTD_MOLDYN_SIMPLEGEOSPHERERENDERER_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "mmcore/moldyn/AbstractSimpleSphereRenderer.h"
#include "vislib/graphics/gl/GLSLGeometryShader.h"


namespace megamol {
namespace stdplugin {
namespace moldyn {
namespace rendering {

/**
 * Renderer for simple sphere glyphs with geometry shader
 */
class SimpleGeoSphereRenderer : public core::moldyn::AbstractSimpleSphereRenderer {
public:

    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char *ClassName(void) {
        return "SimpleGeoSphereRenderer";
    }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static const char *Description(void) {
        return "Renderer for sphere glyphs using geometry shader";
    }

    /**
     * Answers whether this module is available on the current system.
     *
     * @return 'true' if the module is available, 'false' otherwise.
     */
    static bool IsAvailable(void) {
#ifdef _WIN32
#if defined(DEBUG) || defined(_DEBUG)
        HDC dc = ::wglGetCurrentDC();
        HGLRC rc = ::wglGetCurrentContext();
        ASSERT(dc != NULL);
        ASSERT(rc != NULL);
#endif // DEBUG || _DEBUG
#endif // _WIN32
        return vislib::graphics::gl::GLSLShader::AreExtensionsAvailable()
            && vislib::graphics::gl::GLSLGeometryShader::AreExtensionsAvailable()
            && ogl_IsVersionGEQ(2, 0)
            && isExtAvailable("GL_EXT_geometry_shader4")
            && isExtAvailable("GL_EXT_gpu_shader4")
            && isExtAvailable("GL_EXT_bindable_uniform")
            && isExtAvailable("GL_ARB_shader_objects");
    }

    /** Ctor. */
    SimpleGeoSphereRenderer(void);

    /** Dtor. */
    virtual ~SimpleGeoSphereRenderer(void);

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
     * The render callback.
     *
     * @param call The calling call.
     *
     * @return The return value of the function.
     */
    virtual bool Render(core::Call& call);

private:

    /** The sphere shader */
    vislib::graphics::gl::GLSLGeometryShader sphereShader;

};

} /* end namespace rendering */
} /* end namespace moldyn */
} /* end namespace stdplugin */
} /* end namespace megamol */

#endif /* MMSTD_MOLDYN_SIMPLEGEOSPHERERENDERER_H_INCLUDED */
