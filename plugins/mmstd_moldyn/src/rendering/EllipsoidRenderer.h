/*
 * EllipsoidRenderer.h
 *
 * Copyright (C) 2008-2015 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCORE_ELLIPSOIDRENDERER_H_INCLUDED
#define MEGAMOLCORE_ELLIPSOIDRENDERER_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#    pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "mmcore/Call.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/moldyn/EllipsoidalDataCall.h"
#include "mmcore/param/ParamSlot.h"
#include "mmcore/view/AbstractCallRender3D.h"
#include "mmcore/view/CallRender3D.h"
#include "mmcore/view/Renderer3DModule.h"
#include "vislib/graphics/gl/GLSLShader.h"
#include "vislib/graphics/gl/IncludeAllGL.h"


namespace megamol {
namespace stdplugin {
namespace moldyn {
namespace rendering {


/**
 * Renderer for ellipsoidal data
 */
class EllipsoidRenderer : public megamol::core::view::Renderer3DModule {
public:
    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char* ClassName(void) { return "EllipsoidRenderer"; }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static const char* Description(void) { return "Renderer for ellipsoidal data"; }

    /**
     * Answers whether this module is available on the current system.
     *
     * @return 'true' if the module is available, 'false' otherwise.
     */
    static bool IsAvailable(void) { return vislib::graphics::gl::GLSLShader::AreExtensionsAvailable(); }

    /** Ctor. */
    EllipsoidRenderer(void);

    /** Dtor. */
    virtual ~EllipsoidRenderer(void);

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
     * The Open GL Render callback.
     *
     * @param call The calling call.
     * @return The return value of the function.
     */
    virtual bool Render(megamol::core::Call& call);

private:
    /**The ellipsoid shader*/
    vislib::graphics::gl::GLSLShader ellipsoidShader;

    /** The slot to fetch the data */
    megamol::core::CallerSlot getDataSlot;

    // camera information
    vislib::SmartPtr<vislib::graphics::CameraParameters> cameraInfo;

    // attribute locations for GLSL-Shader
    GLint attribLocInParams;
    GLint attribLocQuatC;
    GLint attribLocColor1;
    GLint attribLocColor2;
};

} // namespace rendering
} // namespace moldyn
} // namespace stdplugin
} // namespace megamol

#endif /* MEGAMOLCORE_ELLIPSOIDRENDERER_H_INCLUDED */
