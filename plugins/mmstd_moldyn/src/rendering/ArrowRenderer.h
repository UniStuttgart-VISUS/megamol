/*
 * ArrowRenderer.h
 *
 * Copyright (C) 2009-2017 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#ifndef MMSTD_MOLDYN_ARROWRENDERER_H_INCLUDED
#define MMSTD_MOLDYN_ARROWRENDERER_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "mmcore/view/Renderer3DModule.h"
#include "mmcore/Call.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/param/ParamSlot.h"
#include "vislib/graphics/gl/GLSLShader.h"


namespace megamol {
namespace stdplugin {
namespace moldyn {
namespace rendering {

/**
 * Renderer for simple sphere glyphs
 */
class ArrowRenderer : public core::view::Renderer3DModule {
public:

    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char *ClassName(void) {
        return "ArrowRenderer";
    }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static const char *Description(void) {
        return "Renderer for arrow glyphs.";
    }

    /**
     * Answers whether this module is available on the current system.
     *
     * @return 'true' if the module is available, 'false' otherwise.
     */
    static bool IsAvailable(void) {
        return vislib::graphics::gl::GLSLShader::AreExtensionsAvailable();
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
     * The get capabilities callback. The module should set the members
     * of 'call' to tell the caller its capabilities.
     *
     * @param call The calling call.
     *
     * @return The return value of the function.
     */
    virtual bool GetCapabilities(core::Call& call);

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
     * The render callback.
     *
     * @param call The calling call.
     *
     * @return The return value of the function.
     */
    virtual bool Render(core::Call& call);

private:

    /** The arrow shader */
    vislib::graphics::gl::GLSLShader arrowShader;

    /** The call for data */
    core::CallerSlot getDataSlot;

    /** The call for Transfer function */
    core::CallerSlot getTFSlot;

    ///** The call for clipping plane */
    //core::CallerSlot getClipPlaneSlot;

    /** A simple black-to-white transfer function texture as fallback */
    unsigned int greyTF;

    /** Scaling factor for arrow lengths */
    core::param::ParamSlot lengthScaleSlot;

    /** Length filter for arrow lengths */
    core::param::ParamSlot lengthFilterSlot;

};

} /* end namespace rendering */
} /* end namespace moldyn */
} /* end namespace stdplugin */
} /* end namespace megamol */

#endif /* MMSTD_MOLDYN_ARROWRENDERER_H_INCLUDED */
