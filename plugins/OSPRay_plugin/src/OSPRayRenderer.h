/*
* OSPRaySphereRenderer.h
* Copyright (C) 2009-2017 by MegaMol Team
* Alle Rechte vorbehalten.
*/
#pragma once

#include "vislib/graphics/gl/GLSLShader.h"
#include "AbstractOSPRayRenderer.h"
#include "mmcore/view/CallRender3D.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/param/ParamSlot.h"
#include "mmcore/moldyn/MultiParticleDataCall.h"


namespace megamol {
namespace ospray {

class OSPRayRenderer : public AbstractOSPRayRenderer {
public:

    /**
    * Answer the name of this module.
    *
    * @return The name of this module.
    */
    static const char *ClassName(void) {
        return "OSPRayRenderer";
    }

    /**
    * Answer a human readable description of this module.
    *
    * @return A human readable description of this module.
    */
    static const char *Description(void) {
        return "Renderer for OSPRay structures.";
    }

    /**
    * Answers whether this module is available on the current system.
    *
    * @return 'true' if the module is available, 'false' otherwise.
    */
    static bool IsAvailable(void) {
        return vislib::graphics::gl::GLSLShader::AreExtensionsAvailable();
    }

    /** Dtor. */
    virtual ~OSPRayRenderer(void);

    /** Ctor. */
    OSPRayRenderer(void);

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
    virtual bool Render(megamol::core::Call& call);

private:

    /**
    * The get extents callback. The module should set the members of
    * 'call' to tell the caller the extents of its data (bounding boxes
    * and times).
    *
    * @param call The calling call.
    *
    * @return The return value of the function.
    */
    virtual bool GetExtents(megamol::core::Call& call);

    /** The call for data */
    core::CallerSlot getStructureSlot;


    /** The texture shader */
    vislib::graphics::gl::GLSLShader osprayShader;
    float scale;

   // Interface dirty flag
    bool InterfaceIsDirty();
    void InterfaceResetDirty();

    // rendering conditions
    bool data_has_changed;
    bool material_has_changed;
    bool light_has_changed;
    bool cam_has_changed;

    vislib::SmartPtr<vislib::graphics::CameraParameters> camParams;
    float time;

    osp::vec2i imgSize;

    // OSPRay textures
    const uint32_t* fb;
    std::vector<float> db;
    OSPTexture2D getOSPDepthTextureFromOpenGLPerspective(megamol::core::Call& call);
    void getOpenGLDepthFromOSPPerspective(megamol::core::Call& call, float* db);

    bool renderer_has_changed;
};

} /*end namespace ospray*/
} /*end namespace megamol*/
