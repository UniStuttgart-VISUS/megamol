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
#include <chrono>


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
    virtual bool Render(megamol::core::view::CallRender3D_2& call);

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
    virtual bool GetExtents(megamol::core::view::CallRender3D_2& call);

    /** The call for data */
    core::CallerSlot _getStructureSlot;


    /** The texture shader */
    vislib::graphics::gl::GLSLShader _osprayShader;

    // Interface dirty flag
    bool InterfaceIsDirty();
    void InterfaceResetDirty();

    // rendering conditions
    bool _data_has_changed;
    bool _material_has_changed;
    bool _light_has_changed;
    bool _cam_has_changed;
    bool _transformation_has_changed;

    core::view::Camera_2 _cam;
    float _time;
    size_t _frameID;

    std::array<int,2> _imgSize;

    // OSPRay textures
    uint32_t* _fb;
    std::vector<float> _db;
    void getOpenGLDepthFromOSPPerspective(std::vector<float>& db, cam_type::matrix_type projTemp);

    bool _renderer_has_changed;

    struct {
        unsigned long long int count;
        unsigned long long int amount;
    } _accum_time;
};

} /*end namespace ospray*/
} /*end namespace megamol*/
