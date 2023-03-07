/*
 * OSPRaySphereRenderer.h
 * Copyright (C) 2009-2017 by MegaMol Team
 * Alle Rechte vorbehalten.
 */
#pragma once

#include "AbstractOSPRayRenderer.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/param/ParamSlot.h"


namespace megamol::ospray {

class OSPRayRenderer : public AbstractOSPRayRenderer {
public:
    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char* ClassName() {
        return "OSPRayRenderer";
    }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static const char* Description() {
        return "Renderer for OSPRay structures.";
    }

    /**
     * Answers whether this module is available on the current system.
     *
     * @return 'true' if the module is available, 'false' otherwise.
     */
    static bool IsAvailable() {
        return true;
    }

    /** Dtor. */
    ~OSPRayRenderer() override;

    /** Ctor. */
    OSPRayRenderer();

protected:
    /**
     * Implementation of 'Create'.
     *
     * @return 'true' on success, 'false' otherwise.
     */
    bool create() override;

    /**
     * Implementation of 'Release'.
     */
    void release() override;

    /**
     * The render callback.
     *
     * @param call The calling call.
     *
     * @return The return value of the function.
     */
    bool Render(megamol::core::view::CallRender3D& call) override;

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
    bool GetExtents(megamol::core::view::CallRender3D& call) override;

    bool OnMouseButton(
        core::view::MouseButton button, core::view::MouseButtonAction action, core::view::Modifiers mods) override;

    bool OnMouseMove(double x, double y) override;

    /** The call for data */
    core::CallerSlot _getStructureSlot;

    core::param::ParamSlot _enablePickingSlot;


    // Interface dirty flag
    bool InterfaceIsDirty();
    void InterfaceResetDirty();

    // rendering conditions
    bool _data_has_changed;
    bool _material_has_changed;
    bool _light_has_changed;
    bool _cam_has_changed;
    bool _transformation_has_changed;
    bool _clipping_geo_changed;

    core::view::Camera _cam;
    float _time;
    size_t _frameID;

    std::array<int, 2> _imgSize;

    // OSPRay textures
    std::vector<uint32_t> _fb;
    std::vector<float> _db;
    void getOpenGLDepthFromOSPPerspective(std::vector<float>& db);

    bool _renderer_has_changed;

    struct {
        unsigned long long int count;
        unsigned long long int amount;
    } _accum_time;

    float _mouse_x;
    float _mouse_y;
};

} // namespace megamol::ospray
