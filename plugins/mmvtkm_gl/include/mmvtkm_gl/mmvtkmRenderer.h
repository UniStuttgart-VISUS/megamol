/*
 * mmvtkmRenderer.h
 *
 * Copyright (C) 2020-2021 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#pragma once

#include "mmcore/CallerSlot.h"
#include "mmcore/param/ParamSlot.h"
#include "mmstd_gl/renderer/Renderer3DModuleGL.h"
#include "mmvtkm/mmvtkmDataCall.h"

// #include "vtkm/rendering/Actor.h"
// #include "vtkm/rendering/CanvasRayTracer.h"
// #include "vtkm/rendering/MapperRayTracer.h"
// #include "vtkm/rendering/MapperVolume.h"
// #include "vtkm/rendering/Scene.h"
// #include "vtkm/rendering/View3D.h"


namespace megamol::mmvtkm_gl {

/**
 * Renderer for vtkm data
 */
class mmvtkmDataRenderer : public mmstd_gl::Renderer3DModuleGL {
public:
    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char* ClassName() {
        return "vtkmDataRenderer";
    }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static const char* Description() {
        return "Renderer for vtkm data.";
    }

    /**
     * Answers whether this module is available on the current system.
     *
     * @return 'true' if the module is available, 'false' otherwise.
     */
    static bool IsAvailable() {
        return true;
    }

    /** Ctor. */
    mmvtkmDataRenderer();

    /** Dtor. */
    ~mmvtkmDataRenderer() override;

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
    bool Render(mmstd_gl::CallRender3DGL& call) override;

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

private:
    /** Callback function for psColorTables paramslot */
    bool setLocalUpdate(core::param::ParamSlot& slot);

    /** caller slot */
    core::CallerSlot vtkmDataCallerSlot_;

    /** Paramslot for vtkm colortables */
    core::param::ParamSlot psColorTables_;

    /** Some vtkm data set */
    std::shared_ptr<mmvtkm::VtkmData> vtkmDataSet_;
    mmvtkm::VtkmMetaData vtkmMetaData_;

    /** The vtkm structures used for rendering */
    // vtkm::rendering::Scene scene_;
    // vtkm::rendering::MapperRayTracer mapper_;
    // vtkm::rendering::CanvasRayTracer canvas_;
    // vtkm::rendering::Camera vtkmCamera_;
    // vtkm::rendering::View3D view;
    void* colorArray_;
    float canvasWidth_;
    float canvasHeight_;
    float canvasDepth_;

    /** Various vtkm specific colortables */
    std::vector<const char*> colorTables_{
        "Viridis",               // 0
        "Cool to Warm",          // 1
        "Cool to Warm Extended", // 2
        "Inferno",               // 3
        "Plasma",                // 4
        "Black-Body Radiation",  // 5
        "X Ray",                 // 6
        "Green",                 // 7
        "Black - Blue - White",  // 8
        "Blue to Orange",        // 9
        "Gray to Red",           // 10
        "Cold and Hot",          // 11
        "Blue - Green - Orange", // 12
        "Yellow - Gray - Blue",  // 13
        "Rainbow Uniform",       // 14
        "Jet",                   // 15
        "Rainbow Desaturated"    // 16
    };

    /** Used for version controlling */
    bool localUpdate_;
};

} // namespace megamol::mmvtkm_gl
