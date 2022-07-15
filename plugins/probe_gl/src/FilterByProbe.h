/*
 * FilterByProbe.h
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VISUS).
 * All rights reserved.
 */

#ifndef FILTER_BY_PROBE_H_INCLUDED
#define FILTER_BY_PROBE_H_INCLUDED

#include "mmcore/CallerSlot.h"
#include "mmstd_gl/renderer/CallRender3DGL.h"
#include "mmstd_gl/renderer/Renderer3DModuleGL.h"

#include "vislib_gl/graphics/gl/GLSLComputeShader.h"

namespace megamol {
namespace probe_gl {

/**
 * TODO
 */
class FilterByProbe : public megamol::mmstd_gl::Renderer3DModuleGL {
public:
    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char* ClassName(void) {
        return "FilterByProbe";
    }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static const char* Description(void) {
        return "...TODO...";
    }

    /**
     * Answers whether this module is available on the current system.
     *
     * @return 'true' if the module is available, 'false' otherwise.
     */
    static bool IsAvailable(void) {
        /*TODO*/
        return true;
    }

    /** Ctor. */
    FilterByProbe();

    /** Dtor. */
    ~FilterByProbe();

protected:
    /**
     * Implementation of 'Create'.
     *
     * @return 'true' on success, 'false' otherwise.
     */
    bool create();

    /**
     * Implementation of 'Release'.
     */
    void release();

    /**
     * The get extents callback. The module should set the members of
     * 'call' to tell the caller the extents of its data (bounding boxes
     * and times).
     *
     * @param call The calling call.
     *
     * @return The return value of the function.
     */
    bool GetExtents(mmstd_gl::CallRender3DGL& call);

    /**
     * The render callback.
     *
     * @param call The calling call.
     *
     * @return The return value of the function.
     */
    bool Render(mmstd_gl::CallRender3DGL& call);

    /**
     * Method that gets called before the rendering is started for all changed modules
     *
     * @param call The rendering call that contains the camera
     */
    void PreRender(mmstd_gl::CallRender3DGL& call);

private:
    typedef vislib_gl::graphics::gl::GLSLComputeShader GLSLComputeShader;

    uint32_t m_version;

    std::vector<bool> m_probe_selection;

    /** Shader program for setting flags based on incoming user manipulations */
    std::unique_ptr<GLSLComputeShader> m_setFlags_prgm;

    /** Shader program for filtering all data within a flag storage */
    std::unique_ptr<GLSLComputeShader> m_filterAll_prgm;

    /** Shader program for clearing filters on all data within a flag storage */
    std::unique_ptr<GLSLComputeShader> m_filterNone_prgm;

    core::CallerSlot m_probes_slot;

    core::CallerSlot m_kd_tree_slot;

    core::CallerSlot m_event_slot;

    core::CallerSlot m_readFlagsSlot;

    core::CallerSlot m_writeFlagsSlot;
};

} // namespace probe_gl
} // namespace megamol

#endif
