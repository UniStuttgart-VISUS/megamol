/*
 * BaseRenderTaskRenderer.h
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VISUS).
 * All rights reserved.
 */

#ifndef BASE_RENDER_TASK_RENDERER_H_INCLUDED
#define BASE_RENDER_TASK_RENDERER_H_INCLUDED

#include "mmstd_gl/renderer/Renderer3DModuleGL.h"

#include "mesh_gl/GPUMaterialCollection.h"
#include "mesh_gl/GPUMeshCollection.h"
#include "mesh_gl/GPURenderTaskCollection.h"

namespace megamol {
namespace mesh_gl {

class BaseRenderTaskRenderer : public megamol::mmstd_gl::Renderer3DModuleGL {
public:
    /**
     * Answers whether this module is available on the current system.
     *
     * @return 'true' if the module is available, 'false' otherwise.
     */
    static bool IsAvailable(void) {
        return true;
    }

    BaseRenderTaskRenderer() = default;
    ~BaseRenderTaskRenderer() override;

protected:
    /**
     * Implementation of 'Create'.
     *
     * @return 'true' on success, 'false' otherwise.
     */
    bool create() final;

    /**
     * Implementation of 'Release'.
     */
    void release() final;

    /**
     * The get extents callback. The module should set the members of
     * 'call' to tell the caller the extents of its data (bounding boxes
     * and times).
     *
     * @param call The calling call.
     *
     * @return The return value of the function.
     */
    bool GetExtents(mmstd_gl::CallRender3DGL& call) override = 0;

    /**
     * The render callback.
     *
     * @param call The calling call.
     *
     * @return The return value of the function.
     */
    bool Render(mmstd_gl::CallRender3DGL& call) final;

    /**
     * Create initial material collection. Called by module's create function.
     */
    virtual void createMaterialCollection();

    /**
     * Create initial mesh collection. Called by module's create function.
     */
    virtual void createMeshCollection();

    /**
     * Create initial render task collection. Called by module's create function.
     */
    virtual void createRenderTaskCollection();

    /**
     * Update material collection. Called by module's Render function.
     */
    virtual bool updateMaterialCollection();

    /**
     * Update mesh collection. Called by module's Render function.
     */
    virtual bool updateMeshCollection();

    /**
     * Update render task collection. Called by module's Render function.
     */
    virtual void updateRenderTaskCollection(mmstd_gl::CallRender3DGL& call, bool force_update);

    std::shared_ptr<GPUMaterialCollection> material_collection_;

    std::shared_ptr<GPUMeshCollection> mesh_collection_;

    std::shared_ptr<GPURenderTaskCollection> render_task_collection_;
};

inline BaseRenderTaskRenderer::~BaseRenderTaskRenderer() {
    this->Release();
}

inline bool BaseRenderTaskRenderer::create() {
    bool retval = true;
    try {
        createMaterialCollection();
        createMeshCollection();
        createRenderTaskCollection();
    } catch (std::runtime_error const& ex) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "%s [%s, %s, line %d]\n", ex.what(), __FILE__, __FUNCTION__, __LINE__);
        retval = false;
    }
    return retval;
}

inline void BaseRenderTaskRenderer::release() {
    material_collection_.reset();
    mesh_collection_.reset();
    render_task_collection_.reset();
}

inline bool BaseRenderTaskRenderer::Render(mmstd_gl::CallRender3DGL& call) {
    mmstd_gl::CallRender3DGL* cr = &call;
    if (cr == nullptr) {
        return false;
    }

    // Obtain camera information
    core::view::Camera cam = call.GetCamera();
    glm::mat4 view_mx = cam.getViewMatrix();
    glm::mat4 proj_mx = cam.getProjectionMatrix();

    // Update material collection and check whether something has changed in the Neath
    bool something_has_changed = updateMaterialCollection();
    // Update mesh collection and check whether something awaits you
    something_has_changed |= updateMeshCollection();
    // Update render task collection. Force update if material or mesh has changed
    updateRenderTaskCollection(call, something_has_changed);

    // Perform actual rendering after updating mesh data and gltf data
    rendering::processGPURenderTasks(render_task_collection_, view_mx, proj_mx);

    // Clear the way for his ancient majesty, the mighty immediate mode...
    glUseProgram(0);
    glBindVertexArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
    glBindBuffer(GL_DRAW_INDIRECT_BUFFER, 0);

    return true;
}

inline void BaseRenderTaskRenderer::createMaterialCollection() {
    material_collection_ = std::make_shared<GPUMaterialCollection>();
}

inline void BaseRenderTaskRenderer::createMeshCollection() {
    mesh_collection_ = std::make_shared<GPUMeshCollection>();
}

inline void BaseRenderTaskRenderer::createRenderTaskCollection() {
    render_task_collection_ = std::make_shared<GPURenderTaskCollection>();
}

inline bool BaseRenderTaskRenderer::updateMaterialCollection() {
    // empty function that is conveniently used when no update is required after creation
    return false;
}

inline bool BaseRenderTaskRenderer::updateMeshCollection() {
    // empty function that is conveniently used when no update is required after creation
    return false;
}

inline void BaseRenderTaskRenderer::updateRenderTaskCollection(mmstd_gl::CallRender3DGL& call, bool force_update) {
    // empty function that is conveniently used when no update is required after creation
}

} // namespace mesh_gl
} // namespace megamol

#endif // !BASE_RENDER_TASK_RENDERER_H_INCLUDED
