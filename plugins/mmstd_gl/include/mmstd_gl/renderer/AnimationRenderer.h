/**
 * MegaMol
 * Copyright (c) 2022, MegaMol Dev Team
 * All rights reserved.
 */

#pragma once

#include "AnimationEditorData.h"
#include "ModuleGraphSubscription.h"

#include <glowl/BufferObject.hpp>
#include <glowl/GLSLProgram.hpp>

#include "mmcore/MegaMolGraph.h"
#include "mmcore/param/ParamSlot.h"
#include "mmcore/utility/graphics/CameraUtils.h"
#include "mmstd_gl/renderer/CallRender3DGL.h"
#include "mmstd_gl/renderer/Renderer3DModuleGL.h"
#include "mmstd_gl/special/TextureInspector.h"

namespace megamol::mmstd_gl {

/**
 * Module for rendering data from the AnimationEditor.
 *
 */
class AnimationRenderer : public mmstd_gl::Renderer3DModuleGL {
public:
    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static inline const char* ClassName() {
        return "AnimationRenderer";
    }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static inline const char* Description() {
        return "Render an abstract static scene depiction and camera paths, if selected in the GUI";
    }

    /**
     * Answers whether this module is available on the current system.
     *
     * @return 'true' if the module is available, 'false' otherwise.
     */
    static inline bool IsAvailable() {
        return true;
    }

    static void requested_lifetime_resources(frontend_resources::ResourceRequest& req) {
        Renderer3DModuleGL::requested_lifetime_resources(req);
        req.require<core::MegaMolGraph>();
        req.require<frontend_resources::AnimationEditorData>();
    }

    /**
     * Initialises a new instance.
     */
    AnimationRenderer();

    /**
     * Finalises an instance.
     */
    ~AnimationRenderer() override;

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

    /** Callbacks for the rendering output */
    bool GetExtents(mmstd_gl::CallRender3DGL& call) override;
    bool Render(mmstd_gl::CallRender3DGL& call) override;

    /** Callbacks for the observation path */
    bool GetObservationExtents(core::Call& call);
    bool RenderObservation(core::Call& call);
    bool OnObservedMouseButton(core::Call& call);
    bool OnObservedMouseMove(core::Call& call);
    bool OnObservedMouseScroll(core::Call& call);
    bool OnObservedChar(core::Call& call);
    bool OnObservedKey(core::Call& call);

private:
    std::unique_ptr<glowl::GLSLProgram> render_points_program;
    std::unique_ptr<glowl::GLSLProgram> fbo_to_points_program;
    std::unique_ptr<glowl::GLSLProgram> campath_program;
    std::unique_ptr<glowl::GLSLProgram> keys_program;
    std::unique_ptr<glowl::GLSLProgram> orientations_program;

    std::unique_ptr<glowl::BufferObject> the_points;
    std::unique_ptr<glowl::BufferObject> animation_positions;
    std::unique_ptr<glowl::BufferObject> animation_keys;
    std::unique_ptr<glowl::BufferObject> animation_orientations;
    GLuint line_vao, keys_vao, orientations_vao;

    core::param::ParamSlot snapshotSlot;
    core::param::ParamSlot numberOfViewsSlot;
    core::param::ParamSlot showLocalCoordSystems;

    core::BoundingBoxes_2 lastBBox;

    bool CheckObservedSlots(
        megamol::mmstd_gl::CallRender3DGL*& in, megamol::mmstd_gl::CallRender3DGL*& out, core::Call& call);

    /** Slot for the observed renderer */
    core::CallerSlot observedRendererSlot;

    /** The render callee slot */
    core::CalleeSlot observedRenderSlot;

    std::vector<std::pair<core::utility::DefaultView, core::utility::DefaultOrientation>> snaps_to_take = {
        {core::utility::DEFAULTVIEW_FACE_FRONT, core::utility::DEFAULTORIENTATION_TOP},
        {core::utility::DEFAULTVIEW_FACE_BACK, core::utility::DEFAULTORIENTATION_TOP},
        {core::utility::DEFAULTVIEW_FACE_LEFT, core::utility::DEFAULTORIENTATION_TOP},
        {core::utility::DEFAULTVIEW_FACE_RIGHT, core::utility::DEFAULTORIENTATION_TOP},
        {core::utility::DEFAULTVIEW_FACE_TOP, core::utility::DEFAULTORIENTATION_TOP},
        {core::utility::DEFAULTVIEW_FACE_BOTTOM, core::utility::DEFAULTORIENTATION_TOP}};

    std::shared_ptr<glowl::FramebufferObject> approx_fbo;
    CallRender3DGL* call_to_hijack = nullptr;
    uint32_t hijack_callback_idx;
    std::vector<float> trajectory_vertices, trajectory_orientations;
    std::vector<unsigned int> key_indices;

    core::MegaMolGraph* theGraph = nullptr;
    frontend_resources::AnimationEditorData* theAnimation = nullptr;

    mmstd_gl::special::TextureInspector tex_inspector_;
};

} // namespace megamol::mmstd_gl
