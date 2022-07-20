/**
 * MegaMol
 * Copyright (c) 2021, MegaMol Dev Team
 * All rights reserved.
 */

#pragma once

#include <array>
#include <memory>
#include <string>
#include <type_traits>
#include <vector>

#include <glowl/glowl.h>

#include "mesh/MeshDataCall.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/param/ParamSlot.h"
#include "mmcore/view/Camera.h"
#include "mmcore/view/MouseFlags.h"
#include "mmstd_gl/renderer/CallRender2DGL.h"
#include "mmstd_gl/renderer/Renderer2DModuleGL.h"
#include "vislib/math/Rectangle.h"

namespace megamol::mesh_gl {
/**
 * Module for rendering a 2D triangle mesh.
 *
 * @author Alexander Straub
 */
class TriangleMeshRenderer2D : public mmstd_gl::Renderer2DModuleGL {
    static_assert(std::is_same<GLfloat, float>::value, "'GLfloat' and 'float' must be the same type!");
    static_assert(std::is_same<GLuint, unsigned int>::value, "'GLuint' and 'unsigned int' must be the same type!");

public:
    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static inline const char* ClassName() {
        return "TriangleMeshRenderer2D";
    }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static inline const char* Description() {
        return "Triangle mesh renderer for 2D data";
    }

    /**
     * Answers whether this module is available on the current system.
     *
     * @return 'true' if the module is available, 'false' otherwise.
     */
    static inline bool IsAvailable() {
        return true;
    }

    /**
     * Initialises a new instance.
     */
    TriangleMeshRenderer2D();

    /**
     * Finalises an instance.
     */
    virtual ~TriangleMeshRenderer2D();

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
     * @return 'true' on success, 'false' otherwise.
     */
    bool Render(mmstd_gl::CallRender2DGL& call) override;

    /**
     * The extent callback.
     *
     * @param call The calling call.
     *
     * @return 'true' on success, 'false' otherwise.
     */
    bool GetExtents(mmstd_gl::CallRender2DGL& call) override;

    /**
     * Forwards key events.
     */
    bool OnKey(core::view::Key key, core::view::KeyAction action, core::view::Modifiers mods) override;

    /**
     * Forwards character events.
     */
    bool OnChar(unsigned int codePoint) override;

    /**
     * Forwards character events.
     */
    bool OnMouseButton(
        core::view::MouseButton button, core::view::MouseButtonAction action, core::view::Modifiers mods) override;

    /**
     * Forwards character events.
     */
    bool OnMouseMove(double x, double y) override;

    /**
     * Forwards scroll events.
     */
    bool OnMouseScroll(double dx, double dy) override;

private:
    /** Input render call */
    core::CallerSlot render_input_slot;

    /** Input slot for the triangle mesh */
    core::CallerSlot triangle_mesh_slot;
    SIZE_T triangle_mesh_hash;

    /** Input slot for data attached to the triangles or their nodes */
    core::CallerSlot mesh_data_slot;
    SIZE_T mesh_data_hash;

    /** Parameter slot for choosing data sets to visualize */
    core::param::ParamSlot data_set;

    /** Parameter slot for choosing validity masks */
    core::param::ParamSlot mask;
    core::param::ParamSlot mask_color;

    /** Parameter slot for setting the default color if no dataset is specified */
    core::param::ParamSlot default_color;

    /** Parameter slot for choosing between filled and wireframe mode */
    core::param::ParamSlot wireframe;

    /** Bounding rectangle */
    vislib::math::Rectangle<float> bounds;

    /** Struct for storing data needed for rendering */
    struct render_data_t {
        bool initialized = false;

        std::unique_ptr<glowl::GLSLProgram> shader_program;

        GLuint vao, vbo, ibo, cbo, mbo;
        GLuint tf, tf_size;

        std::shared_ptr<std::vector<GLfloat>> vertices;
        std::shared_ptr<std::vector<GLuint>> indices;

        std::shared_ptr<mesh::MeshDataCall::data_set> values;

        std::shared_ptr<std::vector<GLfloat>> mask;

    } render_data;

    /** Storing a copy of the camera */
    core::view::Camera camera;
};
} // namespace megamol::mesh_gl
