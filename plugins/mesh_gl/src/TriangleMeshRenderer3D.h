/*
 * TriangleMeshRenderer3D.h
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */
#pragma once

#include "mesh/MeshDataCall.h"

#include "mesh_gl/BaseRenderTaskRenderer.h"

#include "mmcore/Call.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/param/ParamSlot.h"

#include "mmcore_gl/utility/ShaderFactory.h"

#include "vislib/math/Cuboid.h"

#include <glowl/glowl.h>

#include <array>
#include <memory>
#include <type_traits>
#include <vector>

namespace megamol::mesh_gl {

/**
 * Module for uploading a 3D triangle mesh to the GPU.
 *
 * @author Alexander Straub
 */
class TriangleMeshRenderer3D : public BaseRenderTaskRenderer {
    static_assert(std::is_same<GLfloat, float>::value, "'GLfloat' and 'float' must be the same type!");
    static_assert(std::is_same<GLuint, unsigned int>::value, "'GLuint' and 'unsigned int' must be the same type!");

public:
    /**
     * Global unique ID that can e.g. be used for hash calculation.
     *
     * @return Unique ID
     */
    static inline SIZE_T GUID() {
        return 955430898uLL;
    }

    /**
     * Initialises a new instance.
     */
    TriangleMeshRenderer3D();

    /**
     * Finalises an instance.
     */
    ~TriangleMeshRenderer3D() override;

    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char* ClassName() {
        return "TriangleMeshRenderer3D";
    }
    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static const char* Description() {
        return "Upload 3D data to the GPU for use with the mesh plugin";
    }

    /**
     * Request resources to ask for OpenGL state
     */
    static void requested_lifetime_resources(frontend_resources::ResourceRequest& req) {
        Module::requested_lifetime_resources(req);
        req.require<frontend_resources::OpenGL_Context>();
    }

protected:
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

    void createMaterialCollection() override;

    void updateRenderTaskCollection(mmstd_gl::CallRender3DGL& call, bool force_update) override;

private:
    /** Get input data and extent from called modules */
    bool get_input_data();
    bool get_input_extent();

    /** Input slot for the triangle mesh */
    SIZE_T triangle_mesh_hash;
    bool triangle_mesh_changed;

    core::CallerSlot triangle_mesh_slot;

    /** Input slot for data attached to the triangles or their nodes */
    SIZE_T mesh_data_hash;
    bool mesh_data_changed;

    core::CallerSlot mesh_data_slot;

    /** Input slot for a clip plane */
    core::CallerSlot clip_plane_slot;

    /** Parameter slot for choosing data sets to visualize */
    core::param::ParamSlot data_set;

    /** Parameter slot for setting the default color if no dataset is specified */
    core::param::ParamSlot default_color;

    /** Bounding box */
    vislib::math::Cuboid<float> bounding_box;

    /** Shader program */
    std::shared_ptr<glowl::GLSLProgram> active_shader_program;
    std::shared_ptr<glowl::GLSLProgram> shader_program, shader_program_wireframe;
    std::shared_ptr<glowl::GLSLProgram> shader_program_normal, shader_program_normal_wireframe;

    /** Rendering options */
    core::param::ParamSlot wireframe, calculate_normals, culling;
    bool shader_changed;

    /** Per draw data offsets and size */
    struct per_draw_data_t {
        static constexpr std::size_t size_tf = sizeof(GLuint64);
        static constexpr std::size_t size_min_value = sizeof(float);
        static constexpr std::size_t size_max_value = sizeof(float);
        static constexpr std::size_t size_plane = 4 * sizeof(float);
        static constexpr std::size_t size_plane_bool = sizeof(int);
        static constexpr std::size_t size_culling = sizeof(int);

        static constexpr std::size_t offset_tf = 0;
        static constexpr std::size_t offset_min_value = offset_tf + size_tf;
        static constexpr std::size_t offset_max_value = offset_min_value + size_min_value;
        static constexpr std::size_t offset_plane = offset_max_value + size_max_value;
        static constexpr std::size_t offset_plane_bool = offset_plane + size_plane;
        static constexpr std::size_t offset_culling = offset_plane_bool + size_plane_bool;

        static constexpr std::size_t size = offset_culling + size_culling;
    };

    /** Struct for storing data needed for rendering */
    struct render_data_t {
        GLuint transfer_function = 0;

        std::shared_ptr<std::vector<GLfloat>> vertices;
        std::shared_ptr<std::vector<GLfloat>> normals;
        std::shared_ptr<std::vector<GLuint>> indices;

        std::shared_ptr<mesh::MeshDataCall::data_set> values;

        std::shared_ptr<GPUMeshCollection> mesh;

        std::array<uint8_t, per_draw_data_t::size> per_draw_data;

    } render_data;
};
} // namespace megamol::mesh_gl
