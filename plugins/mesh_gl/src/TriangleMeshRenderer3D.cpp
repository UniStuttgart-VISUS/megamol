#include "TriangleMeshRenderer3D.h"

#include "OpenGL_Context.h"

#include "mesh/MeshCalls.h"
#include "mesh/MeshDataCall.h"
#include "mesh/TriangleMeshCall.h"

#include "mesh_gl/GPUMeshCollection.h"

#include "mmcore/BoundingBoxes_2.h"
#include "mmcore/Call.h"
#include "mmcore/CoreInstance.h"
#include "mmcore/param/BoolParam.h"
#include "mmcore/param/ColorParam.h"
#include "mmcore/param/EnumParam.h"
#include "mmcore/param/FilePathParam.h"
#include "mmcore/param/FlexEnumParam.h"
#include "mmcore/param/TransferFunctionParam.h"
#include "mmcore/utility/DataHash.h"
#include "mmstd/renderer/CallClipPlane.h"

#include "mmcore_gl/utility/ShaderFactory.h"

#include <glowl/VertexLayout.hpp>
#include <glowl/glowl.h>

#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace megamol {
namespace mesh_gl {

TriangleMeshRenderer3D::TriangleMeshRenderer3D()
        : triangle_mesh_slot("get_triangle_mesh", "Triangle mesh input")
        , mesh_data_slot("get_mesh_data", "Mesh data input")
        , clip_plane_slot("clip_plane", "Clip plane for clipping the rendered triangle mesh")
        , data_set("data_set", "Data set used for coloring the triangles")
        , default_color("default_color", "Default color if no dataset is specified")
        , wireframe("wireframe", "Render wireframe instead of filled triangles")
        , calculate_normals("calculate_normals", "Calculate normals in geometry shader instead of using input normals")
        , culling("culling", "Culling mode")
        , triangle_mesh_hash(TriangleMeshRenderer3D::GUID())
        , triangle_mesh_changed(false)
        , mesh_data_hash(TriangleMeshRenderer3D::GUID())
        , mesh_data_changed(false)
        , shader_changed(false) {

    // Connect input slots
    this->triangle_mesh_slot.SetCompatibleCall<mesh::TriangleMeshCall::triangle_mesh_description>();
    this->MakeSlotAvailable(&this->triangle_mesh_slot);

    this->mesh_data_slot.SetCompatibleCall<mesh::MeshDataCall::mesh_data_description>();
    this->MakeSlotAvailable(&this->mesh_data_slot);

    this->clip_plane_slot.SetCompatibleCall<core::view::CallClipPlaneDescription>();
    this->MakeSlotAvailable(&this->clip_plane_slot);

    // Connect parameter slots
    this->data_set << new core::param::FlexEnumParam("");
    this->MakeSlotAvailable(&this->data_set);

    this->default_color << new core::param::ColorParam(0.7f, 0.7f, 0.7f, 1.0f);
    this->MakeSlotAvailable(&this->default_color);

    this->wireframe << new core::param::BoolParam(false);
    this->MakeSlotAvailable(&this->wireframe);

    this->calculate_normals << new core::param::BoolParam(true);
    this->MakeSlotAvailable(&this->calculate_normals);

    this->culling << new core::param::EnumParam(0);
    this->culling.Param<core::param::EnumParam>()->SetTypePair(0, "None");
    this->culling.Param<core::param::EnumParam>()->SetTypePair(1, "Backface culling");
    this->culling.Param<core::param::EnumParam>()->SetTypePair(2, "Frontface culling");
    this->MakeSlotAvailable(&this->culling);
}

TriangleMeshRenderer3D::~TriangleMeshRenderer3D() {
    this->Release();
}

bool TriangleMeshRenderer3D::GetExtents(mmstd_gl::CallRender3DGL& call) {
    if (!get_input_extent()) {
        return false;
    }

    core::BoundingBoxes_2 bbox;
    bbox.SetBoundingBox(this->bounding_box);
    bbox.SetClipBox(this->bounding_box);

    call.SetTimeFramesCount(1);
    call.AccessBoundingBoxes() = bbox;

    return true;
}

void TriangleMeshRenderer3D::createMaterialCollection() {

    auto const& ogl_ctx = frontend_resources.get<frontend_resources::OpenGL_Context>();
    if (!ogl_ctx.isVersionGEQ(4, 3) || !ogl_ctx.isExtAvailable("GL_ARB_shader_draw_parameters") ||
        !ogl_ctx.isExtAvailable("GL_ARB_bindless_texture")) {

        Log::DefaultLog.WriteError("GL version too low or required extensions not available.");
    }

    auto const shader_options = msf::ShaderFactoryOptionsOpenGL(GetCoreInstance()->GetShaderPaths());

    try {
        this->shader_program = core::utility::make_shared_glowl_shader("triangle_mesh_renderer_3d", shader_options,
            "mesh_gl/triangle_3d/without_normals.vert.glsl", "mesh_gl/triangle_3d/calc_normals.geom.glsl",
            "mesh_gl/triangle_3d/clip.frag.glsl");

        this->shader_program_wireframe = core::utility::make_shared_glowl_shader("triangle_mesh_renderer_3d_wireframe",
            shader_options, "mesh_gl/triangle_3d/without_normals.vert.glsl",
            "mesh_gl/triangle_3d/calc_normals_wireframe.geom.glsl", "mesh_gl/triangle_3d/clip.frag.glsl");

        this->shader_program_normal = core::utility::make_shared_glowl_shader("triangle_mesh_renderer_3d",
            shader_options, "mesh_gl/triangle_3d/with_normals.vert.glsl", "mesh_gl/triangle_3d/pass_normals.geom.glsl",
            "mesh_gl/triangle_3d/clip.frag.glsl");

        this->shader_program_normal_wireframe = core::utility::make_shared_glowl_shader(
            "triangle_mesh_renderer_3d_wireframe", shader_options, "mesh_gl/triangle_3d/with_normals.vert.glsl",
            "mesh_gl/triangle_3d/pass_normals_wireframe.geom.glsl", "mesh_gl/triangle_3d/clip.frag.glsl");

        this->active_shader_program = this->shader_program;

    } catch (const std::exception& e) {
        Log::DefaultLog.WriteError(("TriangleMeshRenderer3D: " + std::string(e.what())).c_str());
    }

    // Shaders are used directly in render task creation, so no need for the collection
    //material_collection_ = std::make_shared<GPUMaterialCollection>();
}

void TriangleMeshRenderer3D::updateRenderTaskCollection(mmstd_gl::CallRender3DGL& call, bool force_update) {
    if (!get_input_data()) {
        return;
        // TODO throw error?
    }

    if (this->render_data.vertices != nullptr && this->render_data.indices != nullptr) {
        const auto num_vertices = this->render_data.vertices->size() / 3;

        if (num_vertices != this->render_data.values->data->size()) {
            megamol::core::utility::log::Log::DefaultLog.WriteError(
                "Number of vertices and data values do not match. [%s, %s, line %d]\n", __FILE__, __FUNCTION__,
                __LINE__);

            return;
        } else if (this->render_data.normals != nullptr && num_vertices != this->render_data.normals->size() / 3) {
            megamol::core::utility::log::Log::DefaultLog.WriteError(
                "Number of vertices and normals do not match. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);

            return;
        }

        const std::string identifier("triangle_mesh");

        if (this->triangle_mesh_changed || this->mesh_data_changed || this->shader_changed) {
            render_task_collection_->clear();

            // Create mesh
            this->render_data.mesh = std::make_shared<GPUMeshCollection>();

            using vbi_t = typename std::vector<GLfloat>::iterator;
            using ibi_t = typename std::vector<GLuint>::iterator;

            std::vector<glowl::VertexLayout> vertex_descriptors{
                glowl::VertexLayout(3 * sizeof(float), {glowl::VertexLayout::Attribute(3, GL_FLOAT, GL_FALSE, 0)}),
                glowl::VertexLayout(1 * sizeof(float), {glowl::VertexLayout::Attribute(1, GL_FLOAT, GL_FALSE, 0)})};

            if (this->render_data.normals != nullptr) {
                vertex_descriptors.push_back(
                    glowl::VertexLayout(3 * sizeof(float), {glowl::VertexLayout::Attribute(3, GL_FLOAT, GL_TRUE, 0)}));
            }

            std::vector<std::pair<vbi_t, vbi_t>> vertex_buffer{
                {this->render_data.vertices->begin(), this->render_data.vertices->end()},
                {this->render_data.values->data->begin(), this->render_data.values->data->end()}};

            if (this->render_data.normals != nullptr) {
                vertex_buffer.push_back({this->render_data.normals->begin(), this->render_data.normals->end()});
            }

            std::pair<ibi_t, ibi_t> index_buffer{this->render_data.indices->begin(), this->render_data.indices->end()};

            this->render_data.mesh->template addMesh<vbi_t, ibi_t>(identifier, vertex_descriptors, vertex_buffer,
                index_buffer, GL_UNSIGNED_INT, GL_STATIC_DRAW, GL_TRIANGLES);

            // Create render task
            const auto& mesh_data = this->render_data.mesh->getSubMeshData().at(identifier);

            std::memcpy(&this->render_data.per_draw_data[per_draw_data_t::offset_min_value],
                &this->render_data.values->min_value, per_draw_data_t::size_min_value);
            std::memcpy(&this->render_data.per_draw_data[per_draw_data_t::offset_max_value],
                &this->render_data.values->max_value, per_draw_data_t::size_max_value);

            render_task_collection_->addRenderTask(identifier, this->active_shader_program,
                mesh_data.mesh->mesh, mesh_data.sub_mesh_draw_command, this->render_data.per_draw_data);
        }

        if (this->render_data.values->transfer_function_dirty || this->triangle_mesh_changed ||
            this->mesh_data_changed || this->shader_changed) {
            // Create texture for transfer function
            std::vector<GLfloat> texture_data;
            int transfer_function_size, _unused__height;

            const auto valid_tf = core::param::TransferFunctionParam::GetTextureData(
                this->render_data.values->transfer_function, texture_data, transfer_function_size, _unused__height);

            if (!valid_tf) {
                return;
                // TODO throw error?
            }

            if (this->render_data.transfer_function != 0) {
                glDeleteTextures(1, &this->render_data.transfer_function);
            }

            glGenTextures(1, &this->render_data.transfer_function);

            glActiveTexture(GL_TEXTURE0);
            glBindTexture(GL_TEXTURE_1D, this->render_data.transfer_function);

            glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
            glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

            glTexImage1D(GL_TEXTURE_1D, 0, GL_RGBA, static_cast<GLsizei>(transfer_function_size), 0, GL_RGBA, GL_FLOAT,
                static_cast<GLvoid*>(texture_data.data()));

            glBindTexture(GL_TEXTURE_1D, 0);

            const auto transfer_function_handle = glGetTextureHandleARB(this->render_data.transfer_function);
            glMakeTextureHandleResidentARB(transfer_function_handle);

            // Update per draw data
            std::memcpy(&this->render_data.per_draw_data[per_draw_data_t::offset_tf], &transfer_function_handle,
                per_draw_data_t::size_tf);

            render_task_collection_->updatePerDrawData(identifier, this->render_data.per_draw_data);
        }

        {
            auto cp = this->clip_plane_slot.CallAs<core::view::CallClipPlane>();

            if (cp != nullptr && (*cp)(0)) {
                // Set clip plane flag to enabled
                const int use_plane = 1;

                std::memcpy(&this->render_data.per_draw_data[per_draw_data_t::offset_plane_bool], &use_plane,
                    per_draw_data_t::size_plane_bool);

                // Get clip plane
                const auto& plane = cp->GetPlane();
                const std::array<float, 4> abcd_plane{plane.A(), plane.B(), plane.C(), plane.D()};

                std::memcpy(&this->render_data.per_draw_data[per_draw_data_t::offset_plane], abcd_plane.data(),
                    per_draw_data_t::size_plane);
            } else {
                // Set clip plane flag to disabled
                const int use_plane = 0;

                std::memcpy(&this->render_data.per_draw_data[per_draw_data_t::offset_plane_bool], &use_plane,
                    per_draw_data_t::size_plane_bool);
            }

            render_task_collection_->updatePerDrawData(identifier, this->render_data.per_draw_data);
        }

        {
            // Set culling mode: 0 - none, 1 - backface culling, 2 - frontface culling
            const int culling_mode = static_cast<int>(this->culling.Param<core::param::EnumParam>()->Value());

            std::memcpy(&this->render_data.per_draw_data[per_draw_data_t::offset_culling], &culling_mode,
                per_draw_data_t::size_culling);
        }
    }

    this->triangle_mesh_changed = false;
    this->mesh_data_changed = false;
    this->render_data.values->transfer_function_dirty = false;
    this->shader_changed = false;
}

std::vector<std::string> TriangleMeshRenderer3D::requested_lifetime_resources() {
    std::vector<std::string> resources = Module::requested_lifetime_resources();
    resources.emplace_back("OpenGL_Context");
    return resources;
}

bool TriangleMeshRenderer3D::get_input_data() {
    auto tmc_ptr = this->triangle_mesh_slot.CallAs<mesh::TriangleMeshCall>();
    auto mdc_ptr = this->mesh_data_slot.CallAs<mesh::MeshDataCall>();

    if (tmc_ptr == nullptr) {
        return false;
    }

    auto& tmc = *tmc_ptr;

    if (!tmc(0)) {
        if (tmc.DataHash() != this->triangle_mesh_hash) {
            megamol::core::utility::log::Log::DefaultLog.WriteError(
                "Error getting triangle mesh. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);

            this->triangle_mesh_hash = tmc.DataHash();
        }

        return false;
    }

    if (tmc.DataHash() != this->triangle_mesh_hash) {
        this->render_data.vertices = tmc.get_vertices();
        this->render_data.normals = tmc.get_normals();
        this->render_data.indices = tmc.get_indices();

        this->triangle_mesh_hash = tmc.DataHash();
        this->triangle_mesh_changed = true;
    }

    if (this->wireframe.IsDirty() || this->calculate_normals.IsDirty()) {
        this->wireframe.ResetDirty();
        this->calculate_normals.ResetDirty();

        if (this->wireframe.Param<core::param::BoolParam>()->Value() &&
            this->calculate_normals.Param<core::param::BoolParam>()->Value()) {

            this->active_shader_program = shader_program_wireframe;
        } else if (!this->wireframe.Param<core::param::BoolParam>()->Value() &&
                   this->calculate_normals.Param<core::param::BoolParam>()->Value()) {

            this->active_shader_program = shader_program;
        } else if (this->wireframe.Param<core::param::BoolParam>()->Value() &&
                   !this->calculate_normals.Param<core::param::BoolParam>()->Value()) {

            this->active_shader_program = shader_program_normal_wireframe;
        } else {
            this->active_shader_program = shader_program_normal;
        }

        this->shader_changed = true;
    }

    if (mdc_ptr != nullptr && (*mdc_ptr)(0) && !mdc_ptr->get_data_sets().empty() &&
        mdc_ptr->DataHash() != this->mesh_data_hash) {

        this->data_set.Param<core::param::FlexEnumParam>()->ClearValues();
        this->data_set.Param<core::param::FlexEnumParam>()->AddValue("");

        for (const auto& data_set_name : mdc_ptr->get_data_sets()) {
            this->data_set.Param<core::param::FlexEnumParam>()->AddValue(data_set_name);
        }

        this->data_set.ForceSetDirty();
        this->mesh_data_hash = mdc_ptr->DataHash();
    }

    if (mdc_ptr != nullptr &&
        (!this->data_set.Param<core::param::FlexEnumParam>()->Value().empty() && this->data_set.IsDirty())) {
        this->render_data.values = mdc_ptr->get_data(this->data_set.Param<core::param::FlexEnumParam>()->Value());

        this->data_set.ResetDirty();
        this->mesh_data_changed = true;
    }

    if (this->render_data.values == nullptr || (this->data_set.Param<core::param::FlexEnumParam>()->Value().empty() &&
                                                   (this->data_set.IsDirty() || this->default_color.IsDirty()))) {

        this->render_data.values = std::make_shared<mesh::MeshDataCall::data_set>();

        this->render_data.values->min_value = 0.0f;
        this->render_data.values->max_value = 1.0f;

        const auto& color = this->default_color.Param<core::param::ColorParam>()->Value();
        this->default_color.ResetDirty();
        this->data_set.ResetDirty();

        std::stringstream ss;
        ss << "{\"Interpolation\":\"LINEAR\",\"Nodes\":["
           << "[" << color[0] << "," << color[1] << "," << color[2] << "," << color[3] << ",0.0,0.05000000074505806],"
           << "[" << color[0] << "," << color[1] << "," << color[2] << "," << color[3] << ",1.0,0.05000000074505806]]"
           << ",\"TextureSize\":2,\"ValueRange\":[0.0,1.0]}";

        this->render_data.values->transfer_function = ss.str();
        this->render_data.values->transfer_function_dirty = true;

        this->render_data.values->data = std::make_shared<std::vector<GLfloat>>();

        if (this->render_data.vertices != nullptr) {
            this->render_data.values->data->resize(this->render_data.vertices->size() / 3, 1.0f);
        }

        this->mesh_data_changed = true;
    }

    return true;
}

bool TriangleMeshRenderer3D::get_input_extent() {
    auto tmc_ptr = this->triangle_mesh_slot.CallAs<mesh::TriangleMeshCall>();

    if (tmc_ptr == nullptr) {
        return false;
    }

    auto& tmc = *tmc_ptr;

    if (!tmc(1)) {
        if (tmc.DataHash() != this->triangle_mesh_hash) {
            megamol::core::utility::log::Log::DefaultLog.WriteError(
                "Error getting extents for the triangle mesh. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);

            this->triangle_mesh_hash = tmc.DataHash();
        }

        return false;
    }

    if (tmc.get_dimension() != mesh::TriangleMeshCall::dimension_t::THREE) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "Input triangle mesh must be three-dimensional. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);

        return false;
    }

    this->bounding_box = tmc.get_bounding_box();

    return true;
}

} // namespace mesh_gl
} // namespace megamol
