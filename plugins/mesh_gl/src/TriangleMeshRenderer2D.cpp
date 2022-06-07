#include "TriangleMeshRenderer2D.h"

#include "mesh/MeshDataCall.h"
#include "mesh/TriangleMeshCall.h"

#include "mmcore/CoreInstance.h"
#include "mmcore/param/BoolParam.h"
#include "mmcore/param/ColorParam.h"
#include "mmcore/param/FlexEnumParam.h"
#include "mmcore/param/TransferFunctionParam.h"
#include "mmcore/view/MouseFlags.h"

#include "mmcore_gl/utility/ShaderFactory.h"
#include "mmcore_gl/view/CallRender2DGL.h"

#include <glowl/glowl.h>

#include <exception>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace megamol {
namespace mesh_gl {
TriangleMeshRenderer2D::TriangleMeshRenderer2D()
        : render_input_slot("render_input_slot", "Render input slot")
        , triangle_mesh_slot("get_triangle_mesh", "Triangle mesh input")
        , triangle_mesh_hash(-1)
        , mesh_data_slot("get_mesh_data", "Mesh data input")
        , mesh_data_hash(-1)
        , data_set("data_set", "Data set used for coloring the triangles")
        , mask("mask", "Validity mask to selectively hide unwanted vertices or triangles")
        , mask_color("mask_color", "Color for invalid values")
        , default_color("default_color", "Default color if no dataset is selected")
        , wireframe("wireframe", "Render as wireframe instead of filling the triangles") {

    // Connect input slots
    this->render_input_slot.SetCompatibleCall<core_gl::view::CallRender2DGLDescription>();
    this->MakeSlotAvailable(&this->render_input_slot);

    this->triangle_mesh_slot.SetCompatibleCall<mesh::TriangleMeshCall::triangle_mesh_description>();
    this->MakeSlotAvailable(&this->triangle_mesh_slot);

    this->mesh_data_slot.SetCompatibleCall<mesh::MeshDataCall::mesh_data_description>();
    this->MakeSlotAvailable(&this->mesh_data_slot);

    // Connect parameter slots
    this->data_set << new core::param::FlexEnumParam("");
    this->MakeSlotAvailable(&this->data_set);

    this->mask << new core::param::FlexEnumParam("");
    this->MakeSlotAvailable(&this->mask);

    this->mask_color << new core::param::ColorParam(1.0f, 1.0f, 1.0f, 1.0f);
    this->MakeSlotAvailable(&this->mask_color);

    this->default_color << new core::param::ColorParam(0.7f, 0.7f, 0.7f, 1.0f);
    this->MakeSlotAvailable(&this->default_color);

    this->wireframe << new core::param::BoolParam(false);
    this->MakeSlotAvailable(&this->wireframe);
}

TriangleMeshRenderer2D::~TriangleMeshRenderer2D() {
    this->Release();
}

bool TriangleMeshRenderer2D::create() {
    auto const shader_options = msf::ShaderFactoryOptionsOpenGL(GetCoreInstance()->GetShaderPaths());

    try {
        this->render_data.shader_program = core::utility::make_glowl_shader("triangle_mesh_renderer_2d", shader_options,
            "mesh_gl/triangle_2d/triangle_2d.vert.glsl", "mesh_gl/triangle_2d/triangle_2d.frag.glsl");
    } catch (const std::exception& e) {
        Log::DefaultLog.WriteError(("TriangleMeshRenderer2D: " + std::string(e.what())).c_str());
        return false;
    }

    return true;
}

void TriangleMeshRenderer2D::release() {
    // Remove shaders, buffers and arrays
    if (this->render_data.initialized) {
        glDeleteVertexArrays(1, &this->render_data.vao);
        glDeleteBuffers(1, &this->render_data.vbo);
        glDeleteBuffers(1, &this->render_data.ibo);
        glDeleteBuffers(1, &this->render_data.cbo);
        glDeleteBuffers(1, &this->render_data.mbo);

        glDeleteTextures(1, &this->render_data.tf);
    }

    return;
}

bool TriangleMeshRenderer2D::Render(core_gl::view::CallRender2DGL& call) {
    // Call input renderer, if connected
    auto* input_renderer = this->render_input_slot.CallAs<core_gl::view::CallRender2DGL>();

    if (input_renderer != nullptr && (*input_renderer)(core::view::AbstractCallRender::FnRender)) {
        (*input_renderer) = call;
    }

    // Initialize renderer by creating shaders and buffers
    if (!this->render_data.initialized) {
        // Create arrays and buffers
        glGenVertexArrays(1, &this->render_data.vao);
        glGenBuffers(1, &this->render_data.vbo);
        glGenBuffers(1, &this->render_data.ibo);
        glGenBuffers(1, &this->render_data.cbo);
        glGenBuffers(1, &this->render_data.mbo);

        // Create transfer function texture
        glGenTextures(1, &this->render_data.tf);

        this->render_data.initialized = true;
    }

    // Get camera transformation matrices
    this->camera = call.GetCamera();

    // Update triangles (connection mandatory)
    auto get_triangles = this->triangle_mesh_slot.CallAs<mesh::TriangleMeshCall>();

    if (get_triangles == nullptr || !(*get_triangles)(0))
        return false;

    if (get_triangles->DataHash() != this->triangle_mesh_hash) {
        // Set hash
        this->triangle_mesh_hash = get_triangles->DataHash();

        // Get vertices and indices
        this->render_data.vertices = get_triangles->get_vertices();
        this->render_data.indices = get_triangles->get_indices();

        // Prepare OpenGL buffers
        if (this->render_data.vertices != nullptr && this->render_data.indices != nullptr) {
            glBindVertexArray(this->render_data.vao);

            glBindBuffer(GL_ARRAY_BUFFER, this->render_data.vbo);
            glBufferData(GL_ARRAY_BUFFER, this->render_data.vertices->size() * sizeof(GLfloat),
                this->render_data.vertices->data(), GL_STATIC_DRAW);

            glEnableVertexAttribArray(0);
            glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, nullptr);

            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, this->render_data.ibo);
            glBufferData(GL_ELEMENT_ARRAY_BUFFER, this->render_data.indices->size() * sizeof(GLuint),
                this->render_data.indices->data(), GL_STATIC_DRAW);

            glBindVertexArray(0);
        }
    }

    // Update data (connection optional)
    if (this->render_data.vertices != nullptr && this->render_data.indices != nullptr) {
        auto get_data = this->mesh_data_slot.CallAs<mesh::MeshDataCall>();

        bool new_data = false;
        bool new_mask = false;

        if (get_data != nullptr && (*get_data)(0)) {
            if (get_data->DataHash() != this->mesh_data_hash || this->data_set.IsDirty()) {
                // Set hash and reset parameter
                this->data_set.ResetDirty();

                this->render_data.values =
                    get_data->get_data(this->data_set.Param<core::param::FlexEnumParam>()->Value());

                new_data = true;
            }

            if (get_data->DataHash() != this->mesh_data_hash || this->mask.IsDirty()) {
                // Set hash and reset parameter
                this->mask.ResetDirty();

                this->render_data.mask = get_data->get_mask(this->mask.Param<core::param::FlexEnumParam>()->Value());

                new_mask = true;
            }

            this->mesh_data_hash = get_data->DataHash();
        }

        if (this->render_data.values == nullptr) {
            this->render_data.values = std::make_shared<mesh::MeshDataCall::data_set>();

            const auto color = this->default_color.Param<core::param::ColorParam>()->Value();

            std::stringstream ss;
            ss << "{\"Interpolation\":\"LINEAR\",\"Nodes\":["
               << "[" << color[0] << "," << color[1] << "," << color[2] << "," << color[3]
               << ",0.0,0.05000000074505806],"
               << "[" << color[0] << "," << color[1] << "," << color[2] << "," << color[3]
               << ",1.0,0.05000000074505806]]"
               << ",\"TextureSize\":2,\"ValueRange\":[0.0,1.0]}";

            this->render_data.values->min_value = 0.0f;
            this->render_data.values->max_value = 1.0f;
            this->render_data.values->data =
                std::make_shared<std::vector<GLfloat>>(this->render_data.vertices->size() / 2, 1.0f);
        }

        if (this->render_data.mask == nullptr) {
            this->render_data.mask =
                std::make_shared<std::vector<GLfloat>>(this->render_data.vertices->size() / 2, 1.0f);

            new_mask = true;
        }

        const auto num_vertices = this->render_data.vertices->size() / 3;

        if (num_vertices != this->render_data.values->data->size()) {
            megamol::core::utility::log::Log::DefaultLog.WriteError(
                "Number of vertices and data values do not match. [%s, %s, line %d]\n", __FILE__, __FUNCTION__,
                __LINE__);

            return false;
        }

        // Prepare OpenGL buffers
        glBindVertexArray(this->render_data.vao);

        glBindBuffer(GL_ARRAY_BUFFER, this->render_data.cbo);
        glBufferData(GL_ARRAY_BUFFER, this->render_data.values->data->size() * sizeof(GLfloat),
            this->render_data.values->data->data(), GL_STATIC_DRAW);

        glEnableVertexAttribArray(1);
        glVertexAttribPointer(1, 1, GL_FLOAT, GL_FALSE, 0, nullptr);

        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, this->render_data.ibo);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, this->render_data.indices->size() * sizeof(GLuint),
            this->render_data.indices->data(), GL_STATIC_DRAW);

        glBindVertexArray(0);

        if (new_mask) {
            if (this->render_data.mask != nullptr && num_vertices != this->render_data.mask->size()) {
                megamol::core::utility::log::Log::DefaultLog.WriteError(
                    "Number of vertices and mask values do not match. [%s, %s, line %d]\n", __FILE__, __FUNCTION__,
                    __LINE__);

                return false;
            }

            glBindVertexArray(this->render_data.vao);

            glBindBuffer(GL_ARRAY_BUFFER, this->render_data.mbo);
            glBufferData(GL_ARRAY_BUFFER, this->render_data.mask->size() * sizeof(GLfloat),
                this->render_data.mask->data(), GL_STATIC_DRAW);

            glEnableVertexAttribArray(2);
            glVertexAttribPointer(2, 1, GL_FLOAT, GL_FALSE, 0, nullptr);

            glBindVertexArray(0);
        }

        if (new_data || this->render_data.values->transfer_function_dirty) {
            // Create texture for transfer function
            std::vector<GLfloat> texture_data;
            int width, _unused__height;

            const auto valid_tf = core::param::TransferFunctionParam::GetTextureData(
                this->render_data.values->transfer_function, texture_data, width, _unused__height);

            this->render_data.tf_size = static_cast<GLuint>(width);

            if (!valid_tf) {
                return false;
            }

            // Create transfer funtion texture
            glActiveTexture(GL_TEXTURE0);
            glBindTexture(GL_TEXTURE_1D, this->render_data.tf);

            glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
            glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

            glTexImage1D(GL_TEXTURE_1D, 0, GL_RGBA, static_cast<GLsizei>(this->render_data.tf_size), 0, GL_RGBA,
                GL_FLOAT, static_cast<GLvoid*>(texture_data.data()));

            glBindTexture(GL_TEXTURE_1D, 0);

            this->render_data.values->transfer_function_dirty = false;
        }
    }

    // Render triangle mesh
    if (this->render_data.vertices != nullptr && this->render_data.indices != nullptr) {
        // Set wireframe or filled rendering
        if (this->wireframe.Param<core::param::BoolParam>()->Value()) {
            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
        } else {
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
        }

        // Render
        this->render_data.shader_program->use();
        glDisable(GL_DEPTH_TEST);
        glDepthMask(GL_FALSE);

        glUniformMatrix4fv(this->render_data.shader_program->getUniformLocation("model_view_matrix"), 1, GL_FALSE,
            glm::value_ptr(this->camera.getViewMatrix()));
        glUniformMatrix4fv(this->render_data.shader_program->getUniformLocation("projection_matrix"), 1, GL_FALSE,
            glm::value_ptr(this->camera.getProjectionMatrix()));

        glUniform1f(
            this->render_data.shader_program->getUniformLocation("min_value"), this->render_data.values->min_value);
        glUniform1f(
            this->render_data.shader_program->getUniformLocation("max_value"), this->render_data.values->max_value);

        const auto mask_color = this->mask_color.Param<core::param::ColorParam>()->Value();
        glUniform4f(this->render_data.shader_program->getUniformLocation("mask_color"), mask_color[0], mask_color[1],
            mask_color[2], mask_color[3]);

        glBindVertexArray(this->render_data.vao);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_1D, this->render_data.tf);
        glDrawElements(GL_TRIANGLES, static_cast<GLsizei>(this->render_data.indices->size()), GL_UNSIGNED_INT, nullptr);
        glBindTexture(GL_TEXTURE_1D, 0);
        glBindVertexArray(0);

        glDepthMask(GL_TRUE);
        glEnable(GL_DEPTH_TEST);
        glUseProgram(0);

        // Reset to filled mode
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
    }

    return true;
}

bool TriangleMeshRenderer2D::GetExtents(core_gl::view::CallRender2DGL& call) {
    // Get and set bounding rectangle (connection mandatory)
    auto get_triangles = this->triangle_mesh_slot.CallAs<mesh::TriangleMeshCall>();

    if (get_triangles == nullptr || !(*get_triangles)(1))
        return false;

    if (get_triangles->get_dimension() != mesh::TriangleMeshCall::dimension_t::TWO) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "The dimension of the data does not fit the renderer. [%s, %s, line %d]\n", __FILE__, __FUNCTION__,
            __LINE__);

        return false;
    }

    this->bounds = get_triangles->get_bounding_rectangle();

    // Get bounding rectangle of input renderer, if available
    auto* input_renderer = this->render_input_slot.CallAs<core_gl::view::CallRender2DGL>();

    if (input_renderer != nullptr && (*input_renderer)(core::view::AbstractCallRender::FnGetExtents)) {
        this->bounds.SetLeft(std::min(this->bounds.Left(), input_renderer->GetBoundingBoxes().BoundingBox().Left()));
        this->bounds.SetRight(std::max(this->bounds.Right(), input_renderer->GetBoundingBoxes().BoundingBox().Right()));
        this->bounds.SetBottom(
            std::min(this->bounds.Bottom(), input_renderer->GetBoundingBoxes().BoundingBox().Bottom()));
        this->bounds.SetTop(std::max(this->bounds.Top(), input_renderer->GetBoundingBoxes().BoundingBox().Top()));
    }

    call.AccessBoundingBoxes().SetBoundingBox(
        this->bounds.GetLeft(), this->bounds.GetBottom(), this->bounds.GetRight(), this->bounds.GetTop());

    // Get data sets (connection optional)
    auto get_data = this->mesh_data_slot.CallAs<mesh::MeshDataCall>();

    if (get_data != nullptr && (*get_data)(1) && get_data->DataHash() != this->mesh_data_hash) {
        // Get available data sets
        {
            const auto previous_value = this->data_set.Param<core::param::FlexEnumParam>()->Value();

            this->data_set.Param<core::param::FlexEnumParam>()->ClearValues();

            this->data_set.Param<core::param::FlexEnumParam>()->AddValue("");
            this->data_set.Param<core::param::FlexEnumParam>()->SetValue("");

            for (const auto& data_set : get_data->get_data_sets()) {
                this->data_set.Param<core::param::FlexEnumParam>()->AddValue(data_set);

                if (data_set == previous_value) {
                    this->data_set.Param<core::param::FlexEnumParam>()->SetValue(previous_value, false);
                }
            }
        }

        // Get available masks
        {
            const auto previous_value = this->mask.Param<core::param::FlexEnumParam>()->Value();

            this->mask.Param<core::param::FlexEnumParam>()->ClearValues();

            this->mask.Param<core::param::FlexEnumParam>()->AddValue("");
            this->mask.Param<core::param::FlexEnumParam>()->SetValue("");

            for (const auto& mask : get_data->get_masks()) {
                this->mask.Param<core::param::FlexEnumParam>()->AddValue(mask);

                if (mask == previous_value) {
                    this->mask.Param<core::param::FlexEnumParam>()->SetValue(previous_value, false);
                }
            }
        }
    }

    return true;
}

bool TriangleMeshRenderer2D::OnKey(core::view::Key key, core::view::KeyAction action, core::view::Modifiers mods) {
    auto* input_renderer = this->render_input_slot.template CallAs<core_gl::view::CallRender2DGL>();
    if (input_renderer == nullptr)
        return false;

    core::view::InputEvent evt;
    evt.tag = core::view::InputEvent::Tag::Key;
    evt.keyData.key = key;
    evt.keyData.action = action;
    evt.keyData.mods = mods;

    input_renderer->SetInputEvent(evt);
    return (*input_renderer)(core::view::InputCall::FnOnKey);
}

bool TriangleMeshRenderer2D::OnChar(unsigned int codePoint) {
    auto* input_renderer = this->render_input_slot.template CallAs<core_gl::view::CallRender2DGL>();
    if (input_renderer == nullptr)
        return false;

    core::view::InputEvent evt;
    evt.tag = core::view::InputEvent::Tag::Char;
    evt.charData.codePoint = codePoint;

    input_renderer->SetInputEvent(evt);
    return (*input_renderer)(core::view::InputCall::FnOnChar);
}

bool TriangleMeshRenderer2D::OnMouseButton(
    core::view::MouseButton button, core::view::MouseButtonAction action, core::view::Modifiers mods) {
    auto* input_renderer = this->render_input_slot.template CallAs<core_gl::view::CallRender2DGL>();
    if (input_renderer == nullptr)
        return false;

    core::view::InputEvent evt;
    evt.tag = core::view::InputEvent::Tag::MouseButton;
    evt.mouseButtonData.button = button;
    evt.mouseButtonData.action = action;
    evt.mouseButtonData.mods = mods;

    input_renderer->SetInputEvent(evt);
    return (*input_renderer)(core::view::InputCall::FnOnMouseButton);
}

bool TriangleMeshRenderer2D::OnMouseMove(double x, double y) {
    auto* input_renderer = this->render_input_slot.template CallAs<core_gl::view::CallRender2DGL>();
    if (input_renderer == nullptr)
        return false;

    core::view::InputEvent evt;
    evt.tag = core::view::InputEvent::Tag::MouseMove;
    evt.mouseMoveData.x = x;
    evt.mouseMoveData.y = y;

    input_renderer->SetInputEvent(evt);
    return (*input_renderer)(core::view::InputCall::FnOnMouseMove);
}

bool TriangleMeshRenderer2D::OnMouseScroll(double dx, double dy) {
    auto* input_renderer = this->render_input_slot.template CallAs<core_gl::view::CallRender2DGL>();
    if (input_renderer == nullptr)
        return false;

    core::view::InputEvent evt;
    evt.tag = core::view::InputEvent::Tag::MouseScroll;
    evt.mouseScrollData.dx = dx;
    evt.mouseScrollData.dy = dy;

    input_renderer->SetInputEvent(evt);
    return (*input_renderer)(core::view::InputCall::FnOnMouseScroll);
}
} // namespace mesh_gl
} // namespace megamol
