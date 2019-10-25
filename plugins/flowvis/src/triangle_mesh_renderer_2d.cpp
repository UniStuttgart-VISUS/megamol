#include "stdafx.h"
#include "triangle_mesh_renderer_2d.h"

#include "mesh_data_call.h"
#include "triangle_mesh_call.h"

#include "flowvis/shader.h"

#include "mmcore/param/BoolParam.h"
#include "mmcore/param/ColorParam.h"
#include "mmcore/param/FlexEnumParam.h"
#include "mmcore/param/TransferFunctionParam.h"
#include "mmcore/view/CallRender2D.h"
#include "mmcore/view/MouseFlags.h"

#include "vislib/sys/Log.h"

#include "glad/glad.h"

#include <exception>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace megamol
{
    namespace flowvis
    {
        triangle_mesh_renderer_2d::triangle_mesh_renderer_2d() :
            render_input_slot("render_input_slot", "Render input slot"),
            triangle_mesh_slot("get_triangle_mesh", "Triangle mesh input"), triangle_mesh_hash(-1),
            mesh_data_slot("get_mesh_data", "Mesh data input"), mesh_data_hash(-1),
            data_set("data_set", "Data set used for coloring the triangles"),
            mask("mask", "Validity mask to selectively hide unwanted vertices or triangles"),
            mask_color("mask_color", "Color for invalid values"),
            wireframe("wireframe", "Render as wireframe instead of filling the triangles")
        {
            // Connect input slots
            this->render_input_slot.SetCompatibleCall<core::view::CallRender2DDescription>();
            this->MakeSlotAvailable(&this->render_input_slot);

            this->triangle_mesh_slot.SetCompatibleCall<triangle_mesh_call::triangle_mesh_description>();
            this->MakeSlotAvailable(&this->triangle_mesh_slot);

            this->mesh_data_slot.SetCompatibleCall<mesh_data_call::mesh_data_description>();
            this->MakeSlotAvailable(&this->mesh_data_slot);

            // Connect parameter slots
            this->data_set << new core::param::FlexEnumParam("");
            this->MakeSlotAvailable(&this->data_set);

            this->mask << new core::param::FlexEnumParam("");
            this->MakeSlotAvailable(&this->mask);

            this->mask_color << new core::param::ColorParam(1.0f, 1.0f, 1.0f, 1.0f);
            this->MakeSlotAvailable(&this->mask_color);

            this->wireframe << new core::param::BoolParam(false);
            this->MakeSlotAvailable(&this->wireframe);
        }

        triangle_mesh_renderer_2d::~triangle_mesh_renderer_2d()
        {
            this->Release();
        }

        bool triangle_mesh_renderer_2d::create()
        {
            return true;
        }

        void triangle_mesh_renderer_2d::release()
        {
            // Remove shaders, buffers and arrays
            if (this->render_data.initialized)
            {
                glDetachShader(this->render_data.prog, this->render_data.vs);
                glDetachShader(this->render_data.prog, this->render_data.fs);
                glDeleteProgram(this->render_data.prog);

                glDeleteVertexArrays(1, &this->render_data.vao);
                glDeleteBuffers(1, &this->render_data.vbo);
                glDeleteBuffers(1, &this->render_data.ibo);
                glDeleteBuffers(1, &this->render_data.cbo);
                glDeleteBuffers(1, &this->render_data.mbo);

                glDeleteTextures(1, &this->render_data.tf);
            }

            return;
        }

        bool triangle_mesh_renderer_2d::Render(core::view::CallRender2D& call)
        {
            // Call input renderer, if connected
            auto* input_renderer = this->render_input_slot.CallAs<core::view::CallRender2D>();

            if (input_renderer != nullptr && (*input_renderer)(core::view::AbstractCallRender::FnRender))
            {
                (*input_renderer) = call;
            }

            // Initialize renderer by creating shaders and buffers
            if (!this->render_data.initialized)
            {
                // Create shaders and link them
                const std::string vertex_shader =
                    "#version 330 \n" \
                    "layout(location = 0) in vec2 in_position; \n" \
                    "layout(location = 1) in float in_value; \n" \
                    "layout(location = 2) in float in_mask; \n" \
                    "uniform mat4 model_view_matrix; \n" \
                    "uniform mat4 projection_matrix; \n" \
                    "uniform float min_value; \n" \
                    "uniform float max_value; \n" \
                    "uniform vec4 mask_color; \n" \
                    "uniform sampler1D transfer_function; \n" \
                    "out vec4 vertex_color; \n" \
                    "void main() { \n" \
                    "    gl_Position = projection_matrix * model_view_matrix * vec4(in_position, 0.0f, 1.0f); \n" \
                    "    vertex_color = in_mask == 1.0f ? " \
                    "        texture(transfer_function, (min_value == max_value) ? 0.5f : ((in_value - min_value) / (max_value - min_value)))" \
                    "        : mask_color; \n" \
                    "}";

                const std::string fragment_shader =
                    "#version 330\n" \
                    "in vec4 vertex_color; \n" \
                    "out vec4 fragColor; \n" \
                    "void main() { \n" \
                    "    fragColor = vertex_color; \n" \
                    "}";

                try
                {
                    this->render_data.vs = utility::make_shader(vertex_shader, GL_VERTEX_SHADER);
                    this->render_data.fs = utility::make_shader(fragment_shader, GL_FRAGMENT_SHADER);

                    this->render_data.prog = utility::make_program({ this->render_data.vs, this->render_data.fs });
                }
                catch (const std::exception& e)
                {
                    vislib::sys::Log::DefaultLog.WriteError(e.what());

                    return false;
                }

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
            glGetFloatv(GL_MODELVIEW_MATRIX, this->camera.model_view.data());
            glGetFloatv(GL_PROJECTION_MATRIX, this->camera.projection.data());

            // Update triangles (connection mandatory)
            auto get_triangles = this->triangle_mesh_slot.CallAs<triangle_mesh_call>();

            if (get_triangles == nullptr || !(*get_triangles)(0)) return false;

            if (get_triangles->DataHash() != this->triangle_mesh_hash)
            {
                // Set hash
                this->triangle_mesh_hash = get_triangles->DataHash();

                // Get vertices and indices
                this->render_data.vertices = get_triangles->get_vertices();
                this->render_data.indices = get_triangles->get_indices();

                // Prepare OpenGL buffers
                if (this->render_data.vertices != nullptr && this->render_data.indices != nullptr)
                {
                    glBindVertexArray(this->render_data.vao);

                    glBindBuffer(GL_ARRAY_BUFFER, this->render_data.vbo);
                    glBufferData(GL_ARRAY_BUFFER, this->render_data.vertices->size() * sizeof(GLfloat), this->render_data.vertices->data(), GL_STATIC_DRAW);

                    glEnableVertexAttribArray(0);
                    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, nullptr);

                    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, this->render_data.ibo);
                    glBufferData(GL_ELEMENT_ARRAY_BUFFER, this->render_data.indices->size() * sizeof(GLuint), this->render_data.indices->data(), GL_STATIC_DRAW);

                    glBindVertexArray(0);
                }
            }

            // Update data (connection optional)
            if (this->render_data.vertices != nullptr && this->render_data.indices != nullptr)
            {
                auto get_data = this->mesh_data_slot.CallAs<mesh_data_call>();

                bool new_data = false;
                bool new_mask = false;

                if (get_data != nullptr && (*get_data)(0))
                {
                    if (get_data->DataHash() != this->mesh_data_hash || this->data_set.IsDirty())
                    {
                        // Set hash and reset parameter
                        this->data_set.ResetDirty();

                        this->render_data.values = get_data->get_data(this->data_set.Param<core::param::FlexEnumParam>()->Value());

                        new_data = true;
                    }

                    if (get_data->DataHash() != this->mesh_data_hash || this->mask.IsDirty())
                    {
                        // Set hash and reset parameter
                        this->mask.ResetDirty();

                        this->render_data.mask = get_data->get_mask(this->mask.Param<core::param::FlexEnumParam>()->Value());

                        new_mask = true;
                    }

                    this->mesh_data_hash = get_data->DataHash();
                }

                if (this->render_data.values == nullptr)
                {
                    this->render_data.values = std::make_shared<mesh_data_call::data_set>();
                    this->render_data.values->transfer_function = "{\"Interpolation\":\"LINEAR\",\"Nodes\":[[0.0,0.0,0.0,0.0,0.0],[1.0,1.0,1.0,1.0,1.0]],\"TextureSize\":2}";
                    this->render_data.values->min_value = 0.0f;
                    this->render_data.values->max_value = 1.0f;
                    this->render_data.values->data = std::make_shared<std::vector<GLfloat>>(this->render_data.vertices->size() / 2, 1.0f);
                }

                if (this->render_data.mask == nullptr)
                {
                    this->render_data.mask = std::make_shared<std::vector<GLfloat>>(this->render_data.vertices->size() / 2, 1.0f);

                    new_mask = true;
                }

                // Prepare OpenGL buffers
                glBindVertexArray(this->render_data.vao);

                glBindBuffer(GL_ARRAY_BUFFER, this->render_data.cbo);
                glBufferData(GL_ARRAY_BUFFER, this->render_data.values->data->size() * sizeof(GLfloat), this->render_data.values->data->data(), GL_STATIC_DRAW);

                glEnableVertexAttribArray(1);
                glVertexAttribPointer(1, 1, GL_FLOAT, GL_FALSE, 0, nullptr);

                glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, this->render_data.ibo);
                glBufferData(GL_ELEMENT_ARRAY_BUFFER, this->render_data.indices->size() * sizeof(GLuint), this->render_data.indices->data(), GL_STATIC_DRAW);

                glBindVertexArray(0);

                if (new_mask)
                {
                    glBindVertexArray(this->render_data.vao);

                    glBindBuffer(GL_ARRAY_BUFFER, this->render_data.mbo);
                    glBufferData(GL_ARRAY_BUFFER, this->render_data.mask->size() * sizeof(GLfloat), this->render_data.mask->data(), GL_STATIC_DRAW);

                    glEnableVertexAttribArray(2);
                    glVertexAttribPointer(2, 1, GL_FLOAT, GL_FALSE, 0, nullptr);

                    glBindVertexArray(0);
                }

                if (new_data || this->render_data.values->transfer_function_dirty)
                {
                    // Get transfer function texture data
                    std::vector<GLfloat> texture_data;
                    std::array<float, 2> _unused__texture_range;

                    core::param::TransferFunctionParam::TransferFunctionTexture(this->render_data.values->transfer_function,
                        texture_data, this->render_data.tf_size, _unused__texture_range);

                    // Create transfer funtion texture
                    glActiveTexture(GL_TEXTURE0);
                    glBindTexture(GL_TEXTURE_1D, this->render_data.tf);

                    glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
                    glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
                    glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

                    glTexImage1D(GL_TEXTURE_1D, 0, GL_RGBA, static_cast<GLsizei>(this->render_data.tf_size), 0, GL_RGBA, GL_FLOAT, static_cast<GLvoid*>(texture_data.data()));

                    glBindTexture(GL_TEXTURE_1D, 0);

                    this->render_data.values->transfer_function_dirty = false;
                }
            }

            // Render triangle mesh
            if (this->render_data.vertices != nullptr && this->render_data.indices != nullptr)
            {
                // Set wireframe or filled rendering
                if (this->wireframe.Param<core::param::BoolParam>()->Value())
                {
                    glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
                }
                else
                {
                    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
                }

                // Render
                glUseProgram(this->render_data.prog);
                glDisable(GL_DEPTH_TEST);
                glDepthMask(GL_FALSE);

                glUniformMatrix4fv(glGetUniformLocation(this->render_data.prog, "model_view_matrix"), 1, GL_FALSE, this->camera.model_view.data());
                glUniformMatrix4fv(glGetUniformLocation(this->render_data.prog, "projection_matrix"), 1, GL_FALSE, this->camera.projection.data());
                
                glUniform1f(glGetUniformLocation(this->render_data.prog, "min_value"), this->render_data.values->min_value);
                glUniform1f(glGetUniformLocation(this->render_data.prog, "max_value"), this->render_data.values->max_value);

                const auto mask_color = this->mask_color.Param<core::param::ColorParam>()->Value();
                glUniform4f(glGetUniformLocation(this->render_data.prog, "mask_color"), mask_color[0], mask_color[1], mask_color[2], mask_color[3]);

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

        bool triangle_mesh_renderer_2d::GetExtents(core::view::CallRender2D& call)
        {
            // Get and set bounding rectangle (connection mandatory)
            auto get_triangles = this->triangle_mesh_slot.CallAs<triangle_mesh_call>();

            if (get_triangles == nullptr || !(*get_triangles)(1)) return false;

            if (get_triangles->get_dimension() != triangle_mesh_call::dimension_t::TWO) {
                vislib::sys::Log::DefaultLog.WriteError("The dimension of the data does not fit the renderer");
                return false;
            }

            this->bounds = get_triangles->get_bounding_rectangle();

            // Get bounding rectangle of input renderer, if available
            auto* input_renderer = this->render_input_slot.CallAs<core::view::CallRender2D>();

            if (input_renderer != nullptr && (*input_renderer)(core::view::AbstractCallRender::FnGetExtents))
            {
                this->bounds.SetLeft(std::min(this->bounds.Left(), input_renderer->GetBoundingBox().Left()));
                this->bounds.SetRight(std::max(this->bounds.Right(), input_renderer->GetBoundingBox().Right()));
                this->bounds.SetBottom(std::min(this->bounds.Bottom(), input_renderer->GetBoundingBox().Bottom()));
                this->bounds.SetTop(std::max(this->bounds.Top(), input_renderer->GetBoundingBox().Top()));
            }

            call.SetBoundingBox(this->bounds);

            // Get data sets (connection optional)
            auto get_data = this->mesh_data_slot.CallAs<mesh_data_call>();

            if (get_data != nullptr && (*get_data)(1) && get_data->DataHash() != this->mesh_data_hash)
            {
                // Get available data sets
                {
                    const auto previous_value = this->data_set.Param<core::param::FlexEnumParam>()->Value();

                    this->data_set.Param<core::param::FlexEnumParam>()->ClearValues();

                this->data_set.Param<core::param::FlexEnumParam>()->AddValue("");
                this->data_set.Param<core::param::FlexEnumParam>()->SetValue("");

                    for (const auto& data_set : get_data->get_data_sets())
                    {
                        this->data_set.Param<core::param::FlexEnumParam>()->AddValue(data_set);

                        if (data_set == previous_value)
                        {
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

                    for (const auto& mask : get_data->get_masks())
                    {
                        this->mask.Param<core::param::FlexEnumParam>()->AddValue(mask);

                        if (mask == previous_value)
                        {
                            this->mask.Param<core::param::FlexEnumParam>()->SetValue(previous_value, false);
                        }
                    }
                }
            }

            return true;
        }

        bool triangle_mesh_renderer_2d::OnKey(core::view::Key key, core::view::KeyAction action, core::view::Modifiers mods)
        {
            auto* input_renderer = this->render_input_slot.template CallAs<core::view::CallRender2D>();
            if (input_renderer == nullptr) return false;

            core::view::InputEvent evt;
            evt.tag = core::view::InputEvent::Tag::Key;
            evt.keyData.key = key;
            evt.keyData.action = action;
            evt.keyData.mods = mods;

            input_renderer->SetInputEvent(evt);
            return (*input_renderer)(core::view::InputCall::FnOnKey);
        }

        bool triangle_mesh_renderer_2d::OnChar(unsigned int codePoint)
        {
            auto* input_renderer = this->render_input_slot.template CallAs<core::view::CallRender2D>();
            if (input_renderer == nullptr) return false;

            core::view::InputEvent evt;
            evt.tag = core::view::InputEvent::Tag::Char;
            evt.charData.codePoint = codePoint;

            input_renderer->SetInputEvent(evt);
            return (*input_renderer)(core::view::InputCall::FnOnChar);
        }

        bool triangle_mesh_renderer_2d::OnMouseButton(core::view::MouseButton button, core::view::MouseButtonAction action, core::view::Modifiers mods)
        {
            auto* input_renderer = this->render_input_slot.template CallAs<core::view::CallRender2D>();
            if (input_renderer == nullptr) return false;

            core::view::InputEvent evt;
            evt.tag = core::view::InputEvent::Tag::MouseButton;
            evt.mouseButtonData.button = button;
            evt.mouseButtonData.action = action;
            evt.mouseButtonData.mods = mods;

            input_renderer->SetInputEvent(evt);
            return (*input_renderer)(core::view::InputCall::FnOnMouseButton);
        }

        bool triangle_mesh_renderer_2d::OnMouseMove(double x, double y)
        {
            auto* input_renderer = this->render_input_slot.template CallAs<core::view::CallRender2D>();
            if (input_renderer == nullptr) return false;

            core::view::InputEvent evt;
            evt.tag = core::view::InputEvent::Tag::MouseMove;
            evt.mouseMoveData.x = x;
            evt.mouseMoveData.y = y;

            input_renderer->SetInputEvent(evt);
            return (*input_renderer)(core::view::InputCall::FnOnMouseMove);
        }

        bool triangle_mesh_renderer_2d::OnMouseScroll(double dx, double dy)
        {
            auto* input_renderer = this->render_input_slot.template CallAs<core::view::CallRender2D>();
            if (input_renderer == nullptr) return false;

            core::view::InputEvent evt;
            evt.tag = core::view::InputEvent::Tag::MouseScroll;
            evt.mouseScrollData.dx = dx;
            evt.mouseScrollData.dy = dy;

            input_renderer->SetInputEvent(evt);
            return (*input_renderer)(core::view::InputCall::FnOnMouseScroll);
        }
    }
}
