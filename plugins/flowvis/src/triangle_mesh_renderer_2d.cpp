#include "stdafx.h"
#include "triangle_mesh_renderer_2d.h"

#include "mmcore/param/BoolParam.h"
#include "mmcore/param/FlexEnumParam.h"
#include "mmcore/view/CallRender2D.h"
#include "mmcore/view/MouseFlags.h"

#include "vislib/sys/Log.h"

#include "glad/glad.h"

#include "mesh_data_call.h"
#include "triangle_mesh_call.h"

#include <algorithm>
#include <exception>
#include <stdexcept>
#include <string>
#include <vector>

namespace megamol
{
    namespace flowvis
    {
        triangle_mesh_renderer_2d::triangle_mesh_renderer_2d() :
            triangle_mesh_slot("get_triangle_mesh", "Triangle mesh input"), triangle_mesh_hash(0),
            mesh_data_slot("get_mesh_data", "Mesh data input"), mesh_data_hash(0),
            data_set("data_set", "Data set used for coloring the triangles"),
            wireframe("wireframe", "Render as wireframe instead of filling the triangles"),
            mouse_state({ false, false, -1.0, -1.0 })
        {
            // Connect input slots
            this->triangle_mesh_slot.SetCompatibleCall<triangle_mesh_call::triangle_mesh_description>();
            this->MakeSlotAvailable(&this->triangle_mesh_slot);

            this->mesh_data_slot.SetCompatibleCall<mesh_data_call::mesh_data_description>();
            this->MakeSlotAvailable(&this->mesh_data_slot);

            // Connect parameter slots
            this->data_set << new core::param::FlexEnumParam("");
            this->MakeSlotAvailable(&this->data_set);

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
            }

            return;
        }

        bool triangle_mesh_renderer_2d::Render(core::view::CallRender2D& call)
        {
            // Initialize renderer by creating shaders and buffers
            if (!this->render_data.initialized)
            {
                // Create shaders and link them
                const std::string vertex_shader =
                    "#version 330 \n" \
                    "layout(location = 0) in vec2 in_position; \n" \
                    "layout(location = 1) in vec4 in_value; \n" \
                    "uniform mat4 model_view_matrix; \n" \
                    "uniform mat4 projection_matrix; \n" \
                    "out vec4 vertex_color; \n" \
                    "void main() { \n" \
                    "    gl_Position = projection_matrix * model_view_matrix * vec4(in_position, 0.0f, 1.0f); \n" \
                    "    vertex_color = in_value; \n" \
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
                    this->render_data.vs = make_shader(vertex_shader, GL_VERTEX_SHADER);
                    this->render_data.fs = make_shader(fragment_shader, GL_FRAGMENT_SHADER);

                    this->render_data.prog = make_program({ this->render_data.vs, this->render_data.fs });
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
                glBindVertexArray(this->render_data.vao);

                glBindBuffer(GL_ARRAY_BUFFER, this->render_data.vbo);
                glBufferData(GL_ARRAY_BUFFER, this->render_data.vertices->size() * sizeof(GLfloat), this->render_data.vertices->data(), GL_STATIC_DRAW);

                glEnableVertexAttribArray(0);
                glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, nullptr);

                glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, this->render_data.ibo);
                glBufferData(GL_ELEMENT_ARRAY_BUFFER, this->render_data.indices->size() * sizeof(GLuint), this->render_data.indices->data(), GL_STATIC_DRAW);

                glBindVertexArray(0);
            }

            // Update data (connection optional)
            {
                auto get_data = this->mesh_data_slot.CallAs<mesh_data_call>();

                bool new_data = false;

                if (get_data != nullptr && (*get_data)(0) && (get_data->DataHash() != this->mesh_data_hash || this->data_set.IsDirty()))
                {
                    // Set hash and reset parameter
                    this->mesh_data_hash = get_data->DataHash();
                    this->data_set.ResetDirty();

                    this->render_data.colors = get_data->get_data(this->data_set.Param<core::param::FlexEnumParam>()->Value());

                    new_data = true;
                }

                if (this->render_data.colors == nullptr)
                {
                    this->render_data.colors = std::make_shared<std::vector<GLfloat>>(this->render_data.vertices->size() * 2, 1.0f);

                    new_data = true;
                }

                // Prepare OpenGL buffers
                if (new_data)
                {
                    glBindVertexArray(this->render_data.vao);

                    glBindBuffer(GL_ARRAY_BUFFER, this->render_data.cbo);
                    glBufferData(GL_ARRAY_BUFFER, this->render_data.colors->size() * sizeof(GLfloat), this->render_data.colors->data(), GL_STATIC_DRAW);

                    glEnableVertexAttribArray(1);
                    glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, 0, nullptr);

                    glBindVertexArray(0);
                }
            }

            // Render triangle mesh
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

                glBindVertexArray(this->render_data.vao);
                glDrawElements(GL_TRIANGLES, static_cast<GLsizei>(this->render_data.indices->size()), GL_UNSIGNED_INT, nullptr);
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

            this->bounds = get_triangles->get_bounding_rectangle();
            call.SetBoundingBox(this->bounds);

            // Get available data sets (connection optional)
            auto get_data = this->mesh_data_slot.CallAs<mesh_data_call>();

            if (get_data != nullptr && (*get_data)(1) && get_data->DataHash() != this->mesh_data_hash)
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

            return true;
        }

        bool triangle_mesh_renderer_2d::OnMouseButton(core::view::MouseButton button, core::view::MouseButtonAction action, core::view::Modifiers mods)
        {
            // Save mouse state
            this->mouse_state.left_pressed = button == core::view::MouseButton::BUTTON_LEFT && action == core::view::MouseButtonAction::PRESS;
            this->mouse_state.control_pressed = mods.test(core::view::Modifier::CTRL);

            // If control is pressed, left mouse button is released and it is inside the data's extent, consume the event
            if (!this->mouse_state.left_pressed && this->mouse_state.control_pressed &&
                this->mouse_state.x >= this->bounds.Left() && this->mouse_state.x <= this->bounds.Right() &&
                this->mouse_state.y >= this->bounds.Bottom() && this->mouse_state.y <= this->bounds.Top())
            {
                vislib::sys::Log::DefaultLog.WriteInfo("Event at %.2f x %.2f!", this->mouse_state.x, this->mouse_state.y);
                // TODO

                return true;
            }

            return false;
        }

        bool triangle_mesh_renderer_2d::OnMouseMove(double, double, double world_x, double world_y)
        {
            // Track mouse position
            this->mouse_state.x = world_x;
            this->mouse_state.y = world_y;

            // Claim mouse event if control key is pressed
            return this->mouse_state.control_pressed;
        }

        GLuint triangle_mesh_renderer_2d::make_shader(const std::string& shader, GLenum type) const
        {
            const GLchar* vertex_shader_ptr = shader.c_str();
            const GLint vertex_shader_length = static_cast<GLint>(shader.length());

            GLuint shader_handle = glCreateShader(type);
            glShaderSource(shader_handle, 1, &vertex_shader_ptr, &vertex_shader_length);
            glCompileShader(shader_handle);

            // Check compile status
            GLint compile_status;
            glGetShaderiv(shader_handle, GL_COMPILE_STATUS, &compile_status);

            if (compile_status == GL_FALSE)
            {
                int info_log_length = 0;
                glGetShaderiv(shader_handle, GL_INFO_LOG_LENGTH, &info_log_length);

                if (info_log_length > 1)
                {
                    int chars_written = 0;
                    std::vector<GLchar> info_log(info_log_length);

                    glGetShaderInfoLog(shader_handle, info_log_length, &chars_written, info_log.data());

                    throw std::runtime_error(info_log.data());
                }
                else
                {
                    throw std::runtime_error("Unknown shader compile error");
                }
            }

            return shader_handle;
        }

        GLuint triangle_mesh_renderer_2d::make_program(const std::vector<GLuint>& shader_handles) const
        {
            GLuint program_handle = glCreateProgram();

            for (const auto shader_handle : shader_handles)
            {
                glAttachShader(program_handle, shader_handle);
            }
            
            glLinkProgram(program_handle);
            glUseProgram(0);

            // Check link status
            GLint link_status;
            glGetShaderiv(program_handle, GL_LINK_STATUS, &link_status);

            if (link_status == GL_FALSE)
            {
                int info_log_length = 0;
                glGetShaderiv(program_handle, GL_INFO_LOG_LENGTH, &info_log_length);

                if (info_log_length > 1)
                {
                    int chars_written = 0;
                    std::vector<GLchar> info_log(info_log_length);

                    glGetShaderInfoLog(program_handle, info_log_length, &chars_written, info_log.data());

                    throw std::runtime_error(info_log.data());
                }
                else
                {
                    throw std::runtime_error("Unknown shader compile error");
                }
            }

            return program_handle;
        }
    }
}
