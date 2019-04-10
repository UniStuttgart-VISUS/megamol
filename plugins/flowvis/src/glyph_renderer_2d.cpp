#include "stdafx.h"
#include "glyph_renderer_2d.h"

#include "glyph_data_call.h"
#include "mouse_click_call.h"

#include "flowvis/shader.h"

#include "mmcore/param/IntParam.h"
#include "mmcore/param/LinearTransferFunctionParam.h"
#include "mmcore/view/CallRender2D.h"
#include "mmcore/view/MouseFlags.h"

#include "glad/glad.h"

#include <algorithm>
#include <string>
#include <vector>

namespace megamol
{
    namespace flowvis
    {
        glyph_renderer_2d::glyph_renderer_2d() :
            render_input_slot("render_input_slot", "Render input slot"),
            glyph_slot("get_glyphs", "Glyph input"), glyph_hash(-1),
            mouse_slot("mouse_slot", "Mouse events"),
            point_size("point_size", "Point size"),
            line_width("line_width", "Line width"),
            transfer_function("transfer_function", "Transfer function"),
            mouse_state({ false, false, -1.0, -1.0 })
        {
            // Connect input slots
            this->render_input_slot.SetCompatibleCall<core::view::CallRender2DDescription>();
            this->MakeSlotAvailable(&this->render_input_slot);

            this->glyph_slot.SetCompatibleCall<glyph_data_call::glyph_data_description>();
            this->MakeSlotAvailable(&this->glyph_slot);

            this->mouse_slot.SetCompatibleCall<mouse_click_call::mouse_click_description>();
            this->MakeSlotAvailable(&this->mouse_slot);

            // Connect parameter slots
            this->point_size << new core::param::IntParam(1);
            this->MakeSlotAvailable(&this->point_size);

            this->line_width << new core::param::IntParam(1);
            this->MakeSlotAvailable(&this->line_width);

            this->transfer_function << new core::param::LinearTransferFunctionParam("");
            this->MakeSlotAvailable(&this->transfer_function);

            // Set initial transfer function
            this->transfer_function.ForceSetDirty();
        }

        glyph_renderer_2d::~glyph_renderer_2d()
        {
            this->Release();
        }

        bool glyph_renderer_2d::create()
        {
            return true;
        }

        void glyph_renderer_2d::release()
        {
            // Remove shaders, buffers and arrays
            if (this->render_data.initialized)
            {
                glDetachShader(this->render_data.prog, this->render_data.vs);
                glDetachShader(this->render_data.prog, this->render_data.fs);
                glDeleteProgram(this->render_data.prog);

                glDeleteVertexArrays(1, &this->render_data.point.vao);
                glDeleteVertexArrays(1, &this->render_data.line.vao);
                glDeleteBuffers(1, &this->render_data.point.vbo);
                glDeleteBuffers(1, &this->render_data.line.vbo);
                glDeleteBuffers(1, &this->render_data.point.ibo);
                glDeleteBuffers(1, &this->render_data.line.ibo);
                glDeleteBuffers(1, &this->render_data.point.cbo);
                glDeleteBuffers(1, &this->render_data.line.cbo);

                glDeleteTextures(1, &this->render_data.tf);
            }

            return;
        }

        bool glyph_renderer_2d::Render(core::view::CallRender2D& call)
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
                    "uniform mat4 model_view_matrix; \n" \
                    "uniform mat4 projection_matrix; \n" \
                    "uniform float min_value; \n" \
                    "uniform float max_value; \n" \
                    "uniform sampler1D transfer_function; \n" \
                    "out vec4 vertex_color; \n" \
                    "void main() { \n" \
                    "    gl_Position = projection_matrix * model_view_matrix * vec4(in_position, 0.0f, 1.0f); \n" \
                    "    vertex_color = texture(transfer_function, (min_value == max_value) ? 0.5f : ((in_value - min_value) / (max_value - min_value))); \n" \
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
                glGenVertexArrays(1, &this->render_data.point.vao);
                glGenVertexArrays(1, &this->render_data.line.vao);
                glGenBuffers(1, &this->render_data.point.vbo);
                glGenBuffers(1, &this->render_data.line.vbo);
                glGenBuffers(1, &this->render_data.point.ibo);
                glGenBuffers(1, &this->render_data.line.ibo);
                glGenBuffers(1, &this->render_data.point.cbo);
                glGenBuffers(1, &this->render_data.line.cbo);

                // Create transfer function texture
                glGenTextures(1, &this->render_data.tf);

                this->render_data.initialized = true;
            }

            // Get camera transformation matrices
            glGetFloatv(GL_MODELVIEW_MATRIX, this->camera.model_view.data());
            glGetFloatv(GL_PROJECTION_MATRIX, this->camera.projection.data());

            // Update glyphs (connection mandatory)
            auto get_glyphs = this->glyph_slot.CallAs<glyph_data_call>();

            if (get_glyphs == nullptr || !(*get_glyphs)(0)) return false;

            if (get_glyphs->DataHash() != this->glyph_hash || this->render_data.point_vertices == nullptr)
            {
                // Set hash
                this->glyph_hash = get_glyphs->DataHash();

                // Get vertices and indices
                this->render_data.point_vertices = get_glyphs->get_point_vertices();
                this->render_data.line_vertices = get_glyphs->get_line_vertices();

                this->render_data.point_indices = get_glyphs->get_point_indices();
                this->render_data.line_indices = get_glyphs->get_line_indices();

                // Get values
                this->render_data.point_values = get_glyphs->get_point_values();
                this->render_data.line_values = get_glyphs->get_line_values();

                this->render_data.min_value = std::numeric_limits<float>::max();
                this->render_data.max_value = std::numeric_limits<float>::min();

                if (!this->render_data.point_values->empty())
                {
                    this->render_data.min_value = std::min(this->render_data.min_value, *std::min_element(this->render_data.point_values->begin(), this->render_data.point_values->end()));
                    this->render_data.max_value = std::max(this->render_data.max_value, *std::max_element(this->render_data.point_values->begin(), this->render_data.point_values->end()));
                }

                if (!this->render_data.line_values->empty())
                {
                    this->render_data.min_value = std::min(this->render_data.min_value, *std::min_element(this->render_data.line_values->begin(), this->render_data.line_values->end()));
                    this->render_data.max_value = std::max(this->render_data.max_value, *std::max_element(this->render_data.line_values->begin(), this->render_data.line_values->end()));
                }

                // Prepare OpenGL buffers for points
                if (!this->render_data.point_indices->empty())
                {
                    glBindVertexArray(this->render_data.point.vao);

                    glBindBuffer(GL_ARRAY_BUFFER, this->render_data.point.vbo);
                    glBufferData(GL_ARRAY_BUFFER, this->render_data.point_vertices->size() * sizeof(GLfloat), this->render_data.point_vertices->data(), GL_STATIC_DRAW);

                    glEnableVertexAttribArray(0);
                    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, nullptr);

                    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, this->render_data.point.ibo);
                    glBufferData(GL_ELEMENT_ARRAY_BUFFER, this->render_data.point_indices->size() * sizeof(GLuint), this->render_data.point_indices->data(), GL_STATIC_DRAW);

                    glBindBuffer(GL_ARRAY_BUFFER, this->render_data.point.cbo);
                    glBufferData(GL_ARRAY_BUFFER, this->render_data.point_values->size() * sizeof(GLfloat), this->render_data.point_values->data(), GL_STATIC_DRAW);

                    glEnableVertexAttribArray(1);
                    glVertexAttribPointer(1, 1, GL_FLOAT, GL_FALSE, 0, nullptr);

                    glBindVertexArray(0);
                }

                // Prepare OpenGL buffers for lines
                if (!this->render_data.line_indices->empty())
                {
                    glBindVertexArray(this->render_data.line.vao);

                    glBindBuffer(GL_ARRAY_BUFFER, this->render_data.line.vbo);
                    glBufferData(GL_ARRAY_BUFFER, this->render_data.line_vertices->size() * sizeof(GLfloat), this->render_data.line_vertices->data(), GL_STATIC_DRAW);

                    glEnableVertexAttribArray(0);
                    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, nullptr);

                    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, this->render_data.line.ibo);
                    glBufferData(GL_ELEMENT_ARRAY_BUFFER, this->render_data.line_indices->size() * sizeof(GLuint), this->render_data.line_indices->data(), GL_STATIC_DRAW);

                    glBindBuffer(GL_ARRAY_BUFFER, this->render_data.line.cbo);
                    glBufferData(GL_ARRAY_BUFFER, this->render_data.line_values->size() * sizeof(GLfloat), this->render_data.line_values->data(), GL_STATIC_DRAW);

                    glEnableVertexAttribArray(1);
                    glVertexAttribPointer(1, 1, GL_FLOAT, GL_FALSE, 0, nullptr);

                    glBindVertexArray(0);
                }
            }

            // Set transfer function
            if (this->transfer_function.IsDirty())
            {
                // Get transfer function texture data
                std::vector<GLfloat> texture_data;

                core::param::LinearTransferFunctionParam::TransferFunctionTexture(
                    this->transfer_function.Param<core::param::LinearTransferFunctionParam>()->Value(),
                    texture_data, this->render_data.tf_size);

                // Create transfer funtion texture
                glActiveTexture(GL_TEXTURE0);
                glBindTexture(GL_TEXTURE_1D, this->render_data.tf);

                glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
                glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
                glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

                glTexImage1D(GL_TEXTURE_1D, 0, GL_RGBA, static_cast<GLsizei>(this->render_data.tf_size), 0, GL_RGBA, GL_FLOAT, static_cast<GLvoid*>(texture_data.data()));

                glBindTexture(GL_TEXTURE_1D, 0);

                this->transfer_function.ResetDirty();
            }

            // Render
            glUseProgram(this->render_data.prog);
            glDisable(GL_DEPTH_TEST);
            glDepthMask(GL_FALSE);

            glUniformMatrix4fv(glGetUniformLocation(this->render_data.prog, "model_view_matrix"), 1, GL_FALSE, this->camera.model_view.data());
            glUniformMatrix4fv(glGetUniformLocation(this->render_data.prog, "projection_matrix"), 1, GL_FALSE, this->camera.projection.data());

            glUniform1f(glGetUniformLocation(this->render_data.prog, "min_value"), this->render_data.min_value);
            glUniform1f(glGetUniformLocation(this->render_data.prog, "max_value"), this->render_data.max_value);

            if (!this->render_data.point_indices->empty())
            {
                glBindVertexArray(this->render_data.point.vao);
                glActiveTexture(GL_TEXTURE0);
                glBindTexture(GL_TEXTURE_1D, this->render_data.tf);
                glPointSize(static_cast<float>(this->point_size.Param<core::param::IntParam>()->Value()));
                glDrawElements(GL_POINTS, static_cast<GLsizei>(this->render_data.point_indices->size()), GL_UNSIGNED_INT, nullptr);
                glPointSize(1.0f);
                glBindTexture(GL_TEXTURE_1D, 0);
                glBindVertexArray(0);
            }

            if (!this->render_data.line_indices->empty())
            {
                glBindVertexArray(this->render_data.line.vao);
                glActiveTexture(GL_TEXTURE0);
                glBindTexture(GL_TEXTURE_1D, this->render_data.tf);
                glLineWidth(static_cast<float>(this->line_width.Param<core::param::IntParam>()->Value()));
                glDrawElements(GL_LINES, static_cast<GLsizei>(this->render_data.line_indices->size()), GL_UNSIGNED_INT, nullptr);
                glLineWidth(1.0f);
                glBindTexture(GL_TEXTURE_1D, 0);
                glBindVertexArray(0);
            }

            glDepthMask(GL_TRUE);
            glEnable(GL_DEPTH_TEST);
            glUseProgram(0);

            return true;
        }

        bool glyph_renderer_2d::GetExtents(core::view::CallRender2D& call)
        {
            // Get and set bounding rectangle (connection mandatory)
            auto get_glyphs = this->glyph_slot.CallAs<glyph_data_call>();

            if (get_glyphs == nullptr || !(*get_glyphs)(1)) return false;

            this->bounds = get_glyphs->get_bounding_rectangle();

            // Get bounding rectangle of input renderer, if available
            auto* input_renderer = this->render_input_slot.CallAs<core::view::CallRender2D>();

            if (input_renderer != nullptr && (*input_renderer)(core::view::AbstractCallRender::FnGetExtents))
            {
                if (get_glyphs->has_bounding_rectangle())
                {
                    this->bounds.SetLeft(std::min(this->bounds.Left(), input_renderer->GetBoundingBox().Left()));
                    this->bounds.SetRight(std::max(this->bounds.Right(), input_renderer->GetBoundingBox().Right()));
                    this->bounds.SetBottom(std::min(this->bounds.Bottom(), input_renderer->GetBoundingBox().Bottom()));
                    this->bounds.SetTop(std::max(this->bounds.Top(), input_renderer->GetBoundingBox().Top()));
                }
                else
                {
                    this->bounds.SetLeft(input_renderer->GetBoundingBox().Left());
                    this->bounds.SetRight(input_renderer->GetBoundingBox().Right());
                    this->bounds.SetBottom(input_renderer->GetBoundingBox().Bottom());
                    this->bounds.SetTop(input_renderer->GetBoundingBox().Top());
                }
            }

            call.SetBoundingBox(this->bounds);

            return true;
        }

        bool glyph_renderer_2d::OnKey(core::view::Key key, core::view::KeyAction action, core::view::Modifiers mods)
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

        bool glyph_renderer_2d::OnChar(unsigned int codePoint)
        {
            auto* input_renderer = this->render_input_slot.template CallAs<core::view::CallRender2D>();
            if (input_renderer == nullptr) return false;

            core::view::InputEvent evt;
            evt.tag = core::view::InputEvent::Tag::Char;
            evt.charData.codePoint = codePoint;

            input_renderer->SetInputEvent(evt);
            return (*input_renderer)(core::view::InputCall::FnOnChar);
        }

        bool glyph_renderer_2d::OnMouseButton(core::view::MouseButton button, core::view::MouseButtonAction action, core::view::Modifiers mods)
        {
            // Save mouse state
            this->mouse_state.left_pressed = button == core::view::MouseButton::BUTTON_LEFT && action == core::view::MouseButtonAction::PRESS;
            this->mouse_state.control_pressed = mods.test(core::view::Modifier::CTRL);

            // If control is pressed, left mouse button is released and it is inside the data's extent, consume the event
            if (!this->mouse_state.left_pressed && this->mouse_state.control_pressed &&
                this->mouse_state.x >= this->bounds.Left() && this->mouse_state.x <= this->bounds.Right() &&
                this->mouse_state.y >= this->bounds.Bottom() && this->mouse_state.y <= this->bounds.Top())
            {
                auto* mouse_call = this->mouse_slot.CallAs<mouse_click_call>();

                if (mouse_call != nullptr)
                {
                    mouse_call->set_coordinates(std::make_pair(static_cast<float>(this->mouse_state.x), static_cast<float>(this->mouse_state.y)));
                    
                    (*mouse_call)(0);
                }
            }

            // Forward event
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

        bool glyph_renderer_2d::OnMouseMove(double x, double y, double world_x, double world_y)
        {
            // Track mouse position
            this->mouse_state.x = world_x;
            this->mouse_state.y = world_y;

            // Forward event
            auto* input_renderer = this->render_input_slot.template CallAs<core::view::CallRender2D>();
            if (input_renderer == nullptr) return false;

            core::view::InputEvent evt;
            evt.tag = core::view::InputEvent::Tag::MouseMove;
            evt.mouseMoveData.x = x;
            evt.mouseMoveData.y = y;
            evt.mouseMoveData.world_x = world_x;
            evt.mouseMoveData.world_y = world_y;

            input_renderer->SetInputEvent(evt);
            return (*input_renderer)(core::view::InputCall::FnOnMouseMove);
        }

        bool glyph_renderer_2d::OnMouseScroll(double dx, double dy)
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
