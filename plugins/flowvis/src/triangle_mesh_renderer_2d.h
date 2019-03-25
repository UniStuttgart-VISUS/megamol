#pragma once

#include "mmcore/CallerSlot.h"
#include "mmcore/param/ParamSlot.h"
#include "mmcore/view/CallRender2D.h"
#include "mmcore/view/MouseFlags.h"
#include "mmcore/view/Renderer2DModule.h"

#include "vislib/math/Rectangle.h"

#include "glad/glad.h"

#include <array>
#include <memory>
#include <vector>

namespace megamol
{
    namespace flowvis
    {
        class triangle_mesh_renderer_2d : public core::view::Renderer2DModule
        {
        public:
            /**
             * Answer the name of this module.
             *
             * @return The name of this module.
             */
            static inline const char* ClassName() { return "triangle_mesh_renderer_2d"; }

            /**
             * Answer a human readable description of this module.
             *
             * @return A human readable description of this module.
             */
            static inline const char* Description() { return "Triangle mesh renderer for 2D data"; }

            /**
             * Answers whether this module is available on the current system.
             *
             * @return 'true' if the module is available, 'false' otherwise.
             */
            static inline bool IsAvailable() { return true; }

            /**
             * Initialises a new instance.
             */
            triangle_mesh_renderer_2d();

            /**
             * Finalises an instance.
             */
            virtual ~triangle_mesh_renderer_2d();

        protected:
            /**
             * Implementation of 'Create'.
             *
             * @return 'true' on success, 'false' otherwise.
             */
            virtual bool create() override;

            /**
             * Implementation of 'Release'.
             */
            virtual void release() override;

            /**
             * The render callback.
             *
             * @param call The calling call.
             *
             * @return 'true' on success, 'false' otherwise.
             */
            virtual bool Render(core::view::CallRender2D& call) override;

            /**
             * The extent callback.
             *
             * @param call The calling call.
             *
             * @return 'true' on success, 'false' otherwise.
             */
            virtual bool GetExtents(core::view::CallRender2D& call) override;

            /**
             * The render callback.
             *
             * @param x X coordinate of the mouse event
             * @param y Y coordinate of the mouse event
             * @param flags Mouse flags
             *
             * @return 'true' if mouse event was handled, 'false' otherwise.
             */
            virtual bool MouseEvent(float x, float y, core::view::MouseFlags flags) override;

        private:
            /**
            * Create shader, additionally performing checks.
            * Throws an exception if it fails to compile the shader.
            *
            * @param shader Shader text
            * @param type Shader type
            *
            * @return Shader handle
            */
            GLuint make_shader(const std::string& shader, GLenum type) const;

            /**
            * Create program, additionally performing checks.
            * Throws an exception if it fails to link the shaders.
            *
            * @param shader Shader text
            * @param type Shader type
            *
            * @return Shader handle
            */
            GLuint make_program(const std::vector<GLuint>& shader_handles) const;

            /** Input slot for the triangle mesh */
            core::CallerSlot triangle_mesh_slot;
            SIZE_T triangle_mesh_hash;

            /** Input slot for data attached to the triangles or their nodes */
            core::CallerSlot mesh_data_slot;
            SIZE_T mesh_data_hash;

            /** Parameter slot for choosing data sets to visualize */
            core::param::ParamSlot data_set;

            /** Parameter slot for choosing between filled and wireframe mode */
            core::param::ParamSlot wireframe;

            /** Bounding rectangle */
            vislib::math::Rectangle<float> bounds;

            /** Struct for storing data needed for rendering */
            struct render_data_t
            {
                bool initialized = false;

                GLuint vao, vbo, ibo, cbo;
                GLuint vs, fs, prog;

                std::shared_ptr<std::vector<GLfloat>> vertices;
                std::shared_ptr<std::vector<GLuint>> indices;
                std::shared_ptr<std::vector<GLfloat>> colors;

            } render_data;

            /** Struct for storing data needed for creating transformation matrices */
            struct camera_t
            {
                std::array<GLfloat, 16> model_view, projection;

            } camera;
        };
    }
}