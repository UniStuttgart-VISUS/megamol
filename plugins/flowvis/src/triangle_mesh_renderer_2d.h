/*
 * triangle_mesh_renderer_2d.h
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */
#pragma once

#include "mesh_data_call.h"

#include "mmcore/CallerSlot.h"
#include "mmcore/param/ParamSlot.h"
#include "mmcore/view/CallRender2D.h"
#include "mmcore/view/MouseFlags.h"
#include "mmcore/view/Renderer2DModule.h"

#include "vislib/math/Rectangle.h"

#include "glad/glad.h"

#include <array>
#include <memory>
#include <string>
#include <type_traits>
#include <vector>

namespace megamol
{
    namespace flowvis
    {
        /**
        * Module for rendering a 2D triangle mesh.
        *
        * @author Alexander Straub
        */
        class triangle_mesh_renderer_2d : public core::view::Renderer2DModule
        {
            static_assert(std::is_same<GLfloat, float>::value, "'GLfloat' and 'float' must be the same type!");
            static_assert(std::is_same<GLuint, unsigned int>::value, "'GLuint' and 'unsigned int' must be the same type!");

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
            * Forwards key events.
            */
            virtual bool OnKey(core::view::Key key, core::view::KeyAction action, core::view::Modifiers mods) override;

            /**
            * Forwards character events.
            */
            virtual bool OnChar(unsigned int codePoint) override;

            /**
            * Forwards character events.
            */
            virtual bool OnMouseButton(core::view::MouseButton button, core::view::MouseButtonAction action, core::view::Modifiers mods) override;

            /**
            * Forwards character events.
            */
            virtual bool OnMouseMove(double x, double y, double world_x, double world_y) override;

            /**
            * Forwards scroll events.
            */
            virtual bool OnMouseScroll(double dx, double dy) override;

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

            /** Parameter slot for choosing between filled and wireframe mode */
            core::param::ParamSlot wireframe;

            /** Bounding rectangle */
            vislib::math::Rectangle<float> bounds;

            /** Struct for storing data needed for rendering */
            struct render_data_t
            {
                bool initialized = false;

                GLuint vs, fs, prog;
                GLuint vao, vbo, ibo, cbo, mbo;
                GLuint tf, tf_size;

                std::shared_ptr<std::vector<GLfloat>> vertices;
                std::shared_ptr<std::vector<GLuint>> indices;

                std::shared_ptr<mesh_data_call::data_set> values;

                std::shared_ptr<std::vector<GLfloat>> mask;

            } render_data;

            /** Struct for storing data needed for creating transformation matrices */
            struct camera_t
            {
                std::array<GLfloat, 16> model_view, projection;

            } camera;
        };
    }
}