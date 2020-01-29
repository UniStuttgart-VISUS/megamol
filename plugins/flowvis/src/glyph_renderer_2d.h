/*
 * glyph_renderer_2d.h
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */
#pragma once

#include "glyph_data_call.h"

#include "mmcore/CallerSlot.h"
#include "mmcore/param/ParamSlot.h"
#include "mmcore/view/CallRender2D.h"
#include "mmcore/view/MouseFlags.h"
#include "mmcore/view/Renderer2DModule.h"

#include "vislib/math/Rectangle.h"

#include "glad/glad.h"

#include <memory>
#include <type_traits>
#include <vector>

namespace megamol
{
    namespace flowvis
    {
        /**
        * Module for rendering 2D glyphs.
        *
        * @author Alexander Straub
        */
        class glyph_renderer_2d : public core::view::Renderer2DModule
        {
            static_assert(std::is_same<GLfloat, float>::value, "'GLfloat' and 'float' must be the same type!");
            static_assert(std::is_same<GLuint, unsigned int>::value, "'GLuint' and 'unsigned int' must be the same type!");

        public:
            /**
             * Answer the name of this module.
             *
             * @return The name of this module.
             */
            static inline const char* ClassName() { return "glyph_renderer_2d"; }

            /**
             * Answer a human readable description of this module.
             *
             * @return A human readable description of this module.
             */
            static inline const char* Description() { return "2D glyph renderer"; }

            /**
             * Answers whether this module is available on the current system.
             *
             * @return 'true' if the module is available, 'false' otherwise.
             */
            static inline bool IsAvailable() { return true; }

            /**
             * Initialises a new instance.
             */
            glyph_renderer_2d();

            /**
             * Finalises an instance.
             */
            virtual ~glyph_renderer_2d();

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
             * The mouse button callback.
             *
             * @param button Mouse button that caused the event
             * @param action Type of interaction with the mouse button
             * @param mods Modifiers, such as control or shift keys on the keyboard
             *
             * @return 'true' if mouse event was handled, 'false' otherwise.
             */
            virtual bool OnMouseButton(core::view::MouseButton button, core::view::MouseButtonAction action, core::view::Modifiers mods) override;

            /**
             * The mouse move callback.
             *
             * @param x Current x-coordinate of the mouse in screen space
             * @param y Current y-coordinate of the mouse in screen space
             *
             * @return 'true' if mouse event was handled, 'false' otherwise.
             */
            virtual bool OnMouseMove(double x, double y) override;

            /**
            * Forwards scroll events.
            */
            virtual bool OnMouseScroll(double dx, double dy) override;

        private:
            /** Input render call */
            core::CallerSlot render_input_slot;

            /** Input slot for the glyphs */
            core::CallerSlot glyph_slot;
            SIZE_T glyph_hash;

            /** Input slot for the mouse event */
            core::CallerSlot mouse_slot;

            /** Parameter slots for defining the size and resolution of the glyphs */
            core::param::ParamSlot num_triangles;
            core::param::ParamSlot radius;
            core::param::ParamSlot width;

            /** Parameter slot for the transfer function */
            core::param::ParamSlot transfer_function;

            /** Parameter slots for fixed value range */
            core::param::ParamSlot range_fixed;
            core::param::ParamSlot range_min;
            core::param::ParamSlot range_max;

            /** Bounding rectangle */
            vislib::math::Rectangle<float> bounds;

            /** Mouse interaction */
            struct mouse_state_t
            {
                bool left_pressed;
                bool control_pressed;

                double x, y;

            } mouse_state;

            /** Struct for storing data needed for rendering */
            struct render_data_t
            {
                bool initialized = false;

                GLuint vs, fs, gs_p, gs_l, prog_p, prog_l;
                struct glyph_data_t { GLuint vao, vbo, ibo, cbo; } point, line;
                GLuint tf, tf_size;

                std::shared_ptr<std::vector<GLfloat>> point_vertices, line_vertices;
                std::shared_ptr<std::vector<GLuint>> point_indices, line_indices;

                std::shared_ptr<std::vector<GLfloat>> point_values, line_values;

                float min_value, max_value;

            } render_data;

            /** Struct for storing data needed for creating transformation matrices */
            struct camera_t
            {
                std::array<GLfloat, 16> model_view, projection;

            } camera;
        };
    }
}
