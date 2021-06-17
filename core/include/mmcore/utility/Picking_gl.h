/*
 * Picking_gl.h
 *
 * Copyright (C) 2020 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOL_GUI_PICKING_GL_INCLUDED
#define MEGAMOL_GUI_PICKING_GL_INCLUDED
#pragma once


#include "mmcore/utility/RenderUtils.h"


namespace megamol {
namespace core {
namespace utility {


    enum InteractionType {
        MOVE_ALONG_AXIS_SCREEN,
        MOVE_ALONG_AXIS_3D,
        MOVE_IN_3D_PLANE,
        ROTATE_AROUND_AXIS,
        SELECT,
        DESELECT,
        HIGHLIGHT
    };

    struct Interaction {
        InteractionType type;
        unsigned int obj_id;
        float axis_x;
        float axis_y;
        float axis_z;
        float origin_x;
        float origin_y;
        float origin_z;
    };

    struct Manipulation {
        InteractionType type;
        unsigned int obj_id;
        float axis_x;
        float axis_y;
        float axis_z;
        float value;
    };

    typedef std::shared_ptr<glowl::GLSLProgram> ShaderPtr_t;
    typedef std::vector<Interaction> InteractVector_t;
    typedef std::vector<Manipulation> ManipVector_t;

    /** ************************************************************************
     * OpenGL implementation of picking
     *
     * (Code adapted from megamol::mesh::Render3DUI)
     */
    class PickingBuffer {
    public:

        PickingBuffer();
        ~PickingBuffer();

        // Call only once per frame
        bool EnableInteraction(glm::vec2 vp_dim);

        // Call only once per frame
        bool DisableInteraction();

        bool ProcessMouseMove(double x, double y);

        bool ProcessMouseClick(megamol::core::view::MouseButton button, megamol::core::view::MouseButtonAction action,
                               megamol::core::view::Modifiers mods);

        void AddInteractionObject(unsigned int obj_id, std::vector<Interaction> const& interactions) {
            this->available_interactions.insert({obj_id, interactions});
        }

        ManipVector_t& GetPendingManipulations() {
            return this->pending_manipulations;
        }

    private:
        // VARIABLES --------------------------------------------------------------

        double cursor_x, cursor_y;
        glm::ivec2 viewport_dim;

        /**
         * Set to true if cursor is on interactable object during current frame with respective obj id as second value
         * Set to fale if cursor is on "background" during current frame with -1 as second value
         */
        std::tuple<bool, int, float> cursor_on_interaction_obj;

        /**
         * Set to true if cursor is on interactable object and mouse interaction (click, move) is ongoing with
         * respective obj id as second value Set to fale if cursor is on "background" during current frame with -1 as
         * second value
         */
        std::tuple<bool, unsigned int, float> active_interaction_obj;

        std::map<unsigned int, std::vector<Interaction>> available_interactions;
        ManipVector_t pending_manipulations;

        std::shared_ptr<glowl::FramebufferObject> fbo;
        std::shared_ptr<glowl::GLSLProgram> fbo_shader;

        bool enabled;

        // FUNCTIONS --------------------------------------------------------------

        std::vector<Interaction> get_available_interactions(unsigned int obj_id) {
            std::vector<Interaction> retval;
            auto query = this->available_interactions.find(obj_id);
            if (query != this->available_interactions.end()) {
                retval = query->second;
            }
            return retval;
        }
    };

} // namespace utility
} // namespace core
} // namespace megamol

#endif // MEGAMOL_GUI_PICKING_GL_INCLUDED
