/*
 * WidgetPicking_gl.h
 *
 * Copyright (C) 2020 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOL_GUI_WIDGETPICKING_GL_INCLUDED
#define MEGAMOL_GUI_WIDGETPICKING_GL_INCLUDED


#include "GUIUtils.h"

#include "mmcore/thecam/math/functions.h"
#include "mmcore/utility/log/Log.h"
#include "mmcore/view/Input.h"
#include "mmcore/view/RenderUtils.h"

#include <glm/gtc/matrix_transform.hpp>
#include <queue>

#include <tuple>


namespace megamol {
namespace gui {


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

    typedef std::shared_ptr<glowl::GLSLProgram> ShaderPtr;
    typedef std::vector<Interaction> InteractVector;
    typedef std::vector<Manipulation> ManipVector;

    class GUIWindows;

    /**
     * OpenGL implementation of widget picking.
     *
     * (Code adapted from megamol::mesh::Render3DUI)
     *
     */
    class PickingBuffer {
    public:
        friend class GUIWindows;

        void AddInteractionObject(unsigned int obj_id, std::vector<Interaction> const& interactions) {
            this->available_interactions.insert({obj_id, interactions});
        }

        ManipVector& GetPendingManipulations(void) {
            return this->pending_manipulations;
        }

    protected:
        // FUNCTIONS --------------------------------------------------------------
        /// Should only be callable by friend class who owns the object

        PickingBuffer(void);
        ~PickingBuffer(void);

        // Call only once per frame
        bool EnableInteraction(glm::vec2 vp_dim);

        // Call only once per frame
        bool DisableInteraction(void);

        bool ProcessMouseMove(double x, double y);

        bool ProcessMouseClick(megamol::core::view::MouseButton button, megamol::core::view::MouseButtonAction action,
            megamol::core::view::Modifiers mods);

    private:
        // VARIABLES --------------------------------------------------------------

        double cursor_x, cursor_y;
        glm::vec2 viewport_dim;

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

        std::map<int, std::vector<Interaction>> available_interactions;
        ManipVector pending_manipulations;

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

} // namespace gui
} // namespace megamol

#endif // MEGAMOL_GUI_WIDGETPICKING_GL_INCLUDED
