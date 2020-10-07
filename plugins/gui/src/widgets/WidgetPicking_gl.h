/*
 * WidgetPicking_gl.h
 *
 * Copyright (C) 2020 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOL_GUI_WIDGETPICKING_GL_INCLUDED
#define MEGAMOL_GUI_WIDGETPICKING_GL_INCLUDED


#include "GUIUtils.h"

#include "mmcore/utility/log/Log.h"
#include "mmcore/view/Input.h"

#include <glm/gtc/matrix_transform.hpp>
#include <queue>

#define GLOWL_OPENGL_INCLUDE_GLAD
#include "glowl/FramebufferObject.hpp"
#include "glowl/GLSLProgram.hpp"

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
    int obj_id;
    float axis_x;
    float axis_y;
    float axis_z;
    float origin_x;
    float origin_y;
    float origin_z;
};

struct Manipulation {
    InteractionType type;
    int obj_id;
    float axis_x;
    float axis_y;
    float axis_Z;
    float value;
};

typedef std::shared_ptr<glowl::GLSLProgram> ShaderPtr;
typedef std::vector<Interaction> InteractVector;
typedef std::vector<Manipulation> ManipVector;

/**
 * OpenGL implementation of widget picking.
 *
 * (Code adapted from megamol::mesh::Render3DUI)
 *
 */
class PickingBuffer {
public:
    PickingBuffer(void);
    ~PickingBuffer(void);

    bool ProcessMouseMove(double x, double y);

    bool ProcessMouseClick(megamol::core::view::MouseButton button, megamol::core::view::MouseButtonAction action,
        megamol::core::view::Modifiers mods);

    bool EnableInteraction(glm::vec2 vp_dim);

    bool DisableInteraction(void);

    void AddInteractionObject(int obj_id, std::vector<Interaction> const& interactions) {
        this->available_interactions.insert({obj_id, interactions});
    }

    ManipVector& GetPendingManipulations(void) { return this->pending_manipulations; }

    static bool CreatShader(ShaderPtr& shader_ptr, const std::string& vertex_src, const std::string& fragment_src);

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
     * Set to true if cursor is on interactable object and mouse interaction (click, move) is ongoing with respective
     * obj id as second value Set to fale if cursor is on "background" during current frame with -1 as second value
     */
    std::tuple<bool, int, float> active_interaction_obj;

    std::map<int, std::vector<Interaction>> available_interactions;
    ManipVector pending_manipulations;

    std::unique_ptr<glowl::FramebufferObject> fbo;

    bool enabled;

    std::shared_ptr<glowl::GLSLProgram> fbo_shader;

    // FUNCTIONS --------------------------------------------------------------

    std::vector<Interaction> get_available_interactions(int obj_id) {
        std::vector<Interaction> retval;
        auto query = this->available_interactions.find(obj_id);
        if (query != this->available_interactions.end()) {
            retval = query->second;
        }
        return retval;
    }
};


class PickableTriangle {
public:
    PickableTriangle(void);
    ~PickableTriangle(void) = default;

    void Draw(unsigned int id, glm::vec2 pixel_dir, glm::vec2 vp_dim, ManipVector& pending_manipulations);

    InteractVector GetInteractions(unsigned int id) {
        InteractVector interactions;
        interactions.emplace_back(
            Interaction({InteractionType::MOVE_ALONG_AXIS_SCREEN, static_cast<int>(id), 1.0f, 0.0f, 0.0f, 0.0f, 0.0f}));
        interactions.emplace_back(
            Interaction({InteractionType::MOVE_ALONG_AXIS_SCREEN, static_cast<int>(id), 0.0f, 1.0f, 0.0f, 0.0f, 0.0f}));
        interactions.emplace_back(
            Interaction({InteractionType::SELECT, static_cast<int>(id), 0.0f, 0.0f, 0.0f, 0.0f, 0.0f}));
        interactions.emplace_back(
            Interaction({InteractionType::DESELECT, static_cast<int>(id), 0.0f, 0.0f, 0.0f, 0.0f, 0.0f}));
        interactions.emplace_back(
            Interaction({InteractionType::HIGHLIGHT, static_cast<int>(id), 0.0f, 0.0f, 0.0f, 0.0f, 0.0f}));
        return interactions;
    }

private:
    // VARIABLES --------------------------------------------------------------

    std::shared_ptr<glowl::GLSLProgram> shader;

    glm::vec2 pixel_direction;
    bool selected;
};


} // namespace gui
} // namespace megamol

#endif // MEGAMOL_GUI_WIDGETPICKING_GL_INCLUDED
