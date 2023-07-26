/**
 * MegaMol
 * Copyright (c) 2020, MegaMol Dev Team
 * All rights reserved.
 */
/// Implementation see mmcore/utility/Picking.cpp and mmcore_gl/utility/Picking_gl.cpp

#pragma once

#include <map>
#include <vector>

#include <glm/glm.hpp>

#include "GL_STUB.h"
#include "mmcore/view/InputCall.h"

// forward declaration
namespace glowl {
class FramebufferObject;
class GLSLProgram;
} // namespace glowl

#define PICKING_INTERACTION_TUPLE_INIT \
    { false, -1, FLT_MAX }


namespace megamol::core::utility {

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
    ~PickingBuffer() GL_STUB();

    // Call only once per frame
    bool EnableInteraction(glm::vec2 vp_dim) GL_STUB(true);

    // Call only once per frame
    bool DisableInteraction() GL_STUB(true);

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
    bool enabled;

#ifdef MEGAMOL_USE_OPENGL
    int prev_fbo = 0;
    std::shared_ptr<glowl::FramebufferObject> fbo = nullptr;
    std::shared_ptr<glowl::GLSLProgram> shader = nullptr;
#endif

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

} // namespace megamol::core::utility
