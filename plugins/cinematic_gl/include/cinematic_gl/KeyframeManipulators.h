/*
 * KeyframeManipulators.h
 *
 * Copyright (C) 2018 by VISUS (Universitaet Stuttgart).
 * Alle Rechte vorbehalten.
 */

#pragma once


#include "CinematicUtils.h"
#include "cinematic/Keyframe.h"
#include "mmcore/param/ButtonParam.h"
#include "mmcore/param/EnumParam.h"
#include "mmcore/param/ParamSlot.h"
#include "mmcore/view/Camera.h"
#include "vislib/math/Cuboid.h"
#include <glm/glm.hpp>


namespace megamol::cinematic_gl {


/*
 * Keyframe Manipulators.
 */
class KeyframeManipulators {

public:
    /** CTOR */
    KeyframeManipulators();

    /** DTOR */
    ~KeyframeManipulators();

    void UpdateExtents(vislib::math::Cuboid<float>& inout_bbox);

    bool UpdateRendering(const std::shared_ptr<std::vector<cinematic::Keyframe>> keyframes,
        cinematic::Keyframe selected_keyframe, glm::vec3 first_ctrl_pos, glm::vec3 last_ctrl_pos,
        core::view::Camera const& camera, glm::vec2 viewport_dim, glm::mat4 mvp);

    bool PushRendering(CinematicUtils& utils);

    int GetSelectedKeyframePositionIndex(float mouse_x, float mouse_y);

    bool CheckForHitManipulator(float mouse_x, float mouse_y);

    bool ProcessHitManipulator(float mouse_x, float mouse_y);

    inline cinematic::Keyframe GetManipulatedSelectedKeyframe() const {
        return this->state.selected_keyframe;
    }

    inline glm::vec3 GetFirstControlPointPosition() const {
        return this->state.first_ctrl_point;
    }

    inline glm::vec3 GetLastControlPointPosition() const {
        return this->state.last_ctrl_point;
    }

    inline std::vector<megamol::core::param::ParamSlot*>& GetParams() {
        return this->paramSlots;
    }

    inline void ResetHitManipulator() {
        this->state.hit = nullptr;
    }

private:
    /**********************************************************************
     * types and classes
     **********************************************************************/

    enum VisibleGroup : int {
        SELECTED_KEYFRAME_AND_CTRLPOINT_POSITION = 0,
        SELECTED_KEYFRAME_LOOKAT_AND_UP_VECTOR = 1,
        VISIBLEGROUP_COUNT = 2
    };

    class Manipulator {
    public:
        enum Rigging { NONE, X_DIRECTION, Y_DIRECTION, Z_DIRECTION, VECTOR_DIRECTION, ROTATION };

        enum Variety {
            SELECTOR_KEYFRAME_POSITION,
            MANIPULATOR_FIRST_CTRLPOINT_POSITION,
            MANIPULATOR_LAST_CTRLPOINT_POSITION,
            MANIPULATOR_SELECTED_KEYFRAME_POSITION,
            MANIPULATOR_SELECTED_KEYFRAME_POSITION_USING_LOOKAT,
            MANIPULATOR_SELECTED_KEYFRAME_LOOKAT_VECTOR,
            MANIPULATOR_SELECTED_KEYFRAME_UP_VECTOR,
        };

        bool show;
        Variety variety;
        Rigging rigging;
        // Storing position for selectable keyframes and
        // direction for manipulators:
        glm::vec3 vector;
    };

    struct CurrentState {
        cinematic::Keyframe selected_keyframe;
        glm::vec2 viewport;
        glm::mat4 mvp;
        glm::vec3 first_ctrl_point;
        glm::vec3 last_ctrl_point;
        core::view::Camera cam;
        vislib::math::Cuboid<float> bbox;
        std::shared_ptr<Manipulator> hit;
        glm::vec2 last_mouse;
        int selected_index;
        float point_radius;
        float line_width;
        float line_length;
        float lookat_length;
    };

    /**********************************************************************
     * variables
     **********************************************************************/

    core::param::ParamSlot visibleGroupParam;
    core::param::ParamSlot toggleVisibleGroupParam;
    core::param::ParamSlot toggleOusideBboxParam;

    std::vector<megamol::core::param::ParamSlot*> paramSlots;
    VisibleGroup visibleGroup;
    bool toggleOusideBbox;

    std::vector<Manipulator> manipulators;
    std::vector<Manipulator> selectors;
    CurrentState state;

    /**********************************************************************
     * functions
     **********************************************************************/

    glm::vec3 getManipulatorOrigin(Manipulator& manipulator);

    glm::vec3 getActualManipulatorPosition(Manipulator& manipulator);

    glm::vec4 getManipulatorColor(Manipulator& manipulator, CinematicUtils& utils);

    bool checkMousePointIntersection(Manipulator& manipulator, glm::vec2 mouse, glm::vec3 cam_up);

    glm::vec2 world2ScreenSpace(glm::vec3 vec);
};

} // namespace megamol::cinematic_gl
