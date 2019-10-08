/**
* KeyframeManipulators.cpp
*
* Copyright (C) 2017 by VISUS (Universitaet Stuttgart).
* Alle Rechte vorbehalten.
*/

#include "stdafx.h"
#include "KeyframeManipulators.h"


using namespace megamol;
using namespace megamol::core;
using namespace megamol::cinematic;


#define MANIPULATOR_RADIUS (20.0f)


KeyframeManipulators::KeyframeManipulators(void)
    : toggleVisibleGroupParam("manipulators::toggleVisibleGroup", "Toggle visibility of different manipulator groups.")
    , visibleGroupParam("manipulators::visibleGroup", "Select visible manipulator group.")
    , toggleOusideBboxParam("manipulators::showOutsideBBox", "Show manipulators always outside of model bounding box.")
    , paramSlots()
    , visibleGroup(VisibleGroup::SELECTED_KEYFRAME_AND_CTRLPOINT_POSITION)
    , toggleOusideBbox(false)
    , manipulators()
    , selectors()
    , state() {

    this->paramSlots.clear();
    this->selectors.clear();
    this->manipulators.clear();

    this->toggleOusideBboxParam.SetParameter(new param::ButtonParam(core::view::Key::KEY_W, core::view::Modifier::CTRL));
    this->paramSlots.emplace_back(&this->toggleOusideBboxParam);

    this->toggleVisibleGroupParam.SetParameter(new param::ButtonParam(core::view::Key::KEY_Q, core::view::Modifier::CTRL));
    this->paramSlots.emplace_back(&this->toggleVisibleGroupParam);

    param::EnumParam* vmg = new param::EnumParam(this->visibleGroup);
    vmg->SetTypePair(VisibleGroup::SELECTED_KEYFRAME_AND_CTRLPOINT_POSITION, "Keyframe and Ctrl-Point Positions");
    vmg->SetTypePair(VisibleGroup::SELECTED_KEYFRAME_LOOKAT_AND_UP_VECTOR, "LookAt Vector and Up Vector");
    this->visibleGroupParam << vmg;
    this->paramSlots.emplace_back(&this->visibleGroupParam);

    Manipulator m;
    m.show = false;
    m.position = glm::vec3(0.0f, 0.0f, 0.0f);
    m.rotation_axis = glm::vec3(0.0f, 0.0f, 0.0f);

    m.variety = Manipulator::Variety::MANIPULATOR_FIRST_CTRLPOINT_POSITION;
    m.rigging = Manipulator::Rigging::X_DIRECTION;
    this->manipulators.push_back(m);
    m.rigging = Manipulator::Rigging::Y_DIRECTION;
    this->manipulators.push_back(m);
    m.rigging = Manipulator::Rigging::Z_DIRECTION;
    this->manipulators.push_back(m);

    m.variety = Manipulator::Variety::MANIPULATOR_LAST_CTRLPOINT_POSITION;
    m.rigging = Manipulator::Rigging::X_DIRECTION;
    this->manipulators.push_back(m);
    m.rigging = Manipulator::Rigging::Y_DIRECTION;
    this->manipulators.push_back(m);
    m.rigging = Manipulator::Rigging::Z_DIRECTION;
    this->manipulators.push_back(m);

    m.variety = Manipulator::Variety::MANIPULATOR_SELECTED_KEYFRAME_POSITION;
    m.rigging = Manipulator::Rigging::X_DIRECTION;
    this->manipulators.push_back(m);
    m.rigging = Manipulator::Rigging::Y_DIRECTION;
    this->manipulators.push_back(m);
    m.rigging = Manipulator::Rigging::Z_DIRECTION;
    this->manipulators.push_back(m);

    m.variety = Manipulator::Variety::MANIPULATOR_SELECTED_KEYFRAME_POSITION_USING_LOOKAT;
    m.rigging = Manipulator::Rigging::VECTOR_DIRECTION;
    this->manipulators.push_back(m);

    m.variety = Manipulator::Variety::MANIPULATOR_SELECTED_KEYFRAME_LOOKAT_VECTOR;
    m.rigging = Manipulator::Rigging::X_DIRECTION;
    this->manipulators.push_back(m);
    m.rigging = Manipulator::Rigging::Y_DIRECTION;
    this->manipulators.push_back(m);
    m.rigging = Manipulator::Rigging::Z_DIRECTION;
    this->manipulators.push_back(m);

    m.variety = Manipulator::Variety::MANIPULATOR_SELECTED_KEYFRAME_UP_VECTOR;
    m.rigging = Manipulator::Rigging::ROTATION;
    this->manipulators.push_back(m);
}


KeyframeManipulators::~KeyframeManipulators(void) {

}


void KeyframeManipulators::UpdateExtents(vislib::math::Cuboid<float>& inout_bbox) {

    // Store copy of unmodified model bounding box
    this->state.bbox = inout_bbox;
    // Grow bounding box of model to manipulator positions
    for (unsigned int i = 0; i < this->manipulators.size(); i++) {
        inout_bbox.GrowToPoint(G2P(this->manipulators[i].position));
    }
}


bool KeyframeManipulators::UpdateRendering(const std::shared_ptr<std::vector<Keyframe>> keyframes, Keyframe selected_keyframe, glm::vec3 first_ctrl_pos, glm::vec3 last_ctrl_pos,
    const camera_state_type& snapshot, glm::vec2 viewport_dim, glm::mat4 mvp) {

    // Update parameters
    if (this->visibleGroupParam.IsDirty()) {
        this->visibleGroup = static_cast<VisibleGroup>(this->visibleGroupParam.Param<core::param::EnumParam>()->Value());
        this->visibleGroupParam.ResetDirty();
    }

    if (this->toggleVisibleGroupParam.IsDirty()) {
        this->visibleGroup = static_cast<VisibleGroup>((this->visibleGroup + 1) % VisibleGroup::VISIBLEGROUP_COUNT);
        this->toggleVisibleGroupParam.ResetDirty();
    }

    if (this->toggleOusideBboxParam.IsDirty()) {
        this->toggleOusideBbox = !this->toggleOusideBbox;
        this->toggleOusideBboxParam.ResetDirty();
    }

    // Update current state
    this->state.viewport = viewport_dim;
    this->state.mvp = mvp;
    this->state.cam_snapshot = snapshot;
    this->state.selected_keyframe = selected_keyframe;
    this->state.first_ctrl_point = first_ctrl_pos;
    this->state.last_ctrl_point = last_ctrl_pos;
    this->state.selected_index = -1;

    // Update keyframe position selectors
    auto count = keyframes->size();
    this->selectors.resize(count);
    std::array<float, 3> pos;
    for (size_t i = 0; i < count; ++i) {
        this->selectors[i].show = true;
        this->selectors[i].variety = Manipulator::Variety::SELECTOR_KEYFRAME_POSITION;
        this->selectors[i].rigging = Manipulator::Rigging::NONE;
        this->selectors[i].rotation_axis = glm::vec3(0.0f, 0.0f, 0.0f);
        pos = keyframes->operator[](i).GetCameraState().position;
        this->selectors[i].position = glm::vec3(pos[0], pos[1], pos[2]);
        if (keyframes->operator[](i) == this->state.selected_keyframe) {
            this->state.selected_index = i;
        
        }
    }

    // Update selected keyframe manipulators
    count = this->manipulators.size();
    for (size_t i = 0; i < count; ++i) {






    }

    return true;
}


bool KeyframeManipulators::PushRendering(CinematicUtils &utils) {

    auto pos = this->state.cam_snapshot.position;
    glm::vec3 camera_position = glm::vec3(pos[0], pos[1], pos[2]);

    glm::vec4 color;

    // Push keyframe position selectors
    auto count = this->selectors.size();
    for (size_t i = 0; i < count; ++i) {
        color = utils.Color(CinematicUtils::Colors::KEYFRAME);
        if (i == this->state.selected_index) {
            color = utils.Color(CinematicUtils::Colors::KEYFRAME_SELECTED);
        }
        utils.PushPointPrimitive(this->selectors[i].position, 2.0f * MANIPULATOR_RADIUS, camera_position, color);
    }

    // Push intermediate keyframe psition
    if (this->state.selected_index < 0) {


    }

    // Push manipulators of selected keyframe
    count = this->manipulators.size();
    for (size_t i = 0; i < count; ++i) {





    }

    return true;
}



int KeyframeManipulators::GetSelectedKeyframePositionIndex(float mouse_x, float mouse_y) {

    int index = -1;
    auto count = this->selectors.size();
    for (size_t i = 0; i < count; ++i) {
        if (this->selectors[i].show) {
            glm::vec2 pos = this->world2ScreenSpace(this->selectors[i].position);
            glm::vec2 mouse = glm::vec2(mouse_x, mouse_y);
            if (glm::length(pos - mouse) <= MANIPULATOR_RADIUS) {
                return i;
            }
        }
    }
    return index;
}


bool KeyframeManipulators::CheckForHitManipulator(float mouse_x, float mouse_y) {

    this->state.hit = nullptr;
    for (auto &m : this->manipulators) {
        if (m.show) {
            glm::vec2 pos = this->world2ScreenSpace(m.position);
            glm::vec2 mouse = glm::vec2(mouse_x, mouse_y);
            if (glm::length(pos - mouse) <= MANIPULATOR_RADIUS) {
                this->state.mouse = mouse;
                this->state.hit = std::make_shared<Manipulator>(m); ///XXX Tut das?
                return true;
            }
        }
    }
    return false;
}


bool KeyframeManipulators::ProcessHitManipulator(float mouse_x, float mouse_y) {

    if (this->state.hit == nullptr) return false;






    return true;
}


glm::vec2 KeyframeManipulators::world2ScreenSpace(glm::vec3 vec) {

	glm::vec4 world = { vec.x, vec.y, vec.z, 1.0f };
    world = this->state.mvp * world;
    world = world / world.w;
    glm::vec2 screen;
    screen.x = (screen.x + 1.0f) / 2.0f * this->state.viewport.x;
    screen.y = glm::abs(screen.y - 1.0f) / 2.0f * this->state.viewport.y; // (flipped y-axis)
    return screen;
}
