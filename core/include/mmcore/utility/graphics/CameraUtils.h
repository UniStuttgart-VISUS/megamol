/**
 * MegaMol
 * Copyright (c) 2021, MegaMol Dev Team
 * All rights reserved.
 */

#pragma once

#include "mmcore/BoundingBoxes_2.h"
#include "mmcore/view/Camera.h"


namespace megamol::core::utility {

/** Enum for default views from the respective direction */
enum DefaultView : int {
    DEFAULTVIEW_FACE_FRONT = 0,
    DEFAULTVIEW_FACE_BACK = 1,
    DEFAULTVIEW_FACE_RIGHT = 2,
    DEFAULTVIEW_FACE_LEFT = 3,
    DEFAULTVIEW_FACE_TOP = 4,
    DEFAULTVIEW_FACE_BOTTOM = 5,
    DEFAULTVIEW_CORNER_TOP_LEFT_FRONT = 6,
    DEFAULTVIEW_CORNER_TOP_RIGHT_FRONT = 7,
    DEFAULTVIEW_CORNER_TOP_LEFT_BACK = 8,
    DEFAULTVIEW_CORNER_TOP_RIGHT_BACK = 9,
    DEFAULTVIEW_CORNER_BOTTOM_LEFT_FRONT = 10,
    DEFAULTVIEW_CORNER_BOTTOM_RIGHT_FRONT = 11,
    DEFAULTVIEW_CORNER_BOTTOM_LEFT_BACK = 12,
    DEFAULTVIEW_CORNER_BOTTOM_RIGHT_BACK = 13,
    DEFAULTVIEW_EDGE_TOP_FRONT = 14,
    DEFAULTVIEW_EDGE_TOP_LEFT = 15,
    DEFAULTVIEW_EDGE_TOP_RIGHT = 16,
    DEFAULTVIEW_EDGE_TOP_BACK = 17,
    DEFAULTVIEW_EDGE_BOTTOM_FRONT = 18,
    DEFAULTVIEW_EDGE_BOTTOM_LEFT = 19,
    DEFAULTVIEW_EDGE_BOTTOM_RIGHT = 20,
    DEFAULTVIEW_EDGE_BOTTOM_BACK = 21,
    DEFAULTVIEW_EDGE_FRONT_LEFT = 22,
    DEFAULTVIEW_EDGE_FRONT_RIGHT = 23,
    DEFAULTVIEW_EDGE_BACK_LEFT = 24,
    DEFAULTVIEW_EDGE_BACK_RIGHT = 25
};

/** Enum for default orientations from the respective direction */
enum DefaultOrientation : int {
    DEFAULTORIENTATION_TOP = 0,
    DEFAULTORIENTATION_RIGHT = 1,
    DEFAULTORIENTATION_BOTTOM = 2,
    DEFAULTORIENTATION_LEFT = 3,
};

glm::vec4 get_default_camera_position(BoundingBoxes_2 const& bboxs, view::Camera::PerspectiveParameters cam_intrinsics,
    glm::quat const& camera_orientation, DefaultView dv);

glm::quat get_default_camera_orientation(DefaultView dv, DefaultOrientation dor);

glm::vec2 get_min_max_dist_to_bbox(BoundingBoxes_2 const& bboxs, const view::Camera::Pose& cam);


} // namespace megamol::core::utility
