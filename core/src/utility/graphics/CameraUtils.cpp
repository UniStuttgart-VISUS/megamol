#include "mmcore/utility/graphics/CameraUtils.h"

#define _USE_MATH_DEFINES
#include <math.h>

#include <array>
#include <vector>

glm::vec4 megamol::core::utility::get_default_camera_position(BoundingBoxes_2 const& bboxs,
    view::Camera::PerspectiveParameters cam_intrinsics, glm::quat const& camera_orientation, DefaultView dv) {

    glm::vec4 default_position = glm::vec4(0.0f, 0.0f, 0.0f, 0.0f);

    // Calculate pseudo width and pseudo height by projecting all eight corners on plane orthogonal to
    // camera position delta and lying in the center.
    glm::vec4 view_vec = glm::normalize(camera_orientation * glm::vec4(0.0, 0.0f, -1.0f, 0.0f));
    glm::vec4 up_vec = glm::normalize(camera_orientation * glm::vec4(0.0, 1.0f, 0.0f, 0.0f));
    glm::vec4 right_vec = glm::normalize(camera_orientation * glm::vec4(1.0, 0.0f, 0.0f, 0.0f));
    std::vector<glm::vec4> corners;
    auto tmp_corner = glm::vec4(bboxs.BoundingBox().Width() / 2.0f, bboxs.BoundingBox().Height() / 2.0f,
        bboxs.BoundingBox().Depth() / 2.0f, 0.0f);
    corners.push_back(glm::vec4(tmp_corner.x, tmp_corner.y, tmp_corner.z, 0.0f));
    corners.push_back(glm::vec4(tmp_corner.x, -tmp_corner.y, tmp_corner.z, 0.0f));
    corners.push_back(glm::vec4(tmp_corner.x, -tmp_corner.y, -tmp_corner.z, 0.0f));
    corners.push_back(glm::vec4(tmp_corner.x, tmp_corner.y, -tmp_corner.z, 0.0f));
    corners.push_back(glm::vec4(-tmp_corner.x, tmp_corner.y, tmp_corner.z, 0.0f));
    corners.push_back(glm::vec4(-tmp_corner.x, -tmp_corner.y, tmp_corner.z, 0.0f));
    corners.push_back(glm::vec4(-tmp_corner.x, -tmp_corner.y, -tmp_corner.z, 0.0f));
    corners.push_back(glm::vec4(-tmp_corner.x, tmp_corner.y, -tmp_corner.z, 0.0f));
    float delta_x_min = FLT_MAX;
    float delta_x_max = -FLT_MAX;
    float delta_y_min = FLT_MAX;
    float delta_y_max = -FLT_MAX;
    float delta_z_min = FLT_MAX;
    float delta_z_max = -FLT_MAX;
    for (auto& corner : corners) {
        float delta_x = glm::dot(corner, right_vec);
        float delta_y = glm::dot(corner, up_vec);
        float delta_z = glm::dot(corner, view_vec);
        delta_x_min = std::min(delta_x_min, delta_x);
        delta_x_max = std::max(delta_x_max, delta_x);
        delta_y_min = std::min(delta_y_min, delta_y);
        delta_y_max = std::max(delta_y_max, delta_y);
        delta_z_min = std::min(delta_z_min, delta_z);
        delta_z_max = std::max(delta_z_max, delta_z);
    }
    auto pseudoWidth = static_cast<double>(delta_x_max - delta_x_min);
    auto pseudoHeight = static_cast<double>(delta_y_max - delta_y_min);
    auto pseudoDepth = static_cast<double>(delta_z_max - delta_z_min);

    // New camera Position
    auto bbc = bboxs.BoundingBox().CalcCenter();
    auto bbcglm = glm::vec4(bbc.GetX(), bbc.GetY(), bbc.GetZ(), 1.0f);
    double halfFovY = static_cast<double>(cam_intrinsics.fovy / 2.0f);
    double halfFovX = static_cast<double>(halfFovY * cam_intrinsics.aspect);
    double distY = (pseudoHeight / (2.0 * tan(halfFovY)));
    double distX = (pseudoWidth / (2.0 * tan(halfFovX)));
    auto face_dist = static_cast<float>((distX > distY) ? distX : distY);
    face_dist = face_dist + (pseudoDepth / 2.0f);
    float edge_dist = face_dist / sqrt(2.0f);
    float corner_dist = edge_dist / sqrt(2.0f);

    switch (dv) {
    // FACES ----------------------------------------------------------------------------------
    case DEFAULTVIEW_FACE_FRONT:
        default_position = bbcglm + glm::vec4(0.0f, 0.0f, face_dist, 0.0f);
        break;
    case DEFAULTVIEW_FACE_BACK:
        default_position = bbcglm + glm::vec4(0.0f, 0.0f, -face_dist, 0.0f);
        break;
    case DEFAULTVIEW_FACE_RIGHT:
        default_position = bbcglm + glm::vec4(face_dist, 0.0f, 0.0f, 0.0f);
        break;
    case DEFAULTVIEW_FACE_LEFT:
        default_position = bbcglm + glm::vec4(-face_dist, 0.0f, 0.0f, 0.0f);
        break;
    case DEFAULTVIEW_FACE_TOP:
        default_position = bbcglm + glm::vec4(0.0f, face_dist, 0.0f, 0.0f);
        break;
    case DEFAULTVIEW_FACE_BOTTOM:
        default_position = bbcglm + glm::vec4(0.0f, -face_dist, 0.0f, 0.0f);
        break;
        // CORNERS --------------------------------------------------------------------------------
    case DEFAULTVIEW_CORNER_TOP_LEFT_FRONT:
        default_position = bbcglm + glm::vec4(-corner_dist, edge_dist, corner_dist, 0.0f);
        break;
    case DEFAULTVIEW_CORNER_TOP_RIGHT_FRONT:
        default_position = bbcglm + glm::vec4(corner_dist, edge_dist, corner_dist, 0.0f);
        break;
    case DEFAULTVIEW_CORNER_TOP_LEFT_BACK:
        default_position = bbcglm + glm::vec4(-corner_dist, edge_dist, -corner_dist, 0.0f);
        break;
    case DEFAULTVIEW_CORNER_TOP_RIGHT_BACK:
        default_position = bbcglm + glm::vec4(corner_dist, edge_dist, -corner_dist, 0.0f);
        break;
    case DEFAULTVIEW_CORNER_BOTTOM_LEFT_FRONT:
        default_position = bbcglm + glm::vec4(-corner_dist, -edge_dist, corner_dist, 0.0f);
        break;
    case DEFAULTVIEW_CORNER_BOTTOM_RIGHT_FRONT:
        default_position = bbcglm + glm::vec4(corner_dist, -edge_dist, corner_dist, 0.0f);
        break;
    case DEFAULTVIEW_CORNER_BOTTOM_LEFT_BACK:
        default_position = bbcglm + glm::vec4(-corner_dist, -edge_dist, -corner_dist, 0.0f);
        break;
    case DEFAULTVIEW_CORNER_BOTTOM_RIGHT_BACK:
        default_position = bbcglm + glm::vec4(corner_dist, -edge_dist, -corner_dist, 0.0f);
        break;
        // EDGES ----------------------------------------------------------------------------------
    case DEFAULTVIEW_EDGE_TOP_FRONT:
        default_position = bbcglm + glm::vec4(0.0f, edge_dist, edge_dist, 0.0f);
        break;
    case DEFAULTVIEW_EDGE_TOP_LEFT:
        default_position = bbcglm + glm::vec4(-edge_dist, edge_dist, 0.0f, 0.0f);
        break;
    case DEFAULTVIEW_EDGE_TOP_RIGHT:
        default_position = bbcglm + glm::vec4(edge_dist, edge_dist, 0.0f, 0.0f);
        break;
    case DEFAULTVIEW_EDGE_TOP_BACK:
        default_position = bbcglm + glm::vec4(0.0f, edge_dist, -edge_dist, 0.0f);
        break;
    case DEFAULTVIEW_EDGE_BOTTOM_FRONT:
        default_position = bbcglm + glm::vec4(0.0f, -edge_dist, edge_dist, 0.0f);
        break;
    case DEFAULTVIEW_EDGE_BOTTOM_LEFT:
        default_position = bbcglm + glm::vec4(-edge_dist, -edge_dist, 0.0f, 0.0f);
        break;
    case DEFAULTVIEW_EDGE_BOTTOM_RIGHT:
        default_position = bbcglm + glm::vec4(edge_dist, -edge_dist, 0.0f, 0.0f);
        break;
    case DEFAULTVIEW_EDGE_BOTTOM_BACK:
        default_position = bbcglm + glm::vec4(0.0f, -edge_dist, -edge_dist, 0.0f);
        break;
    case DEFAULTVIEW_EDGE_FRONT_LEFT:
        default_position = bbcglm + glm::vec4(-edge_dist, 0.0f, edge_dist, 0.0f);
        break;
    case DEFAULTVIEW_EDGE_FRONT_RIGHT:
        default_position = bbcglm + glm::vec4(edge_dist, 0.0f, edge_dist, 0.0f);
        break;
    case DEFAULTVIEW_EDGE_BACK_LEFT:
        default_position = bbcglm + glm::vec4(-edge_dist, 0.0f, -edge_dist, 0.0f);
        break;
    case DEFAULTVIEW_EDGE_BACK_RIGHT:
        default_position = bbcglm + glm::vec4(edge_dist, 0.0f, -edge_dist, 0.0f);
        break;
    default:
        break;
    }
    return default_position;
}

glm::quat megamol::core::utility::get_default_camera_orientation(DefaultView dv, DefaultOrientation dor) {

    glm::quat default_orientation = glm::identity<glm::quat>();

    // New camera orientation
    /// quat rot(theta) around axis(x,y,z) -> q = (sin(theta/2)*x, sin(theta/2)*y, sin(theta/2)*z,
    /// cos(theta/2))
    const float cos45 = sqrt(2.0f) / 2.0f;
    const float sin45 = cos45;
    const float cos22_5 = cos(M_PI / 8.0f);
    const float sin22_5 = sin(M_PI / 8.0f);

    auto qx_p45 = glm::quat(cos22_5, sin22_5, 0.0, 0.0);
    auto qx_n45 = glm::quat(cos22_5, -sin22_5, 0.0, 0.0);
    auto qy_p45 = glm::quat(cos22_5, 0.0, sin22_5, 0.0);
    auto qy_n45 = glm::quat(cos22_5, 0.0, -sin22_5, 0.0);
    auto qz_p45 = glm::quat(cos22_5, 0.0, 0.0, sin22_5);
    auto qz_n45 = glm::quat(cos22_5, 0.0, 0.0, -sin22_5);

    auto qx_p90 = glm::quat(cos45, sin45, 0.0, 0.0);
    auto qx_n90 = glm::quat(cos45, -sin45, 0.0, 0.0);
    auto qy_p90 = glm::quat(cos45, 0.0, sin45, 0.0);
    auto qy_n90 = glm::quat(cos45, 0.0, -sin45, 0.0);
    auto qz_p90 = glm::quat(cos45, 0.0, 0.0, sin45);
    auto qz_n90 = glm::quat(cos45, 0.0, 0.0, -sin45);

    auto qx_p180 = glm::quat(0.0, 1.0, 0.0, 0.0);
    auto qy_p180 = glm::quat(0.0, 0.0, 1.0, 0.0);
    auto qz_p180 = glm::quat(0.0, 0.0, 0.0, 1.0);

    switch (dv) {
    // FACES ----------------------------------------------------------------------------------
    case DEFAULTVIEW_FACE_FRONT:
        switch (dor) {
        case DEFAULTORIENTATION_TOP:
            default_orientation = glm::identity<glm::quat>();
            break;
        case DEFAULTORIENTATION_RIGHT:
            default_orientation = qz_n90;
            break;
        case DEFAULTORIENTATION_BOTTOM:
            default_orientation = qz_p180;
            break;
        case DEFAULTORIENTATION_LEFT:
            default_orientation = qz_p90;
            break;
        default:
            break;
        }
        break;
    case DEFAULTVIEW_FACE_BACK: // 180 deg around y axis
        switch (dor) {
        case DEFAULTORIENTATION_TOP:
            default_orientation = qy_p180;
            break;
        case DEFAULTORIENTATION_RIGHT:
            default_orientation = qy_p180 * qz_n90;
            break;
        case DEFAULTORIENTATION_BOTTOM:
            default_orientation = qy_p180 * qz_p180;
            break;
        case DEFAULTORIENTATION_LEFT:
            default_orientation = qy_p180 * qz_p90;
            break;
        default:
            break;
        }
        break;
    case DEFAULTVIEW_FACE_RIGHT: // 90 deg around y axis
        switch (dor) {
        case DEFAULTORIENTATION_TOP:
            default_orientation = qy_p90;
            break;
        case DEFAULTORIENTATION_RIGHT:
            default_orientation = qy_p90 * qz_n90;
            break;
        case DEFAULTORIENTATION_BOTTOM:
            default_orientation = qy_p90 * qz_p180;
            break;
        case DEFAULTORIENTATION_LEFT:
            default_orientation = qy_p90 * qz_p90;
            break;
        default:
            break;
        }
        break;
    case DEFAULTVIEW_FACE_LEFT: // 90 deg reverse around y axis
        switch (dor) {
        case DEFAULTORIENTATION_TOP:
            default_orientation = qy_n90;
            break;
        case DEFAULTORIENTATION_RIGHT:
            default_orientation = qy_n90 * qz_n90;
            break;
        case DEFAULTORIENTATION_BOTTOM:
            default_orientation = qy_n90 * qz_p180;
            break;
        case DEFAULTORIENTATION_LEFT:
            default_orientation = qy_n90 * qz_p90;
            break;
        default:
            break;
        }
        break;
    case DEFAULTVIEW_FACE_TOP: // 90 deg around x axis
        switch (dor) {
        case DEFAULTORIENTATION_TOP:
            default_orientation = qx_n90;
            break;
        case DEFAULTORIENTATION_RIGHT:
            default_orientation = qx_n90 * qz_n90;
            break;
        case DEFAULTORIENTATION_BOTTOM:
            default_orientation = qx_n90 * qz_p180;
            break;
        case DEFAULTORIENTATION_LEFT:
            default_orientation = qx_n90 * qz_p90;
            break;
        default:
            break;
        }
        break;
    case DEFAULTVIEW_FACE_BOTTOM: // 90 deg reverse around x axis
        switch (dor) {
        case DEFAULTORIENTATION_TOP:
            default_orientation = qx_p90;
            break;
        case DEFAULTORIENTATION_RIGHT:
            default_orientation = qx_p90 * qz_n90;
            break;
        case DEFAULTORIENTATION_BOTTOM:
            default_orientation = qx_p90 * qz_p180;
            break;
        case DEFAULTORIENTATION_LEFT:
            default_orientation = qx_p90 * qz_p90;
            break;
        default:
            break;
        }
        break;
    // CORNERS --------------------------------------------------------------------------------
    case DEFAULTVIEW_CORNER_TOP_LEFT_FRONT:
        switch (dor) {
        case DEFAULTORIENTATION_TOP:
            default_orientation = qy_n45 * qx_n45;
            break;
        case DEFAULTORIENTATION_RIGHT:
            default_orientation = qy_n45 * qx_n45 * qz_n90;
            break;
        case DEFAULTORIENTATION_BOTTOM:
            default_orientation = qy_n45 * qx_n45 * qz_p180;
            break;
        case DEFAULTORIENTATION_LEFT:
            default_orientation = qy_n45 * qx_n45 * qz_p90;
            break;
        default:
            break;
        }
        break;
    case DEFAULTVIEW_CORNER_TOP_RIGHT_FRONT:
        switch (dor) {
        case DEFAULTORIENTATION_TOP:
            default_orientation = qy_p45 * qx_n45;
            break;
        case DEFAULTORIENTATION_RIGHT:
            default_orientation = qy_p45 * qx_n45 * qz_n90;
            break;
        case DEFAULTORIENTATION_BOTTOM:
            default_orientation = qy_p45 * qx_n45 * qz_p180;
            break;
        case DEFAULTORIENTATION_LEFT:
            default_orientation = qy_p45 * qx_n45 * qz_p90;
            break;
        default:
            break;
        }
        break;
    case DEFAULTVIEW_CORNER_TOP_LEFT_BACK:
        switch (dor) {
        case DEFAULTORIENTATION_TOP:
            default_orientation = qy_p180 * qy_p45 * qx_n45;
            break;
        case DEFAULTORIENTATION_RIGHT:
            default_orientation = qy_p180 * qy_p45 * qx_n45 * qz_n90;
            break;
        case DEFAULTORIENTATION_BOTTOM:
            default_orientation = qy_p180 * qy_p45 * qx_n45 * qz_p180;
            break;
        case DEFAULTORIENTATION_LEFT:
            default_orientation = qy_p180 * qy_p45 * qx_n45 * qz_p90;
            break;
        default:
            break;
        }
        break;
    case DEFAULTVIEW_CORNER_TOP_RIGHT_BACK:
        switch (dor) {
        case DEFAULTORIENTATION_TOP:
            default_orientation = qy_p180 * qy_n45 * qx_n45;
            break;
        case DEFAULTORIENTATION_RIGHT:
            default_orientation = qy_p180 * qy_n45 * qx_n45 * qz_n90;
            break;
        case DEFAULTORIENTATION_BOTTOM:
            default_orientation = qy_p180 * qy_n45 * qx_n45 * qz_p180;
            break;
        case DEFAULTORIENTATION_LEFT:
            default_orientation = qy_p180 * qy_n45 * qx_n45 * qz_p90;
            break;
        default:
            break;
        }
        break;
    case DEFAULTVIEW_CORNER_BOTTOM_LEFT_FRONT:
        switch (dor) {
        case DEFAULTORIENTATION_TOP:
            default_orientation = qy_n45 * qx_p45;
            break;
        case DEFAULTORIENTATION_RIGHT:
            default_orientation = qy_n45 * qx_p45 * qz_n90;
            break;
        case DEFAULTORIENTATION_BOTTOM:
            default_orientation = qy_n45 * qx_p45 * qz_p180;
            break;
        case DEFAULTORIENTATION_LEFT:
            default_orientation = qy_n45 * qx_p45 * qz_p90;
            break;
        default:
            break;
        }
        break;
    case DEFAULTVIEW_CORNER_BOTTOM_RIGHT_FRONT:
        switch (dor) {
        case DEFAULTORIENTATION_TOP:
            default_orientation = qy_p45 * qx_p45;
            break;
        case DEFAULTORIENTATION_RIGHT:
            default_orientation = qy_p45 * qx_p45 * qz_n90;
            break;
        case DEFAULTORIENTATION_BOTTOM:
            default_orientation = qy_p45 * qx_p45 * qz_p180;
            break;
        case DEFAULTORIENTATION_LEFT:
            default_orientation = qy_p45 * qx_p45 * qz_p90;
            break;
        default:
            break;
        }
        break;
    case DEFAULTVIEW_CORNER_BOTTOM_LEFT_BACK:
        switch (dor) {
        case DEFAULTORIENTATION_TOP:
            default_orientation = qy_p180 * qy_p45 * qx_p45;
            break;
        case DEFAULTORIENTATION_RIGHT:
            default_orientation = qy_p180 * qy_p45 * qx_p45 * qz_n90;
            break;
        case DEFAULTORIENTATION_BOTTOM:
            default_orientation = qy_p180 * qy_p45 * qx_p45 * qz_p180;
            break;
        case DEFAULTORIENTATION_LEFT:
            default_orientation = qy_p180 * qy_p45 * qx_p45 * qz_p90;
            break;
        default:
            break;
        }
        break;
    case DEFAULTVIEW_CORNER_BOTTOM_RIGHT_BACK:
        switch (dor) {
        case DEFAULTORIENTATION_TOP:
            default_orientation = qy_p180 * qy_n45 * qx_p45;
            break;
        case DEFAULTORIENTATION_RIGHT:
            default_orientation = qy_p180 * qy_n45 * qx_p45 * qz_n90;
            break;
        case DEFAULTORIENTATION_BOTTOM:
            default_orientation = qy_p180 * qy_n45 * qx_p45 * qz_p180;
            break;
        case DEFAULTORIENTATION_LEFT:
            default_orientation = qy_p180 * qy_n45 * qx_p45 * qz_p90;
            break;
        default:
            break;
        }
        break;
    // EDGES ----------------------------------------------------------------------------------
    case DEFAULTVIEW_EDGE_TOP_FRONT:
        switch (dor) {
        case DEFAULTORIENTATION_TOP:
            default_orientation = qx_n45 * qz_p90;
            break;
        case DEFAULTORIENTATION_RIGHT:
            default_orientation = qx_n45;
            break;
        case DEFAULTORIENTATION_BOTTOM:
            default_orientation = qx_n45 * qz_n90;
            break;
        case DEFAULTORIENTATION_LEFT:
            default_orientation = qz_p180 * qx_p45;
            break;
        default:
            break;
        }
        break;
    case DEFAULTVIEW_EDGE_TOP_LEFT: // TODO
        switch (dor) {
        case DEFAULTORIENTATION_TOP:
            default_orientation = qx_n90 * qy_n45;
            break;
        case DEFAULTORIENTATION_RIGHT:
            default_orientation = qy_p180 * qy_p90 * qx_n45;
            break;
        case DEFAULTORIENTATION_BOTTOM:
            default_orientation = qy_p180 * qx_n90 * qy_p45;
            break;
        case DEFAULTORIENTATION_LEFT:
            default_orientation = qx_p180 * qy_n90 * qx_p45;
            break;
        default:
            break;
        }
        break;
    case DEFAULTVIEW_EDGE_TOP_RIGHT: // TODO
        switch (dor) {
        case DEFAULTORIENTATION_TOP:
            default_orientation = qy_p180 * qx_n90 * qy_n45;
            break;
        case DEFAULTORIENTATION_RIGHT:
            default_orientation = qy_p180 * qy_n90 * qx_n45;
            break;
        case DEFAULTORIENTATION_BOTTOM:
            default_orientation = qx_n90 * qy_p45;
            break;
        case DEFAULTORIENTATION_LEFT:
            default_orientation = qx_p180 * qy_p90 * qx_p45;
            break;
        default:
            break;
        }
        break;
    case DEFAULTVIEW_EDGE_TOP_BACK:
        switch (dor) {
        case DEFAULTORIENTATION_TOP:
            default_orientation = qy_p180 * qx_n45 * qz_n90;
            break;
        case DEFAULTORIENTATION_RIGHT:
            default_orientation = qy_p180 * qz_p180 * qx_p45;
            break;
        case DEFAULTORIENTATION_BOTTOM:
            default_orientation = qy_p180 * qx_n45 * qz_p90;
            break;
        case DEFAULTORIENTATION_LEFT:
            default_orientation = qy_p180 * qx_n45;
            break;
        default:
            break;
        }
        break;
    case DEFAULTVIEW_EDGE_BOTTOM_FRONT:
        switch (dor) {
        case DEFAULTORIENTATION_TOP:
            default_orientation = qx_p45 * qz_p90;
            break;
        case DEFAULTORIENTATION_RIGHT:
            default_orientation = qx_p45;
            break;
        case DEFAULTORIENTATION_BOTTOM:
            default_orientation = qx_p45 * qz_n90;
            break;
        case DEFAULTORIENTATION_LEFT:
            default_orientation = qz_p180 * qx_n45;
            break;
        default:
            break;
        }
        break;
    case DEFAULTVIEW_EDGE_BOTTOM_LEFT: // TODO
        switch (dor) {
        case DEFAULTORIENTATION_TOP:
            default_orientation = qy_p180 * qx_p90 * qy_p45;
            break;
        case DEFAULTORIENTATION_RIGHT:
            default_orientation = qy_n90 * qx_p45;
            break;
        case DEFAULTORIENTATION_BOTTOM:
            default_orientation = qx_p90 * qy_n45;
            break;
        case DEFAULTORIENTATION_LEFT:
            default_orientation = qz_p180 * qy_p90 * qx_n45;
            break;
        default:
            break;
        }
        break;
    case DEFAULTVIEW_EDGE_BOTTOM_RIGHT: // TODO
        switch (dor) {
        case DEFAULTORIENTATION_TOP:
            default_orientation = qx_p90 * qy_p45;
            break;
        case DEFAULTORIENTATION_RIGHT:
            default_orientation = qy_p90 * qx_p45;
            break;
        case DEFAULTORIENTATION_BOTTOM:
            default_orientation = qy_p180 * qx_p90 * qy_n45;
            break;
        case DEFAULTORIENTATION_LEFT:
            default_orientation = qz_p180 * qy_n90 * qx_n45;
            break;
        default:
            break;
        }
        break;
    case DEFAULTVIEW_EDGE_BOTTOM_BACK:
        switch (dor) {
        case DEFAULTORIENTATION_TOP:
            default_orientation = qy_p180 * qx_p45 * qz_p90;
            break;
        case DEFAULTORIENTATION_RIGHT:
            default_orientation = qy_p180 * qx_p45;
            break;
        case DEFAULTORIENTATION_BOTTOM:
            default_orientation = qy_p180 * qx_p45 * qz_n90;
            break;
        case DEFAULTORIENTATION_LEFT:
            default_orientation = qy_p180 * qz_p180 * qx_n45;
            break;
        default:
            break;
        }
        break;
    case DEFAULTVIEW_EDGE_FRONT_LEFT:
        switch (dor) {
        case DEFAULTORIENTATION_TOP:
            default_orientation = qy_n45;
            break;
        case DEFAULTORIENTATION_RIGHT:
            default_orientation = qz_n90 * qx_p45;
            break;
        case DEFAULTORIENTATION_BOTTOM:
            default_orientation = qz_p180 * qy_p45;
            break;
        case DEFAULTORIENTATION_LEFT:
            default_orientation = qz_p90 * qx_n45;
            break;
        default:
            break;
        }
        break;
    case DEFAULTVIEW_EDGE_FRONT_RIGHT:
        switch (dor) {
        case DEFAULTORIENTATION_TOP:
            default_orientation = qy_p45;
            break;
        case DEFAULTORIENTATION_RIGHT:
            default_orientation = qz_n90 * qx_n45;
            break;
        case DEFAULTORIENTATION_BOTTOM:
            default_orientation = qz_p180 * qy_n45;
            break;
        case DEFAULTORIENTATION_LEFT:
            default_orientation = qz_p90 * qx_p45;
            break;
        default:
            break;
        }
        break;
    case DEFAULTVIEW_EDGE_BACK_LEFT:
        switch (dor) {
        case DEFAULTORIENTATION_TOP:
            default_orientation = qy_p180 * qy_p45;
            break;
        case DEFAULTORIENTATION_RIGHT:
            default_orientation = qy_p180 * qz_n90 * qx_n45;
            break;
        case DEFAULTORIENTATION_BOTTOM:
            default_orientation = qx_p180 * qy_n45;
            break;
        case DEFAULTORIENTATION_LEFT:
            default_orientation = qy_p180 * qz_p90 * qx_p45;
            break;
        default:
            break;
        }
        break;
    case DEFAULTVIEW_EDGE_BACK_RIGHT:
        switch (dor) {
        case DEFAULTORIENTATION_TOP:
            default_orientation = qy_p180 * qy_n45;
            break;
        case DEFAULTORIENTATION_RIGHT:
            default_orientation = qy_p180 * qz_n90 * qx_p45;
            break;
        case DEFAULTORIENTATION_BOTTOM:
            default_orientation = qx_p180 * qy_p45;
            break;
        case DEFAULTORIENTATION_LEFT:
            default_orientation = qy_p180 * qz_p90 * qx_n45;
            break;
        default:
            break;
        }
        break;
    default:
        break;
    }
    return default_orientation;
}

glm::vec2 megamol::core::utility::get_min_max_dist_to_bbox(
    BoundingBoxes_2 const& bboxs, const view::Camera::Pose& pose) {
    // compute min and max distance from camera to bounding box corners
    auto pointPlaneDist = [](std::array<float, 3> point, std::array<float, 3> point_on_plane,
                              std::array<float, 3> plane_normal) -> float {
        return (std::get<0>(point) - std::get<0>(point_on_plane)) * std::get<0>(plane_normal) +
               (std::get<1>(point) - std::get<1>(point_on_plane)) * std::get<1>(plane_normal) +
               (std::get<2>(point) - std::get<2>(point_on_plane)) * std::get<2>(plane_normal);
    };

    std::array<float, 3> point = {pose.position.x, pose.position.y, pose.position.z};
    std::array<float, 3> plane_normal = {-pose.direction.x, -pose.direction.y, -pose.direction.z};

    float left_bottom_back_dist = pointPlaneDist(point,
        {bboxs.ClipBox().GetLeftBottomBack().GetX(), bboxs.ClipBox().GetLeftBottomBack().GetY(),
            bboxs.ClipBox().GetLeftBottomBack().GetZ()},
        plane_normal);
    float left_bottom_front_dist = pointPlaneDist(point,
        {bboxs.ClipBox().GetLeftBottomFront().GetX(), bboxs.ClipBox().GetLeftBottomFront().GetY(),
            bboxs.ClipBox().GetLeftBottomFront().GetZ()},
        plane_normal);
    float left_top_back_dist = pointPlaneDist(point,
        {bboxs.ClipBox().GetLeftTopBack().GetX(), bboxs.ClipBox().GetLeftTopBack().GetY(),
            bboxs.ClipBox().GetLeftTopBack().GetZ()},
        plane_normal);
    float left_top_front_dist = pointPlaneDist(point,
        {bboxs.ClipBox().GetLeftTopFront().GetX(), bboxs.ClipBox().GetLeftTopFront().GetY(),
            bboxs.ClipBox().GetLeftTopFront().GetZ()},
        plane_normal);

    float right_bottom_back_dist = pointPlaneDist(point,
        {bboxs.ClipBox().GetRightBottomBack().GetX(), bboxs.ClipBox().GetRightBottomBack().GetY(),
            bboxs.ClipBox().GetRightBottomBack().GetZ()},
        plane_normal);
    float right_bottom_front_dist = pointPlaneDist(point,
        {bboxs.ClipBox().GetRightBottomFront().GetX(), bboxs.ClipBox().GetRightBottomFront().GetY(),
            bboxs.ClipBox().GetRightBottomFront().GetZ()},
        plane_normal);
    float right_top_back_dist = pointPlaneDist(point,
        {bboxs.ClipBox().GetRightTopBack().GetX(), bboxs.ClipBox().GetRightTopBack().GetY(),
            bboxs.ClipBox().GetRightTopBack().GetZ()},
        plane_normal);
    float right_top_front_dist = pointPlaneDist(point,
        {bboxs.ClipBox().GetRightTopFront().GetX(), bboxs.ClipBox().GetRightTopFront().GetY(),
            bboxs.ClipBox().GetRightTopFront().GetZ()},
        plane_normal);

    float min_dist = std::min(left_bottom_back_dist,
        std::min(left_bottom_front_dist,
            std::min(left_top_back_dist,
                std::min(left_top_front_dist,
                    std::min(right_bottom_back_dist,
                        std::min(right_bottom_front_dist, std::min(right_top_back_dist, right_top_front_dist)))))));

    float max_dist = std::max(left_bottom_back_dist,
        std::max(left_bottom_front_dist,
            std::max(left_top_back_dist,
                std::max(left_top_front_dist,
                    std::max(right_bottom_back_dist,
                        std::max(right_bottom_front_dist, std::max(right_top_back_dist, right_top_front_dist)))))));

    return glm::vec2(min_dist, max_dist);
}
