/**
 * Keyframe.cpp
 *
 * Copyright (C) 2017 by VISUS (Universitaet Stuttgart).
 * Alle Rechte vorbehalten.
 */

#include "cinematic/Keyframe.h"


using namespace megamol::cinematic;


Keyframe::Keyframe(void) : anim_time(0.0f), sim_time(0.0f), camera_state() {

    // Default intrinsics
    auto intrinsics = core::view::Camera::PerspectiveParameters();
    intrinsics.fovy = 0.5f;
    intrinsics.aspect = 16.0f / 9.0f;
    intrinsics.near_plane = 0.01f;
    intrinsics.far_plane = 100.0f;
    /// intrinsics.image_plane_tile = ;
    this->camera_state.setPerspectiveProjection(intrinsics);
}


Keyframe::Keyframe(float anim_time, float sim_time, core::view::Camera cam_state)
        : anim_time(anim_time)
        , sim_time(sim_time)
        , camera_state(cam_state) {}


Keyframe::~Keyframe(void) {}


bool Keyframe::Serialise(nlohmann::json& inout_json, size_t index) {

    // Append to given json
    inout_json["keyframes"][index]["animation_time"] = this->anim_time;
    inout_json["keyframes"][index]["simulation_time"] = this->sim_time;

    // Identify projection type and serialize accordingly
    auto cam_type = camera_state.get<core::view::Camera::ProjectionType>();

    inout_json["keyframes"][index]["camera_state"]["projection_type"] = cam_type;

    // If projection type is not unknown and camera is not serialized as matrix only do first block
    if (cam_type == core::view::Camera::PERSPECTIVE || cam_type == core::view::Camera::ORTHOGRAPHIC) {
        // Serialize pose
        auto cam_pose = camera_state.get<core::view::Camera::Pose>();
        inout_json["keyframes"][index]["camera_state"]["position"] =
            std::array<float, 3>{cam_pose.position.x, cam_pose.position.y, cam_pose.position.z};
        inout_json["keyframes"][index]["camera_state"]["direction"] =
            std::array<float, 3>{cam_pose.direction.x, cam_pose.direction.y, cam_pose.direction.z};
        inout_json["keyframes"][index]["camera_state"]["up"] =
            std::array<float, 3>{cam_pose.up.x, cam_pose.up.y, cam_pose.up.z};
        inout_json["keyframes"][index]["camera_state"]["right"] =
            std::array<float, 3>{cam_pose.right.x, cam_pose.right.y, cam_pose.right.z};

        // Serialize intrinsics
        if (cam_type == core::view::Camera::PERSPECTIVE) {
            auto cam_intrinsics = camera_state.get<core::view::Camera::PerspectiveParameters>();
            inout_json["keyframes"][index]["camera_state"]["fovy"] = cam_intrinsics.fovy.value();
            inout_json["keyframes"][index]["camera_state"]["aspect"] = cam_intrinsics.aspect.value();
            inout_json["keyframes"][index]["camera_state"]["near_plane"] = cam_intrinsics.near_plane.value();
            inout_json["keyframes"][index]["camera_state"]["far_plane"] = cam_intrinsics.far_plane.value();
            inout_json["keyframes"][index]["camera_state"]["image_plane_tile_start"] = std::array<float, 2>{
                cam_intrinsics.image_plane_tile.tile_start.x, cam_intrinsics.image_plane_tile.tile_start.y};
            inout_json["keyframes"][index]["camera_state"]["image_plane_tile_end"] = std::array<float, 2>{
                cam_intrinsics.image_plane_tile.tile_end.x, cam_intrinsics.image_plane_tile.tile_end.y};

        } else if (cam_type == core::view::Camera::ORTHOGRAPHIC) {
            auto cam_intrinsics = camera_state.get<core::view::Camera::OrthographicParameters>();

            inout_json["keyframes"][index]["camera_state"]["frustrum_height"] = cam_intrinsics.frustrum_height.value();
            inout_json["keyframes"][index]["camera_state"]["aspect"] = cam_intrinsics.aspect.value();
            inout_json["keyframes"][index]["camera_state"]["near_plane"] = cam_intrinsics.near_plane.value();
            inout_json["keyframes"][index]["camera_state"]["far_plane"] = cam_intrinsics.far_plane.value();
            inout_json["keyframes"][index]["camera_state"]["image_plane_tile_start"] = std::array<float, 2>{
                cam_intrinsics.image_plane_tile.tile_start.x, cam_intrinsics.image_plane_tile.tile_start.y};
            inout_json["keyframes"][index]["camera_state"]["image_plane_tile_end"] = std::array<float, 2>{
                cam_intrinsics.image_plane_tile.tile_end.x, cam_intrinsics.image_plane_tile.tile_end.y};
        }

    } else { // Camera::UNKNOWN
        auto view_mx = camera_state.getViewMatrix();
        auto proj_mx = camera_state.getProjectionMatrix();
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "[Camera] Found no valid projection. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
    }

    return true;
}


bool Keyframe::Deserialise(const nlohmann::json& in_json) {

    bool valid = true;
    valid &= megamol::core::utility::get_json_value<float>(in_json, {"animation_time"}, &this->anim_time);
    valid &= megamol::core::utility::get_json_value<float>(in_json, {"simulation_time"}, &this->sim_time);

    int cam_type_tmp = 0;
    valid &= megamol::core::utility::get_json_value<int>(in_json, {"camera_state", "projection_type"}, &cam_type_tmp);
    core::view::Camera::ProjectionType cam_type = static_cast<core::view::Camera::ProjectionType>(cam_type_tmp);

    if (cam_type == core::view::Camera::PERSPECTIVE || cam_type == core::view::Camera::ORTHOGRAPHIC) {

        // Get camera pose
        core::view::Camera::Pose cam_pose;
        std::array<float, 3> position;
        std::array<float, 3> direction;
        std::array<float, 3> up;
        std::array<float, 3> right;

        valid &= megamol::core::utility::get_json_value<float>(
            in_json, {"camera_state", "position"}, position.data(), position.size());
        valid &= megamol::core::utility::get_json_value<float>(
            in_json, {"camera_state", "direction"}, direction.data(), direction.size());
        valid &= megamol::core::utility::get_json_value<float>(in_json, {"camera_state", "up"}, up.data(), up.size());
        valid &= megamol::core::utility::get_json_value<float>(
            in_json, {"camera_state", "right"}, right.data(), right.size());

        cam_pose.position = glm::vec3(std::get<0>(position), std::get<1>(position), std::get<2>(position));
        cam_pose.direction = glm::vec3(std::get<0>(direction), std::get<1>(direction), std::get<2>(direction));
        cam_pose.up = glm::vec3(std::get<0>(up), std::get<1>(up), std::get<2>(up));
        cam_pose.right = glm::vec3(std::get<0>(right), std::get<1>(right), std::get<2>(right));

        // get camera intrinsics (starting with common intrinsics)
        float aspect;
        float near_plane;
        float far_plane;
        std::array<float, 2> image_plane_tile_start;
        std::array<float, 2> image_plane_tile_end;

        valid &= megamol::core::utility::get_json_value<float>(in_json, {"camera_state", "aspect"}, &aspect);
        valid &= megamol::core::utility::get_json_value<float>(in_json, {"camera_state", "near_plane"}, &near_plane);
        valid &= megamol::core::utility::get_json_value<float>(in_json, {"camera_state", "far_plane"}, &far_plane);

        valid &= megamol::core::utility::get_json_value<float>(in_json, {"camera_state", "image_plane_tile_start"},
            image_plane_tile_start.data(), image_plane_tile_start.size());
        valid &= megamol::core::utility::get_json_value<float>(in_json, {"camera_state", "image_plane_tile_end"},
            image_plane_tile_end.data(), image_plane_tile_end.size());

        if (cam_type == core::view::Camera::PERSPECTIVE) {
            core::view::Camera::PerspectiveParameters cam_intrinsics;

            float fovy;
            valid &= megamol::core::utility::get_json_value<float>(in_json, {"camera_state", "fovy"}, &fovy);

            cam_intrinsics.fovy = fovy;
            cam_intrinsics.aspect = aspect;
            cam_intrinsics.near_plane = near_plane;
            cam_intrinsics.far_plane = far_plane;
            cam_intrinsics.image_plane_tile.tile_start =
                glm::vec2(std::get<0>(image_plane_tile_start), std::get<1>(image_plane_tile_start));
            cam_intrinsics.image_plane_tile.tile_end =
                glm::vec2(std::get<0>(image_plane_tile_end), std::get<1>(image_plane_tile_end));

            camera_state = core::view::Camera(cam_pose, cam_intrinsics);

        } else if (cam_type == core::view::Camera::ORTHOGRAPHIC) {
            core::view::Camera::OrthographicParameters cam_intrinsics;

            float frustrum_height;
            valid &= megamol::core::utility::get_json_value<float>(
                in_json, {"camera_state", "frustrum_height"}, &frustrum_height);

            cam_intrinsics.frustrum_height = frustrum_height;
            cam_intrinsics.aspect = aspect;
            cam_intrinsics.near_plane = near_plane;
            cam_intrinsics.far_plane = far_plane;
            cam_intrinsics.image_plane_tile.tile_start =
                glm::vec2(std::get<0>(image_plane_tile_start), std::get<1>(image_plane_tile_start));
            cam_intrinsics.image_plane_tile.tile_end =
                glm::vec2(std::get<0>(image_plane_tile_end), std::get<1>(image_plane_tile_end));

            camera_state = core::view::Camera(cam_pose, cam_intrinsics);
        }

    } else {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "[Camera] Found no valid projection. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
    }

    return valid;
}
