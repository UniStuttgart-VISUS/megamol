/**
 * Keyframe.cpp
*
* Copyright (C) 2017 by VISUS (Universitaet Stuttgart).
* Alle Rechte vorbehalten.
*/

#include "stdafx.h"
#include "Keyframe.h"


using namespace megamol::cinematic;


Keyframe::Keyframe(void)
    : anim_time(0.0f)
    , sim_time(0.0f)
    , camera_state() {

    camera_state.half_aperture_angle_radians = glm::radians(30.0f);
}


Keyframe::Keyframe(float anim_time, float sim_time, camera_state_type cam_state)
    : anim_time(anim_time)
    , sim_time(sim_time)
    , camera_state(cam_state) {

}


Keyframe::~Keyframe(void) {
}


bool Keyframe::Serialise(nlohmann::json& inout_json, size_t index) {

    // Append to given json
    inout_json["keyframes"][index]["animation_time"]                              = this->anim_time;
    inout_json["keyframes"][index]["simulation_time"]                             = this->sim_time;
    inout_json["keyframes"][index]["camera_state"]["centre_offset"]               = this->camera_state.centre_offset;
    inout_json["keyframes"][index]["camera_state"]["convergence_plane"]           = this->camera_state.convergence_plane;
    inout_json["keyframes"][index]["camera_state"]["eye"]                         = static_cast<int>(this->camera_state.eye);
    inout_json["keyframes"][index]["camera_state"]["far_clipping_plane"]          = this->camera_state.far_clipping_plane;
    inout_json["keyframes"][index]["camera_state"]["film_gate"]                   = this->camera_state.film_gate;
    inout_json["keyframes"][index]["camera_state"]["gate_scaling"]                = static_cast<int>(this->camera_state.gate_scaling);
    inout_json["keyframes"][index]["camera_state"]["half_aperture_angle_radians"] = this->camera_state.half_aperture_angle_radians;
    inout_json["keyframes"][index]["camera_state"]["half_disparity"]              = this->camera_state.half_disparity;
    inout_json["keyframes"][index]["camera_state"]["image_tile"]                  = this->camera_state.image_tile;
    inout_json["keyframes"][index]["camera_state"]["near_clipping_plane"]         = this->camera_state.near_clipping_plane;
    inout_json["keyframes"][index]["camera_state"]["orientation"]                 = this->camera_state.orientation;
    inout_json["keyframes"][index]["camera_state"]["position"]                    = this->camera_state.position;
    inout_json["keyframes"][index]["camera_state"]["projection_type"]             = static_cast<int>(this->camera_state.projection_type);
    inout_json["keyframes"][index]["camera_state"]["resolution_gate"]             = this->camera_state.resolution_gate;

    return true;
}


bool Keyframe::Deserialise(const nlohmann::json& in_json) {

    bool valid = true;
    valid &= megamol::core::utility::get_json_value<float>(in_json, { "animation_time" }, &this->anim_time);
    valid &= megamol::core::utility::get_json_value<float>(in_json, { "simulation_time" }, &this->sim_time);
    valid &= megamol::core::utility::get_json_value<float>(in_json, { "camera_state", "centre_offset" }, this->camera_state.centre_offset.data(), this->camera_state.centre_offset.size());
    valid &= megamol::core::utility::get_json_value<float>(in_json, { "camera_state", "convergence_plane" }, &this->camera_state.convergence_plane);
    int eye = 0;
    valid &= megamol::core::utility::get_json_value<int>(in_json, { "camera_state", "eye" }, &eye);
    this->camera_state.eye = static_cast<megamol::core::thecam::Eye>(eye);
    valid &= megamol::core::utility::get_json_value<float>(in_json, { "camera_state", "far_clipping_plane" }, &this->camera_state.far_clipping_plane);
    valid &= megamol::core::utility::get_json_value<float>(in_json, { "camera_state", "film_gate" }, this->camera_state.film_gate.data(), this->camera_state.film_gate.size());
    int gate_scaling = 0;
    valid &= megamol::core::utility::get_json_value<int>(in_json, { "camera_state", "gate_scaling" }, &gate_scaling);
    this->camera_state.gate_scaling = static_cast<megamol::core::thecam::Gate_scaling>(gate_scaling);
    valid &= megamol::core::utility::get_json_value<float>(in_json, { "camera_state", "half_aperture_angle_radians" }, &this->camera_state.half_aperture_angle_radians);
    valid &= megamol::core::utility::get_json_value<float>(in_json, { "camera_state", "half_disparity" }, &this->camera_state.half_disparity);
    valid &= megamol::core::utility::get_json_value<int>(in_json, { "camera_state", "image_tile" }, this->camera_state.image_tile.data(), this->camera_state.image_tile.size());
    valid &= megamol::core::utility::get_json_value<float>(in_json, { "camera_state", "near_clipping_plane" }, &this->camera_state.near_clipping_plane);
    valid &= megamol::core::utility::get_json_value<float>(in_json, { "camera_state", "orientation" }, this->camera_state.orientation.data(), this->camera_state.orientation.size());
    valid &= megamol::core::utility::get_json_value<float>(in_json, { "camera_state", "position" }, this->camera_state.position.data(), this->camera_state.position.size());
    int projection_type = 0;
    valid &= megamol::core::utility::get_json_value<int>(in_json, { "camera_state", "projection_type" }, &projection_type);
    this->camera_state.projection_type = static_cast<megamol::core::thecam::Projection_type>(projection_type);
    valid &= megamol::core::utility::get_json_value<int>(in_json, { "camera_state", "resolution_gate" }, this->camera_state.resolution_gate.data(), this->camera_state.resolution_gate.size());

    return valid;
}
