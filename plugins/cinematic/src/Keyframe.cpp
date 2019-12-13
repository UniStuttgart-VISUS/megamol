/**
 * Keyframe.cpp
*
* Copyright (C) 2017 by VISUS (Universitaet Stuttgart).
* Alle Rechte vorbehalten.
*/

#include "stdafx.h"
#include "Keyframe.h"


using namespace megamol::cinematic;


Keyframe::Keyframe()
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


Keyframe::~Keyframe() {

}


bool Keyframe::Serialise(nlohmann::json& inout_json, size_t index) {

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

    if (in_json.at("animation_time").is_number()) {
        in_json.at("animation_time").get_to(this->anim_time);
    }
    else {
        vislib::sys::Log::DefaultLog.WriteError("JSON ERROR - Couldn't read 'animation_time': %s (%s:%d)", __FUNCTION__, __FILE__, __LINE__);
        return false;
    }

    if (in_json.at("simulation_time").is_number()) {
        in_json.at("simulation_time").get_to(this->sim_time);
    }
    else {
        vislib::sys::Log::DefaultLog.WriteError("JSON ERROR - Couldn't read 'simulation_time': %s (%s:%d)", __FUNCTION__, __FILE__, __LINE__);
        return false;
    }

    if (in_json.at("camera_state").at("centre_offset").is_array()) {
        if (in_json.at("camera_state").at("centre_offset").size() != 2) {
            vislib::sys::Log::DefaultLog.WriteError("JSON ERROR - Array of 'camera_state' - 'centre_offset' should have size 2: %s (%s:%d)", __FUNCTION__, __FILE__, __LINE__);
            return false;
        }
        for (size_t i = 0; i < in_json.at("camera_state").at("centre_offset").size(); i++) {
            if (!in_json.at("camera_state").at("centre_offset")[i].is_number()) {
                vislib::sys::Log::DefaultLog.WriteError("JSON ERROR - Elements of array 'camera_state' - 'centre_offset' should be numbers: %s (%s:%d)", __FUNCTION__, __FILE__, __LINE__);
                return false;
            }
        }
        in_json.at("camera_state").at("centre_offset").get_to(this->camera_state.centre_offset);
    }
    else {
        vislib::sys::Log::DefaultLog.WriteError("JSON ERROR - Couldn't read 'camera_state' - 'centre_offset': %s (%s:%d)", __FUNCTION__, __FILE__, __LINE__);
        return false;
    }

    if (in_json.at("camera_state").at("convergence_plane").is_number()) {
        in_json.at("camera_state").at("convergence_plane").get_to(this->camera_state.convergence_plane);
    }
    else {
        vislib::sys::Log::DefaultLog.WriteError("JSON ERROR - Couldn't read 'camera_state' - 'convergence_plane': %s (%s:%d)", __FUNCTION__, __FILE__, __LINE__);
        return false;
    }

    if (in_json.at("camera_state").at("eye").is_number()) {
        int eye;
        in_json.at("camera_state").at("eye").get_to(eye);
        this->camera_state.eye = static_cast<megamol::core::thecam::Eye>(eye);
    }
    else {
        vislib::sys::Log::DefaultLog.WriteError("JSON ERROR - Couldn't read 'camera_state' - 'eye': %s (%s:%d)", __FUNCTION__, __FILE__, __LINE__);
        return false;
    }

    if (in_json.at("camera_state").at("far_clipping_plane").is_number()) {
        in_json.at("camera_state").at("far_clipping_plane").get_to(this->camera_state.far_clipping_plane);
    }
    else {
        vislib::sys::Log::DefaultLog.WriteError("JSON ERROR - Couldn't read 'camera_state' - 'far_clipping_plane': %s (%s:%d)", __FUNCTION__, __FILE__, __LINE__);
        return false;
    }

    if (in_json.at("camera_state").at("film_gate").is_array()) {
        if (in_json.at("camera_state").at("film_gate").size() != 2) {
            vislib::sys::Log::DefaultLog.WriteError("JSON ERROR - Array of 'camera_state' - 'film_gate' should have size 2: %s (%s:%d)", __FUNCTION__, __FILE__, __LINE__);
            return false;
        }
        for (size_t i = 0; i < in_json.at("camera_state").at("film_gate").size(); i++) {
            if (!in_json.at("camera_state").at("film_gate")[i].is_number()) {
                vislib::sys::Log::DefaultLog.WriteError("JSON ERROR - Elements of array 'camera_state' - 'film_gate' should be numbers: %s (%s:%d)", __FUNCTION__, __FILE__, __LINE__);
                return false;
            }
        }
        in_json.at("camera_state").at("film_gate").get_to(this->camera_state.film_gate);
    }
    else {
        vislib::sys::Log::DefaultLog.WriteError("JSON ERROR - Couldn't read 'camera_state' - 'film_gate': %s (%s:%d)", __FUNCTION__, __FILE__, __LINE__);
        return false;
    }

    if (in_json.at("camera_state").at("gate_scaling").is_number()) {
        int gate_scaling;
        in_json.at("camera_state").at("gate_scaling").get_to(gate_scaling);
        this->camera_state.gate_scaling = static_cast<megamol::core::thecam::Gate_scaling>(gate_scaling);
    }
    else {
        vislib::sys::Log::DefaultLog.WriteError("JSON ERROR - Couldn't read 'camera_state' - 'gate_scaling': %s (%s:%d)", __FUNCTION__, __FILE__, __LINE__);
        return false;
    }

    if (in_json.at("camera_state").at("half_aperture_angle_radians").is_number()) {
        in_json.at("camera_state").at("half_aperture_angle_radians").get_to(this->camera_state.half_aperture_angle_radians);
    }
    else {
        vislib::sys::Log::DefaultLog.WriteError("JSON ERROR - Couldn't read 'camera_state' - 'half_aperture_angle_radians': %s (%s:%d)", __FUNCTION__, __FILE__, __LINE__);
        return false;
    }

    if (in_json.at("camera_state").at("half_disparity").is_number()) {
        in_json.at("camera_state").at("half_disparity").get_to(this->camera_state.half_disparity);
    }
    else {
        vislib::sys::Log::DefaultLog.WriteError("JSON ERROR - Couldn't read 'camera_state' - 'half_disparity': %s (%s:%d)", __FUNCTION__, __FILE__, __LINE__);
        return false;
    }

    if (in_json.at("camera_state").at("image_tile").is_array()) {
        if (in_json.at("camera_state").at("image_tile").size() != 4) {
            vislib::sys::Log::DefaultLog.WriteError("JSON ERROR - Array of 'camera_state' - 'image_tile' should have size 2: %s (%s:%d)", __FUNCTION__, __FILE__, __LINE__);
            return false;
        }
        for (size_t i = 0; i < in_json.at("camera_state").at("image_tile").size(); i++) {
            if (!in_json.at("camera_state").at("image_tile")[i].is_number()) {
                vislib::sys::Log::DefaultLog.WriteError("JSON ERROR - Elements of array 'camera_state' - 'image_tile' should be numbers: %s (%s:%d)", __FUNCTION__, __FILE__, __LINE__);
                return false;
            }
        }
        in_json.at("camera_state").at("image_tile").get_to(this->camera_state.image_tile);
    }
    else {
        vislib::sys::Log::DefaultLog.WriteError("JSON ERROR - Couldn't read 'camera_state' - 'image_tile': %s (%s:%d)", __FUNCTION__, __FILE__, __LINE__);
        return false;
    }

    if (in_json.at("camera_state").at("near_clipping_plane").is_number()) {
        in_json.at("camera_state").at("near_clipping_plane").get_to(this->camera_state.near_clipping_plane);
    }
    else {
        vislib::sys::Log::DefaultLog.WriteError("JSON ERROR - Couldn't read 'camera_state' - 'near_clipping_plane': %s (%s:%d)", __FUNCTION__, __FILE__, __LINE__);
        return false;
    }

    if (in_json.at("camera_state").at("orientation").is_array()) {
        if (in_json.at("camera_state").at("orientation").size() != 4) {
            vislib::sys::Log::DefaultLog.WriteError("JSON ERROR - Array of 'camera_state' - 'orientation' should have size 2: %s (%s:%d)", __FUNCTION__, __FILE__, __LINE__);
            return false;
        }
        for (size_t i = 0; i < in_json.at("camera_state").at("orientation").size(); i++) {
            if (!in_json.at("camera_state").at("orientation")[i].is_number()) {
                vislib::sys::Log::DefaultLog.WriteError("JSON ERROR - Elements of array 'camera_state' - 'orientation' should be numbers: %s (%s:%d)", __FUNCTION__, __FILE__, __LINE__);
                return false;
            }
        }
        in_json.at("camera_state").at("orientation").get_to(this->camera_state.orientation);
    }
    else {
        vislib::sys::Log::DefaultLog.WriteError("JSON ERROR - Couldn't read 'camera_state' - 'orientation': %s (%s:%d)", __FUNCTION__, __FILE__, __LINE__);
        return false;
    }

    if (in_json.at("camera_state").at("position").is_array()) {
        if (in_json.at("camera_state").at("position").size() != 3) {
            vislib::sys::Log::DefaultLog.WriteError("JSON ERROR - Array of 'camera_state' - 'position' should have size 2: %s (%s:%d)", __FUNCTION__, __FILE__, __LINE__);
            return false;
        }
        for (size_t i = 0; i < in_json.at("camera_state").at("position").size(); i++) {
            if (!in_json.at("camera_state").at("position")[i].is_number()) {
                vislib::sys::Log::DefaultLog.WriteError("JSON ERROR - Elements of array 'camera_state' - 'position' should be numbers: %s (%s:%d)", __FUNCTION__, __FILE__, __LINE__);
                return false;
            }
        }
        in_json.at("camera_state").at("position").get_to(this->camera_state.position);
    }
    else {
        vislib::sys::Log::DefaultLog.WriteError("JSON ERROR - Couldn't read 'camera_state' - 'position': %s (%s:%d)", __FUNCTION__, __FILE__, __LINE__);
        return false;
    }

    if (in_json.at("camera_state").at("projection_type").is_number()) {
        int projection_type;
        in_json.at("camera_state").at("projection_type").get_to(projection_type);
        this->camera_state.projection_type = static_cast<megamol::core::thecam::Projection_type>(projection_type);
    }
    else {
        vislib::sys::Log::DefaultLog.WriteError("JSON ERROR - Couldn't read 'camera_state' - 'projection_type': %s (%s:%d)", __FUNCTION__, __FILE__, __LINE__);
        return false;
    }

    if (in_json.at("camera_state").at("resolution_gate").is_array()) {
        if (in_json.at("camera_state").at("resolution_gate").size() != 2) {
            vislib::sys::Log::DefaultLog.WriteError("JSON ERROR - Array of 'camera_state' - 'resolution_gate' should have size 2: %s (%s:%d)", __FUNCTION__, __FILE__, __LINE__);
            return false;
        }
        for (size_t i = 0; i < in_json.at("camera_state").at("resolution_gate").size(); i++) {
            if (!in_json.at("camera_state").at("resolution_gate")[i].is_number()) {
                vislib::sys::Log::DefaultLog.WriteError("JSON ERROR - Elements of array 'camera_state' - 'resolution_gate' should be numbers: %s (%s:%d)", __FUNCTION__, __FILE__, __LINE__);
                return false;
            }
        }
        in_json.at("camera_state").at("resolution_gate").get_to(this->camera_state.resolution_gate);
    }
    else {
        vislib::sys::Log::DefaultLog.WriteError("JSON ERROR - Couldn't read 'camera_state' - 'resolution_gate': %s (%s:%d)", __FUNCTION__, __FILE__, __LINE__);
        return false;
    }

    return true;
}
