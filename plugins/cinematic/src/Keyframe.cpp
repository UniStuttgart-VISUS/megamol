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
    valid &= get_json_value<float>(in_json, { "animation_time" }, &this->anim_time);
    valid &= get_json_value<float>(in_json, { "simulation_time" }, &this->sim_time);
    valid &= get_json_value<float>(in_json, { "camera_state", "centre_offset" }, this->camera_state.centre_offset.data(), this->camera_state.centre_offset.size());
    valid &= get_json_value<float>(in_json, { "camera_state", "convergence_plane" }, &this->camera_state.convergence_plane);
    int eye = 0;
    valid &= get_json_value<int>(in_json, { "camera_state", "eye" }, &eye);
    this->camera_state.eye = static_cast<megamol::core::thecam::Eye>(eye);
    valid &= get_json_value<float>(in_json, { "camera_state", "far_clipping_plane" }, &this->camera_state.far_clipping_plane);
    valid &= get_json_value<float>(in_json, { "camera_state", "film_gate" }, this->camera_state.film_gate.data(), this->camera_state.film_gate.size());
    int gate_scaling = 0;
    valid &= get_json_value<int>(in_json, { "camera_state", "gate_scaling" }, &gate_scaling);
    this->camera_state.gate_scaling = static_cast<megamol::core::thecam::Gate_scaling>(gate_scaling);
    valid &= get_json_value<float>(in_json, { "camera_state", "half_aperture_angle_radians" }, &this->camera_state.half_aperture_angle_radians);
    valid &= get_json_value<float>(in_json, { "camera_state", "half_disparity" }, &this->camera_state.half_disparity);
    valid &= get_json_value<int>(in_json, { "camera_state", "image_tile" }, this->camera_state.image_tile.data(), this->camera_state.image_tile.size());
    valid &= get_json_value<float>(in_json, { "camera_state", "near_clipping_plane" }, &this->camera_state.near_clipping_plane);
    valid &= get_json_value<float>(in_json, { "camera_state", "orientation" }, this->camera_state.orientation.data(), this->camera_state.orientation.size());
    valid &= get_json_value<float>(in_json, { "camera_state", "position" }, this->camera_state.position.data(), this->camera_state.position.size());
    int projection_type = 0;
    valid &= get_json_value<int>(in_json, { "camera_state", "projection_type" }, &projection_type);
    this->camera_state.projection_type = static_cast<megamol::core::thecam::Projection_type>(projection_type);
    valid &= get_json_value<int>(in_json, { "camera_state", "resolution_gate" }, this->camera_state.resolution_gate.data(), this->camera_state.resolution_gate.size());

    return valid;
}


template <typename T>
bool megamol::cinematic::get_json_value<float>(const nlohmann::json& in_json, const std::vector<std::string>& in_nodes, T* out_value, size_t array_size) {

    try {
        auto node_count = in_nodes.size();
        if (node_count == 0) {
            megamol::core::utility::log::Log::DefaultLog.WriteError("JSON ERROR - No in_nodes for reading value given. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
            return false;
        }
        std::string node_name = in_nodes.front();
        auto json_value = in_json.at(in_nodes.front());
        for (size_t i = 1; i < node_count; i++) {
            node_name = node_name + "/" + in_nodes[i];
            json_value = json_value.at(in_nodes[i]);
        }
        if (array_size > 0) {
            if (!json_value.is_array()) {
                megamol::core::utility::log::Log::DefaultLog.WriteError("JSON ERROR - %s is no JSON array. [%s, %s, line %d]\n", node_name.c_str(), __FILE__, __FUNCTION__, __LINE__);
                return false;
            }
            if (json_value.size() != array_size) {
                megamol::core::utility::log::Log::DefaultLog.WriteError("JSON ERROR - %s is no JSON array of size %i. [%s, %s, line %d]\n", node_name.c_str(), array_size, __FILE__, __FUNCTION__, __LINE__);
                return false;
            }
            for (size_t i = 0; i < array_size; i++) {
                if (!json_value[i].is_number()) {
                    megamol::core::utility::log::Log::DefaultLog.WriteError("JSON ERROR - Couldn't read float value from json node '%s' at position %i. [%s, %s, line %d]\n", node_name.c_str(), i, __FILE__, __FUNCTION__, __LINE__);
                    return false;
                }
                out_value[i] = json_value[i];
            }
        }
        else {
            if (!json_value.is_number()) {
                megamol::core::utility::log::Log::DefaultLog.WriteError("JSON ERROR - Couldn't read float value from json node '%s'. [%s, %s, line %d]\n", node_name.c_str(), __FILE__, __FUNCTION__, __LINE__);
                return false;
            }
            json_value.get_to((*out_value));
        }
        return true;
    }
    catch (nlohmann::json::type_error& e) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "JSON ERROR: %s. [%s, %s, line %d]\n", e.what(), __FILE__, __FUNCTION__, __LINE__);
        return false;
    }
    catch (nlohmann::json::exception& e) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "JSON ERROR: %s. [%s, %s, line %d]\n", e.what(), __FILE__, __FUNCTION__, __LINE__);
        return false;
    }
    catch (nlohmann::json::parse_error& e) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "JSON ERROR: %s. [%s, %s, line %d]\n", e.what(), __FILE__, __FUNCTION__, __LINE__);
        return false;
    }
    catch (nlohmann::json::invalid_iterator& e) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "JSON ERROR: %s. [%s, %s, line %d]\n", e.what(), __FILE__, __FUNCTION__, __LINE__);
        return false;
    }
    catch (nlohmann::json::out_of_range& e) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "JSON ERROR: %s. [%s, %s, line %d]\n", e.what(), __FILE__, __FUNCTION__, __LINE__);
        return false;
    }
    catch (nlohmann::json::other_error& e) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "JSON ERROR: %s. [%s, %s, line %d]\n", e.what(), __FILE__, __FUNCTION__, __LINE__);
        return false;
    }
    catch (...) {
        megamol::core::utility::log::Log::DefaultLog.WriteError("JSON ERROR - Unknown Error. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return false;
    }
}
