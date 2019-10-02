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

}


Keyframe::Keyframe(float anim_time, float sim_time, cam_state_type cam_state)
    : anim_time(anim_time)
    , sim_time(sim_time)
    , camera_state(cam_state) {

}


Keyframe::~Keyframe() {

}


bool Keyframe::Serialise(std::string& json_string) {

    json_string = "";
    try {
        nlohmann::json json;

        /// TODO






        json_string = json.dump(2); // Dump with indent of 2 spaces and new lines.
    }
    catch (nlohmann::json::type_error& e) {
        vislib::sys::Log::DefaultLog.WriteError(
            "JSON ERROR - %s: %s (%s:%d)", __FUNCTION__, e.what(), __FILE__, __LINE__);
        return false;
    }
    catch (nlohmann::json::exception& e) {
        vislib::sys::Log::DefaultLog.WriteError(
            "JSON ERROR - %s: %s (%s:%d)", __FUNCTION__, e.what(), __FILE__, __LINE__);
        return false;
    }
    catch (nlohmann::json::parse_error& e) {
        vislib::sys::Log::DefaultLog.WriteError(
            "JSON ERROR - %s: %s (%s:%d)", __FUNCTION__, e.what(), __FILE__, __LINE__);
        return false;
    }
    catch (nlohmann::json::invalid_iterator& e) {
        vislib::sys::Log::DefaultLog.WriteError(
            "JSON ERROR - %s: %s (%s:%d)", __FUNCTION__, e.what(), __FILE__, __LINE__);
        return false;
    }
    catch (nlohmann::json::out_of_range& e) {
        vislib::sys::Log::DefaultLog.WriteError(
            "JSON ERROR - %s: %s (%s:%d)", __FUNCTION__, e.what(), __FILE__, __LINE__);
        return false;
    }
    catch (nlohmann::json::other_error& e) {
        vislib::sys::Log::DefaultLog.WriteError(
            "JSON ERROR - %s: %s (%s:%d)", __FUNCTION__, e.what(), __FILE__, __LINE__);
        return false;
    }
    catch (...) {
        vislib::sys::Log::DefaultLog.WriteError("JSON ERROR - %s: Unknown Error (%s:%d)", __FUNCTION__,  __FILE__, __LINE__);
        return false;
    }

    return true;
}


bool Keyframe::Deserialise(const std::string& json_string) {

    try {
        bool valid = true;
        nlohmann::json json;
        json = nlohmann::json::parse(json_string);

        if (!json.is_object()) {
            vislib::sys::Log::DefaultLog.WriteError("[Keyframe::Deserialise] String is no valid JSON object.");
            return false;
        }

        /// TODO






        
        if (!valid) {
            vislib::sys::Log::DefaultLog.WriteWarn("[Keyframe::Deserialise] Could not deserialise keyframe.");
            return false;
        }
    }
    catch (nlohmann::json::type_error& e) {
        vislib::sys::Log::DefaultLog.WriteError(
            "JSON ERROR - %s: %s (%s:%d)", __FUNCTION__, e.what(), __FILE__, __LINE__);
        return false;
    }
    catch (nlohmann::json::exception& e) {
        vislib::sys::Log::DefaultLog.WriteError(
            "JSON ERROR - %s: %s (%s:%d)", __FUNCTION__, e.what(), __FILE__, __LINE__);
        return false;
    }
    catch (nlohmann::json::parse_error& e) {
        vislib::sys::Log::DefaultLog.WriteError(
            "JSON ERROR - %s: %s (%s:%d)", __FUNCTION__, e.what(), __FILE__, __LINE__);
        return false;
    }
    catch (nlohmann::json::invalid_iterator& e) {
        vislib::sys::Log::DefaultLog.WriteError(
            "JSON ERROR - %s: %s (%s:%d)", __FUNCTION__, e.what(), __FILE__, __LINE__);
        return false;
    }
    catch (nlohmann::json::out_of_range& e) {
        vislib::sys::Log::DefaultLog.WriteError(
            "JSON ERROR - %s: %s (%s:%d)", __FUNCTION__, e.what(), __FILE__, __LINE__);
        return false;
    }
    catch (nlohmann::json::other_error& e) {
        vislib::sys::Log::DefaultLog.WriteError(
            "JSON ERROR - %s: %s (%s:%d)", __FUNCTION__, e.what(), __FILE__, __LINE__);
        return false;
    }
    catch (...) {
        vislib::sys::Log::DefaultLog.WriteError("JSON ERROR - %s: Unknown Error (%s:%d)", __FUNCTION__, __FILE__, __LINE__);
        return false;
    }

    return true;
}


bool Keyframe::camStatesEqual(cam_state_type ls, cam_state_type rs) {

    return ((ls.centre_offset               == rs.centre_offset) &&
            (ls.convergence_plane           == rs.convergence_plane) &&
            (ls.eye                         == rs.eye) &&
            (ls.far_clipping_plane          == rs.far_clipping_plane) &&
            (ls.film_gate                   == rs.film_gate) &&
            (ls.gate_scaling                == rs.gate_scaling) &&
            (ls.half_aperture_angle_radians == rs.half_aperture_angle_radians) &&
            (ls.half_disparity              == rs.half_disparity) &&
            (ls.image_tile                  == ls.image_tile) &&
            (ls.near_clipping_plane         == rs.near_clipping_plane) &&
            (ls.orientation                 == rs.orientation) &&
            (ls.position                    == rs.position) &&
            (ls.projection_type             == rs.projection_type) &&
            (ls.resolution_gate             == rs.resolution_gate));
}