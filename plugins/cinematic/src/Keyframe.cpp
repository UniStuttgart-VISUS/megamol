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


Keyframe::Keyframe(float anim_time, float sim_time, camera_state_type cam_state)
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
