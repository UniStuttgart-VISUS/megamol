/*
 * CameraSerializer.cpp
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */
#include "stdafx.h"
#include "mmcore/nextgen/CameraSerializer.h"

using namespace megamol::core;
using namespace megamol::core::nextgen;

/*
 * CameraSerializer::CameraSerializer
 */
CameraSerializer::CameraSerializer(bool prettyMode) : prettyMode(prettyMode) {
    // intentionally empty
}

/*
 * CameraSerializer::~CameraSerializer
 */
CameraSerializer::~CameraSerializer(void) {
    // intentionally empty
}

/*
 * CameraSerializer::serialize
 */
std::string CameraSerializer::serialize(const Camera_2::minimal_state_type& cam) const {
    nlohmann::json out;
    this->addCamToJsonObject(out, cam);
    if (this->prettyMode) {
        return out.dump(this->prettyIndent);
    }
    return out.dump();
}

/*
 * CameraSerializer::serialize
 */
std::string CameraSerializer::serialize(const std::vector<Camera_2::minimal_state_type>& camVec) const {
    nlohmann::json out;
    for (const auto& cam : camVec) {
        nlohmann::json c;
        this->addCamToJsonObject(c, cam);
        out.push_back(c);
    }
    if (this->prettyMode) {
        return out.dump(this->prettyIndent);
    }
    return out.dump();
}

/*
 * CameraSerializer::deserialize
 */
bool CameraSerializer::deserialize(Camera_2::minimal_state_type& outCamera, const std::string text) const {
    nlohmann::json obj = nlohmann::json::parse(text);
    bool result = this->getCamFromJsonObject(outCamera, obj);
    if (!result) outCamera = {};
    return result;
}

/*
 * CameraSerializer::deserialize
 */
bool CameraSerializer::deserialize(
    std::vector<Camera_2::minimal_state_type>& outCameras, const std::string text) const {
    nlohmann::json obj = nlohmann::json::parse(text);
    outCameras.clear();
    if (!obj.is_array()) {
        vislib::sys::Log::DefaultLog.WriteError("The input text does not contain a json array");
        return false;
    }
    for (nlohmann::json::iterator it = obj.begin(); it != obj.end(); ++it) {
        size_t index = static_cast<size_t>(it - obj.begin());
        auto cur = *it;
        megamol::core::nextgen::Camera_2::minimal_state_type cam;
        bool result = this->getCamFromJsonObject(cam, cur);
        if (!result) {
            cam = {}; // empty the cam if it is garbage
        }
        outCameras.push_back(cam);
    }
    return true;
}

/*
 * CameraSerializer::setPrettyMode
 */
void CameraSerializer::setPrettyMode(bool prettyMode) { this->prettyMode = prettyMode; }

/*
 * CameraSerializer::addCamToJsonObject
 */
void CameraSerializer::addCamToJsonObject(nlohmann::json& outObj, const Camera_2::minimal_state_type& cam) const {
    outObj["centre_offset"] = cam.centre_offset;
    outObj["convergence_plane"] = cam.convergence_plane;
    outObj["eye"] = cam.eye;
    outObj["far_clipping_plane"] = cam.far_clipping_plane;
    outObj["film_gate"] = cam.film_gate;
    outObj["gate_scaling"] = cam.gate_scaling;
    outObj["half_aperture_angle"] = cam.half_aperture_angle_radians;
    outObj["half_disparity"] = cam.half_disparity;
    outObj["image_tile"] = cam.image_tile;
    outObj["near_clipping_plane"] = cam.near_clipping_plane;
    outObj["orientation"] = cam.orientation;
    outObj["position"] = cam.position;
    outObj["projection_type"] = cam.projection_type;
    outObj["resolution_gate"] = cam.resolution_gate;
}

/*
 * CameraSerializer::getCamFromJsonObject
 */
bool CameraSerializer::getCamFromJsonObject(
    Camera_2::minimal_state_type& cam, const nlohmann::json::value_type& val) const {

    try {
        // If we would be sure that the read file is valid we could omit the sanity checks.
        // In fact, one sanity check is missing but very time consuming. We should check each array element for the
        // correct data type.
        if (!val.is_object()) return false;
        if (val.at("centre_offset").is_array() && val.at("centre_offset").size() == cam.centre_offset.size()) {
            val.at("centre_offset").get_to(cam.centre_offset);
        } else {
            return false;
        }
        if (val.at("convergence_plane").is_number_float()) {
            val.at("convergence_plane").get_to(cam.convergence_plane);
        } else {
            return false;
        }
        if (val.at("eye").is_number_integer()) {
            val.at("eye").get_to(cam.eye);
        } else {
            return false;
        }
        if (val.at("far_clipping_plane").is_number_float()) {
            val.at("far_clipping_plane").get_to(cam.far_clipping_plane);
        } else {
            return false;
        }
        if (val.at("film_gate").is_array() && val.at("film_gate").size() == cam.film_gate.size()) {
            val.at("film_gate").get_to(cam.film_gate);
        } else {
            return false;
        }
        if (val.at("gate_scaling").is_number_integer()) {
            val.at("gate_scaling").get_to(cam.gate_scaling);
        } else {
            return false;
        }
        if (val.at("half_aperture_angle").is_number_float()) {
            val.at("half_aperture_angle").get_to(cam.half_aperture_angle_radians);
        } else {
            return false;
        }
        if (val.at("half_disparity").is_number_float()) {
            val.at("half_disparity").get_to(cam.half_disparity);
        } else {
            return false;
        }
        if (val.at("image_tile").is_array() && val.at("image_tile").size() == cam.image_tile.size()) {
            val.at("image_tile").get_to(cam.image_tile);
        } else {
            return false;
        }
        if (val.at("near_clipping_plane").is_number_float()) {
            val.at("near_clipping_plane").get_to(cam.near_clipping_plane);
        } else {
            return false;
        }
        if (val.at("orientation").is_array() && val.at("orientation").size() == cam.orientation.size()) {
            val.at("orientation").get_to(cam.orientation);
        } else {
            return false;
        }
        if (val.at("position").is_array() && val.at("position").size() == cam.position.size()) {
            val.at("position").get_to(cam.position);
        } else {
            return false;
        }
        if (val.at("projection_type").is_number_integer()) {
            val.at("projection_type").get_to(cam.projection_type);
        } else {
            return false;
        }
        if (val.at("resolution_gate").is_array() && val.at("resolution_gate").size() == cam.resolution_gate.size()) {
            val.at("resolution_gate").get_to(cam.resolution_gate);
        } else {
            return false;
        }
        // try to read the additional "valid" value
        if (val.find("valid") != val.end()) {
            if (val.at("valid").is_boolean()) {
                bool valid;
                val.at("valid").get_to(valid);
                return valid;
            }
        }
    } catch (...) {
        return false;
    }

    return true;
}
