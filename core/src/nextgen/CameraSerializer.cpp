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
    return false; // TODO
}

/*
 * CameraSerializer::deserialize
 */
bool CameraSerializer::deserialize(
    std::vector<Camera_2::minimal_state_type>& outCameras, const std::string text) const {
    return false; // TODO
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