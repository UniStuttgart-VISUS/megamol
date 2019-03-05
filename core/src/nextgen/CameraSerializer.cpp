/*
 * CameraSerializer.cpp
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */
#include "stdafx.h"
#include "mmcore/nextgen/CameraSerializer.h"

#include "json.hpp"

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
    out["centre_offset"] = cam.centre_offset;
    out["convergence_plane"] = cam.convergence_plane;
    out["eye"] = cam.eye;
    out["far_clipping_plane"] = cam.far_clipping_plane;
    out["film_gate"] = cam.film_gate;
    out["gate_scaling"] = cam.gate_scaling;
    out["half_aperture_angle"] = cam.half_aperture_angle_radians;
    out["half_disparity"] = cam.half_disparity;
    out["image_tile"] = cam.image_tile;
    out["near_clipping_plane"] = cam.near_clipping_plane;
    out["orientation"] = cam.orientation;
    out["position"] = cam.position;
    out["projection_type"] = cam.projection_type;
    out["resolution_gate"] = cam.resolution_gate;
    if (this->prettyMode) {
        return out.dump(4);
	}
    return out.dump();
}

/*
 * CameraSerializer::serialize
 */
std::string CameraSerializer::serialize(const std::vector<Camera_2::minimal_state_type>& camVec) const {
    // TODO
    return "";
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