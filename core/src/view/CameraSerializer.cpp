/*
 * CameraSerializer.cpp
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */
#include "mmcore/view/CameraSerializer.h"

using namespace megamol::core;
using namespace megamol::core::view;

/*
 * CameraSerializer::CameraSerializer
 */
CameraSerializer::CameraSerializer(bool prettyMode) : prettyMode(prettyMode) {
    // intentionally empty
}

/*
 * CameraSerializer::~CameraSerializer
 */
CameraSerializer::~CameraSerializer() {
    // intentionally empty
}

/*
 * CameraSerializer::serialize
 */
std::string CameraSerializer::serialize(Camera const& cam) const {
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
std::string CameraSerializer::serialize(std::vector<Camera> const& camVec) const {
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
bool CameraSerializer::deserialize(Camera& outCamera, std::string const text) const {
    nlohmann::json obj = nlohmann::json::parse(text, nullptr, false);
    if (!obj.is_discarded()) {
        bool result = this->getCamFromJsonObject(outCamera, obj);
        if (!result)
            outCamera = {};
        return result;
    } else {
        return false;
    }
}

/*
 * CameraSerializer::deserialize
 */
bool CameraSerializer::deserialize(std::vector<Camera>& outCameras, std::string const text) const {
    nlohmann::json obj = nlohmann::json::parse(text);
    outCameras.clear();
    if (!obj.is_array()) {
        megamol::core::utility::log::Log::DefaultLog.WriteError("The input text does not contain a json array");
        return false;
    }
    for (nlohmann::json::iterator it = obj.begin(); it != obj.end(); ++it) {
        size_t index = static_cast<size_t>(it - obj.begin());
        auto cur = *it;
        megamol::core::view::Camera cam;
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
void CameraSerializer::setPrettyMode(bool prettyMode) {
    this->prettyMode = prettyMode;
}

/*
 * CameraSerializer::addCamToJsonObject
 */
void CameraSerializer::addCamToJsonObject(nlohmann::json& outObj, Camera const& cam) const {

    // Identify projection type and serialize accordingly
    auto cam_type = cam.get<Camera::ProjectionType>();

    outObj["projection_type"] = cam_type;

    // If projection type is not unknown and camera is not serialized as matrix only do first block
    if (cam_type == Camera::PERSPECTIVE || cam_type == Camera::ORTHOGRAPHIC) {
        // Serialize pose
        auto cam_pose = cam.get<Camera::Pose>();
        outObj["position"] = std::array<float, 3>{cam_pose.position.x, cam_pose.position.y, cam_pose.position.z};
        outObj["direction"] = std::array<float, 3>{cam_pose.direction.x, cam_pose.direction.y, cam_pose.direction.z};
        outObj["up"] = std::array<float, 3>{cam_pose.up.x, cam_pose.up.y, cam_pose.up.z};
        outObj["right"] = std::array<float, 3>{cam_pose.right.x, cam_pose.right.y, cam_pose.right.z};

        // Serialize intrinsics
        if (cam_type == Camera::PERSPECTIVE) {
            auto cam_intrinsics = cam.get<Camera::PerspectiveParameters>();
            outObj["fovy"] = cam_intrinsics.fovy.value();
            outObj["aspect"] = cam_intrinsics.aspect.value();
            outObj["near_plane"] = cam_intrinsics.near_plane.value();
            outObj["far_plane"] = cam_intrinsics.far_plane.value();
            outObj["image_plane_tile_start"] = std::array<float, 2>{
                cam_intrinsics.image_plane_tile.tile_start.x, cam_intrinsics.image_plane_tile.tile_start.y};
            outObj["image_plane_tile_end"] = std::array<float, 2>{
                cam_intrinsics.image_plane_tile.tile_end.x, cam_intrinsics.image_plane_tile.tile_end.y};

        } else if (cam_type == Camera::ORTHOGRAPHIC) {
            auto cam_intrinsics = cam.get<Camera::OrthographicParameters>();

            outObj["frustrum_height"] = cam_intrinsics.frustrum_height.value();
            outObj["aspect"] = cam_intrinsics.aspect.value();
            outObj["near_plane"] = cam_intrinsics.near_plane.value();
            outObj["far_plane"] = cam_intrinsics.far_plane.value();
            outObj["image_plane_tile_start"] = std::array<float, 2>{
                cam_intrinsics.image_plane_tile.tile_start.x, cam_intrinsics.image_plane_tile.tile_start.y};
            outObj["image_plane_tile_end"] = std::array<float, 2>{
                cam_intrinsics.image_plane_tile.tile_end.x, cam_intrinsics.image_plane_tile.tile_end.y};
        }

    } else { //Camera::UNKNOWN
        auto view_mx = cam.getViewMatrix();
        auto proj_mx = cam.getProjectionMatrix();

        //TODO write matrix entries?
    }

    //  outObj["centre_offset"] = cam.centre_offset;
    //  outObj["convergence_plane"] = cam.convergence_plane;
    //  outObj["eye"] = cam.eye;
    //  outObj["far_clipping_plane"] = cam.far_clipping_plane;
    //  outObj["film_gate"] = cam.film_gate;
    //  outObj["gate_scaling"] = cam.gate_scaling;
    //  outObj["half_aperture_angle"] = cam.half_aperture_angle_radians;
    //  outObj["half_disparity"] = cam.half_disparity;
    //  outObj["image_tile"] = cam.image_tile;
    //  outObj["near_clipping_plane"] = cam.near_clipping_plane;
    //  outObj["orientation"] = cam.orientation;
    //  outObj["position"] = cam.position;
    //  outObj["projection_type"] = cam.projection_type;
    //  outObj["resolution_gate"] = cam.resolution_gate;
}

/*
 * CameraSerializer::getCamFromJsonObject
 */
bool CameraSerializer::getCamFromJsonObject(Camera& cam, nlohmann::json::value_type const& val) const {

    try {
        if (!val.is_object())
            return false;

        Camera::ProjectionType cam_type;

        if (val.at("projection_type").is_number_integer()) {
            val.at("projection_type").get_to(cam_type);
        } else {
            return false;
        }

        if (cam_type == Camera::PERSPECTIVE || cam_type == Camera::ORTHOGRAPHIC) {

            // Get camera pose
            Camera::Pose cam_pose;
            std::array<float, 3> position;
            std::array<float, 3> direction;
            std::array<float, 3> up;
            std::array<float, 3> right;

            if (val.at("position").is_array() && val.at("position").size() == position.size()) {
                val.at("position").get_to(position);
            } else {
                return false;
            }
            if (val.at("direction").is_array() && val.at("direction").size() == direction.size()) {
                val.at("direction").get_to(direction);
            } else {
                return false;
            }
            if (val.at("up").is_array() && val.at("up").size() == up.size()) {
                val.at("up").get_to(up);
            } else {
                return false;
            }
            if (val.at("right").is_array() && val.at("right").size() == right.size()) {
                val.at("right").get_to(right);
            } else {
                return false;
            }

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

            if (val.at("aspect").is_number_float()) {
                val.at("aspect").get_to(aspect);
            } else {
                return false;
            }
            if (val.at("near_plane").is_number_float()) {
                val.at("near_plane").get_to(near_plane);
            } else {
                return false;
            }
            if (val.at("far_plane").is_number_float()) {
                val.at("far_plane").get_to(far_plane);
            } else {
                return false;
            }
            if (val.at("image_plane_tile_start").is_array() &&
                val.at("image_plane_tile_start").size() == image_plane_tile_start.size()) {
                val.at("image_plane_tile_start").get_to(image_plane_tile_start);
            } else {
                return false;
            }
            if (val.at("image_plane_tile_end").is_array() &&
                val.at("image_plane_tile_end").size() == image_plane_tile_end.size()) {
                val.at("image_plane_tile_end").get_to(image_plane_tile_end);
            } else {
                return false;
            }


            if (cam_type == Camera::PERSPECTIVE) {
                Camera::PerspectiveParameters cam_intrinsics;

                float fovy;
                if (val.at("fovy").is_number_float()) {
                    val.at("fovy").get_to(fovy);
                } else {
                    return false;
                }

                cam_intrinsics.fovy = fovy;
                cam_intrinsics.aspect = aspect;
                cam_intrinsics.near_plane = near_plane;
                cam_intrinsics.far_plane = far_plane;
                cam_intrinsics.image_plane_tile.tile_start =
                    glm::vec2(std::get<0>(image_plane_tile_start), std::get<1>(image_plane_tile_start));
                cam_intrinsics.image_plane_tile.tile_end =
                    glm::vec2(std::get<0>(image_plane_tile_end), std::get<1>(image_plane_tile_end));

                cam = Camera(cam_pose, cam_intrinsics);

            } else if (cam_type == Camera::ORTHOGRAPHIC) {
                Camera::OrthographicParameters cam_intrinsics;

                float frustrum_height;
                if (val.at("frustrum_height").is_number_float()) {
                    val.at("frustrum_height").get_to(frustrum_height);
                } else {
                    return false;
                }

                cam_intrinsics.frustrum_height = frustrum_height;
                cam_intrinsics.aspect = aspect;
                cam_intrinsics.near_plane = near_plane;
                cam_intrinsics.far_plane = far_plane;
                cam_intrinsics.image_plane_tile.tile_start =
                    glm::vec2(std::get<0>(image_plane_tile_start), std::get<1>(image_plane_tile_start));
                cam_intrinsics.image_plane_tile.tile_end =
                    glm::vec2(std::get<0>(image_plane_tile_end), std::get<1>(image_plane_tile_end));

                cam = Camera(cam_pose, cam_intrinsics);
            }

        } else {
            //TODO
        }

    } catch (...) { return false; }

    // TODO
    //  try {
    //      // If we would be sure that the read file is valid we could omit the sanity checks.
    //      // In fact, one sanity check is missing but very time consuming. We should check each array element for the
    //      // correct data type.
    //      if (!val.is_object()) return false;
    //      if (val.at("centre_offset").is_array() && val.at("centre_offset").size() == cam.centre_offset.size()) {
    //          val.at("centre_offset").get_to(cam.centre_offset);
    //      } else {
    //          return false;
    //      }
    //      if (val.at("convergence_plane").is_number_float()) {
    //          val.at("convergence_plane").get_to(cam.convergence_plane);
    //      } else {
    //          return false;
    //      }
    //      if (val.at("eye").is_number_integer()) {
    //          val.at("eye").get_to(cam.eye);
    //      } else {
    //          return false;
    //      }
    //      if (val.at("far_clipping_plane").is_number_float()) {
    //          val.at("far_clipping_plane").get_to(cam.far_clipping_plane);
    //      } else {
    //          return false;
    //      }
    //      if (val.at("film_gate").is_array() && val.at("film_gate").size() == cam.film_gate.size()) {
    //          val.at("film_gate").get_to(cam.film_gate);
    //      } else {
    //          return false;
    //      }
    //      if (val.at("gate_scaling").is_number_integer()) {
    //          val.at("gate_scaling").get_to(cam.gate_scaling);
    //      } else {
    //          return false;
    //      }
    //      if (val.at("half_aperture_angle").is_number_float()) {
    //          val.at("half_aperture_angle").get_to(cam.half_aperture_angle_radians);
    //      } else {
    //          return false;
    //      }
    //      if (val.at("half_disparity").is_number_float()) {
    //          val.at("half_disparity").get_to(cam.half_disparity);
    //      } else {
    //          return false;
    //      }
    //      if (val.at("image_tile").is_array() && val.at("image_tile").size() == cam.image_tile.size()) {
    //          val.at("image_tile").get_to(cam.image_tile);
    //      } else {
    //          return false;
    //      }
    //      if (val.at("near_clipping_plane").is_number_float()) {
    //          val.at("near_clipping_plane").get_to(cam.near_clipping_plane);
    //      } else {
    //          return false;
    //      }
    //      if (val.at("orientation").is_array() && val.at("orientation").size() == cam.orientation.size()) {
    //          val.at("orientation").get_to(cam.orientation);
    //      } else {
    //          return false;
    //      }
    //      if (val.at("position").is_array() && val.at("position").size() == cam.position.size()) {
    //          val.at("position").get_to(cam.position);
    //      } else {
    //          return false;
    //      }
    //      if (val.at("projection_type").is_number_integer()) {
    //          val.at("projection_type").get_to(cam.projection_type);
    //      } else {
    //          return false;
    //      }
    //      if (val.at("resolution_gate").is_array() && val.at("resolution_gate").size() == cam.resolution_gate.size()) {
    //          val.at("resolution_gate").get_to(cam.resolution_gate);
    //      } else {
    //          return false;
    //      }
    //      // try to read the additional "valid" value
    //      if (val.find("valid") != val.end()) {
    //          if (val.at("valid").is_boolean()) {
    //              bool valid;
    //              val.at("valid").get_to(valid);
    //              return valid;
    //          }
    //      }
    //  } catch (...) {
    //      return false;
    //  }

    return true;
}
