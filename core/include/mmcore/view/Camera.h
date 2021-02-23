/*
 * Camera.h
 *
 * Copyright (C) 2021 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCORE_CAMERA_H_INCLUDED
#define MEGAMOLCORE_CAMERA_H_INCLUDED

#include <variant>

#include <glm/glm.hpp>

namespace megamol {
namespace core {
namespace view {


    class Camera {
    public:
        enum ProjectionType { PERSPECTIVE, ORTHOGRAPHIC, UNKNOWN };

        struct Pose {
            glm::vec3 position;
            glm::vec3 direction;
            glm::vec3 up;
        };

        // Use image plane tile defintion to shift the image plane
        // or to restrict the camera (projection matrix) to a limited area
        // within the frustrum given by fovy and aspect or by frustrum height and aspect
        struct ImagePlaneTile {
            ImagePlaneTile() : tile_start(glm::vec2(0.0f)), tile_end(glm::vec2(1.0f)){};

            glm::vec2 tile_start; //< lower left corner of image tile in normalized coordinates
            glm::vec2 tile_end;   //< upper right corner of image tile in normalized coordinates
        };

        struct PerspectiveParameters {
            float fovy; //< vertical field of view
            float aspect; //< aspect ratio of camera frustrum

            ImagePlaneTile tile; //< tile on the image plane displayed by camera
        };

        struct OrthographicParameters {
            float frustrum_height;
            float aspect;

            ImagePlaneTile tile; //< tile on the image plane displayed by camera
        };

        Camera(glm::mat4 view_matrix, glm::mat4 projection_matrix);
        Camera(Pose pose, PerspectiveParameters intrinsics);
        Camera(Pose pose, OrthographicParameters intrinsics);
        ~Camera() = default;

        template<typename CameraInfoType>
        CameraInfoType get() const;

        glm::mat4 getViewMatrix() const;

        glm::mat4 getProjectionMatrix() const;
    
    private:
        
        glm::mat4 _view_matrix;
        glm::mat4 _projection_matrix;

        Pose _pose;

        std::variant<std::monostate, PerspectiveParameters, OrthographicParameters> _intrinsics;

    };

    inline Camera::Camera(glm::mat4 view_matrix, glm::mat4 projection_matrix)
            : _view_matrix(view_matrix), _projection_matrix(projection_matrix), _intrinsics(std::monostate()) {
        // TODO compute matrices
    }

    inline Camera::Camera(Pose pose, PerspectiveParameters intrinsics)
            : _pose(pose), _intrinsics(intrinsics) {
        // TODO compute matrices
    }

    inline Camera::Camera(Pose pose, OrthographicParameters intrinsics)
        : _pose(pose), _intrinsics(intrinsics) {
        // TODO compute matrices
    }

    template<typename CameraInfoType>
    inline CameraInfoType Camera::get() const {
        if constexpr (std::is_same_v<CameraInfoType, ProjectionType>) {
            if constexpr (_intrinsics.index() == 0) {
                return Camera::UNKNOWN;
            } else if constexpr (_intrinsics.index() == 1) {
                return Camera::PERSPECTIVE;
            } else if constexpr (_intrinsics.index() == 2) {
                return Camera::ORTHOGRAPHIC;
            }
        } else if constexpr (std::is_same_v<CameraInfoType, Pose>) {
            return _pose;
        }
        else {
            return std::get<CameraInfoType>(_intrinsics);
        }
    }

    inline glm::mat4 Camera::getViewMatrix() const {
        return _view_matrix;
    }

    inline glm::mat4 Camera::getProjectionMatrix() const {
        return _projection_matrix;
    }

} // namespace view
} // namespace core
} // namespace megamol

#endif // !MEGAMOLCORE_CAMERA_H_INCLUDED
