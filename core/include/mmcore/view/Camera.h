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
#include <glm/ext.hpp>

namespace megamol {
namespace core {
namespace view {

    class Camera {
    public:

        enum ProjectionType { PERSPECTIVE, ORTHOGRAPHIC, UNKNOWN };

        struct Pose {
            Pose() = default;
            Pose(Pose const& cpy) : position(cpy.position), direction(cpy.direction), up(cpy.up) {}
            Pose(glm::vec3 const& position, glm::vec3 const& direction, glm::vec3 const& up)
                    : position(position), direction(direction), up(up) {}
            Pose(glm::vec3 const& position, glm::quat const& orientation)
                    : position(position) {
                direction = glm::rotate(orientation, glm::vec3(0.0, 0.0, -1.0));
                up = glm::rotate(orientation, glm::vec3(0.0, 0.0, 0.0));
            }

            glm::vec3 position;
            glm::vec3 direction;
            glm::vec3 up;
        };

        // Use image plane tile defintion to shift the image plane
        // or to restrict the camera (projection matrix) to a limited area
        // within the frustrum given by fovy and aspect or by frustrum height and aspect
        struct ImagePlaneTile {
            ImagePlaneTile() : tile_start(glm::vec2(0.0f)), tile_end(glm::vec2(1.0f)){};
            ImagePlaneTile(glm::vec2 const& start, glm::vec2 const& end) : tile_start(start), tile_end(end){};

            glm::vec2 tile_start; //< lower left corner of image tile in normalized coordinates
            glm::vec2 tile_end;   //< upper right corner of image tile in normalized coordinates
        };

        struct PerspectiveParameters {
            float fovy; //< vertical field of view
            float aspect; //< aspect ratio of the camera frustrum
            float near_plane; //< near clipping plane
            float far_plane; //< far clipping plane

            ImagePlaneTile image_plane_tile; //< tile on the image plane displayed by camera
        };

        struct OrthographicParameters {
            float frustrum_height; //< vertical size of the orthographic frustrum in world space
            float aspect; //< aspect ratio of the camera frustrum
            float near_plane; //< near clipping plane
            float far_plane;  //< far clipping plane

            ImagePlaneTile image_plane_tile; //< tile on the image plane displayed by camera
        };

        Camera();
        Camera(glm::mat4 const& view_matrix, glm::mat4 const& projection_matrix);
        Camera(Pose const& pose, PerspectiveParameters const& intrinsics);
        Camera(Pose const& pose, OrthographicParameters const& intrinsics);
        ~Camera() = default;

        template<typename CameraInfoType>
        CameraInfoType get() const;

        glm::mat4 getViewMatrix() const;

        glm::mat4 getProjectionMatrix() const;

        void setPose(Pose pose);
    
    private:
        
        glm::mat4 _view_matrix;
        glm::mat4 _projection_matrix;

        Pose _pose;

        std::variant<std::monostate, PerspectiveParameters, OrthographicParameters> _intrinsics;

        friend bool operator==(Camera const& lhs, Camera const& rhs);
    };


    inline bool operator==(Camera::Pose const& lhs, Camera::Pose const& rhs) {
        bool retval = true;
        retval &= (lhs.position == rhs.position);
        retval &= (lhs.direction == rhs.direction);
        retval &= (lhs.up == rhs.up);
        return retval;
    }

    inline bool operator==(Camera::ImagePlaneTile const& lhs, Camera::ImagePlaneTile const& rhs) {
        bool retval = true;
        retval &= (lhs.tile_start == rhs.tile_start);
        retval &= (lhs.tile_end == rhs.tile_end);
        return retval;
    }

    inline bool operator==(Camera::PerspectiveParameters const& lhs, Camera::PerspectiveParameters const& rhs) {
        bool retval = true;
        retval &= (lhs.fovy == rhs.fovy);
        retval &= (lhs.aspect == rhs.aspect);
        retval &= (lhs.near_plane == rhs.near_plane);
        retval &= (lhs.far_plane == rhs.far_plane);
        retval &= (lhs.image_plane_tile == rhs.image_plane_tile);
        return retval;
    }

    inline bool operator==(Camera::OrthographicParameters const& lhs, Camera::OrthographicParameters const& rhs) {
        bool retval = true;
        retval &= (lhs.frustrum_height == rhs.frustrum_height);
        retval &= (lhs.aspect == rhs.aspect);
        retval &= (lhs.near_plane == rhs.near_plane);
        retval &= (lhs.far_plane == rhs.far_plane);
        retval &= (lhs.image_plane_tile == rhs.image_plane_tile);
        return retval;
    }

    inline bool operator==(Camera const& lhs, Camera const& rhs) {
        bool retval = true;

        retval &= (lhs._view_matrix == rhs._view_matrix);
        retval &= (lhs._projection_matrix == rhs._projection_matrix);
        retval &= (lhs._pose == rhs._pose);

        return retval;
    }


    inline Camera::Camera() : Camera(glm::mat4(1.0f), glm::mat4(1.0)) {}

    inline Camera::Camera(glm::mat4 const& view_matrix, glm::mat4 const& projection_matrix)
            : _view_matrix(view_matrix), _projection_matrix(projection_matrix), _intrinsics(std::monostate()) {

        const glm::vec4 position  = {0.0f, 0.0f, 0.0f, 1.0f};
        const glm::vec4 direction = {0.0f, 0.0f,-1.0f, 1.0f};
        const glm::vec4 up        = {0.0f, 1.0f, 0.0f, 1.0f};

        _pose.position  =                view_matrix * position;
        _pose.direction = glm::normalize(view_matrix * direction);
        _pose.up        = glm::normalize(view_matrix * up);
    }

    inline Camera::Camera(Pose const& pose, PerspectiveParameters const& intrinsics)
            : _pose(pose), _intrinsics(intrinsics) {
        // compute view matrix from pose
        _view_matrix = glm::lookAt(_pose.position, _pose.position + _pose.direction, _pose.up);

        // compute projection matrix from intrinsics
        //_projection_matrix =
        //    glm::perspective(intrinsics.fovy, intrinsics.aspect, intrinsics.near_plane, intrinsics.far_plane);
        //return;

        // check image_tile and compute projection matrix with adjusted fovy and/or off-center
        // all values normalized at distance 1 from camera
        #undef near // some header already defines near?
        const float near   = intrinsics.near_plane;
        const float global_height = glm::tan(intrinsics.fovy * 0.5f) * 2.0f; // fovy is whole frustum but tan takes only half
        const float global_width  = global_height * intrinsics.aspect;

        // https://docs.microsoft.com/en-us/windows/win32/opengl/glfrustum
        // https://www.khronos.org/registry/OpenGL-Refpages/gl2.1/xhtml/glFrustum.xml
        // note that these values need to be multiplied by near_distance to be fed into glFrustum
        const glm::vec2 global_left_bottom = {0, 0};
        const glm::vec2 global_right_top = {global_width, global_height};

        const auto normalized_tile_to_frustum = [&](glm::vec2 const& tile_point) {
            return tile_point * global_right_top;
        };

        const auto left_bottom_tile = normalized_tile_to_frustum(intrinsics.image_plane_tile.tile_start);
        const auto right_top_tile   = normalized_tile_to_frustum(intrinsics.image_plane_tile.tile_end);

        // center at (0,0) and map onto near plane
        const auto center_and_map = [&](glm::vec2 const& point) {
            const glm::vec2 global_half_frustum = {global_width * 0.5f, global_height * 0.5f};
            return (point - global_half_frustum) * near;
        };

        const auto local_frustum_left_bottom = center_and_map(left_bottom_tile);
        const auto local_frustum_right_top   = center_and_map(right_top_tile);

        _projection_matrix =
            glm::frustum(
                local_frustum_left_bottom.x, // left
                local_frustum_right_top.x,   // right
                local_frustum_left_bottom.y, // botom
                local_frustum_right_top.y,   // top
                intrinsics.near_plane, intrinsics.far_plane);
    }

    inline Camera::Camera(Pose const& pose, OrthographicParameters const& intrinsics)
            : _pose(pose), _intrinsics(intrinsics) {
        // compute view matrix from pose
        _view_matrix = glm::lookAt(_pose.position, _pose.position + _pose.direction, _pose.up);
        // compute projection matrix from intrinsics
        auto l = -1.0f * (intrinsics.frustrum_height / 2.0f) * intrinsics.aspect;
        auto r = (intrinsics.frustrum_height / 2.0f) * intrinsics.aspect;
        auto t = (intrinsics.frustrum_height / 2.0f);
        auto b = -1.0f * (intrinsics.frustrum_height / 2.0f);
        //_projection_matrix = glm::ortho(l, r, b, t, intrinsics.near_plane, intrinsics.far_plane);
        // return;

        // check image_tile and compute projection matrix with adjusted fovy and/or off-center
        const auto global_height = intrinsics.frustrum_height;
        const auto global_width  = intrinsics.frustrum_height * intrinsics.aspect;

        const glm::vec2 global_frustum_size = {global_width, global_height};
        const glm::vec2 global_left_bottom  = -global_frustum_size * 0.5f;
        const glm::vec2 global_right_top    =  global_frustum_size * 0.5f;

        // tile specified in (0,1) - can use for linear interpolation
        const glm::vec2 tile_start = intrinsics.image_plane_tile.tile_start;
        const glm::vec2 tile_end   = intrinsics.image_plane_tile.tile_end;

        const auto normalized_tile_to_frustum = [&](glm::vec2 const& point) {
            return point * global_frustum_size + global_left_bottom;
        };

        const glm::vec2 local_frustum_left_bottom = normalized_tile_to_frustum(tile_start);
        const glm::vec2 local_frustum_right_top   = normalized_tile_to_frustum(tile_end);
        _projection_matrix = glm::ortho(
            local_frustum_left_bottom.x, // left
            local_frustum_right_top.x,   // right
            local_frustum_left_bottom.y, // bottom
            local_frustum_right_top.y,   // top
            intrinsics.near_plane, intrinsics.far_plane);
    }

    template<typename CameraInfoType>
    inline CameraInfoType Camera::get() const {
        if constexpr (std::is_same_v<CameraInfoType, ProjectionType>) {
            if (_intrinsics.index() == 0) {
                return Camera::UNKNOWN;
            } else if (_intrinsics.index() == 1) {
                return Camera::PERSPECTIVE;
            } else if (_intrinsics.index() == 2) {
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

    inline void Camera::setPose(Pose pose) {
        _pose = pose;
        _view_matrix = glm::lookAt(_pose.position, _pose.position + _pose.direction, _pose.up);
    }

} // namespace view
} // namespace core
} // namespace megamol

#endif // !MEGAMOLCORE_CAMERA_H_INCLUDED
