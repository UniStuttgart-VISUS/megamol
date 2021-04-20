#ifndef CAMERA_PARAMETER_SLOTS_H_INCLUDED
#define CAMERA_PARAMETER_SLOTS_H_INCLUDED

#include "mmcore/param/BoolParam.h"
#include "mmcore/param/EnumParam.h"
#include "mmcore/param/FloatParam.h"
#include "mmcore/param/ParamSlot.h"
#include "mmcore/param/Vector3fParam.h"
#include "mmcore/param/Vector4fParam.h"
#include "mmcore/view/Camera.h"

namespace megamol {
namespace core {
    namespace view {

        struct Camera3DParameters {
            /** Enum for default views from the respective direction */
            enum defaultview {
                DEFAULTVIEW_FRONT,
                DEFAULTVIEW_BACK,
                DEFAULTVIEW_RIGHT,
                DEFAULTVIEW_LEFT,
                DEFAULTVIEW_TOP,
                DEFAULTVIEW_BOTTOM,
            };

            /** Enum for default orientations from the respective direction */
            enum defaultorientation {
                DEFAULTORIENTATION_TOP,
                DEFAULTORIENTATION_RIGHT,
                DEFAULTORIENTATION_BOTTOM,
                DEFAULTORIENTATION_LEFT
            };

            Camera3DParameters()
                    : _cameraPositionParam("cam::position", "")
                    , _cameraOrientationParam("cam::orientation", "")
                    , _cameraProjectionTypeParam("cam::projectiontype", "")
                    , _cameraNearPlaneParam("cam::nearplane", "")
                    , _cameraFarPlaneParam("cam::farplane", "")
                    , _cameraHalfApertureDegreesParam("cam::halfaperturedegrees", "")
                    , _cameraSetViewChooserParam("view::defaultView", "Choose a default view to look from")
                    , _cameraSetOrientationChooserParam(
                          "view::defaultOrientation", "Choose a default orientation to look from")
                    , _cameraViewOrientationParam(
                          "view::cubeOrientation", "Current camera orientation used for view cube.") {
                const bool camparamvisibility = true;

                auto camposparam = new param::Vector3fParam(vislib::math::Vector<float, 3>());
                camposparam->SetGUIVisible(camparamvisibility);
                this->_cameraPositionParam.SetParameter(camposparam);

                auto camorientparam = new param::Vector4fParam(vislib::math::Vector<float, 4>());
                camorientparam->SetGUIVisible(camparamvisibility);
                this->_cameraOrientationParam.SetParameter(camorientparam);

                auto projectionParam = new param::EnumParam(static_cast<int>(Camera::ProjectionType::PERSPECTIVE));
                projectionParam->SetTypePair(static_cast<int>(Camera::ProjectionType::PERSPECTIVE), "Perspective");
                projectionParam->SetTypePair(static_cast<int>(Camera::ProjectionType::ORTHOGRAPHIC), "Orthographic");
                projectionParam->SetGUIVisible(camparamvisibility);
                this->_cameraProjectionTypeParam.SetParameter(projectionParam);

                auto farplaneparam = new param::FloatParam(100.0f, 0.0f);
                farplaneparam->SetGUIVisible(camparamvisibility);
                this->_cameraFarPlaneParam.SetParameter(farplaneparam);

                auto nearplaneparam = new param::FloatParam(0.1f, 0.0f);
                nearplaneparam->SetGUIVisible(camparamvisibility);
                this->_cameraNearPlaneParam.SetParameter(nearplaneparam);

                auto apertureparam = new param::FloatParam(35.0f, 0.0f);
                apertureparam->SetGUIVisible(camparamvisibility);
                this->_cameraHalfApertureDegreesParam.SetParameter(apertureparam);

                auto defaultViewParam = new param::EnumParam(0);
                defaultViewParam->SetTypePair(defaultview::DEFAULTVIEW_FRONT, "Front");
                defaultViewParam->SetTypePair(defaultview::DEFAULTVIEW_BACK, "Back");
                defaultViewParam->SetTypePair(defaultview::DEFAULTVIEW_RIGHT, "Right");
                defaultViewParam->SetTypePair(defaultview::DEFAULTVIEW_LEFT, "Left");
                defaultViewParam->SetTypePair(defaultview::DEFAULTVIEW_TOP, "Top");
                defaultViewParam->SetTypePair(defaultview::DEFAULTVIEW_BOTTOM, "Bottom");
                defaultViewParam->SetGUIVisible(camparamvisibility);
                this->_cameraSetViewChooserParam.SetParameter(defaultViewParam);
                // this->_cameraSetViewChooserParam.SetUpdateCallback(&AbstractView::OnResetView);

                auto defaultOrientationParam = new param::EnumParam(0);
                defaultOrientationParam->SetTypePair(defaultorientation::DEFAULTORIENTATION_TOP, "Top");
                defaultOrientationParam->SetTypePair(defaultorientation::DEFAULTORIENTATION_RIGHT, "Right");
                defaultOrientationParam->SetTypePair(defaultorientation::DEFAULTORIENTATION_BOTTOM, "Bottom");
                defaultOrientationParam->SetTypePair(defaultorientation::DEFAULTORIENTATION_LEFT, "Left");
                defaultOrientationParam->SetGUIVisible(camparamvisibility);
                this->_cameraSetOrientationChooserParam.SetParameter(defaultOrientationParam);
                // this->_cameraSetOrientationChooserParam.SetUpdateCallback(&AbstractView::OnResetView);

                this->_cameraViewOrientationParam.SetParameter(
                    new param::Vector4fParam(vislib::math::Vector<float, 4>(0.0f, 0.0f, 0.0f, 1.0f)));
                this->_cameraViewOrientationParam.Parameter()->SetGUIReadOnly(true);
                this->_cameraViewOrientationParam.Parameter()->SetGUIVisible(false);
            }

            std::vector<AbstractSlot*> getParameterSlots() {
                return {&_cameraPositionParam, &_cameraOrientationParam, &_cameraProjectionTypeParam,
                    &_cameraNearPlaneParam, &_cameraFarPlaneParam, &_cameraHalfApertureDegreesParam,
                    &_cameraViewOrientationParam, &_cameraSetViewChooserParam, &_cameraSetOrientationChooserParam};
            }

            void setParametersFromCamera(Camera const& cam) {
                auto cam_pose = cam.get<Camera::Pose>();

                glm::vec3 pos = cam_pose.position;
                const bool makeDirty = false;
                this->_cameraPositionParam.Param<param::Vector3fParam>()->SetValue(
                    vislib::math::Vector<float, 3>(pos.x, pos.y, pos.z), makeDirty);
                this->_cameraPositionParam.QueueUpdateNotification();

                glm::quat orientation = cam_pose.to_quat();
                this->_cameraOrientationParam.Param<param::Vector4fParam>()->SetValue(
                    vislib::math::Vector<float, 4>(orientation.x, orientation.y, orientation.z, orientation.w),
                    makeDirty);
                this->_cameraOrientationParam.QueueUpdateNotification();

                auto cam_proj_type = cam.get<Camera::ProjectionType>();
                this->_cameraProjectionTypeParam.Param<param::EnumParam>()->SetValue(
                    static_cast<int>(cam_proj_type), makeDirty);
                this->_cameraProjectionTypeParam.QueueUpdateNotification();

                /*this->cameraNearPlaneParam.Param<param::FloatParam>()->SetValue(cam.near_clipping_plane(), makeDirty);
                this->cameraFarPlaneParam.Param<param::FloatParam>()->SetValue(cam.far_clipping_plane(), makeDirty);*/
                /*this->cameraEyeParam.Param<param::EnumParam>()->SetValue(static_cast<int>(cam.eye()), makeDirty);
                this->cameraGateScalingParam.Param<param::EnumParam>()->SetValue(static_cast<int>(cam.gate_scaling()),
                makeDirty); this->cameraFilmGateParam.Param<param::Vector2fParam>()->SetValue(
                vislib::math::Vector<float, 2>(cam.film_gate().width(), cam.film_gate().height()), makeDirty);*/
                /*this->cameraResolutionXParam.Param<param::IntParam>()->SetValue(cam.resolution_gate().width());
                this->cameraResolutionYParam.Param<param::IntParam>()->SetValue(cam.resolution_gate().height());*/

                if (cam_proj_type == Camera::ProjectionType::PERSPECTIVE) {
                    auto cam_intrinsics = cam.get<Camera::PerspectiveParameters>();
                    this->_cameraHalfApertureDegreesParam.Param<param::FloatParam>()->SetValue(
                        cam_intrinsics.fovy * 180.0f / 3.14159265359 /*TODO*/, makeDirty);
                    this->_cameraHalfApertureDegreesParam.QueueUpdateNotification();
                }
            }

            Camera getCameraFromParameters(Camera const& cam, glm::vec3 cam_look_at) {
                // initialize new camera with current camera
                Camera new_cam = cam;

                auto cam_pose = cam.get<Camera::Pose>();
                if (this->_cameraPositionParam.IsDirty()) {
                    auto val = this->_cameraPositionParam.Param<param::Vector3fParam>()->Value();
                    cam_pose.position = glm::vec3(val.GetX(), val.GetY(), val.GetZ());
                    this->_cameraPositionParam.ResetDirty();
                }

                if (this->_cameraOrientationParam.IsDirty()) {
                    auto val = this->_cameraOrientationParam.Param<param::Vector4fParam>()->Value();
                    const auto orientation = glm::quat(val.GetW(), val.GetX(), val.GetY(), val.GetZ());
                    cam_pose = Camera::Pose(cam_pose.position, orientation);
                    this->_cameraOrientationParam.ResetDirty();
                }
                new_cam.setPose(cam_pose);

                // BIG TODO: manipulation of intrinsics via GUI

                if (this->_cameraProjectionTypeParam.IsDirty()) {
                    auto curr_proj_type = cam.get<Camera::ProjectionType>();
                    auto cam_pose = cam.get<Camera::Pose>();
                    float orbitalAltitude = glm::length(cam_pose.position - cam_look_at);
                    float fovy;
                    float vertical_height;
                    float aspect;
                    float near_plane;
                    float far_plane;
                    Camera::ImagePlaneTile tile;
                    if (curr_proj_type == Camera::ProjectionType::PERSPECTIVE) {
                        fovy = cam.get<Camera::PerspectiveParameters>().fovy;
                        aspect = cam.get<Camera::AspectRatio>();
                        near_plane = cam.get<Camera::PerspectiveParameters>().near_plane;
                        far_plane = cam.get<Camera::PerspectiveParameters>().far_plane;
                        vertical_height = std::tan(fovy) * orbitalAltitude;
                        tile = cam.get<Camera::PerspectiveParameters>().image_plane_tile;
                    } else if (curr_proj_type == Camera::ProjectionType::ORTHOGRAPHIC) {
                        aspect = cam.get<Camera::AspectRatio>();
                        near_plane = cam.get<Camera::OrthographicParameters>().near_plane;
                        far_plane = cam.get<Camera::OrthographicParameters>().far_plane;
                        vertical_height = cam.get<Camera::OrthographicParameters>().frustrum_height;
                        fovy = std::atan(vertical_height / orbitalAltitude);
                        tile = cam.get<Camera::OrthographicParameters>().image_plane_tile;
                    } else {
                        return new_cam; // Oh bother...
                    }

                    auto val = static_cast<Camera::ProjectionType>(
                        this->_cameraProjectionTypeParam.Param<param::EnumParam>()->Value());
                    if (val == Camera::PERSPECTIVE) {
                        Camera::PerspectiveParameters cam_intrinsics;
                        cam_intrinsics.aspect = aspect;
                        cam_intrinsics.fovy = fovy;
                        cam_intrinsics.near_plane = near_plane;
                        cam_intrinsics.far_plane = far_plane;
                        cam_intrinsics.image_plane_tile = tile;

                        new_cam = Camera(cam_pose, cam_intrinsics);
                    } else if (val == Camera::ORTHOGRAPHIC) {
                        Camera::OrthographicParameters cam_intrinsics;
                        cam_intrinsics.aspect = aspect;
                        cam_intrinsics.frustrum_height = vertical_height;
                        cam_intrinsics.near_plane = near_plane;
                        cam_intrinsics.far_plane = far_plane;
                        cam_intrinsics.image_plane_tile = tile;

                        new_cam = Camera(cam_pose, cam_intrinsics);
                    }

                    this->_cameraProjectionTypeParam.ResetDirty();
                }

                return new_cam;
                //// setting of near plane and far plane might make no sense as we are setting them new each frame
                /// anyway
                // if (this->cameraNearPlaneParam.IsDirty()) {
                //    auto val = this->cameraNearPlaneParam.Param<param::FloatParam>()->Value();
                //    this->_camera.near_clipping_plane(val);
                //    this->cameraNearPlaneParam.ResetDirty();
                //    result = true;
                //}
                // if (this->cameraFarPlaneParam.IsDirty()) {
                //    auto val = this->cameraFarPlaneParam.Param<param::FloatParam>()->Value();
                //    this->_camera.far_clipping_plane(val);
                //    this->cameraFarPlaneParam.ResetDirty();
                //    result = true;
                //}
                //  if (this->_cameraHalfApertureDegreesParam.IsDirty()) {
                //      auto val = this->_cameraHalfApertureDegreesParam.Param<param::FloatParam>()->Value();
                //      this->_camera.half_aperture_angle_radians(val * M_PI / 180.0f);
                //      this->_cameraHalfApertureDegreesParam.ResetDirty();
                //      result = true;
                //  }
            }

            /** Invisible parameters for lua manipulation */
            param::ParamSlot _cameraPositionParam;
            param::ParamSlot _cameraOrientationParam;
            param::ParamSlot _cameraProjectionTypeParam;
            param::ParamSlot _cameraNearPlaneParam;
            param::ParamSlot _cameraFarPlaneParam;
            param::ParamSlot _cameraHalfApertureDegreesParam;

            /** Standard camera views */
            param::ParamSlot _cameraViewOrientationParam;
            param::ParamSlot _cameraSetViewChooserParam;
            param::ParamSlot _cameraSetOrientationChooserParam;
        };

        struct Camera2DParameters {
            std::vector<AbstractSlot*> getParameterSlots() {
                return {};
            }
        };

    } // namespace view
} // namespace core
} // namespace megamol

#endif // !CAMERA_PARAMETER_SLOTS_H_INCLUDED
