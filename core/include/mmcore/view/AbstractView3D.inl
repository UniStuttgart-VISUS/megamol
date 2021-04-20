/*
 * AbstractView3D<FBO_TYPE, resize_func, CAM_CONTROLLER_TYPE, CAM_PARAMS_TYPE>.cpp
 *
 * Copyright (C) 2021 by VISUS (Universitaet Stuttgart).
 * Alle Rechte vorbehalten.
 */

using namespace megamol::core::view;

/*
 * AbstractView3D<FBO_TYPE, resize_func, CAM_CONTROLLER_TYPE, CAM_PARAMS_TYPE>::AbstractView3D<FBO_TYPE, resize_func, CAM_CONTROLLER_TYPE, CAM_PARAMS_TYPE>
 */
template<typename FBO_TYPE, RESIZEFUNC<typename FBO_TYPE> resize_func, typename CAM_CONTROLLER_TYPE,
    typename CAM_PARAMS_TYPE>
AbstractView3D<FBO_TYPE, resize_func, CAM_CONTROLLER_TYPE, CAM_PARAMS_TYPE>::AbstractView3D<FBO_TYPE, resize_func,
    CAM_CONTROLLER_TYPE, CAM_PARAMS_TYPE>(void)
        : BaseView<FBO_TYPE, resize_func, CAM_CONTROLLER_TYPE, CAM_PARAMS_TYPE>() {
}

/*
 * AbstractView3D<FBO_TYPE, resize_func, CAM_CONTROLLER_TYPE, CAM_PARAMS_TYPE>::~AbstractView3D<FBO_TYPE, resize_func, CAM_CONTROLLER_TYPE, CAM_PARAMS_TYPE>
 */
template<typename FBO_TYPE, RESIZEFUNC<typename FBO_TYPE> resize_func, typename CAM_CONTROLLER_TYPE,
    typename CAM_PARAMS_TYPE>
AbstractView3D<FBO_TYPE, resize_func, CAM_CONTROLLER_TYPE, CAM_PARAMS_TYPE>::~AbstractView3D<FBO_TYPE, resize_func, CAM_CONTROLLER_TYPE, CAM_PARAMS_TYPE>(void) {
    this->Release();
}

/*
 * AbstractView3D<FBO_TYPE, resize_func, CAM_CONTROLLER_TYPE, CAM_PARAMS_TYPE>::ResetView
 */
template<typename FBO_TYPE, RESIZEFUNC<typename FBO_TYPE> resize_func, typename CAM_CONTROLLER_TYPE,
    typename CAM_PARAMS_TYPE>
void AbstractView3D<FBO_TYPE, resize_func, CAM_CONTROLLER_TYPE, CAM_PARAMS_TYPE>::ResetView(float window_aspect) {
    if (this->_cameraIsMutable) { // check if view is in control of the camera
        Camera::PerspectiveParameters cam_intrinsics;
        cam_intrinsics.near_plane = 0.1f;
        cam_intrinsics.far_plane = 100.0f;
        cam_intrinsics.fovy = 0.5;
        cam_intrinsics.aspect = window_aspect;
        cam_intrinsics.image_plane_tile = Camera::ImagePlaneTile(); // view is in control -> no tiling -> use default tile values

        if (!this->_bboxs.IsBoundingBoxValid()) {
            this->_bboxs.SetBoundingBox(-1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f);
        }
        float dist = (0.5f * sqrtf((this->_bboxs.BoundingBox().Width() * this->_bboxs.BoundingBox().Width()) +
                                   (this->_bboxs.BoundingBox().Depth() * this->_bboxs.BoundingBox().Depth()) +
                                   (this->_bboxs.BoundingBox().Height() * this->_bboxs.BoundingBox().Height()))) /
                     tanf(cam_intrinsics.fovy / 2.0f);
        double fovx = cam_intrinsics.fovy * cam_intrinsics.aspect;
        double distX = static_cast<double>(this->_bboxs.BoundingBox().Width()) / (2.0 * tan(fovx/2.0));
        double distY = static_cast<double>(this->_bboxs.BoundingBox().Height()) /
                       (2.0 * tan(static_cast<double>(cam_intrinsics.fovy / 2.0f)));
        dist = static_cast<float>((distX > distY) ? distX : distY);
        dist = dist + (this->_bboxs.BoundingBox().Depth() / 2.0f);
        auto bbc = this->_bboxs.BoundingBox().CalcCenter();

        auto bbcglm = glm::vec3(bbc.GetX(), bbc.GetY(), bbc.GetZ());

        Camera::Pose cam_pose;
        cam_pose.position = bbcglm + glm::vec3(0.0f, 0.0f, dist);
        cam_pose.direction = glm::vec3(0.0, 0.0, -1.0);
        cam_pose.up = glm::vec3(0.0, 1.0, 0.0);

        this->_camera = Camera(cam_pose,cam_intrinsics);

        this->_camera_controller.setRotationalCenter(glm::vec3(bbc.GetX(), bbc.GetY(), bbc.GetZ()));

        ////////////////

    //double pseudoWidth = this->_bboxs.BoundingBox().Width();
    //double pseudoHeight = this->_bboxs.BoundingBox().Height();
    //double pseudoDepth = this->_bboxs.BoundingBox().Depth();
    //auto dor_axis = glm::vec3(0.0f, 0.0f, 0.0f);
    //defaultview dv = static_cast<defaultview>(this->_cameraSetViewChooserParam.Param<param::EnumParam>()->Value());
    //switch (dv) {
    //case DEFAULTVIEW_FRONT:
    //    dor_axis = glm::vec3(0.0f, 0.0f, -1.0f);
    //    break;
    //case DEFAULTVIEW_BACK:
    //    dor_axis = glm::vec3(0.0f, 0.0f, 1.0f);
    //    break;
    //case DEFAULTVIEW_RIGHT:
    //    dor_axis = glm::vec3(-1.0f, 0.0f, 0.0f);
    //    pseudoWidth = this->_bboxs.BoundingBox().Depth();
    //    pseudoHeight = this->_bboxs.BoundingBox().Height();
    //    pseudoDepth = this->_bboxs.BoundingBox().Width();
    //    break;
    //case DEFAULTVIEW_LEFT:
    //    dor_axis = glm::vec3(1.0f, 0.0f, 0.0f);
    //    pseudoWidth = this->_bboxs.BoundingBox().Depth();
    //    pseudoHeight = this->_bboxs.BoundingBox().Height();
    //    pseudoDepth = this->_bboxs.BoundingBox().Width();
    //    break;
    //case DEFAULTVIEW_TOP:
    //    dor_axis = glm::vec3(0.0f, -1.0f, 0.0f);
    //    pseudoWidth = this->_bboxs.BoundingBox().Width();
    //    pseudoHeight = this->_bboxs.BoundingBox().Depth();
    //    pseudoDepth = this->_bboxs.BoundingBox().Height();
    //    break;
    //case DEFAULTVIEW_BOTTOM:
    //    dor_axis = glm::vec3(0.0f, 1.0f, 0.0f);
    //    pseudoWidth = this->_bboxs.BoundingBox().Width();
    //    pseudoHeight = this->_bboxs.BoundingBox().Depth();
    //    pseudoDepth = this->_bboxs.BoundingBox().Height();
    //    break;
    //default:;
    //}
    //auto dim = this->_camera.resolution_gate();
    //double halfFovX =
    //    (static_cast<double>(dim.width()) * static_cast<double>(this->_camera.aperture_angle_radians() / 2.0f)) /
    //    static_cast<double>(dim.height());
    //double distX = pseudoWidth / (2.0 * tan(halfFovX));
    //double distY = pseudoHeight / (2.0 * tan(static_cast<double>(this->_camera.aperture_angle_radians() / 2.0f)));
    //float dist = static_cast<float>((distX > distY) ? distX : distY);
    //dist = dist + (pseudoDepth / 2.0f);
    //auto bbc = this->_bboxs.BoundingBox().CalcCenter();
    //auto bbcglm = glm::vec4(bbc.GetX(), bbc.GetY(), bbc.GetZ(), 1.0f);
    //const double cos0 = 0.0;
    //const double cos45 = sqrt(2.0) / 2.0;
    //const double cos90 = 1.0;
    //const double sin0 = 1.0;
    //const double sin45 = cos45;
    //const double sin90 = 0.0;
    //defaultorientation dor =
    //    static_cast<defaultorientation>(this->_cameraSetOrientationChooserParam.Param<param::EnumParam>()->Value());
    //auto dor_rotation = cam_type::quaternion_type(0.0f, 0.0f, 0.0f, 1.0f);
    //switch (dor) {
    //case DEFAULTORIENTATION_TOP: // 0 degree
    //    break;
    //case DEFAULTORIENTATION_RIGHT: // 90 degree
    //    dor_axis *= sin45;
    //    dor_rotation = cam_type::quaternion_type(dor_axis.x, dor_axis.y, dor_axis.z, cos45);
    //    break;
    //case DEFAULTORIENTATION_BOTTOM: { // 180 degree
    //    // Using euler angles to get quaternion for 180 degree rotation
    //    glm::quat flip_quat = glm::quat(dor_axis * static_cast<float>(M_PI));
    //    dor_rotation = cam_type::quaternion_type(flip_quat.x, flip_quat.y, flip_quat.z, flip_quat.w);
    //} break;
    //case DEFAULTORIENTATION_LEFT: // 270 degree (= -90 degree)
    //    dor_axis *= -sin45;
    //    dor_rotation = cam_type::quaternion_type(dor_axis.x, dor_axis.y, dor_axis.z, cos45);
    //    break;
    //default:;
    //}
    //if (!this->_valuesFromOutside) {
    //    // quat rot(theta) around axis(x,y,z) -> q = (sin(theta/2)*x, sin(theta/2)*y, sin(theta/2)*z, cos(theta/2))
    //    switch (dv) {
    //    case DEFAULTVIEW_FRONT:
    //        this->_camera.orientation(dor_rotation * cam_type::quaternion_type::create_identity());
    //        this->_camera.position(bbcglm + glm::vec4(0.0f, 0.0f, dist, 0.0f));
    //        break;
    //    case DEFAULTVIEW_BACK: // 180 deg around y axis
    //        this->_camera.orientation(dor_rotation * cam_type::quaternion_type(0, 1.0, 0, 0.0f));
    //        this->_camera.position(bbcglm + glm::vec4(0.0f, 0.0f, -dist, 0.0f));
    //        break;
    //    case DEFAULTVIEW_RIGHT: // 90 deg around y axis
    //        this->_camera.orientation(dor_rotation * cam_type::quaternion_type(0, sin45 * 1.0, 0, cos45));
    //        this->_camera.position(bbcglm + glm::vec4(dist, 0.0f, 0.0f, 0.0f));
    //        break;
    //    case DEFAULTVIEW_LEFT: // 90 deg reverse around y axis
    //        this->_camera.orientation(dor_rotation * cam_type::quaternion_type(0, -sin45 * 1.0, 0, cos45));
    //        this->_camera.position(bbcglm + glm::vec4(-dist, 0.0f, 0.0f, 0.0f));
    //        break;
    //    case DEFAULTVIEW_TOP: // 90 deg around x axis
    //        this->_camera.orientation(dor_rotation * cam_type::quaternion_type(-sin45 * 1.0, 0, 0, cos45));
    //        this->_camera.position(bbcglm + glm::vec4(0.0f, dist, 0.0f, 0.0f));
    //        break;
    //    case DEFAULTVIEW_BOTTOM: // 90 deg reverse around x axis
    //        this->_camera.orientation(dor_rotation * cam_type::quaternion_type(sin45 * 1.0, 0, 0, cos45));
    //        this->_camera.position(bbcglm + glm::vec4(0.0f, -dist, 0.0f, 0.0f));
    //        break;
    //    default:;
    //    }
    //}
    //
    //this->_rotCenter = glm::vec3(bbc.GetX(), bbc.GetY(), bbc.GetZ());

    ///////////////////////

    } else {
        // TODO print warning
    }
}

/*
 * AbstractView3D<FBO_TYPE, resize_func, CAM_CONTROLLER_TYPE, CAM_PARAMS_TYPE>::OnRenderView
 */
template<typename FBO_TYPE, RESIZEFUNC<typename FBO_TYPE> resize_func, typename CAM_CONTROLLER_TYPE,
    typename CAM_PARAMS_TYPE>
bool AbstractView3D<FBO_TYPE, resize_func, CAM_CONTROLLER_TYPE, CAM_PARAMS_TYPE>::OnRenderView(Call& call) {
    AbstractCallRenderView* crv = dynamic_cast<AbstractCallRenderView*>(&call);
    if (crv == nullptr) return false;

    double time = crv->Time();
    if (time < 0.0f) time = this->DefaultTime(crv->InstanceTime());
    double instanceTime = crv->InstanceTime();

    this->Render(time, instanceTime, false);

    return true;
}

/*
 * AbstractView3D<FBO_TYPE, resize_func, CAM_CONTROLLER_TYPE, CAM_PARAMS_TYPE>::create
 */
template<typename FBO_TYPE, RESIZEFUNC<typename FBO_TYPE> resize_func, typename CAM_CONTROLLER_TYPE,
    typename CAM_PARAMS_TYPE>
bool AbstractView3D<FBO_TYPE, resize_func, CAM_CONTROLLER_TYPE, CAM_PARAMS_TYPE>::create(void) {
    const auto arcball_key = "arcball";

    if (!this->GetCoreInstance()->IsmmconsoleFrontendCompatible()) {
        // new frontend has global key-value resource
        auto maybe = this->frontend_resources[0].getResource<megamol::frontend_resources::GlobalValueStore>().maybe_get(arcball_key);
        if (maybe.has_value()) {
            this->_camera_controller.setArcballDefault(vislib::CharTraitsA::ParseBool(maybe.value().c_str()));
        }

    } else {
        mmcValueType wpType;
        this->_camera_controller.setArcballDefault(false);
        auto value = this->GetCoreInstance()->Configuration().GetValue(MMC_CFGID_VARIABLE, _T(arcball_key), &wpType);
        if (value != nullptr) {
            try {
                switch (wpType) {
                case MMC_TYPE_BOOL:
                    this->_camera_controller.setArcballDefault(*static_cast<const bool*>(value));
                    break;

                case MMC_TYPE_CSTR:
                    this->_camera_controller.setArcballDefault(vislib::CharTraitsA::ParseBool(static_cast<const char*>(value)));
                    break;

                case MMC_TYPE_WSTR:
                    this->_camera_controller.setArcballDefault(vislib::CharTraitsW::ParseBool(static_cast<const wchar_t*>(value)));
                    break;
                }
            } catch (...) {}
        }
    }

    this->_firstImg = true;
    return true;
}

/*
 * AbstractView3D<FBO_TYPE, resize_func, CAM_CONTROLLER_TYPE, CAM_PARAMS_TYPE>::cameraOvrCallback
 */
template<typename FBO_TYPE, RESIZEFUNC<typename FBO_TYPE> resize_func, typename CAM_CONTROLLER_TYPE,
    typename CAM_PARAMS_TYPE>
bool AbstractView3D<FBO_TYPE, resize_func, CAM_CONTROLLER_TYPE, CAM_PARAMS_TYPE>::cameraOvrCallback(param::ParamSlot& p) {
    auto up_vis = this->_cameraOvrUpParam.Param<param::Vector3fParam>()->Value();
    auto lookat_vis = this->_cameraOvrLookatParam.Param<param::Vector3fParam>()->Value();

    glm::vec3 up(up_vis.X(), up_vis.Y(), up_vis.Z());
    up = glm::normalize(up);
    glm::vec3 lookat(lookat_vis.X(), lookat_vis.Y(), lookat_vis.Z());

    auto cam_pose = this->_camera.get<Camera::Pose>();
    glm::mat3 view;
    view[2] = -glm::normalize(lookat - cam_pose.position);
    view[0] = glm::normalize(glm::cross(up, view[2]));
    view[1] = glm::normalize(glm::cross(view[2], view[0]));

    auto orientation = glm::quat_cast(view);

    this->_camera.setPose(Camera::Pose(cam_pose.position,orientation));
    this->_rotCenter = lookat;

    return true;
}
