/*
 * AbstractView3D.cpp
 *
 * Copyright (C) 2021 by VISUS (Universitaet Stuttgart).
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "mmcore/view/AbstractView3D.h"
#ifdef _WIN32
#    include <windows.h>
#endif /* _WIN32 */
#include <chrono>
#include <fstream>
#include <glm/gtx/string_cast.hpp>
#include "mmcore/CoreInstance.h"
#include "mmcore/param/BoolParam.h"
#include "mmcore/param/ButtonParam.h"
#include "mmcore/param/EnumParam.h"
#include "mmcore/param/FloatParam.h"
#include "mmcore/param/Vector2fParam.h"
#include "mmcore/param/Vector3fParam.h"
#include "mmcore/param/Vector4fParam.h"
#include "vislib/math/Point.h"
#include "vislib/math/Quaternion.h"
#include "vislib/math/Vector.h"
#include "vislib/sys/KeyCode.h"
#include "vislib/sys/sysfunctions.h"
#include "mmcore/view/CallRenderView.h"

#include "glm/gtc/matrix_transform.hpp"

using namespace megamol::core::view;

/*
 * AbstractView3D::AbstractView3D
 */
AbstractView3D::AbstractView3D(void)
        : AbstractView()
    , _showLookAt("showLookAt", "Flag showing the look at point")
    , _stereoFocusDistSlot("stereo::focusDist", "focus distance for stereo projection")
    , _stereoEyeDistSlot("stereo::eyeDist", "eye distance for stereo projection")
    , _viewKeyMoveStepSlot("viewKey::MoveStep", "The move step size in world coordinates")
    , _viewKeyRunFactorSlot("viewKey::RunFactor", "The factor for step size multiplication when running (shift)")
    , _viewKeyAngleStepSlot("viewKey::AngleStep", "The angle rotate step in degrees")
    , _viewKeyFixToWorldUpSlot("viewKey::FixToWorldUp","Fix rotation manipulator to world up vector")
    , _mouseSensitivitySlot("viewKey::MouseSensitivity", "used for WASD mode")
    , _viewKeyRotPointSlot("viewKey::RotPoint", "The point around which the view will be rotated")
    , _enableMouseSelectionSlot("enableMouseSelection", "Enable selecting and picking with the mouse")
    , _hookOnChangeOnlySlot("hookOnChange", "whether post-hooks are triggered when the frame would be identical")
    , _cameraSetViewChooserParam("view::defaultView", "Choose a default view to look from")
    , _cameraSetOrientationChooserParam("view::defaultOrientation", "Choose a default orientation to look from")
    , _cameraViewOrientationParam("view::cubeOrientation", "Current camera orientation used for view cube.")
    , _showViewCubeParam("view::showViewCube", "Shows view cube.")
    , _cameraPositionParam("cam::position", "")
    , _cameraOrientationParam("cam::orientation", "")
    , _cameraProjectionTypeParam("cam::projectiontype", "")
    , _cameraNearPlaneParam("cam::nearplane", "")
    , _cameraFarPlaneParam("cam::farplane", "")
    , _cameraConvergencePlaneParam("cam::convergenceplane", "")
    , _cameraEyeParam("cam::eye", "")
    , _cameraGateScalingParam("cam::gatescaling", "")
    , _cameraFilmGateParam("cam::filmgate", "")
    , _cameraResolutionXParam("cam::resgate::x", "")
    , _cameraResolutionYParam("cam::resgate::y", "")
    , _cameraCenterOffsetParam("cam::centeroffset", "")
    , _cameraHalfApertureDegreesParam("cam::halfaperturedegrees", "")
    , _cameraHalfDisparityParam("cam::halfdisparity", "")
    , _cameraOvrUpParam("cam::ovr::up", "")
    , _cameraOvrLookatParam("cam::ovr::lookat", "")
    , _cameraOvrParam("cam::ovr::override", "")
    , _valuesFromOutside(false)
    , _cameraControlOverrideActive(false) {

    using vislib::sys::KeyCode;

    this->_camera.resolution_gate(cam_type::screen_size_type(100, 100));
    this->_camera.image_tile(cam_type::screen_rectangle_type(std::array<int, 4>{0, 100, 100, 0}));

    this->_showLookAt.SetParameter(new param::BoolParam(false));
    this->MakeSlotAvailable(&this->_showLookAt);

    this->_viewKeyMoveStepSlot.SetParameter(new param::FloatParam(0.5f, 0.001f));
    this->MakeSlotAvailable(&this->_viewKeyMoveStepSlot);

    this->_viewKeyRunFactorSlot.SetParameter(new param::FloatParam(2.0f, 0.1f));
    this->MakeSlotAvailable(&this->_viewKeyRunFactorSlot);

    this->_viewKeyAngleStepSlot.SetParameter(new param::FloatParam(90.0f, 0.1f, 360.0f));
    this->MakeSlotAvailable(&this->_viewKeyAngleStepSlot);

    this->_viewKeyFixToWorldUpSlot.SetParameter(new param::BoolParam(true));
    this->MakeSlotAvailable(&this->_viewKeyFixToWorldUpSlot);

    this->_mouseSensitivitySlot.SetParameter(new param::FloatParam(3.0f, 0.001f, 10.0f));
    this->_mouseSensitivitySlot.SetUpdateCallback(&AbstractView3D::mouseSensitivityChanged);
    this->MakeSlotAvailable(&this->_mouseSensitivitySlot);

    // TODO clean up vrpsev memory after use
    param::EnumParam* vrpsev = new param::EnumParam(1);
    vrpsev->SetTypePair(0, "Position");
    vrpsev->SetTypePair(1, "Look-At");
    this->_viewKeyRotPointSlot.SetParameter(vrpsev);
    this->MakeSlotAvailable(&this->_viewKeyRotPointSlot);

    this->_enableMouseSelectionSlot.SetParameter(new param::ButtonParam(Key::KEY_TAB));
    this->_enableMouseSelectionSlot.SetUpdateCallback(&AbstractView3D::onToggleButton);
    this->MakeSlotAvailable(&this->_enableMouseSelectionSlot);

    this->_hookOnChangeOnlySlot.SetParameter(new param::BoolParam(false));
    this->MakeSlotAvailable(&this->_hookOnChangeOnlySlot);

    const bool camparamvisibility = true;

    auto camposparam = new param::Vector3fParam(vislib::math::Vector<float, 3>());
    camposparam->SetGUIVisible(camparamvisibility);
    this->_cameraPositionParam.SetParameter(camposparam);
    this->MakeSlotAvailable(&this->_cameraPositionParam);

    auto camorientparam = new param::Vector4fParam(vislib::math::Vector<float, 4>());
    camorientparam->SetGUIVisible(camparamvisibility);
    this->_cameraOrientationParam.SetParameter(camorientparam);
    this->MakeSlotAvailable(&this->_cameraOrientationParam);

    auto projectionParam = new param::EnumParam(static_cast<int>(core::thecam::Projection_type::perspective));
    projectionParam->SetTypePair(static_cast<int>(core::thecam::Projection_type::perspective), "Perspective");
    projectionParam->SetTypePair(static_cast<int>(core::thecam::Projection_type::orthographic), "Orthographic");
    projectionParam->SetTypePair(static_cast<int>(core::thecam::Projection_type::parallel), "Parallel");
    projectionParam->SetTypePair(static_cast<int>(core::thecam::Projection_type::off_axis), "Off-Axis");
    projectionParam->SetTypePair(static_cast<int>(core::thecam::Projection_type::converged), "Converged");
    projectionParam->SetGUIVisible(camparamvisibility);
    this->_cameraProjectionTypeParam.SetParameter(projectionParam);
    this->MakeSlotAvailable(&this->_cameraProjectionTypeParam);

    auto camconvergenceparam = new param::FloatParam(0.1f, 0.0f);
    camconvergenceparam->SetGUIVisible(camparamvisibility);
    this->_cameraConvergencePlaneParam.SetParameter(camconvergenceparam);
    this->MakeSlotAvailable(&this->_cameraConvergencePlaneParam);

    auto centeroffsetparam = new param::Vector2fParam(vislib::math::Vector<float, 2>());
    centeroffsetparam->SetGUIVisible(camparamvisibility);
    this->_cameraCenterOffsetParam.SetParameter(centeroffsetparam);
    this->MakeSlotAvailable(&this->_cameraCenterOffsetParam);

    auto apertureparam = new param::FloatParam(35.0f, 0.0f);
    apertureparam->SetGUIVisible(camparamvisibility);
    this->_cameraHalfApertureDegreesParam.SetParameter(apertureparam);
    this->MakeSlotAvailable(&this->_cameraHalfApertureDegreesParam);

    auto disparityparam = new param::FloatParam(1.0f, 0.0f);
    disparityparam->SetGUIVisible(camparamvisibility);
    this->_cameraHalfDisparityParam.SetParameter(disparityparam);
    this->MakeSlotAvailable(&this->_cameraHalfDisparityParam);

    this->_cameraOvrUpParam << new param::Vector3fParam(vislib::math::Vector<float, 3>());
    this->MakeSlotAvailable(&this->_cameraOvrUpParam);

    this->_cameraOvrLookatParam << new param::Vector3fParam(vislib::math::Vector<float, 3>());
    this->MakeSlotAvailable(&this->_cameraOvrLookatParam);

    this->_cameraOvrParam << new param::ButtonParam();
    this->_cameraOvrParam.SetUpdateCallback(&AbstractView3D::cameraOvrCallback);
    this->MakeSlotAvailable(&this->_cameraOvrParam);

    this->_translateManipulator.set_target(this->_camera);
    this->_translateManipulator.enable();

    this->_rotateManipulator.set_target(this->_camera);
    this->_rotateManipulator.enable();

    this->_arcballManipulator.set_target(this->_camera);
    this->_arcballManipulator.enable();
    this->_rotCenter = glm::vec3(0.0f, 0.0f, 0.0f);

    this->_turntableManipulator.set_target(this->_camera);
    this->_turntableManipulator.enable();

    this->_orbitAltitudeManipulator.set_target(this->_camera);
    this->_orbitAltitudeManipulator.enable();

    auto defaultViewParam = new param::EnumParam(0);
    defaultViewParam->SetTypePair(defaultview::DEFAULTVIEW_FRONT, "Front");
    defaultViewParam->SetTypePair(defaultview::DEFAULTVIEW_BACK, "Back");
    defaultViewParam->SetTypePair(defaultview::DEFAULTVIEW_RIGHT, "Right");
    defaultViewParam->SetTypePair(defaultview::DEFAULTVIEW_LEFT, "Left");
    defaultViewParam->SetTypePair(defaultview::DEFAULTVIEW_TOP, "Top");
    defaultViewParam->SetTypePair(defaultview::DEFAULTVIEW_BOTTOM, "Bottom");
    defaultViewParam->SetGUIVisible(camparamvisibility);
    this->_cameraSetViewChooserParam.SetParameter(defaultViewParam),
        this->MakeSlotAvailable(&this->_cameraSetViewChooserParam);
    this->_cameraSetViewChooserParam.SetUpdateCallback(&AbstractView::OnResetView);

    auto defaultOrientationParam = new param::EnumParam(0);
    defaultOrientationParam->SetTypePair(defaultorientation::DEFAULTORIENTATION_TOP, "Top");
    defaultOrientationParam->SetTypePair(defaultorientation::DEFAULTORIENTATION_RIGHT, "Right");
    defaultOrientationParam->SetTypePair(defaultorientation::DEFAULTORIENTATION_BOTTOM, "Bottom");
    defaultOrientationParam->SetTypePair(defaultorientation::DEFAULTORIENTATION_LEFT, "Left");
    defaultOrientationParam->SetGUIVisible(camparamvisibility);
    this->_cameraSetOrientationChooserParam.SetParameter(defaultOrientationParam),
        this->MakeSlotAvailable(&this->_cameraSetOrientationChooserParam);
    this->_cameraSetOrientationChooserParam.SetUpdateCallback(&AbstractView::OnResetView);

    this->_cameraViewOrientationParam.SetParameter(
        new param::Vector4fParam(vislib::math::Vector<float, 4>(0.0f, 0.0f, 0.0f, 1.0f)));
    this->MakeSlotAvailable(&this->_cameraViewOrientationParam);
    this->_cameraViewOrientationParam.Parameter()->SetGUIReadOnly(true);
    this->_cameraViewOrientationParam.Parameter()->SetGUIVisible(false);

    this->_showViewCubeParam.SetParameter(new param::BoolParam(false));
    this->MakeSlotAvailable(&this->_showViewCubeParam);

    // none of the saved camera states are valid right now
    for (auto& e : this->_savedCameras) {
        e.second = false;
    }

    this->ResetView();
}

/*
 * AbstractView3D::~AbstractView3D
 */
AbstractView3D::~AbstractView3D(void) {
    this->Release();
}

/*
 * AbstractView3D::GetCameraSyncNumber
 */
unsigned int AbstractView3D::GetCameraSyncNumber(void) const {
    // TODO implement
    return 0;
}

/*
 * AbstractView3D::beforeRender
 */
void AbstractView3D::beforeRender(const mmcRenderViewContext& context) {

    AbstractView::beforeRender(context);

    // get camera values from params(?)
    this->adaptCameraValues(this->_camera);

    // handle 3D view specific camera implementation
    this->handleCameraMovement();

    // camera settings
    if (this->_stereoEyeDistSlot.IsDirty()) {
        // TODO
        param::FloatParam* fp = this->_stereoEyeDistSlot.Param<param::FloatParam>();
        // this->camParams->SetStereoDisparity(fp->Value());
        // fp->SetValue(this->camParams->StereoDisparity());
        this->_stereoEyeDistSlot.ResetDirty();
    }
    if (this->_stereoFocusDistSlot.IsDirty()) {
        // TODO
        param::FloatParam* fp = this->_stereoFocusDistSlot.Param<param::FloatParam>();
        // this->camParams->SetFocalDistance(fp->Value());
        // fp->SetValue(this->camParams->FocalDistance(false));
        this->_stereoFocusDistSlot.ResetDirty();
    }

    // set current camera orientation for view cube 
    auto cam_orientation = static_cast<glm::quat>(this->_camera.orientation());
    this->_cameraViewOrientationParam.Param<param::Vector4fParam>()->SetValue(
        vislib::math::Vector<float, 4>(cam_orientation.x, cam_orientation.y, cam_orientation.z, cam_orientation.w));
}

/*
 * AbstractView3D::afterRender
 */
void AbstractView3D::afterRender(const mmcRenderViewContext& context) {
    // set camera values to params
    this->setCameraValues(this->_camera);

    AbstractView::afterRender(context);
}

/*
 * AbstractView3D::ResetView
 */
void AbstractView3D::ResetView(void) {
    if (!this->_valuesFromOutside) {
        this->_camera.near_clipping_plane(0.1f);
        this->_camera.far_clipping_plane(100.0f);
        this->_camera.aperture_angle(30.0f);
        this->_camera.disparity(0.05f);
        this->_camera.eye(thecam::Eye::mono);
        this->_camera.projection_type(thecam::Projection_type::perspective);
    }
    // TODO set distance between eyes
    if (!this->_bboxs.IsBoundingBoxValid()) {
        this->_bboxs.SetBoundingBox(-1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f);
    }
    double pseudoWidth = this->_bboxs.BoundingBox().Width();
    double pseudoHeight = this->_bboxs.BoundingBox().Height();
    double pseudoDepth = this->_bboxs.BoundingBox().Depth();
    auto dor_axis = glm::vec3(0.0f, 0.0f, 0.0f);
    defaultview dv = static_cast<defaultview>(this->_cameraSetViewChooserParam.Param<param::EnumParam>()->Value());
    switch (dv) {
    case DEFAULTVIEW_FRONT:
        dor_axis = glm::vec3(0.0f, 0.0f, -1.0f);
        break;
    case DEFAULTVIEW_BACK:
        dor_axis = glm::vec3(0.0f, 0.0f, 1.0f);
        break;
    case DEFAULTVIEW_RIGHT:
        dor_axis = glm::vec3(-1.0f, 0.0f, 0.0f);
        pseudoWidth = this->_bboxs.BoundingBox().Depth();
        pseudoHeight = this->_bboxs.BoundingBox().Height();
        pseudoDepth = this->_bboxs.BoundingBox().Width();
        break;
    case DEFAULTVIEW_LEFT:
        dor_axis = glm::vec3(1.0f, 0.0f, 0.0f);
        pseudoWidth = this->_bboxs.BoundingBox().Depth();
        pseudoHeight = this->_bboxs.BoundingBox().Height();
        pseudoDepth = this->_bboxs.BoundingBox().Width();
        break;
    case DEFAULTVIEW_TOP:
        dor_axis = glm::vec3(0.0f, -1.0f, 0.0f);
        pseudoWidth = this->_bboxs.BoundingBox().Width();
        pseudoHeight = this->_bboxs.BoundingBox().Depth();
        pseudoDepth = this->_bboxs.BoundingBox().Height();
        break;
    case DEFAULTVIEW_BOTTOM:
        dor_axis = glm::vec3(0.0f, 1.0f, 0.0f);
        pseudoWidth = this->_bboxs.BoundingBox().Width();
        pseudoHeight = this->_bboxs.BoundingBox().Depth();
        pseudoDepth = this->_bboxs.BoundingBox().Height();
        break;
    default:;
    }
    auto dim = this->_camera.resolution_gate();
    double halfFovX =
        (static_cast<double>(dim.width()) * static_cast<double>(this->_camera.aperture_angle_radians() / 2.0f)) /
        static_cast<double>(dim.height());
    double distX = pseudoWidth / (2.0 * tan(halfFovX));
    double distY = pseudoHeight / (2.0 * tan(static_cast<double>(this->_camera.aperture_angle_radians() / 2.0f)));
    float dist = static_cast<float>((distX > distY) ? distX : distY);
    dist = dist + (pseudoDepth / 2.0f);
    auto bbc = this->_bboxs.BoundingBox().CalcCenter();
    auto bbcglm = glm::vec4(bbc.GetX(), bbc.GetY(), bbc.GetZ(), 1.0f);
    const double cos0 = 0.0;
    const double cos45 = sqrt(2.0) / 2.0;
    const double cos90 = 1.0;
    const double sin0 = 1.0;
    const double sin45 = cos45;
    const double sin90 = 0.0;
    defaultorientation dor =
        static_cast<defaultorientation>(this->_cameraSetOrientationChooserParam.Param<param::EnumParam>()->Value());
    auto dor_rotation = cam_type::quaternion_type(0.0f, 0.0f, 0.0f, 1.0f);
    switch (dor) {
    case DEFAULTORIENTATION_TOP: // 0 degree
        break;
    case DEFAULTORIENTATION_RIGHT: // 90 degree
        dor_axis *= sin45;
        dor_rotation = cam_type::quaternion_type(dor_axis.x, dor_axis.y, dor_axis.z, cos45);
        break;
    case DEFAULTORIENTATION_BOTTOM: { // 180 degree
        // Using euler angles to get quaternion for 180 degree rotation
        glm::quat flip_quat = glm::quat(dor_axis * static_cast<float>(M_PI));
        dor_rotation = cam_type::quaternion_type(flip_quat.x, flip_quat.y, flip_quat.z, flip_quat.w);
    } break;
    case DEFAULTORIENTATION_LEFT: // 270 degree (= -90 degree)
        dor_axis *= -sin45;
        dor_rotation = cam_type::quaternion_type(dor_axis.x, dor_axis.y, dor_axis.z, cos45);
        break;
    default:;
    }
    if (!this->_valuesFromOutside) {
        // quat rot(theta) around axis(x,y,z) -> q = (sin(theta/2)*x, sin(theta/2)*y, sin(theta/2)*z, cos(theta/2))
        switch (dv) {
        case DEFAULTVIEW_FRONT:
            this->_camera.orientation(dor_rotation * cam_type::quaternion_type::create_identity());
            this->_camera.position(bbcglm + glm::vec4(0.0f, 0.0f, dist, 0.0f));
            break;
        case DEFAULTVIEW_BACK: // 180 deg around y axis
            this->_camera.orientation(dor_rotation * cam_type::quaternion_type(0, 1.0, 0, 0.0f));
            this->_camera.position(bbcglm + glm::vec4(0.0f, 0.0f, -dist, 0.0f));
            break;
        case DEFAULTVIEW_RIGHT: // 90 deg around y axis
            this->_camera.orientation(dor_rotation * cam_type::quaternion_type(0, sin45 * 1.0, 0, cos45));
            this->_camera.position(bbcglm + glm::vec4(dist, 0.0f, 0.0f, 0.0f));
            break;
        case DEFAULTVIEW_LEFT: // 90 deg reverse around y axis
            this->_camera.orientation(dor_rotation * cam_type::quaternion_type(0, -sin45 * 1.0, 0, cos45));
            this->_camera.position(bbcglm + glm::vec4(-dist, 0.0f, 0.0f, 0.0f));
            break;
        case DEFAULTVIEW_TOP: // 90 deg around x axis
            this->_camera.orientation(dor_rotation * cam_type::quaternion_type(-sin45 * 1.0, 0, 0, cos45));
            this->_camera.position(bbcglm + glm::vec4(0.0f, dist, 0.0f, 0.0f));
            break;
        case DEFAULTVIEW_BOTTOM: // 90 deg reverse around x axis
            this->_camera.orientation(dor_rotation * cam_type::quaternion_type(sin45 * 1.0, 0, 0, cos45));
            this->_camera.position(bbcglm + glm::vec4(0.0f, -dist, 0.0f, 0.0f));
            break;
        default:;
        }
    }

    this->_rotCenter = glm::vec3(bbc.GetX(), bbc.GetY(), bbc.GetZ());

    // TODO Further manipulators? better value?
    this->_valuesFromOutside = false;
}

/*
 * AbstractView3D::OnRenderView
 */
bool AbstractView3D::OnRenderView(Call& call) {
    AbstractCallRenderView* crv = dynamic_cast<AbstractCallRenderView*>(&call);
    if (crv == nullptr) return false;

    float time = crv->Time();
    if (time < 0.0f) time = this->DefaultTime(crv->InstanceTime());
    mmcRenderViewContext context;
    ::ZeroMemory(&context, sizeof(context));
    context.Time = time;
    context.InstanceTime = crv->InstanceTime();

    if (crv->IsTileSet()) {
        this->_camera.resolution_gate(cam_type::screen_size_type(crv->VirtualWidth(), crv->VirtualHeight()));
        this->_camera.image_tile(cam_type::screen_rectangle_type::from_bottom_left(
            crv->TileX(), crv->TileY(), crv->TileWidth(), crv->TileHeight()));
    }

    this->Render(context, &call);

    return true;
}

/*
 * AbstractView3D::UpdateFreeze
 */
void AbstractView3D::UpdateFreeze(bool freeze) {
    // intentionally empty?
}

/*
 * AbstractView3D::OnKey
 */
bool AbstractView3D::OnKey(Key key, KeyAction action, Modifiers mods) {
    auto* cr = this->_rhsRenderSlot.CallAs<AbstractCallRender>();
    if (cr != nullptr) {
        InputEvent evt;
        evt.tag = InputEvent::Tag::Key;
        evt.keyData.key = key;
        evt.keyData.action = action;
        evt.keyData.mods = mods;
        cr->SetInputEvent(evt);
        if ((*cr)(AbstractCallRender::FnOnKey))
            return true;
    }

    if (action == KeyAction::PRESS || action == KeyAction::REPEAT) {
        this->_pressedKeyMap[key] = true;
    } else if (action == KeyAction::RELEASE) {
        this->_pressedKeyMap[key] = false;
    }

    if (key == Key::KEY_LEFT_ALT || key == Key::KEY_RIGHT_ALT) {
        if (action == KeyAction::PRESS || action == KeyAction::REPEAT) {
            this->modkeys.set(Modifier::ALT);
            _cameraControlOverrideActive = true;
        } else if (action == KeyAction::RELEASE) {
            this->modkeys.reset(Modifier::ALT);
            _cameraControlOverrideActive = false;
        }
    }
    if (key == Key::KEY_LEFT_SHIFT || key == Key::KEY_RIGHT_SHIFT) {
        if (action == KeyAction::PRESS || action == KeyAction::REPEAT) {
            this->modkeys.set(Modifier::SHIFT);
        } else if (action == KeyAction::RELEASE) {
            this->modkeys.reset(Modifier::SHIFT);
        }
    }
    if (key == Key::KEY_LEFT_CONTROL || key == Key::KEY_RIGHT_CONTROL) {
        if (action == KeyAction::PRESS || action == KeyAction::REPEAT) {
            this->modkeys.set(Modifier::CTRL);
            _cameraControlOverrideActive = true;
        } else if (action == KeyAction::RELEASE) {
            this->modkeys.reset(Modifier::CTRL);
            _cameraControlOverrideActive = false;
        }
    }

    if (action == KeyAction::PRESS && (key >= Key::KEY_0 && key <= Key::KEY_9)) {
        int index =
            static_cast<int>(key) - static_cast<int>(Key::KEY_0); // ugly hack, maybe this can be done better
        index = (index - 1) % 10;                                       // put key '1' at index 0
        index = index < 0 ? index + 10 : index;                         // wrap key '0' to a positive index '9'

        if (mods.test(Modifier::CTRL)) {
            this->_savedCameras[index].first = this->_camera.get_minimal_state(this->_savedCameras[index].first);
            this->_savedCameras[index].second = true;
            if (this->_autoSaveCamSettingsSlot.Param<param::BoolParam>()->Value()) {
                this->onStoreCamera(this->_storeCameraSettingsSlot); // manually trigger the storing
            }
        } else {
            if (this->_savedCameras[index].second) {
                // As a change of camera position should not change the display resolution, we actively save and restore
                // the old value of the resolution
                auto oldResolution = this->_camera.resolution_gate; // save old resolution
                this->_camera = this->_savedCameras[index].first;    // override current camera
                this->_camera.resolution_gate = oldResolution;      // restore old resolution
            }
        }
    }

    return false;
}

/*
 * AbstractView3D::OnChar
 */
bool AbstractView3D::OnChar(unsigned int codePoint) {
    auto* cr = this->_rhsRenderSlot.CallAs<AbstractCallRender>();
    if (cr == NULL) return false;

    InputEvent evt;
    evt.tag = InputEvent::Tag::Char;
    evt.charData.codePoint = codePoint;
    cr->SetInputEvent(evt);
    if (!(*cr)(AbstractCallRender::FnOnChar))
        return false;

    return true;
}

/*
 * AbstractView3D::release
 */
void AbstractView3D::release(void) { }

/*
 * AbstractView3D::mouseSensitivityChanged
 */
bool AbstractView3D::mouseSensitivityChanged(param::ParamSlot& p) { return true; }

/*
 * AbstractView3D::OnMouseButton
 */
 bool AbstractView3D::OnMouseButton(MouseButton button, MouseButtonAction action, Modifiers mods) {
     return true;
 }

/*
 * AbstractView3D::OnMouseMove
 */
bool AbstractView3D::OnMouseMove(double x, double y) {
    return true;
}

/*
 * AbstractView3D::OnMouseScroll
 */
bool AbstractView3D::OnMouseScroll(double dx, double dy) {
    return true;
}

/*
 * AbstractView3D::OnMouseScroll
 */
void AbstractView3D::unpackMouseCoordinates(float& x, float& y) {}

/*
 * AbstractView3D::create
 */
bool AbstractView3D::create(void) {
    mmcValueType wpType;
    this->_arcballDefault = false;
    auto value = this->GetCoreInstance()->Configuration().GetValue(MMC_CFGID_VARIABLE, _T("arcball"), &wpType);
    if (value != nullptr) {
        try {
            switch (wpType) {
            case MMC_TYPE_BOOL:
                this->_arcballDefault = *static_cast<const bool*>(value);
                break;

            case MMC_TYPE_CSTR:
                this->_arcballDefault = vislib::CharTraitsA::ParseBool(static_cast<const char*>(value));
                break;

            case MMC_TYPE_WSTR:
                this->_arcballDefault = vislib::CharTraitsW::ParseBool(static_cast<const wchar_t*>(value));
                break;
            }
        } catch (...) {}
    }
    this->_firstImg = true;
    return true;
}

/*
 * AbstractView3D::onToggleButton
 */
bool AbstractView3D::onToggleButton(param::ParamSlot& p) {
    // TODO implement
    return true;
}

/*
 * AbstractView3D::handleCameraMovement
 */
void AbstractView3D::handleCameraMovement(void) {
    float step = this->_viewKeyMoveStepSlot.Param<param::FloatParam>()->Value();
    float dt = std::chrono::duration<float>(this->_lastFrameDuration).count();
    step *= dt;

    const float runFactor = this->_viewKeyRunFactorSlot.Param<param::FloatParam>()->Value();
    if (this->modkeys.test(Modifier::SHIFT)) {
        step *= runFactor;
    }

    bool anymodpressed = !this->modkeys.none();
    float rotationStep = this->_viewKeyAngleStepSlot.Param<param::FloatParam>()->Value();
    rotationStep *= dt;

    glm::vec3 currCamPos(static_cast<glm::vec4>(this->_camera.eye_position()));
    float orbitalAltitude = glm::length(currCamPos - _rotCenter);

    if (this->_translateManipulator.manipulating()) {

        if (this->_pressedKeyMap.count(Key::KEY_W) > 0 && this->_pressedKeyMap[Key::KEY_W]) {
            this->_translateManipulator.move_forward(step);
        }
        if (this->_pressedKeyMap.count(Key::KEY_S) > 0 && this->_pressedKeyMap[Key::KEY_S]) {
            this->_translateManipulator.move_forward(-step);
        }
        if (this->_pressedKeyMap.count(Key::KEY_A) > 0 && this->_pressedKeyMap[Key::KEY_A]) {
            this->_translateManipulator.move_horizontally(-step);
        }
        if (this->_pressedKeyMap.count(Key::KEY_D) > 0 && this->_pressedKeyMap[Key::KEY_D]) {
            this->_translateManipulator.move_horizontally(step);
        }
        if (this->_pressedKeyMap.count(Key::KEY_C) > 0 && this->_pressedKeyMap[Key::KEY_C]) {
            this->_translateManipulator.move_vertically(step);
        }
        if (this->_pressedKeyMap.count(Key::KEY_V) > 0 && this->_pressedKeyMap[Key::KEY_V]) {
            this->_translateManipulator.move_vertically(-step);
        }
        if (this->_pressedKeyMap.count(Key::KEY_Q) > 0 && this->_pressedKeyMap[Key::KEY_Q]) {
            this->_rotateManipulator.roll(-rotationStep);
        }
        if (this->_pressedKeyMap.count(Key::KEY_E) > 0 && this->_pressedKeyMap[Key::KEY_E]) {
            this->_rotateManipulator.roll(rotationStep);
        }
        if (this->_pressedKeyMap.count(Key::KEY_UP) > 0 && this->_pressedKeyMap[Key::KEY_UP]) {
            this->_rotateManipulator.pitch(-rotationStep);
        }
        if (this->_pressedKeyMap.count(Key::KEY_DOWN) > 0 && this->_pressedKeyMap[Key::KEY_DOWN]) {
            this->_rotateManipulator.pitch(rotationStep);
        }
        if (this->_pressedKeyMap.count(Key::KEY_LEFT) > 0 && this->_pressedKeyMap[Key::KEY_LEFT]) {
            this->_rotateManipulator.yaw(rotationStep);
        }
        if (this->_pressedKeyMap.count(Key::KEY_RIGHT) > 0 && this->_pressedKeyMap[Key::KEY_RIGHT]) {
            this->_rotateManipulator.yaw(-rotationStep);
        }
    }

    glm::vec3 newCamPos(static_cast<glm::vec4>(this->_camera.eye_position()));
    glm::vec3 camDir(static_cast<glm::vec4>(this->_camera.view_vector()));
    _rotCenter = newCamPos + orbitalAltitude * glm::normalize(camDir);
}

/*
 * AbstractView3D::setCameraValues
 */
void AbstractView3D::setCameraValues(const Camera_2& cam) {
    glm::vec4 pos = cam.position();
    const bool makeDirty = false;
    this->_cameraPositionParam.Param<param::Vector3fParam>()->SetValue(
        vislib::math::Vector<float, 3>(pos.x, pos.y, pos.z), makeDirty);
    this->_cameraPositionParam.QueueUpdateNotification();

    glm::quat orient = cam.orientation();
    this->_cameraOrientationParam.Param<param::Vector4fParam>()->SetValue(
        vislib::math::Vector<float, 4>(orient.x, orient.y, orient.z, orient.w), makeDirty);
    this->_cameraOrientationParam.QueueUpdateNotification();

    this->_cameraProjectionTypeParam.Param<param::EnumParam>()->SetValue(static_cast<int>(cam.projection_type()), makeDirty);
    this->_cameraProjectionTypeParam.QueueUpdateNotification();

    this->_cameraConvergencePlaneParam.Param<param::FloatParam>()->SetValue(cam.convergence_plane(), makeDirty);
    this->_cameraConvergencePlaneParam.QueueUpdateNotification();

    /*this->cameraNearPlaneParam.Param<param::FloatParam>()->SetValue(cam.near_clipping_plane(), makeDirty);
    this->cameraFarPlaneParam.Param<param::FloatParam>()->SetValue(cam.far_clipping_plane(), makeDirty);*/
    /*this->cameraEyeParam.Param<param::EnumParam>()->SetValue(static_cast<int>(cam.eye()), makeDirty);
    this->cameraGateScalingParam.Param<param::EnumParam>()->SetValue(static_cast<int>(cam.gate_scaling()), makeDirty);
    this->cameraFilmGateParam.Param<param::Vector2fParam>()->SetValue(
        vislib::math::Vector<float, 2>(cam.film_gate().width(), cam.film_gate().height()), makeDirty);*/
    /*this->cameraResolutionXParam.Param<param::IntParam>()->SetValue(cam.resolution_gate().width());
    this->cameraResolutionYParam.Param<param::IntParam>()->SetValue(cam.resolution_gate().height());*/

    this->_cameraCenterOffsetParam.Param<param::Vector2fParam>()->SetValue(
        vislib::math::Vector<float, 2>(cam.centre_offset().x(), cam.centre_offset().y()), makeDirty);
    this->_cameraCenterOffsetParam.QueueUpdateNotification();

    this->_cameraHalfApertureDegreesParam.Param<param::FloatParam>()->SetValue(
        cam.half_aperture_angle_radians() * 180.0f / M_PI, makeDirty);
    this->_cameraHalfApertureDegreesParam.QueueUpdateNotification();

    this->_cameraHalfDisparityParam.Param<param::FloatParam>()->SetValue(cam.half_disparity(), makeDirty);
    this->_cameraHalfDisparityParam.QueueUpdateNotification();
}

/*
 * AbstractView3D::adaptCameraValues
 */
bool AbstractView3D::adaptCameraValues(Camera_2& cam) {
    bool result = false;
    if (this->_cameraPositionParam.IsDirty()) {
        auto val = this->_cameraPositionParam.Param<param::Vector3fParam>()->Value();
        this->_camera.position(glm::vec4(val.GetX(), val.GetY(), val.GetZ(), 1.0f));
        this->_cameraPositionParam.ResetDirty();
        result = true;
    }
    if (this->_cameraOrientationParam.IsDirty()) {
        auto val = this->_cameraOrientationParam.Param<param::Vector4fParam>()->Value();
        this->_camera.orientation(glm::quat(val.GetW(), val.GetX(), val.GetY(), val.GetZ()));
        this->_cameraOrientationParam.ResetDirty();
        result = true;
    }
    if (this->_cameraProjectionTypeParam.IsDirty()) {
        auto val =
            static_cast<thecam::Projection_type>(this->_cameraProjectionTypeParam.Param<param::EnumParam>()->Value());
        this->_camera.projection_type(val);
        this->_cameraProjectionTypeParam.ResetDirty();
        result = true;
    }
    //// setting of near plane and far plane might make no sense as we are setting them new each frame anyway
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
    if (this->_cameraConvergencePlaneParam.IsDirty()) {
        auto val = this->_cameraConvergencePlaneParam.Param<param::FloatParam>()->Value();
        this->_camera.convergence_plane(val);
        this->_cameraConvergencePlaneParam.ResetDirty();
        result = true;
    }
    /*if (this->cameraEyeParam.IsDirty()) {
        auto val = static_cast<thecam::Eye>(this->cameraEyeParam.Param<param::EnumParam>()->Value());
        this->_camera.eye(val);
        this->cameraEyeParam.ResetDirty();
        result = true;
    }*/
    /*if (this->cameraGateScalingParam.IsDirty()) {
        auto val = static_cast<thecam::Gate_scaling>(this->cameraGateScalingParam.Param<param::EnumParam>()->Value());
        this->_camera.gate_scaling(val);
        this->cameraGateScalingParam.ResetDirty();
        result = true;
    }
    if (this->cameraFilmGateParam.IsDirty()) {
        auto val = this->cameraFilmGateParam.Param<param::Vector2fParam>()->Value();
        this->_camera.film_gate(thecam::math::size<float, 2>(val.GetX(), val.GetY()));
        this->cameraFilmGateParam.ResetDirty();
        result = true;
    }*/
    /*if (this->cameraResolutionXParam.IsDirty()) {
        auto val = this->cameraResolutionXParam.Param<param::IntParam>()->Value();
        this->_camera.resolution_gate({val, this->_camera.resolution_gate().height()});
        this->cameraResolutionXParam.ResetDirty();
        result = true;
    }
    if (this->cameraResolutionYParam.IsDirty()) {
        auto val = this->cameraResolutionYParam.Param<param::IntParam>()->Value();
        this->_camera.resolution_gate({this->_camera.resolution_gate().width(), val});
        this->cameraResolutionYParam.ResetDirty();
        result = true;
    }*/
    if (this->_cameraCenterOffsetParam.IsDirty()) {
        auto val = this->_cameraCenterOffsetParam.Param<param::Vector2fParam>()->Value();
        this->_camera.centre_offset(thecam::math::vector<float, 2>(val.GetX(), val.GetY()));
        this->_cameraCenterOffsetParam.ResetDirty();
        result = true;
    }
    if (this->_cameraHalfApertureDegreesParam.IsDirty()) {
        auto val = this->_cameraHalfApertureDegreesParam.Param<param::FloatParam>()->Value();
        this->_camera.half_aperture_angle_radians(val * M_PI / 180.0f);
        this->_cameraHalfApertureDegreesParam.ResetDirty();
        result = true;
    }
    if (this->_cameraHalfDisparityParam.IsDirty()) {
        auto val = this->_cameraHalfDisparityParam.Param<param::FloatParam>()->Value();
        this->_camera.half_disparity(val);
        this->_cameraHalfDisparityParam.ResetDirty();
        result = true;
    }
    return result;
}

/*
 * AbstractView3D::cameraOvrCallback
 */
bool AbstractView3D::cameraOvrCallback(param::ParamSlot& p) {
    auto up_vis = this->_cameraOvrUpParam.Param<param::Vector3fParam>()->Value();
    auto lookat_vis = this->_cameraOvrLookatParam.Param<param::Vector3fParam>()->Value();

    glm::vec3 up(up_vis.X(), up_vis.Y(), up_vis.Z());
    up = glm::normalize(up);
    glm::vec3 lookat(lookat_vis.X(), lookat_vis.Y(), lookat_vis.Z());

    glm::mat3 view;
    view[2] = -glm::normalize(lookat - glm::vec3(static_cast<glm::vec4>(this->_camera.eye_position())));
    view[0] = glm::normalize(glm::cross(up, view[2]));
    view[1] = glm::normalize(glm::cross(view[2], view[0]));

    auto orientation = glm::quat_cast(view);

    this->_camera.orientation(orientation);
    this->_rotCenter = lookat;

    return true;
}
