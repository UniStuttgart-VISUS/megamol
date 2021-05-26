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

#include "GlobalValueStore.h"

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
    , _cameraViewOrientationParam("view::cubeOrientation", "Current camera orientation used for view cube orientation.")
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
    defaultViewParam->SetTypePair(DEFAULTVIEW_FACE_FRONT,               "FACE - Front");
    defaultViewParam->SetTypePair(DEFAULTVIEW_FACE_BACK,                "FACE - Back");
    defaultViewParam->SetTypePair(DEFAULTVIEW_FACE_RIGHT,               "FACE - Right");
    defaultViewParam->SetTypePair(DEFAULTVIEW_FACE_LEFT,                "FACE - Left");
    defaultViewParam->SetTypePair(DEFAULTVIEW_FACE_TOP,                 "FACE - Top");
    defaultViewParam->SetTypePair(DEFAULTVIEW_FACE_BOTTOM,              "FACE - Bottom");
    defaultViewParam->SetTypePair(DEFAULTVIEW_CORNER_TOP_LEFT_FRONT,    "CORNER - Top Left Front");
    defaultViewParam->SetTypePair(DEFAULTVIEW_CORNER_TOP_RIGHT_FRONT,   "CORNER - Top Right Front");
    defaultViewParam->SetTypePair(DEFAULTVIEW_CORNER_TOP_LEFT_BACK ,    "CORNER - Top Left Back");
    defaultViewParam->SetTypePair(DEFAULTVIEW_CORNER_TOP_RIGHT_BACK ,   "CORNER - Top Right Back");
    defaultViewParam->SetTypePair(DEFAULTVIEW_CORNER_BOTTOM_LEFT_FRONT, "CORNER - Bottom Left Front");
    defaultViewParam->SetTypePair(DEFAULTVIEW_CORNER_BOTTOM_RIGHT_FRONT,"CORNER - Bottom Right Front");
    defaultViewParam->SetTypePair(DEFAULTVIEW_CORNER_BOTTOM_LEFT_BACK,  "CORNER - Bottom Left Back");
    defaultViewParam->SetTypePair(DEFAULTVIEW_CORNER_BOTTOM_RIGHT_BACK, "CORNER - Bottom Right Back");
    defaultViewParam->SetTypePair(DEFAULTVIEW_EDGE_TOP_FRONT,           "EDGE - Top Front");
    defaultViewParam->SetTypePair(DEFAULTVIEW_EDGE_TOP_LEFT,            "EDGE - Top Left");
    defaultViewParam->SetTypePair(DEFAULTVIEW_EDGE_TOP_RIGHT,           "EDGE - Top Right");
    defaultViewParam->SetTypePair(DEFAULTVIEW_EDGE_TOP_BACK,            "EDGE - Top Back");
    defaultViewParam->SetTypePair(DEFAULTVIEW_EDGE_BOTTOM_FRONT,        "EDGE - Bottom Front");
    defaultViewParam->SetTypePair(DEFAULTVIEW_EDGE_BOTTOM_LEFT,         "EDGE - Bottom Left");
    defaultViewParam->SetTypePair(DEFAULTVIEW_EDGE_BOTTOM_RIGHT,        "EDGE - Bottom Right");
    defaultViewParam->SetTypePair(DEFAULTVIEW_EDGE_BOTTOM_BACK,         "EDGE - Bottom Back");
    defaultViewParam->SetTypePair(DEFAULTVIEW_EDGE_FRONT_LEFT,          "EDGE - Front Left");
    defaultViewParam->SetTypePair(DEFAULTVIEW_EDGE_FRONT_RIGHT ,        "EDGE - Front Right");
    defaultViewParam->SetTypePair(DEFAULTVIEW_EDGE_BACK_LEFT,           "EDGE - Back Left");
    defaultViewParam->SetTypePair(DEFAULTVIEW_EDGE_BACK_RIGHT,          "EDGE - Back Right");
    defaultViewParam->SetGUIVisible(camparamvisibility);
    this->_cameraSetViewChooserParam.SetParameter(defaultViewParam),
    this->_cameraSetViewChooserParam.SetUpdateCallback(&AbstractView::OnResetView);
    this->MakeSlotAvailable(&this->_cameraSetViewChooserParam);

    auto defaultOrientationParam = new param::EnumParam(0);
    defaultOrientationParam->SetTypePair(DEFAULTORIENTATION_TOP, "Top");
    defaultOrientationParam->SetTypePair(DEFAULTORIENTATION_RIGHT, "Right");
    defaultOrientationParam->SetTypePair(DEFAULTORIENTATION_BOTTOM, "Bottom");
    defaultOrientationParam->SetTypePair(DEFAULTORIENTATION_LEFT, "Left");
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
       
        param::FloatParam* fp = this->_stereoEyeDistSlot.Param<param::FloatParam>();
        // this->camParams->SetStereoDisparity(fp->Value());
        // fp->SetValue(this->camParams->StereoDisparity());
        this->_stereoEyeDistSlot.ResetDirty();
    }
    if (this->_stereoFocusDistSlot.IsDirty()) {
       
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

    // TODO set distance between eyes
    if (!this->_bboxs.IsBoundingBoxValid()) {
        this->_bboxs.SetBoundingBox(-1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f);
    }

    if (!this->_valuesFromOutside) {
        this->_camera.near_clipping_plane(0.1f);
        this->_camera.far_clipping_plane(100.0f);
        this->_camera.aperture_angle(30.0f);
        this->_camera.disparity(0.05f);
        this->_camera.eye(thecam::Eye::mono);
        this->_camera.projection_type(thecam::Projection_type::perspective);

        auto camera_orientation = this->get_default_camera_orientation();
        this->_camera.orientation(camera_orientation);
        this->_camera.position(this->get_default_camera_position(camera_orientation));
    }

    auto bbc = this->_bboxs.BoundingBox().CalcCenter();
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
    const auto arcball_key = "arcball";

    if (!this->GetCoreInstance()->IsmmconsoleFrontendCompatible()) {
        // new frontend has global key-value resource
        auto maybe = this->frontend_resources.get<megamol::frontend_resources::GlobalValueStore>().maybe_get(arcball_key);
        if (maybe.has_value()) {
            this->_arcballDefault = vislib::CharTraitsA::ParseBool(maybe.value().c_str());
        }

    } else {
        mmcValueType wpType;
        this->_arcballDefault = false;
        auto value = this->GetCoreInstance()->Configuration().GetValue(MMC_CFGID_VARIABLE, _T(arcball_key), &wpType);
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

/*
 * AbstractView3D::get_default_camera_position
 */
glm::vec4 AbstractView3D::get_default_camera_position(glm::quat camera_orientation) {

    glm::vec4 default_position = glm::vec4(0.0f, 0.0f, 0.0f, 0.0f);
    auto dv = static_cast<DefaultView>(this->_cameraSetViewChooserParam.Param<param::EnumParam>()->Value());

    // Calculate pseudo width and pseudo height by projecting all eight corners on plane orthogonal to camera position delta and lying in the center.
    glm::vec4 view_vec  = glm::normalize(camera_orientation * glm::vec4(0.0, 0.0f, 1.0f, 0.0f));
    glm::vec4 up_vec    = glm::normalize(camera_orientation * glm::vec4(0.0, 1.0f, 0.0f, 0.0f));
    glm::vec4 right_vec = glm::normalize(camera_orientation * glm::vec4(1.0, 0.0f, 0.0f, 0.0f));
    std::vector<glm::vec4> corners;
    auto tmp_corner = glm::vec4(this->_bboxs.BoundingBox().Width()/2.0f, this->_bboxs.BoundingBox().Height()/2.0f, this->_bboxs.BoundingBox().Depth()/2.0f, 0.0f);
    corners.push_back(glm::vec4(tmp_corner.x, tmp_corner.y, tmp_corner.z, 0.0f));
    corners.push_back(glm::vec4(tmp_corner.x, -tmp_corner.y, tmp_corner.z, 0.0f));
    corners.push_back(glm::vec4(tmp_corner.x, -tmp_corner.y, -tmp_corner.z, 0.0f));
    corners.push_back(glm::vec4(tmp_corner.x, tmp_corner.y, -tmp_corner.z, 0.0f));
    corners.push_back(glm::vec4(-tmp_corner.x, tmp_corner.y, tmp_corner.z, 0.0f));
    corners.push_back(glm::vec4(-tmp_corner.x, -tmp_corner.y, tmp_corner.z, 0.0f));
    corners.push_back(glm::vec4(-tmp_corner.x, -tmp_corner.y, -tmp_corner.z, 0.0f));
    corners.push_back(glm::vec4(-tmp_corner.x, tmp_corner.y, -tmp_corner.z, 0.0f));
    float delta_x_min = FLT_MAX;
    float delta_x_max = -FLT_MAX;
    float delta_y_min = FLT_MAX;
    float delta_y_max = -FLT_MAX;
    float delta_z_min = FLT_MAX;
    float delta_z_max = -FLT_MAX;
    for (auto& corner : corners)  {
        float delta_x = glm::dot(corner, right_vec);
        float delta_y = glm::dot(corner, up_vec);
        float delta_z = glm::dot(corner, view_vec);
        delta_x_min = std::min(delta_x_min, delta_x);
        delta_x_max = std::max(delta_x_max, delta_x);
        delta_y_min = std::min(delta_y_min, delta_y);
        delta_y_max = std::max(delta_y_max, delta_y);
        delta_z_min = std::min(delta_z_min, delta_z);
        delta_z_max = std::max(delta_z_max, delta_z);
    }
    auto pseudoWidth  = static_cast<double>(delta_x_max - delta_x_min);
    auto pseudoHeight = static_cast<double>(delta_y_max - delta_y_min);
    auto pseudoDepth  = static_cast<double>(delta_z_max - delta_z_min);

    // New camera Position
    auto dim = this->_camera.resolution_gate();
    auto bbc = this->_bboxs.BoundingBox().CalcCenter();
    auto bbcglm = glm::vec4(bbc.GetX(), bbc.GetY(), bbc.GetZ(), 1.0f);
    double halfFovY = static_cast<double>(this->_camera.aperture_angle_radians() / 2.0f);
    double halfFovX = (static_cast<double>(dim.width()) * halfFovY) / static_cast<double>(dim.height());
    double distY = (pseudoHeight / (2.0 * tan(halfFovY)));
    double distX = (pseudoWidth / (2.0 * tan(halfFovX)));
    auto face_dist = static_cast<float>((distX > distY) ? distX : distY);
    face_dist = face_dist + (pseudoDepth / 2.0f);
    float edge_dist   = face_dist / sqrt(2.0f);
    float corner_dist = edge_dist / sqrt(2.0f);

    switch (dv) {
        // FACES ----------------------------------------------------------------------------------
        case DEFAULTVIEW_FACE_FRONT:
            default_position = bbcglm + glm::vec4(0.0f, 0.0f, face_dist, 0.0f);
            break;
        case DEFAULTVIEW_FACE_BACK:
            default_position = bbcglm + glm::vec4(0.0f, 0.0f, -face_dist, 0.0f);
            break;
        case DEFAULTVIEW_FACE_RIGHT:
            default_position = bbcglm + glm::vec4(face_dist, 0.0f, 0.0f, 0.0f);
            break;
        case DEFAULTVIEW_FACE_LEFT:
            default_position = bbcglm + glm::vec4(-face_dist, 0.0f, 0.0f, 0.0f);
            break;
        case DEFAULTVIEW_FACE_TOP:
            default_position = bbcglm + glm::vec4(0.0f, face_dist, 0.0f, 0.0f);
            break;
        case DEFAULTVIEW_FACE_BOTTOM:
            default_position = bbcglm + glm::vec4(0.0f, -face_dist, 0.0f, 0.0f);
            break;
            // CORNERS --------------------------------------------------------------------------------
        case DEFAULTVIEW_CORNER_TOP_LEFT_FRONT:
            default_position = bbcglm + glm::vec4(-corner_dist, edge_dist, corner_dist, 0.0f);
            break;
        case DEFAULTVIEW_CORNER_TOP_RIGHT_FRONT:
            default_position = bbcglm + glm::vec4(corner_dist, edge_dist, corner_dist, 0.0f);
            break;
        case DEFAULTVIEW_CORNER_TOP_LEFT_BACK:
            default_position = bbcglm + glm::vec4(-corner_dist, edge_dist, -corner_dist, 0.0f);
            break;
        case DEFAULTVIEW_CORNER_TOP_RIGHT_BACK:
            default_position = bbcglm + glm::vec4(corner_dist, edge_dist, -corner_dist, 0.0f);
            break;
        case DEFAULTVIEW_CORNER_BOTTOM_LEFT_FRONT:
            default_position = bbcglm + glm::vec4(-corner_dist, -edge_dist, corner_dist, 0.0f);
            break;
        case DEFAULTVIEW_CORNER_BOTTOM_RIGHT_FRONT:
            default_position = bbcglm + glm::vec4(corner_dist, -edge_dist, corner_dist, 0.0f);
            break;
        case DEFAULTVIEW_CORNER_BOTTOM_LEFT_BACK:
            default_position = bbcglm + glm::vec4(-corner_dist, -edge_dist, -corner_dist, 0.0f);
            break;
        case DEFAULTVIEW_CORNER_BOTTOM_RIGHT_BACK:
            default_position = bbcglm + glm::vec4(corner_dist, -edge_dist, -corner_dist, 0.0f);
            break;
            // EDGES ----------------------------------------------------------------------------------
        case DEFAULTVIEW_EDGE_TOP_FRONT:
            default_position = bbcglm + glm::vec4(0.0f, edge_dist, edge_dist, 0.0f);
            break;
        case DEFAULTVIEW_EDGE_TOP_LEFT:
            default_position = bbcglm + glm::vec4(-edge_dist, edge_dist, 0.0f, 0.0f);
            break;
        case DEFAULTVIEW_EDGE_TOP_RIGHT :
            default_position = bbcglm + glm::vec4(edge_dist, edge_dist, 0.0f, 0.0f);
            break;
        case DEFAULTVIEW_EDGE_TOP_BACK:
            default_position = bbcglm + glm::vec4(0.0f, edge_dist, -edge_dist, 0.0f);
            break;
        case DEFAULTVIEW_EDGE_BOTTOM_FRONT:
            default_position = bbcglm + glm::vec4(0.0f, -edge_dist, edge_dist, 0.0f);
            break;
        case DEFAULTVIEW_EDGE_BOTTOM_LEFT:
            default_position = bbcglm + glm::vec4(-edge_dist, -edge_dist, 0.0f, 0.0f);
            break;
        case DEFAULTVIEW_EDGE_BOTTOM_RIGHT :
            default_position = bbcglm + glm::vec4(edge_dist, -edge_dist, 0.0f, 0.0f);
            break;
        case DEFAULTVIEW_EDGE_BOTTOM_BACK:
            default_position = bbcglm + glm::vec4(0.0f, -edge_dist, -edge_dist, 0.0f);
            break;
        case DEFAULTVIEW_EDGE_FRONT_LEFT:
            default_position = bbcglm + glm::vec4(-edge_dist, 0.0f, edge_dist, 0.0f);
            break;
        case DEFAULTVIEW_EDGE_FRONT_RIGHT:
            default_position = bbcglm + glm::vec4(edge_dist, 0.0f, edge_dist, 0.0f);
            break;
        case DEFAULTVIEW_EDGE_BACK_LEFT:
            default_position = bbcglm + glm::vec4(-edge_dist, 0.0f, -edge_dist, 0.0f);
            break;
        case DEFAULTVIEW_EDGE_BACK_RIGHT:
            default_position = bbcglm + glm::vec4(edge_dist, 0.0f, -edge_dist, 0.0f);
            break;
        default: break;
    }
    return default_position;
}

/*
 * AbstractView3D::get_default_camera_orientation
 */
glm::quat AbstractView3D::get_default_camera_orientation() {

    glm::quat default_orientation = cam_type::quaternion_type::create_identity();
    auto dv = static_cast<DefaultView>(this->_cameraSetViewChooserParam.Param<param::EnumParam>()->Value());
    auto dor = static_cast<DefaultOrientation>(this->_cameraSetOrientationChooserParam.Param<param::EnumParam>()->Value());

    // New camera orientation
    /// quat rot(theta) around axis(x,y,z) -> q = (sin(theta/2)*x, sin(theta/2)*y, sin(theta/2)*z, cos(theta/2))
    const float cos45 = sqrt(2.0f) / 2.0f;
    const float sin45 = cos45;
    const float cos22_5 = cos(M_PI/8.0f);
    const float sin22_5 = sin(M_PI/8.0f);

    auto qx_p45 = cam_type::quaternion_type(sin22_5, 0.0, 0.0, cos22_5);
    auto qx_n45 = cam_type::quaternion_type(-sin22_5, 0.0, 0.0, cos22_5);
    auto qy_p45 = cam_type::quaternion_type(0.0, sin22_5, 0.0, cos22_5);
    auto qy_n45 = cam_type::quaternion_type(0.0, -sin22_5, 0.0, cos22_5);
    auto qz_p45 = cam_type::quaternion_type(0.0, 0.0, sin22_5, cos22_5);
    auto qz_n45 = cam_type::quaternion_type(0.0, 0.0, -sin22_5, cos22_5);

    auto qx_p90 = cam_type::quaternion_type(sin45, 0.0, 0.0, cos45);
    auto qx_n90 = cam_type::quaternion_type(-sin45, 0.0, 0.0, cos45);
    auto qy_p90 = cam_type::quaternion_type(0.0, sin45, 0.0, cos45);
    auto qy_n90 = cam_type::quaternion_type(0.0, -sin45, 0.0, cos45);    
    auto qz_p90 = cam_type::quaternion_type(0.0, 0.0, sin45, cos45);
    auto qz_n90 = cam_type::quaternion_type(0.0, 0.0, -sin45, cos45);

    auto qx_p180 = cam_type::quaternion_type(1.0, 0.0, 0.0, 0.0);
    auto qy_p180 = cam_type::quaternion_type(0.0, 1.0, 0.0, 0.0);
    auto qz_p180 = cam_type::quaternion_type(0.0, 0.0, 1.0, 0.0);

    switch (dv) {
        // FACES ----------------------------------------------------------------------------------
        case DEFAULTVIEW_FACE_FRONT:
            switch (dor) {
            case DEFAULTORIENTATION_TOP:
                default_orientation = cam_type::quaternion_type::create_identity();
                break;
            case DEFAULTORIENTATION_RIGHT:
                default_orientation = qz_n90;
                break;
            case DEFAULTORIENTATION_BOTTOM:
                default_orientation = qz_p180;
                break;
            case DEFAULTORIENTATION_LEFT:
                default_orientation = qz_p90;
                break;
            default: break;
            }
            break;
        case DEFAULTVIEW_FACE_BACK: // 180 deg around y axis
            switch (dor) {
            case DEFAULTORIENTATION_TOP:
                default_orientation = qy_p180;
                break;
            case DEFAULTORIENTATION_RIGHT:
                default_orientation = qy_p180 * qz_n90;
                break;
            case DEFAULTORIENTATION_BOTTOM:
                default_orientation = qy_p180 * qz_p180;
                break;
            case DEFAULTORIENTATION_LEFT:
                default_orientation = qy_p180 * qz_p90;
                break;
            default: break;
            }
            break;
        case DEFAULTVIEW_FACE_RIGHT: // 90 deg around y axis
            switch (dor) {
            case DEFAULTORIENTATION_TOP:
                default_orientation = qy_p90;
                break;
            case DEFAULTORIENTATION_RIGHT:
                default_orientation = qy_p90 * qz_n90;
                break;
            case DEFAULTORIENTATION_BOTTOM:
                default_orientation = qy_p90 * qz_p180;
                break;
            case DEFAULTORIENTATION_LEFT:
                default_orientation = qy_p90 * qz_p90;
                break;
            default: break;
            }
            break;
        case DEFAULTVIEW_FACE_LEFT: // 90 deg reverse around y axis
            switch (dor) {
            case DEFAULTORIENTATION_TOP:
                default_orientation = qy_n90;
                break;
            case DEFAULTORIENTATION_RIGHT:
                default_orientation = qy_n90 * qz_n90;
                break;
            case DEFAULTORIENTATION_BOTTOM:
                default_orientation = qy_n90 * qz_p180;
                break;
            case DEFAULTORIENTATION_LEFT:
                default_orientation = qy_n90 * qz_p90;
                break;
            default: break;
            }
            break;
        case DEFAULTVIEW_FACE_TOP: // 90 deg around x axis
            switch (dor) {
            case DEFAULTORIENTATION_TOP:
                default_orientation = qx_n90;
                break;
            case DEFAULTORIENTATION_RIGHT:
                default_orientation = qx_n90 * qz_n90;
                break;
            case DEFAULTORIENTATION_BOTTOM:
                default_orientation = qx_n90 * qz_p180;
                break;
            case DEFAULTORIENTATION_LEFT:
                default_orientation = qx_n90 * qz_p90;
                break;
            default: break;
            }
            break;
        case DEFAULTVIEW_FACE_BOTTOM: // 90 deg reverse around x axis
            switch (dor) {
            case DEFAULTORIENTATION_TOP:
                default_orientation = qx_p90;
                break;
            case DEFAULTORIENTATION_RIGHT:
                default_orientation = qx_p90 * qz_n90;
                break;
            case DEFAULTORIENTATION_BOTTOM:
                default_orientation = qx_p90 * qz_p180;
                break;
            case DEFAULTORIENTATION_LEFT:
                default_orientation = qx_p90 * qz_p90;
                break;
            default: break;
            }
            break;
        // CORNERS --------------------------------------------------------------------------------
        case DEFAULTVIEW_CORNER_TOP_LEFT_FRONT:
            switch (dor) {
                case DEFAULTORIENTATION_TOP:
                    default_orientation = qy_n45 * qx_n45;
                    break;
                case DEFAULTORIENTATION_RIGHT:
                    default_orientation =  qy_n45 * qx_n45 * qz_n90;
                    break;
                case DEFAULTORIENTATION_BOTTOM:
                    default_orientation = qy_n45 * qx_n45 * qz_p180;
                    break;
                case DEFAULTORIENTATION_LEFT:
                    default_orientation = qy_n45 * qx_n45 * qz_p90;
                    break;
                default: break;
            }
            break;
        case DEFAULTVIEW_CORNER_TOP_RIGHT_FRONT:
            switch (dor) {
                case DEFAULTORIENTATION_TOP:
                    default_orientation = qy_p45 * qx_n45;
                    break;
                case DEFAULTORIENTATION_RIGHT:
                    default_orientation = qy_p45 * qx_n45 * qz_n90;
                    break;
                case DEFAULTORIENTATION_BOTTOM:
                    default_orientation = qy_p45 * qx_n45 * qz_p180;
                    break;
                case DEFAULTORIENTATION_LEFT:
                    default_orientation = qy_p45 * qx_n45 * qz_p90;
                    break;
                default: break;
            }
            break;
        case DEFAULTVIEW_CORNER_TOP_LEFT_BACK:
            switch (dor) {
                case DEFAULTORIENTATION_TOP:
                    default_orientation = qy_p180 * qy_p45 * qx_n45;
                    break;
                case DEFAULTORIENTATION_RIGHT:
                    default_orientation = qy_p180 * qy_p45 * qx_n45 * qz_n90;
                    break;
                case DEFAULTORIENTATION_BOTTOM:
                    default_orientation = qy_p180 * qy_p45 * qx_n45 * qz_p180;
                    break;
                case DEFAULTORIENTATION_LEFT:
                    default_orientation = qy_p180 * qy_p45 * qx_n45 * qz_p90;
                    break;
                default: break;
            }
            break;
        case DEFAULTVIEW_CORNER_TOP_RIGHT_BACK:
            switch (dor) {
                case DEFAULTORIENTATION_TOP:
                    default_orientation = qy_p180 * qy_n45 * qx_n45;
                    break;
                case DEFAULTORIENTATION_RIGHT:
                    default_orientation = qy_p180 * qy_n45 * qx_n45 * qz_n90;
                    break;
                case DEFAULTORIENTATION_BOTTOM:
                    default_orientation = qy_p180 * qy_n45 * qx_n45 * qz_p180;
                    break;
                case DEFAULTORIENTATION_LEFT:
                    default_orientation = qy_p180 * qy_n45 * qx_n45 * qz_p90;
                    break;
                default: break;
            }
            break;
        case DEFAULTVIEW_CORNER_BOTTOM_LEFT_FRONT:
            switch (dor) {
                case DEFAULTORIENTATION_TOP:
                    default_orientation = qy_n45 * qx_p45;
                    break;
                case DEFAULTORIENTATION_RIGHT:
                    default_orientation = qy_n45 * qx_p45 * qz_n90;
                    break;
                case DEFAULTORIENTATION_BOTTOM:
                    default_orientation = qy_n45 * qx_p45 * qz_p180;
                    break;
                case DEFAULTORIENTATION_LEFT:
                    default_orientation = qy_n45 * qx_p45 * qz_p90;
                    break;
                default: break;
            }
            break;
        case DEFAULTVIEW_CORNER_BOTTOM_RIGHT_FRONT:
            switch (dor) {
                case DEFAULTORIENTATION_TOP:
                    default_orientation = qy_p45 * qx_p45;
                    break;
                case DEFAULTORIENTATION_RIGHT:
                    default_orientation = qy_p45 * qx_p45 * qz_n90;
                    break;
                case DEFAULTORIENTATION_BOTTOM:
                    default_orientation = qy_p45 * qx_p45 * qz_p180;
                    break;
                case DEFAULTORIENTATION_LEFT:
                    default_orientation = qy_p45 * qx_p45 * qz_p90;
                    break;
                default: break;
            }
            break;
        case DEFAULTVIEW_CORNER_BOTTOM_LEFT_BACK:
            switch (dor) {
                case DEFAULTORIENTATION_TOP:
                    default_orientation = qy_p180 * qy_p45 * qx_p45;
                    break;
                case DEFAULTORIENTATION_RIGHT:
                    default_orientation = qy_p180 * qy_p45 * qx_p45 * qz_n90;
                    break;
                case DEFAULTORIENTATION_BOTTOM:
                    default_orientation = qy_p180 * qy_p45 * qx_p45 * qz_p180;
                    break;
                case DEFAULTORIENTATION_LEFT:
                    default_orientation = qy_p180 * qy_p45 * qx_p45 * qz_p90;
                    break;
                default: break;
            }
            break;
        case DEFAULTVIEW_CORNER_BOTTOM_RIGHT_BACK:
            switch (dor) {
                case DEFAULTORIENTATION_TOP:
                    default_orientation = qy_p180 * qy_n45 * qx_p45;
                    break;
                case DEFAULTORIENTATION_RIGHT:
                    default_orientation = qy_p180 * qy_n45 * qx_p45 * qz_n90;
                    break;
                case DEFAULTORIENTATION_BOTTOM:
                    default_orientation = qy_p180 * qy_n45 * qx_p45 * qz_p180;
                    break;
                case DEFAULTORIENTATION_LEFT:
                    default_orientation = qy_p180 * qy_n45 * qx_p45 * qz_p90;
                    break;
                default: break;
            }
            break;
        // EDGES ----------------------------------------------------------------------------------
        case DEFAULTVIEW_EDGE_TOP_FRONT:
            switch (dor) {
                case DEFAULTORIENTATION_TOP:
                    default_orientation = qx_n45 * qz_p90;
                    break;
                case DEFAULTORIENTATION_RIGHT:
                    default_orientation = qx_n45;
                    break;
                case DEFAULTORIENTATION_BOTTOM:
                    default_orientation = qx_n45 * qz_n90;
                    break;
                case DEFAULTORIENTATION_LEFT:
                    default_orientation = qz_p180 * qx_p45;
                    break;
                default: break;
            }
            break;
        case DEFAULTVIEW_EDGE_TOP_LEFT: // TODO
            switch (dor) {
                case DEFAULTORIENTATION_TOP:
                    default_orientation = qx_n90 * qy_n45;
                    break;
                case DEFAULTORIENTATION_RIGHT:
                    default_orientation = qy_p180 * qy_p90 * qx_n45;
                    break;
                case DEFAULTORIENTATION_BOTTOM:
                    default_orientation = qy_p180 * qx_n90 * qy_p45;
                    break;
                case DEFAULTORIENTATION_LEFT:
                    default_orientation = qx_p180 * qy_n90 * qx_p45;
                    break;
                default: break;
            }
            break;
        case DEFAULTVIEW_EDGE_TOP_RIGHT: // TODO
            switch (dor) {
                case DEFAULTORIENTATION_TOP:
                    default_orientation = qy_p180 * qx_n90 * qy_n45;
                    break;
                case DEFAULTORIENTATION_RIGHT:
                    default_orientation = qy_p180 * qy_n90 * qx_n45;
                    break;
                case DEFAULTORIENTATION_BOTTOM:
                    default_orientation = qx_n90 * qy_p45;
                    break;
                case DEFAULTORIENTATION_LEFT:
                    default_orientation = qx_p180 * qy_p90 * qx_p45;
                    break;
                default: break;
            }
            break;
        case DEFAULTVIEW_EDGE_TOP_BACK:
            switch (dor) {
                case DEFAULTORIENTATION_TOP:
                    default_orientation = qy_p180 * qx_n45 * qz_n90;
                    break;
                case DEFAULTORIENTATION_RIGHT:
                    default_orientation = qy_p180 * qz_p180 * qx_p45;
                    break;
                case DEFAULTORIENTATION_BOTTOM:
                    default_orientation = qy_p180 * qx_n45 * qz_p90;
                    break;
                case DEFAULTORIENTATION_LEFT:
                    default_orientation = qy_p180 * qx_n45;
                    break;
                default: break;
            }
            break;
        case DEFAULTVIEW_EDGE_BOTTOM_FRONT:
            switch (dor) {
                case DEFAULTORIENTATION_TOP:
                    default_orientation = qx_p45 * qz_p90;
                    break;
                case DEFAULTORIENTATION_RIGHT:
                    default_orientation = qx_p45;
                    break;
                case DEFAULTORIENTATION_BOTTOM:
                    default_orientation = qx_p45 * qz_n90;
                    break;
                case DEFAULTORIENTATION_LEFT:
                    default_orientation = qz_p180 * qx_n45;
                    break;
                default: break;
            }
            break;
        case DEFAULTVIEW_EDGE_BOTTOM_LEFT: // TODO
            switch (dor) {
                case DEFAULTORIENTATION_TOP:
                    default_orientation = qy_p180 * qx_p90 * qy_p45;
                    break;
                case DEFAULTORIENTATION_RIGHT:
                    default_orientation = qy_n90 * qx_p45;
                    break;
                case DEFAULTORIENTATION_BOTTOM:
                    default_orientation = qx_p90 * qy_n45;
                    break;
                case DEFAULTORIENTATION_LEFT:
                    default_orientation = qz_p180 * qy_p90 * qx_n45;
                    break;
                default: break;
            }
            break;
        case DEFAULTVIEW_EDGE_BOTTOM_RIGHT: // TODO
            switch (dor) {
                case DEFAULTORIENTATION_TOP:
                    default_orientation = qx_p90 * qy_p45;
                    break;
                case DEFAULTORIENTATION_RIGHT:
                    default_orientation = qy_p90 * qx_p45;
                    break;
                case DEFAULTORIENTATION_BOTTOM:
                    default_orientation = qy_p180 * qx_p90 * qy_n45;
                    break;
                case DEFAULTORIENTATION_LEFT:
                    default_orientation = qz_p180 * qy_n90 * qx_n45;
                    break;
                default: break;
            }
            break;
        case DEFAULTVIEW_EDGE_BOTTOM_BACK:
            switch (dor) {
                case DEFAULTORIENTATION_TOP:
                    default_orientation = qy_p180 * qx_p45 * qz_p90;
                    break;
                case DEFAULTORIENTATION_RIGHT:
                    default_orientation = qy_p180 * qx_p45;
                    break;
                case DEFAULTORIENTATION_BOTTOM:
                    default_orientation = qy_p180 * qx_p45 * qz_n90;
                    break;
                case DEFAULTORIENTATION_LEFT:
                    default_orientation = qy_p180 * qz_p180 * qx_n45;
                    break;
                default: break;
            }
            break;
        case DEFAULTVIEW_EDGE_FRONT_LEFT:
            switch (dor) {
                case DEFAULTORIENTATION_TOP:
                    default_orientation = qy_n45;
                    break;
                case DEFAULTORIENTATION_RIGHT:
                    default_orientation = qz_n90 * qx_p45;
                    break;
                case DEFAULTORIENTATION_BOTTOM:
                    default_orientation = qz_p180 * qy_p45;
                    break;
                case DEFAULTORIENTATION_LEFT:
                    default_orientation = qz_p90 * qx_n45;
                    break;
                default: break;
            }
            break;
        case DEFAULTVIEW_EDGE_FRONT_RIGHT:
            switch (dor) {
                case DEFAULTORIENTATION_TOP:
                    default_orientation = qy_p45;
                    break;
                case DEFAULTORIENTATION_RIGHT:
                    default_orientation = qz_n90 * qx_n45;
                    break;
                case DEFAULTORIENTATION_BOTTOM:
                    default_orientation = qz_p180 * qy_n45;
                    break;
                case DEFAULTORIENTATION_LEFT:
                    default_orientation = qz_p90 * qx_p45;
                    break;
                default: break;
            }
            break;
        case DEFAULTVIEW_EDGE_BACK_LEFT:
            switch (dor) {
                case DEFAULTORIENTATION_TOP:
                    default_orientation = qy_p180 * qy_p45;
                    break;
                case DEFAULTORIENTATION_RIGHT:
                    default_orientation = qy_p180 * qz_n90 * qx_n45;
                    break;
                case DEFAULTORIENTATION_BOTTOM:
                    default_orientation = qx_p180 * qy_n45;
                    break;
                case DEFAULTORIENTATION_LEFT:
                    default_orientation = qy_p180 * qz_p90 * qx_p45;
                    break;
                default: break;
            }
            break;
        case DEFAULTVIEW_EDGE_BACK_RIGHT:
            switch (dor) {
                case DEFAULTORIENTATION_TOP:
                    default_orientation = qy_p180 * qy_n45;
                    break;
                case DEFAULTORIENTATION_RIGHT:
                    default_orientation = qy_p180 * qz_n90 * qx_p45;
                    break;
                case DEFAULTORIENTATION_BOTTOM:
                    default_orientation = qx_p180 * qy_p45;
                    break;
                case DEFAULTORIENTATION_LEFT:
                    default_orientation = qy_p180 * qz_p90 * qx_n45;
                    break;
                default: break;
            }
            break;
        default: break;
    }
    return default_orientation;
}
