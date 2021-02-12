/*
 * AbstractView3D.cpp
 *
 * Copyright (C) 2018, 2020 by VISUS (Universitaet Stuttgart).
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
#include "mmcore/param/StringParam.h"
#include "mmcore/param/Vector2fParam.h"
#include "mmcore/param/Vector3fParam.h"
#include "mmcore/param/Vector4fParam.h"
#include "vislib/String.h"
#include "vislib/StringSerialiser.h"
#include "vislib/math/Point.h"
#include "vislib/math/Quaternion.h"
#include "vislib/math/Vector.h"
#include "vislib/sys/KeyCode.h"
#include "mmcore/utility/log/Log.h"
#include "vislib/sys/sysfunctions.h"
#include "mmcore/view/CallRenderView.h"

#include "glm/gtc/matrix_transform.hpp"

using namespace megamol::core;
using namespace megamol::core::view;

/*
 * AbstractView3D::AbstractView3D
 */
AbstractView3D::AbstractView3D(void)
    : view::AbstractRenderingView()
    , rendererSlot("rendering", "Connects the view to a Renderer")
    , showLookAt("showLookAt", "Flag showing the look at point")
    , cameraSettingsSlot("camstore::settings", "Holds the camera settings of the currently stored camera.")
    , storeCameraSettingsSlot("camstore::storecam", "Triggers the storage of the camera settings. This only works for "
                                                    "multiple cameras if you use .lua project files")
    , restoreCameraSettingsSlot("camstore::restorecam", "Triggers the restore of the camera settings. This only works "
                                                        "for multiple cameras if you use .lua project files")
    , overrideCamSettingsSlot("camstore::overrideSettings",
          "When activated, existing camera settings files will be overwritten by this "
          "module. This only works if you use .lua project files")
    , autoSaveCamSettingsSlot("camstore::autoSaveSettings",
          "When activated, the camera settings will be stored to disk whenever a camera checkpoint is saved or MegaMol "
          "is closed. This only works if you use .lua project files")
    , autoLoadCamSettingsSlot("camstore::autoLoadSettings",
          "When activated, the view will load the camera settings from disk at startup. "
          "This only works if you use .lua project files")
    , resetViewSlot("resetView", "Triggers the reset of the view")
    , firstImg(false)
    , stereoFocusDistSlot("stereo::focusDist", "focus distance for stereo projection")
    , stereoEyeDistSlot("stereo::eyeDist", "eye distance for stereo projection")
    , viewKeyMoveStepSlot("viewKey::MoveStep", "The move step size in world coordinates")
    , viewKeyRunFactorSlot("viewKey::RunFactor", "The factor for step size multiplication when running (shift)")
    , viewKeyAngleStepSlot("viewKey::AngleStep", "The angle rotate step in degrees")
    , viewKeyFixToWorldUpSlot("viewKey::FixToWorldUp","Fix rotation manipulator to world up vector")
    , mouseSensitivitySlot("viewKey::MouseSensitivity", "used for WASD mode")
    , viewKeyRotPointSlot("viewKey::RotPoint", "The point around which the view will be rotated")
    , enableMouseSelectionSlot("enableMouseSelection", "Enable selecting and picking with the mouse")
    , showViewCubeSlot("viewcube::show", "Shows the view cube helper")
    , resetViewOnBBoxChangeSlot("resetViewOnBBoxChange", "whether to reset the view when the bounding boxes change")
    , timeCtrl()
    , hookOnChangeOnlySlot("hookOnChange", "whether post-hooks are triggered when the frame would be identical")
    , cameraPositionParam("cam::position", "")
    , cameraOrientationParam("cam::orientation", "")
    , cameraProjectionTypeParam("cam::projectiontype", "")
    , cameraNearPlaneParam("cam::nearplane", "")
    , cameraFarPlaneParam("cam::farplane", "")
    , cameraConvergencePlaneParam("cam::convergenceplane", "")
    , cameraEyeParam("cam::eye", "")
    , cameraGateScalingParam("cam::gatescaling", "")
    , cameraFilmGateParam("cam::filmgate", "")
    , cameraResolutionXParam("cam::resgate::x", "")
    , cameraResolutionYParam("cam::resgate::y", "")
    , cameraCenterOffsetParam("cam::centeroffset", "")
    , cameraHalfApertureDegreesParam("cam::halfaperturedegrees", "")
    , cameraHalfDisparityParam("cam::halfdisparity", "")
    , cameraOvrUpParam("cam::ovr::up", "")
    , cameraOvrLookatParam("cam::ovr::lookat", "")
    , cameraOvrParam("cam::ovr::override", "")
    , valuesFromOutside(false)
    , cameraControlOverrideActive(false) {

    using vislib::sys::KeyCode;

    this->cam.resolution_gate(cam_type::screen_size_type(100, 100));
    this->cam.image_tile(cam_type::screen_rectangle_type(std::array<int, 4>{0, 100, 100, 0}));

    // this triggers the initialization
    this->bboxs.Clear();

    this->showLookAt.SetParameter(new param::BoolParam(false));
    this->MakeSlotAvailable(&this->showLookAt);

    this->cameraSettingsSlot.SetParameter(new param::StringParam(""));
    this->MakeSlotAvailable(&this->cameraSettingsSlot);

    this->storeCameraSettingsSlot.SetParameter(
        new param::ButtonParam(view::Key::KEY_C, (view::Modifier::SHIFT | view::Modifier::ALT)));
    this->storeCameraSettingsSlot.SetUpdateCallback(&AbstractView3D::onStoreCamera);
    this->MakeSlotAvailable(&this->storeCameraSettingsSlot);

    this->restoreCameraSettingsSlot.SetParameter(new param::ButtonParam(view::Key::KEY_C, view::Modifier::ALT));
    this->restoreCameraSettingsSlot.SetUpdateCallback(&AbstractView3D::onRestoreCamera);
    this->MakeSlotAvailable(&this->restoreCameraSettingsSlot);

    this->overrideCamSettingsSlot.SetParameter(new param::BoolParam(false));
    this->MakeSlotAvailable(&this->overrideCamSettingsSlot);

    this->autoSaveCamSettingsSlot.SetParameter(new param::BoolParam(false));
    this->MakeSlotAvailable(&this->autoSaveCamSettingsSlot);

    this->autoLoadCamSettingsSlot.SetParameter(new param::BoolParam(true));
    this->MakeSlotAvailable(&this->autoLoadCamSettingsSlot);

    this->resetViewSlot.SetParameter(new param::ButtonParam(view::Key::KEY_HOME));
    this->resetViewSlot.SetUpdateCallback(&AbstractView3D::onResetView);
    this->MakeSlotAvailable(&this->resetViewSlot);

    this->ResetView();

    this->viewKeyMoveStepSlot.SetParameter(new param::FloatParam(0.5f, 0.001f));
    this->MakeSlotAvailable(&this->viewKeyMoveStepSlot);

    this->viewKeyRunFactorSlot.SetParameter(new param::FloatParam(2.0f, 0.1f));
    this->MakeSlotAvailable(&this->viewKeyRunFactorSlot);

    this->viewKeyAngleStepSlot.SetParameter(new param::FloatParam(90.0f, 0.1f, 360.0f));
    this->MakeSlotAvailable(&this->viewKeyAngleStepSlot);

    this->viewKeyFixToWorldUpSlot.SetParameter(new param::BoolParam(true));
    this->MakeSlotAvailable(&this->viewKeyFixToWorldUpSlot);

    this->mouseSensitivitySlot.SetParameter(new param::FloatParam(3.0f, 0.001f, 10.0f));
    this->mouseSensitivitySlot.SetUpdateCallback(&AbstractView3D::mouseSensitivityChanged);
    this->MakeSlotAvailable(&this->mouseSensitivitySlot);

    // TODO clean up vrpsev memory after use
    param::EnumParam* vrpsev = new param::EnumParam(1);
    vrpsev->SetTypePair(0, "Position");
    vrpsev->SetTypePair(1, "Look-At");
    this->viewKeyRotPointSlot.SetParameter(vrpsev);
    this->MakeSlotAvailable(&this->viewKeyRotPointSlot);

    this->enableMouseSelectionSlot.SetParameter(new param::ButtonParam(view::Key::KEY_TAB));
    this->enableMouseSelectionSlot.SetUpdateCallback(&AbstractView3D::onToggleButton);
    this->MakeSlotAvailable(&this->enableMouseSelectionSlot);

    this->resetViewOnBBoxChangeSlot.SetParameter(new param::BoolParam(false));
    this->MakeSlotAvailable(&this->resetViewOnBBoxChangeSlot);

    this->showViewCubeSlot.SetParameter(new param::BoolParam(true));
    this->MakeSlotAvailable(&this->showViewCubeSlot);

    for (unsigned int i = 0; this->timeCtrl.GetSlot(i) != NULL; i++) {
        this->MakeSlotAvailable(this->timeCtrl.GetSlot(i));
    }

    this->hookOnChangeOnlySlot.SetParameter(new param::BoolParam(false));
    this->MakeSlotAvailable(&this->hookOnChangeOnlySlot);

    const bool camparamvisibility = true;

    auto camposparam = new param::Vector3fParam(vislib::math::Vector<float, 3>());
    camposparam->SetGUIVisible(camparamvisibility);
    this->cameraPositionParam.SetParameter(camposparam);
    this->MakeSlotAvailable(&this->cameraPositionParam);

    auto camorientparam = new param::Vector4fParam(vislib::math::Vector<float, 4>());
    camorientparam->SetGUIVisible(camparamvisibility);
    this->cameraOrientationParam.SetParameter(camorientparam);
    this->MakeSlotAvailable(&this->cameraOrientationParam);

    auto projectionParam = new param::EnumParam(static_cast<int>(core::thecam::Projection_type::perspective));
    projectionParam->SetTypePair(static_cast<int>(core::thecam::Projection_type::perspective), "Perspective");
    projectionParam->SetTypePair(static_cast<int>(core::thecam::Projection_type::orthographic), "Orthographic");
    projectionParam->SetTypePair(static_cast<int>(core::thecam::Projection_type::parallel), "Parallel");
    projectionParam->SetTypePair(static_cast<int>(core::thecam::Projection_type::off_axis), "Off-Axis");
    projectionParam->SetTypePair(static_cast<int>(core::thecam::Projection_type::converged), "Converged");
    projectionParam->SetGUIVisible(camparamvisibility);
    this->cameraProjectionTypeParam.SetParameter(projectionParam);
    this->MakeSlotAvailable(&this->cameraProjectionTypeParam);

    auto camconvergenceparam = new param::FloatParam(0.1f, 0.0f);
    camconvergenceparam->SetGUIVisible(camparamvisibility);
    this->cameraConvergencePlaneParam.SetParameter(camconvergenceparam);
    this->MakeSlotAvailable(&this->cameraConvergencePlaneParam);

    auto centeroffsetparam = new param::Vector2fParam(vislib::math::Vector<float, 2>());
    centeroffsetparam->SetGUIVisible(camparamvisibility);
    this->cameraCenterOffsetParam.SetParameter(centeroffsetparam);
    this->MakeSlotAvailable(&this->cameraCenterOffsetParam);

    auto apertureparam = new param::FloatParam(35.0f, 0.0f);
    apertureparam->SetGUIVisible(camparamvisibility);
    this->cameraHalfApertureDegreesParam.SetParameter(apertureparam);
    this->MakeSlotAvailable(&this->cameraHalfApertureDegreesParam);

    auto disparityparam = new param::FloatParam(1.0f, 0.0f);
    disparityparam->SetGUIVisible(camparamvisibility);
    this->cameraHalfDisparityParam.SetParameter(disparityparam);
    this->MakeSlotAvailable(&this->cameraHalfDisparityParam);

    this->cameraOvrUpParam << new param::Vector3fParam(vislib::math::Vector<float, 3>());
    this->MakeSlotAvailable(&this->cameraOvrUpParam);

    this->cameraOvrLookatParam << new param::Vector3fParam(vislib::math::Vector<float, 3>());
    this->MakeSlotAvailable(&this->cameraOvrLookatParam);

    this->cameraOvrParam << new param::ButtonParam();
    this->cameraOvrParam.SetUpdateCallback(&AbstractView3D::cameraOvrCallback);
    this->MakeSlotAvailable(&this->cameraOvrParam);

    this->translateManipulator.set_target(this->cam);
    this->translateManipulator.enable();

    this->rotateManipulator.set_target(this->cam);
    this->rotateManipulator.enable();

    this->arcballManipulator.set_target(this->cam);
    this->arcballManipulator.enable();
    this->rotCenter = glm::vec3(0.0f, 0.0f, 0.0f);

    this->turntableManipulator.set_target(this->cam);
    this->turntableManipulator.enable();

    this->orbitAltitudeManipulator.set_target(this->cam);
    this->orbitAltitudeManipulator.enable();


    // none of the saved camera states are valid right now
    for (auto& e : this->savedCameras) {
        e.second = false;
    }
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
unsigned int view::AbstractView3D::GetCameraSyncNumber(void) const {
    // TODO implement
    return 0;
}

/*
 * AbstractView3D::beforeRender
 */
void AbstractView3D::beforeRender(const mmcRenderViewContext& context) {
    float time = static_cast<float>(context.Time);
    float instTime = static_cast<float>(context.InstanceTime);

    if (this->doHookCode()) {
        this->doBeforeRenderHook();
    }

    glm::ivec4 currentViewport;
    AbstractCallRender* cr3d = this->rendererSlot.CallAs<AbstractCallRender>();
    this->handleCameraMovement();

    AbstractRenderingView::beginFrame();

    auto bkgndCol = (this->overrideBkgndCol != glm::vec4(0, 0, 0, 0)) ? this->overrideBkgndCol : this->BkgndColour();

    if (cr3d == NULL) {
        this->endFrame(true);
        return; // empty enought
    }

    this->removeTitleRenderer();
    cr3d->SetBackgroundColor(glm::vec4(bkgndCol[0], bkgndCol[1], bkgndCol[2], 0.0f));
    // cr3d->SetFramebuffer(_framebuffer);

    // camera settings
    if (this->stereoEyeDistSlot.IsDirty()) {
        // TODO
        param::FloatParam* fp = this->stereoEyeDistSlot.Param<param::FloatParam>();
        // this->camParams->SetStereoDisparity(fp->Value());
        // fp->SetValue(this->camParams->StereoDisparity());
        this->stereoEyeDistSlot.ResetDirty();
    }
    if (this->stereoFocusDistSlot.IsDirty()) {
        // TODO
        param::FloatParam* fp = this->stereoFocusDistSlot.Param<param::FloatParam>();
        // this->camParams->SetFocalDistance(fp->Value());
        // fp->SetValue(this->camParams->FocalDistance(false));
        this->stereoFocusDistSlot.ResetDirty();
    }

    if (cr3d != nullptr) {
        (*cr3d)(view::AbstractCallRender::FnGetExtents);
        this->valuesFromOutside = this->adaptCameraValues(this->cam);
        if (this->firstImg || (!(cr3d->AccessBoundingBoxes() == this->bboxs) &&
                                  !(!cr3d->AccessBoundingBoxes().IsAnyValid() && !this->bboxs.IsBoundingBoxValid() &&
                                      !this->bboxs.IsClipBoxValid()))) {
            this->bboxs = cr3d->AccessBoundingBoxes();
            glm::vec3 bbcenter = glm::make_vec3(this->bboxs.BoundingBox().CalcCenter().PeekCoordinates());

            if (this->firstImg) {
                this->ResetView();
                this->firstImg = false;
                if (this->autoLoadCamSettingsSlot.Param<param::BoolParam>()->Value()) {
                    this->onRestoreCamera(this->restoreCameraSettingsSlot);
                }
                this->lastFrameTime = std::chrono::high_resolution_clock::now();
            } else if (resetViewOnBBoxChangeSlot.Param<param::BoolParam>()->Value()) {
                this->ResetView();
            }
        }

        this->timeCtrl.SetTimeExtend(cr3d->TimeFramesCount(), false);
        if (time > static_cast<float>(cr3d->TimeFramesCount())) {
            time = static_cast<float>(cr3d->TimeFramesCount());
        }

        // old code was ...SetTime(this->frozenValues ? this->frozenValues->time : time);
        cr3d->SetTime(time);

        // TODO
        // cr3d->SetCameraParameters(this->cam.Parameters()); // < here we use the 'active' parameters!
        cr3d->SetLastFrameTime(AbstractRenderingView::lastFrameTime());

        auto currentTime = std::chrono::high_resolution_clock::now();
        this->lastFrameDuration =
            std::chrono::duration_cast<std::chrono::microseconds>(currentTime - this->lastFrameTime);
        this->lastFrameTime = currentTime;
    }

    this->cam.CalcClipping(this->bboxs.ClipBox(), 0.1f);

    cam_type::snapshot_type camsnap;
    cam_type::matrix_type viewCam, projCam;
    this->cam.calc_matrices(camsnap, viewCam, projCam);

    glm::mat4 view = viewCam;
    glm::mat4 proj = projCam;
    glm::mat4 mvp = projCam * viewCam;
}

/*
 * AbstractView3D::afterRender
 */
void AbstractView3D::afterRender(const mmcRenderViewContext& context) {
    this->setCameraValues(this->cam);

    AbstractRenderingView::endFrame();

    // this->lastFrameParams->CopyFrom(this->OnGetCamParams, false);

    if (this->doHookCode() && frameIsNew) {
        this->doAfterRenderHook();
    }
}

/*
 * AbstractView3D::ResetView
 */
void AbstractView3D::ResetView(void) {
    if (!this->valuesFromOutside) {
        this->cam.near_clipping_plane(0.1f);
        this->cam.far_clipping_plane(100.0f);
        this->cam.aperture_angle(30.0f);
        this->cam.disparity(0.05f);
        this->cam.eye(thecam::Eye::mono);
        this->cam.projection_type(thecam::Projection_type::perspective);
    }
    // TODO set distance between eyes
    if (!this->bboxs.IsBoundingBoxValid()) {
        this->bboxs.SetBoundingBox(-1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f);
    }
    float dist = (0.5f * sqrtf((this->bboxs.BoundingBox().Width() * this->bboxs.BoundingBox().Width()) +
                               (this->bboxs.BoundingBox().Depth() * this->bboxs.BoundingBox().Depth()) +
                               (this->bboxs.BoundingBox().Height() * this->bboxs.BoundingBox().Height()))) /
                 tanf(this->cam.aperture_angle_radians() / 2.0f);
    auto dim = this->cam.resolution_gate();
    double halfFovX =
        (static_cast<double>(dim.width()) * static_cast<double>(this->cam.aperture_angle_radians() / 2.0f)) /
        static_cast<double>(dim.height());
    double distX = static_cast<double>(this->bboxs.BoundingBox().Width()) / (2.0 * tan(halfFovX));
    double distY = static_cast<double>(this->bboxs.BoundingBox().Height()) /
                   (2.0 * tan(static_cast<double>(this->cam.aperture_angle_radians() / 2.0f)));
    dist = static_cast<float>((distX > distY) ? distX : distY);
    dist = dist + (this->bboxs.BoundingBox().Depth() / 2.0f);
    auto bbc = this->bboxs.BoundingBox().CalcCenter();

    auto bbcglm = glm::vec4(bbc.GetX(), bbc.GetY(), bbc.GetZ(), 1.0f);

    if (!this->valuesFromOutside) {
        this->cam.position(bbcglm + glm::vec4(0.0f, 0.0f, dist, 0.0f));
        this->cam.orientation(cam_type::quaternion_type::create_identity());
    }

    this->rotCenter = glm::vec3(bbc.GetX(), bbc.GetY(), bbc.GetZ());

    glm::mat4 vm = this->cam.view_matrix();
    glm::mat4 pm = this->cam.projection_matrix();

    // TODO Further manipulators? better value?
    this->valuesFromOutside = false;
}

/*
 * AbstractView3D::Resize
 */
void AbstractView3D::Resize(unsigned int width, unsigned int height) {
    if (this->cam.resolution_gate().width() != width || this->cam.resolution_gate().height() != height) {
        this->cam.resolution_gate(cam_type::screen_size_type(static_cast<LONG>(width), static_cast<LONG>(height)));
    }
    if (this->cam.image_tile().width() != width || this->cam.image_tile().height() != height) {
        this->cam.image_tile(cam_type::screen_rectangle_type(
            std::array<int, 4>({0, static_cast<int>(height), static_cast<int>(width), 0})));
    }
}

/*
 * AbstractView3D::OnRenderView
 */
bool AbstractView3D::OnRenderView(Call& call) {
    view::CallRenderView* crv = dynamic_cast<view::CallRenderView*>(&call);
    if (crv == nullptr) return false;

    if (crv->IsBackgroundSet()) {
        this->overrideBkgndCol = crv->BackgroundColor();
    }

    float time = crv->Time();
    if (time < 0.0f) time = this->DefaultTime(crv->InstanceTime());
    mmcRenderViewContext context;
    ::ZeroMemory(&context, sizeof(context));
    context.Time = time;
    context.InstanceTime = crv->InstanceTime();

    if (crv->IsTileSet()) {
        this->cam.resolution_gate(cam_type::screen_size_type(crv->VirtualWidth(), crv->VirtualHeight()));
        this->cam.image_tile(cam_type::screen_rectangle_type::from_bottom_left(
            crv->TileX(), crv->TileY(), crv->TileWidth(), crv->TileHeight()));
    }

    this->Render(context);

    this->overrideBkgndCol = glm::vec4(0,0,0,0);
    this->overrideViewport = glm::vec4(0, 0, 0, 0);

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
bool view::AbstractView3D::OnKey(view::Key key, view::KeyAction action, view::Modifiers mods) {
    auto* cr = this->rendererSlot.CallAs<AbstractCallRender>();
    if (cr != nullptr) {
        view::InputEvent evt;
        evt.tag = view::InputEvent::Tag::Key;
        evt.keyData.key = key;
        evt.keyData.action = action;
        evt.keyData.mods = mods;
        cr->SetInputEvent(evt);
        if ((*cr)(AbstractCallRender::FnOnKey))
            return true;
    }

    if (action == view::KeyAction::PRESS || action == view::KeyAction::REPEAT) {
        this->pressedKeyMap[key] = true;
    } else if (action == view::KeyAction::RELEASE) {
        this->pressedKeyMap[key] = false;
    }

    if (key == view::Key::KEY_LEFT_ALT || key == view::Key::KEY_RIGHT_ALT) {
        if (action == view::KeyAction::PRESS || action == view::KeyAction::REPEAT) {
            this->modkeys.set(view::Modifier::ALT);
            cameraControlOverrideActive = true;
        } else if (action == view::KeyAction::RELEASE) {
            this->modkeys.reset(view::Modifier::ALT);
            cameraControlOverrideActive = false;
        }
    }
    if (key == view::Key::KEY_LEFT_SHIFT || key == view::Key::KEY_RIGHT_SHIFT) {
        if (action == view::KeyAction::PRESS || action == view::KeyAction::REPEAT) {
            this->modkeys.set(view::Modifier::SHIFT);
        } else if (action == view::KeyAction::RELEASE) {
            this->modkeys.reset(view::Modifier::SHIFT);
        }
    }
    if (key == view::Key::KEY_LEFT_CONTROL || key == view::Key::KEY_RIGHT_CONTROL) {
        if (action == view::KeyAction::PRESS || action == view::KeyAction::REPEAT) {
            this->modkeys.set(view::Modifier::CTRL);
            cameraControlOverrideActive = true;
        } else if (action == view::KeyAction::RELEASE) {
            this->modkeys.reset(view::Modifier::CTRL);
            cameraControlOverrideActive = false;
        }
    }

    if (action == view::KeyAction::PRESS && (key >= view::Key::KEY_0 && key <= view::Key::KEY_9)) {
        int index =
            static_cast<int>(key) - static_cast<int>(view::Key::KEY_0); // ugly hack, maybe this can be done better
        index = (index - 1) % 10;                                       // put key '1' at index 0
        index = index < 0 ? index + 10 : index;                         // wrap key '0' to a positive index '9'

        if (mods.test(view::Modifier::CTRL)) {
            this->savedCameras[index].first = this->cam.get_minimal_state(this->savedCameras[index].first);
            this->savedCameras[index].second = true;
            if (this->autoSaveCamSettingsSlot.Param<param::BoolParam>()->Value()) {
                this->onStoreCamera(this->storeCameraSettingsSlot); // manually trigger the storing
            }
        } else {
            if (this->savedCameras[index].second) {
                // As a change of camera position should not change the display resolution, we actively save and restore
                // the old value of the resolution
                auto oldResolution = this->cam.resolution_gate; // save old resolution
                this->cam = this->savedCameras[index].first;    // override current camera
                this->cam.resolution_gate = oldResolution;      // restore old resolution
            }
        }
    }

    return false;
}

/*
 * AbstractView3D::OnChar
 */
bool view::AbstractView3D::OnChar(unsigned int codePoint) {
    auto* cr = this->rendererSlot.CallAs<view::AbstractCallRender>();
    if (cr == NULL) return false;

    view::InputEvent evt;
    evt.tag = view::InputEvent::Tag::Char;
    evt.charData.codePoint = codePoint;
    cr->SetInputEvent(evt);
    if (!(*cr)(view::AbstractCallRender::FnOnChar))
        return false;

    return true;
}

/*
 * AbstractView3D::release
 */
void AbstractView3D::release(void) { this->removeTitleRenderer(); }

/*
 * AbstractView3D::mouseSensitivityChanged
 */
bool AbstractView3D::mouseSensitivityChanged(param::ParamSlot& p) { return true; }

/*
 * AbstractView3D::OnMouseButton
 */
 bool AbstractView3D::OnMouseButton(view::MouseButton button, view::MouseButtonAction action, view::Modifiers mods) {
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
    this->arcballDefault = false;
    auto value = this->GetCoreInstance()->Configuration().GetValue(MMC_CFGID_VARIABLE, _T("arcball"), &wpType);
    if (value != nullptr) {
        try {
            switch (wpType) {
            case MMC_TYPE_BOOL:
                this->arcballDefault = *static_cast<const bool*>(value);
                break;

            case MMC_TYPE_CSTR:
                this->arcballDefault = vislib::CharTraitsA::ParseBool(static_cast<const char*>(value));
                break;

            case MMC_TYPE_WSTR:
                this->arcballDefault = vislib::CharTraitsW::ParseBool(static_cast<const wchar_t*>(value));
                break;
            }
        } catch (...) {}
    }
    this->firstImg = true;
    return true;
}

/*
 * AbstractView3D::onStoreCamera
 */
bool AbstractView3D::onStoreCamera(param::ParamSlot& p) {
    // save the current camera, too
    Camera_2::minimal_state_type minstate;
    this->cam.get_minimal_state(minstate);
    this->savedCameras[10].first = minstate;
    this->savedCameras[10].second = true;
    this->serializer.setPrettyMode(false);
    std::string camstring = this->serializer.serialize(this->savedCameras[10].first);
    this->cameraSettingsSlot.Param<param::StringParam>()->SetValue(camstring.c_str());

    auto path = this->determineCameraFilePath();
    if (path.empty()) {
        megamol::core::utility::log::Log::DefaultLog.WriteWarn(
            "The camera output file path could not be determined. This is probably due to the usage of .mmprj project "
            "files. Please use a .lua project file instead");
        return false;
    }

    if (!this->overrideCamSettingsSlot.Param<param::BoolParam>()->Value()) {
        // check if the file already exists
        std::ifstream file(path);
        if (file.good()) {
            file.close();
            megamol::core::utility::log::Log::DefaultLog.WriteWarn(
                "The camera output file path already contains a camera file with the name '%s'. Override mode is "
                "deactivated, so no camera is stored",
                path.c_str());
            return false;
        }
    }


    this->serializer.setPrettyMode();
    auto outString = this->serializer.serialize(this->savedCameras);

    std::ofstream file(path);
    if (file.is_open()) {
        file << outString;
        file.close();
    } else {
        megamol::core::utility::log::Log::DefaultLog.WriteWarn(
            "The camera output file could not be written to '%s' because the file could not be opened.", path.c_str());
        return false;
    }

    megamol::core::utility::log::Log::DefaultLog.WriteInfo("Camera statistics successfully written to '%s'", path.c_str());
    return true;
}

/*
 * AbstractView3D::onRestoreCamera
 */
bool AbstractView3D::onRestoreCamera(param::ParamSlot& p) {
    if (!this->cameraSettingsSlot.Param<param::StringParam>()->Value().IsEmpty()) {
        std::string camstring(this->cameraSettingsSlot.Param<param::StringParam>()->Value());
        cam_type::minimal_state_type minstate;
        if (!this->serializer.deserialize(minstate, camstring)) {
            megamol::core::utility::log::Log::DefaultLog.WriteWarn(
                "The entered camera string was not valid. No change of the camera has been performed");
        } else {
            this->cam = minstate;
            return true;
        }
    }

    auto path = this->determineCameraFilePath();
    if (path.empty()) {
        megamol::core::utility::log::Log::DefaultLog.WriteWarn(
            "The camera file path could not be determined. This is probably due to the usage of .mmprj project "
            "files. Please use a .lua project file instead");
        return false;
    }

    std::ifstream file(path);
    std::string text;
    if (file.is_open()) {
        text.assign(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
    } else {
        megamol::core::utility::log::Log::DefaultLog.WriteWarn("The camera output file at '%s' could not be opened.", path.c_str());
        return false;
    }
    auto copy = this->savedCameras;
    bool success = this->serializer.deserialize(copy, text);
    if (!success) {
        megamol::core::utility::log::Log::DefaultLog.WriteWarn(
            "The reading of the camera parameters did not work properly. No changes were made.");
        return false;
    }
    this->savedCameras = copy;
    if (this->savedCameras.back().second) {
        this->cam = this->savedCameras.back().first;
    } else {
        megamol::core::utility::log::Log::DefaultLog.WriteWarn("The stored default cam was not valid. The old default cam is used");
    }
    return true;
}

/*
 * AbstractView3D::onResetView
 */
bool AbstractView3D::onResetView(param::ParamSlot& p) {
    this->ResetView();
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
 * AbstractView3D::determineCameraFilePath
 */
std::string AbstractView3D::determineCameraFilePath(void) const {
    auto path = this->GetCoreInstance()->GetLuaState()->GetScriptPath();
    if (path.empty()) return path; // early exit for mmprj projects
    auto dotpos = path.find_last_of('.');
    path = path.substr(0, dotpos);
    path.append("_cam.json");
    return path;
}

/*
 * AbstractView3D::handleCameraMovement
 */
void AbstractView3D::handleCameraMovement(void) {
    float step = this->viewKeyMoveStepSlot.Param<param::FloatParam>()->Value();
    float dt = std::chrono::duration<float>(this->lastFrameDuration).count();
    step *= dt;

    const float runFactor = this->viewKeyRunFactorSlot.Param<param::FloatParam>()->Value();
    if (this->modkeys.test(view::Modifier::SHIFT)) {
        step *= runFactor;
    }

    bool anymodpressed = !this->modkeys.none();
    float rotationStep = this->viewKeyAngleStepSlot.Param<param::FloatParam>()->Value();
    rotationStep *= dt;

    glm::vec3 currCamPos(static_cast<glm::vec4>(this->cam.eye_position()));
    float orbitalAltitude = glm::length(currCamPos - rotCenter);

    if (this->translateManipulator.manipulating()) {

        if (this->pressedKeyMap.count(view::Key::KEY_W) > 0 && this->pressedKeyMap[view::Key::KEY_W]) {
            this->translateManipulator.move_forward(step);
        }
        if (this->pressedKeyMap.count(view::Key::KEY_S) > 0 && this->pressedKeyMap[view::Key::KEY_S]) {
            this->translateManipulator.move_forward(-step);
        }
        if (this->pressedKeyMap.count(view::Key::KEY_A) > 0 && this->pressedKeyMap[view::Key::KEY_A]) {
            this->translateManipulator.move_horizontally(-step);
        }
        if (this->pressedKeyMap.count(view::Key::KEY_D) > 0 && this->pressedKeyMap[view::Key::KEY_D]) {
            this->translateManipulator.move_horizontally(step);
        }
        if (this->pressedKeyMap.count(view::Key::KEY_C) > 0 && this->pressedKeyMap[view::Key::KEY_C]) {
            this->translateManipulator.move_vertically(step);
        }
        if (this->pressedKeyMap.count(view::Key::KEY_V) > 0 && this->pressedKeyMap[view::Key::KEY_V]) {
            this->translateManipulator.move_vertically(-step);
        }
        if (this->pressedKeyMap.count(view::Key::KEY_Q) > 0 && this->pressedKeyMap[view::Key::KEY_Q]) {
            this->rotateManipulator.roll(-rotationStep);
        }
        if (this->pressedKeyMap.count(view::Key::KEY_E) > 0 && this->pressedKeyMap[view::Key::KEY_E]) {
            this->rotateManipulator.roll(rotationStep);
        }
        if (this->pressedKeyMap.count(view::Key::KEY_UP) > 0 && this->pressedKeyMap[view::Key::KEY_UP]) {
            this->rotateManipulator.pitch(-rotationStep);
        }
        if (this->pressedKeyMap.count(view::Key::KEY_DOWN) > 0 && this->pressedKeyMap[view::Key::KEY_DOWN]) {
            this->rotateManipulator.pitch(rotationStep);
        }
        if (this->pressedKeyMap.count(view::Key::KEY_LEFT) > 0 && this->pressedKeyMap[view::Key::KEY_LEFT]) {
            this->rotateManipulator.yaw(rotationStep);
        }
        if (this->pressedKeyMap.count(view::Key::KEY_RIGHT) > 0 && this->pressedKeyMap[view::Key::KEY_RIGHT]) {
            this->rotateManipulator.yaw(-rotationStep);
        }
    }

    glm::vec3 newCamPos(static_cast<glm::vec4>(this->cam.eye_position()));
    glm::vec3 camDir(static_cast<glm::vec4>(this->cam.view_vector()));
    rotCenter = newCamPos + orbitalAltitude * glm::normalize(camDir);
}

/*
 * AbstractView3D::setCameraValues
 */
void AbstractView3D::setCameraValues(const view::Camera_2& cam) {
    glm::vec4 pos = cam.position();
    const bool makeDirty = false;
    this->cameraPositionParam.Param<param::Vector3fParam>()->SetValue(
        vislib::math::Vector<float, 3>(pos.x, pos.y, pos.z), makeDirty);
    this->cameraPositionParam.QueueUpdateNotification();

    glm::quat orient = cam.orientation();
    this->cameraOrientationParam.Param<param::Vector4fParam>()->SetValue(
        vislib::math::Vector<float, 4>(orient.x, orient.y, orient.z, orient.w), makeDirty);
    this->cameraOrientationParam.QueueUpdateNotification();

    this->cameraProjectionTypeParam.Param<param::EnumParam>()->SetValue(static_cast<int>(cam.projection_type()), makeDirty);
    this->cameraProjectionTypeParam.QueueUpdateNotification();

    this->cameraConvergencePlaneParam.Param<param::FloatParam>()->SetValue(cam.convergence_plane(), makeDirty);
    this->cameraConvergencePlaneParam.QueueUpdateNotification();

    /*this->cameraNearPlaneParam.Param<param::FloatParam>()->SetValue(cam.near_clipping_plane(), makeDirty);
    this->cameraFarPlaneParam.Param<param::FloatParam>()->SetValue(cam.far_clipping_plane(), makeDirty);*/
    /*this->cameraEyeParam.Param<param::EnumParam>()->SetValue(static_cast<int>(cam.eye()), makeDirty);
    this->cameraGateScalingParam.Param<param::EnumParam>()->SetValue(static_cast<int>(cam.gate_scaling()), makeDirty);
    this->cameraFilmGateParam.Param<param::Vector2fParam>()->SetValue(
        vislib::math::Vector<float, 2>(cam.film_gate().width(), cam.film_gate().height()), makeDirty);*/
    /*this->cameraResolutionXParam.Param<param::IntParam>()->SetValue(cam.resolution_gate().width());
    this->cameraResolutionYParam.Param<param::IntParam>()->SetValue(cam.resolution_gate().height());*/

    this->cameraCenterOffsetParam.Param<param::Vector2fParam>()->SetValue(
        vislib::math::Vector<float, 2>(cam.centre_offset().x(), cam.centre_offset().y()), makeDirty);
    this->cameraCenterOffsetParam.QueueUpdateNotification();

    this->cameraHalfApertureDegreesParam.Param<param::FloatParam>()->SetValue(
        cam.half_aperture_angle_radians() * 180.0f / M_PI, makeDirty);
    this->cameraHalfApertureDegreesParam.QueueUpdateNotification();

    this->cameraHalfDisparityParam.Param<param::FloatParam>()->SetValue(cam.half_disparity(), makeDirty);
    this->cameraHalfDisparityParam.QueueUpdateNotification();
}

/*
 * AbstractView3D::adaptCameraValues
 */
bool AbstractView3D::adaptCameraValues(view::Camera_2& cam) {
    bool result = false;
    if (this->cameraPositionParam.IsDirty()) {
        auto val = this->cameraPositionParam.Param<param::Vector3fParam>()->Value();
        this->cam.position(glm::vec4(val.GetX(), val.GetY(), val.GetZ(), 1.0f));
        this->cameraPositionParam.ResetDirty();
        result = true;
    }
    if (this->cameraOrientationParam.IsDirty()) {
        auto val = this->cameraOrientationParam.Param<param::Vector4fParam>()->Value();
        this->cam.orientation(glm::quat(val.GetW(), val.GetX(), val.GetY(), val.GetZ()));
        this->cameraOrientationParam.ResetDirty();
        result = true;
    }
    if (this->cameraProjectionTypeParam.IsDirty()) {
        auto val =
            static_cast<thecam::Projection_type>(this->cameraProjectionTypeParam.Param<param::EnumParam>()->Value());
        this->cam.projection_type(val);
        this->cameraProjectionTypeParam.ResetDirty();
        result = true;
    }
    //// setting of near plane and far plane might make no sense as we are setting them new each frame anyway
    // if (this->cameraNearPlaneParam.IsDirty()) {
    //    auto val = this->cameraNearPlaneParam.Param<param::FloatParam>()->Value();
    //    this->cam.near_clipping_plane(val);
    //    this->cameraNearPlaneParam.ResetDirty();
    //    result = true;
    //}
    // if (this->cameraFarPlaneParam.IsDirty()) {
    //    auto val = this->cameraFarPlaneParam.Param<param::FloatParam>()->Value();
    //    this->cam.far_clipping_plane(val);
    //    this->cameraFarPlaneParam.ResetDirty();
    //    result = true;
    //}
    if (this->cameraConvergencePlaneParam.IsDirty()) {
        auto val = this->cameraConvergencePlaneParam.Param<param::FloatParam>()->Value();
        this->cam.convergence_plane(val);
        this->cameraConvergencePlaneParam.ResetDirty();
        result = true;
    }
    /*if (this->cameraEyeParam.IsDirty()) {
        auto val = static_cast<thecam::Eye>(this->cameraEyeParam.Param<param::EnumParam>()->Value());
        this->cam.eye(val);
        this->cameraEyeParam.ResetDirty();
        result = true;
    }*/
    /*if (this->cameraGateScalingParam.IsDirty()) {
        auto val = static_cast<thecam::Gate_scaling>(this->cameraGateScalingParam.Param<param::EnumParam>()->Value());
        this->cam.gate_scaling(val);
        this->cameraGateScalingParam.ResetDirty();
        result = true;
    }
    if (this->cameraFilmGateParam.IsDirty()) {
        auto val = this->cameraFilmGateParam.Param<param::Vector2fParam>()->Value();
        this->cam.film_gate(thecam::math::size<float, 2>(val.GetX(), val.GetY()));
        this->cameraFilmGateParam.ResetDirty();
        result = true;
    }*/
    /*if (this->cameraResolutionXParam.IsDirty()) {
        auto val = this->cameraResolutionXParam.Param<param::IntParam>()->Value();
        this->cam.resolution_gate({val, this->cam.resolution_gate().height()});
        this->cameraResolutionXParam.ResetDirty();
        result = true;
    }
    if (this->cameraResolutionYParam.IsDirty()) {
        auto val = this->cameraResolutionYParam.Param<param::IntParam>()->Value();
        this->cam.resolution_gate({this->cam.resolution_gate().width(), val});
        this->cameraResolutionYParam.ResetDirty();
        result = true;
    }*/
    if (this->cameraCenterOffsetParam.IsDirty()) {
        auto val = this->cameraCenterOffsetParam.Param<param::Vector2fParam>()->Value();
        this->cam.centre_offset(thecam::math::vector<float, 2>(val.GetX(), val.GetY()));
        this->cameraCenterOffsetParam.ResetDirty();
        result = true;
    }
    if (this->cameraHalfApertureDegreesParam.IsDirty()) {
        auto val = this->cameraHalfApertureDegreesParam.Param<param::FloatParam>()->Value();
        this->cam.half_aperture_angle_radians(val * M_PI / 180.0f);
        this->cameraHalfApertureDegreesParam.ResetDirty();
        result = true;
    }
    if (this->cameraHalfDisparityParam.IsDirty()) {
        auto val = this->cameraHalfDisparityParam.Param<param::FloatParam>()->Value();
        this->cam.half_disparity(val);
        this->cameraHalfDisparityParam.ResetDirty();
        result = true;
    }
    return result;
}

/*
 * AbstractView3D::cameraOvrCallback
 */
bool AbstractView3D::cameraOvrCallback(param::ParamSlot& p) {
    auto up_vis = this->cameraOvrUpParam.Param<param::Vector3fParam>()->Value();
    auto lookat_vis = this->cameraOvrLookatParam.Param<param::Vector3fParam>()->Value();

    glm::vec3 up(up_vis.X(), up_vis.Y(), up_vis.Z());
    up = glm::normalize(up);
    glm::vec3 lookat(lookat_vis.X(), lookat_vis.Y(), lookat_vis.Z());

    glm::mat3 view;
    view[2] = -glm::normalize(lookat - glm::vec3(static_cast<glm::vec4>(this->cam.eye_position())));
    view[0] = glm::normalize(glm::cross(up, view[2]));
    view[1] = glm::normalize(glm::cross(view[2], view[0]));

    auto orientation = glm::quat_cast(view);

    this->cam.orientation(orientation);
    this->rotCenter = lookat;

    return true;
}
