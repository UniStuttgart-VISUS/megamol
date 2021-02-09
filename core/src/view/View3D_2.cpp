/*
 * View3D_2.cpp
 *
 * Copyright (C) 2018, 2020 by VISUS (Universitaet Stuttgart).
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "mmcore/view/View3D_2.h"
#include "vislib/graphics/gl/IncludeAllGL.h"
#ifdef _WIN32
#    include <windows.h>
#endif /* _WIN32 */
#include <chrono>
#include <fstream>
#include <glm/gtx/string_cast.hpp>
#include "mmcore/CoreInstance.h"
#include "mmcore/misc/PngBitmapCodec.h"
#include "mmcore/param/BoolParam.h"
#include "mmcore/param/ButtonParam.h"
#include "mmcore/param/EnumParam.h"
#include "mmcore/param/FilePathParam.h"
#include "mmcore/param/FloatParam.h"
#include "mmcore/param/IntParam.h"
#include "mmcore/param/StringParam.h"
#include "mmcore/param/Vector2fParam.h"
#include "mmcore/param/Vector3fParam.h"
#include "mmcore/param/Vector4fParam.h"
#include "mmcore/utility/ColourParser.h"
#include "mmcore/view/AbstractCallRender.h"
#include "mmcore/view/CallRender3D_2.h"
#include "mmcore/view/CallRenderView.h"
#include "mmcore/view/CameraParamOverride.h"
#include "vislib/Exception.h"
#include "vislib/String.h"
#include "vislib/StringSerialiser.h"
#include "vislib/Trace.h"
#include "vislib/graphics/BitmapImage.h"
#include "vislib/graphics/CameraParamsStore.h"
#include "vislib/math/Point.h"
#include "vislib/math/Quaternion.h"
#include "vislib/math/Vector.h"
#include "vislib/math/mathfunctions.h"
#include "vislib/sys/KeyCode.h"
#include "mmcore/utility/log/Log.h"
#include "vislib/sys/sysfunctions.h"
#include <glm/gtx/string_cast.hpp>

#include "glm/gtc/matrix_transform.hpp"

using namespace megamol::core;
using namespace megamol::core::view;

/*
 * View3D_2::View3D_2
 */
View3D_2::View3D_2(void)
    : view::AbstractRenderingView()
    /*, view::AbstractCamParamSync()*/
    , cursor2d()
    , rendererSlot("rendering", "Connects the view to a Renderer")
    , bboxs()
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
    , firstImg(false)
    , stereoFocusDistSlot("stereo::focusDist", "focus distance for stereo projection")
    , stereoEyeDistSlot("stereo::eyeDist", "eye distance for stereo projection")
    , overrideCall(NULL)
    , viewKeyMoveStepSlot("viewKey::MoveStep", "The move step size in world coordinates")
    , viewKeyRunFactorSlot("viewKey::RunFactor", "The factor for step size multiplication when running (shift)")
    , viewKeyAngleStepSlot("viewKey::AngleStep", "The angle rotate step in degrees")
    , viewKeyFixToWorldUpSlot("viewKey::FixToWorldUp","Fix rotation manipulator to world up vector")
    , mouseSensitivitySlot("viewKey::MouseSensitivity", "used for WASD mode")
    , viewKeyRotPointSlot("viewKey::RotPoint", "The point around which the view will be rotated")
    , enableMouseSelectionSlot("enableMouseSelection", "Enable selecting and picking with the mouse")
    , resetViewOnBBoxChangeSlot("resetViewOnBBoxChange", "Whether to reset the view when the bounding boxes change")
    , cameraSetViewChooserParam("view::defaultView", "Choose a default view to look from")
    , cameraViewOrientationParam("view::cubeOrientation", "Current camera orientation used for view cube.")
    , showViewCubeParam("view::showViewCube", "Shows view cube.")
    , resetViewSlot("resetView", "Triggers the reset of the view")
    , mouseX(0.0f)
    , mouseY(0.0f)
    , mouseFlags(0)
    , timeCtrl()
    , toggleMouseSelection(false)
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

    // this->camParams = this->cam.Parameters();
    // this->camOverrides = new CameraParamOverride(this->camParams);

    // vislib::graphics::ImageSpaceType defWidth(static_cast<vislib::graphics::ImageSpaceType>(100));
    // vislib::graphics::ImageSpaceType defHeight(static_cast<vislib::graphics::ImageSpaceType>(100));

    // this->camParams->SetVirtualViewSize(defWidth, defHeight);
    // this->camParams->SetTileRect(vislib::math::Rectangle<float>(0.0f, 0.0f, defWidth, defHeight));

    this->cam.resolution_gate(cam_type::screen_size_type(100, 100));
    this->cam.image_tile(cam_type::screen_rectangle_type(std::array<int, 4>{0, 100, 100, 0}));

    this->rendererSlot.SetCompatibleCall<CallRender3D_2Description>();
    this->MakeSlotAvailable(&this->rendererSlot);

    // this triggers the initialization
    this->bboxs.Clear();

    this->showLookAt.SetParameter(new param::BoolParam(false));
    this->MakeSlotAvailable(&this->showLookAt);

    this->cameraSettingsSlot.SetParameter(new param::StringParam(""));
    this->MakeSlotAvailable(&this->cameraSettingsSlot);

    this->storeCameraSettingsSlot.SetParameter(
        new param::ButtonParam(view::Key::KEY_C, (view::Modifier::SHIFT | view::Modifier::ALT)));
    this->storeCameraSettingsSlot.SetUpdateCallback(&View3D_2::onStoreCamera);
    this->MakeSlotAvailable(&this->storeCameraSettingsSlot);

    this->restoreCameraSettingsSlot.SetParameter(new param::ButtonParam(view::Key::KEY_C, view::Modifier::ALT));
    this->restoreCameraSettingsSlot.SetUpdateCallback(&View3D_2::onRestoreCamera);
    this->MakeSlotAvailable(&this->restoreCameraSettingsSlot);

    this->overrideCamSettingsSlot.SetParameter(new param::BoolParam(false));
    this->MakeSlotAvailable(&this->overrideCamSettingsSlot);

    this->autoSaveCamSettingsSlot.SetParameter(new param::BoolParam(false));
    this->MakeSlotAvailable(&this->autoSaveCamSettingsSlot);

    this->autoLoadCamSettingsSlot.SetParameter(new param::BoolParam(true));
    this->MakeSlotAvailable(&this->autoLoadCamSettingsSlot);

    // TODO
    // this->stereoEyeDistSlot << new param::FloatParam(this->camParams->StereoDisparity(), 0.0f);
    // this->MakeSlotAvailable(&this->stereoEyeDistSlot);

    // TODO
    // this->stereoFocusDistSlot << new param::FloatParam(this->camParams->FocalDistance(false), 0.0f);
    // this->MakeSlotAvailable(&this->stereoFocusDistSlot);

    this->viewKeyMoveStepSlot.SetParameter(new param::FloatParam(0.5f, 0.001f));
    this->MakeSlotAvailable(&this->viewKeyMoveStepSlot);

    this->viewKeyRunFactorSlot.SetParameter(new param::FloatParam(2.0f, 0.1f));
    this->MakeSlotAvailable(&this->viewKeyRunFactorSlot);

    this->viewKeyAngleStepSlot.SetParameter(new param::FloatParam(90.0f, 0.1f, 360.0f));
    this->MakeSlotAvailable(&this->viewKeyAngleStepSlot);

    this->viewKeyFixToWorldUpSlot.SetParameter(new param::BoolParam(true));
    this->MakeSlotAvailable(&this->viewKeyFixToWorldUpSlot);

    this->mouseSensitivitySlot.SetParameter(new param::FloatParam(3.0f, 0.001f, 10.0f));
    this->mouseSensitivitySlot.SetUpdateCallback(&View3D_2::mouseSensitivityChanged);
    this->MakeSlotAvailable(&this->mouseSensitivitySlot);

    // TODO clean up vrpsev memory after use
    param::EnumParam* vrpsev = new param::EnumParam(1);
    vrpsev->SetTypePair(0, "Position");
    vrpsev->SetTypePair(1, "Look-At");
    this->viewKeyRotPointSlot.SetParameter(vrpsev);
    this->MakeSlotAvailable(&this->viewKeyRotPointSlot);

    this->enableMouseSelectionSlot.SetParameter(new param::ButtonParam(view::Key::KEY_TAB));
    this->enableMouseSelectionSlot.SetUpdateCallback(&View3D_2::onToggleButton);
    this->MakeSlotAvailable(&this->enableMouseSelectionSlot);

    for (unsigned int i = 0; this->timeCtrl.GetSlot(i) != NULL; i++) {
        this->MakeSlotAvailable(this->timeCtrl.GetSlot(i));
    }

    // this->MakeSlotAvailable(&this->slotGetCamParams);
    // this->MakeSlotAvailable(&this->slotSetCamParams);

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

    /*auto camnearparam = new param::FloatParam(0.1f, 0.0f);
    camnearparam->SetGUIVisible(camparamvisibility);
    this->cameraNearPlaneParam.SetParameter(camnearparam);
    this->MakeSlotAvailable(&this->cameraNearPlaneParam);

    auto camfarparam = new param::FloatParam(0.1f, 0.0f);
    camfarparam->SetGUIVisible(camparamvisibility);
    this->cameraFarPlaneParam.SetParameter(camfarparam);
    this->MakeSlotAvailable(&this->cameraFarPlaneParam);*/

    auto camconvergenceparam = new param::FloatParam(0.1f, 0.0f);
    camconvergenceparam->SetGUIVisible(camparamvisibility);
    this->cameraConvergencePlaneParam.SetParameter(camconvergenceparam);
    this->MakeSlotAvailable(&this->cameraConvergencePlaneParam);

    /*auto eyeparam = new param::EnumParam(static_cast<int>(core::thecam::Eye::mono));
    eyeparam->SetTypePair(static_cast<int>(core::thecam::Eye::left), "Left");
    eyeparam->SetTypePair(static_cast<int>(core::thecam::Eye::mono), "Mono");
    eyeparam->SetTypePair(static_cast<int>(core::thecam::Eye::right), "Right");
    eyeparam->SetGUIVisible(camparamvisibility);
    this->cameraEyeParam.SetParameter(eyeparam);
    this->MakeSlotAvailable(&this->cameraEyeParam);*/

    /*auto gateparam = new param::EnumParam(static_cast<int>(core::thecam::Gate_scaling::none));
    gateparam->SetTypePair(static_cast<int>(core::thecam::Gate_scaling::none), "None");
    gateparam->SetTypePair(static_cast<int>(core::thecam::Gate_scaling::fill), "Fill");
    gateparam->SetTypePair(static_cast<int>(core::thecam::Gate_scaling::overscan), "Overscan");
    gateparam->SetGUIVisible(camparamvisibility);
    this->cameraGateScalingParam.SetParameter(gateparam);
    this->MakeSlotAvailable(&this->cameraGateScalingParam);

    auto filmparam = new param::Vector2fParam(vislib::math::Vector<float, 2>());
    filmparam->SetGUIVisible(camparamvisibility);
    this->cameraFilmGateParam.SetParameter(filmparam);
    this->MakeSlotAvailable(&this->cameraFilmGateParam);

    auto resxparam = new param::IntParam(100);
    resxparam->SetGUIVisible(camparamvisibility);
    this->cameraResolutionXParam.SetParameter(resxparam);
    this->MakeSlotAvailable(&this->cameraResolutionXParam);

    auto resyparam = new param::IntParam(100);
    resyparam->SetGUIVisible(camparamvisibility);
    this->cameraResolutionYParam.SetParameter(resyparam);
    this->MakeSlotAvailable(&this->cameraResolutionYParam);*/

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
    this->cameraOvrParam.SetUpdateCallback(&View3D_2::cameraOvrCallback);
    this->MakeSlotAvailable(&this->cameraOvrParam);

    this->resetViewSlot.SetParameter(new param::ButtonParam(view::Key::KEY_HOME));
    this->resetViewSlot.SetUpdateCallback(&View3D_2::onResetView);
    this->MakeSlotAvailable(&this->resetViewSlot);

    this->resetViewOnBBoxChangeSlot.SetParameter(new param::BoolParam(false));
    this->MakeSlotAvailable(&this->resetViewOnBBoxChangeSlot);

    auto defaultViewParam = new param::EnumParam(0);
    defaultViewParam->SetTypePair(defaultview::DEFAULTVIEW_FRONT, "Front");
    defaultViewParam->SetTypePair(defaultview::DEFAULTVIEW_BACK, "Back");
    defaultViewParam->SetTypePair(defaultview::DEFAULTVIEW_RIGHT, "Right");
    defaultViewParam->SetTypePair(defaultview::DEFAULTVIEW_LEFT, "Left");
    defaultViewParam->SetTypePair(defaultview::DEFAULTVIEW_TOP, "Top");
    defaultViewParam->SetTypePair(defaultview::DEFAULTVIEW_BOTTOM, "Bottom");
    defaultViewParam->SetGUIVisible(camparamvisibility);
    this->cameraSetViewChooserParam.SetParameter(defaultViewParam),
    this->MakeSlotAvailable(&this->cameraSetViewChooserParam);
    this->cameraSetViewChooserParam.SetUpdateCallback(&View3D_2::onResetView);

    this->cameraViewOrientationParam.SetParameter(new param::Vector4fParam(vislib::math::Vector<float, 4>(0.0f, 0.0f, 0.0f, 1.0f)));
    this->MakeSlotAvailable(&this->cameraViewOrientationParam);
    this->cameraViewOrientationParam.Parameter()->SetGUIReadOnly(true);
    this->cameraViewOrientationParam.Parameter()->SetGUIVisible(false);

    this->showViewCubeParam.SetParameter(new param::BoolParam(false));
    this->MakeSlotAvailable(&this->showViewCubeParam);

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


    this->ResetView();

    // none of the saved camera states are valid right now
    for (auto& e : this->savedCameras) {
        e.second = false;
    }
}

/*
 * View3D_2::~View3D_2
 */
View3D_2::~View3D_2(void) {
    this->Release();
    this->overrideCall = nullptr; // DO NOT DELETE
}

/*
 * View3D_2::GetCameraSyncNumber
 */
unsigned int view::View3D_2::GetCameraSyncNumber(void) const {
    // TODO implement
    return 0;
}


/*
 * View3D_2::SerialiseCamera
 */
void view::View3D_2::SerialiseCamera(vislib::Serialiser& serialiser) const {
    // TODO currently emtpy because the old serialization sucks
}


/*
 * View3D_2::DeserialiseCamera
 */
void view::View3D_2::DeserialiseCamera(vislib::Serialiser& serialiser) {
    // TODO currently empty because the old serialization sucks
}

/*
 * View3D_2::Render
 */
void View3D_2::Render(const mmcRenderViewContext& context) {
    float time = static_cast<float>(context.Time);
    float instTime = static_cast<float>(context.InstanceTime);

    if (this->doHookCode()) {
        this->doBeforeRenderHook();
    }

    glm::ivec4 currentViewport;
    CallRender3D_2* cr3d = this->rendererSlot.CallAs<CallRender3D_2>();
    if (cr3d != nullptr) {
        cr3d->SetMouseSelection(this->toggleMouseSelection);
    }

    this->handleCameraMovement();

    auto cam_orientation = static_cast<glm::quat>(this->cam.orientation());
    this->cameraViewOrientationParam.Param<param::Vector4fParam>()->SetValue(
        vislib::math::Vector<float, 4>(cam_orientation.x, cam_orientation.y, cam_orientation.z, cam_orientation.w));

    AbstractRenderingView::beginFrame();

    // TODO Conditionally synchronise camera from somewhere else.
    // this->SyncCamParams(this->cam.Parameters());

    // clear viewport
    ///*if (this->overrideViewport != nullptr) {
    //    if ((this->overrideViewport[0] >= 0) && (this->overrideViewport[1] >= 0) && (this->overrideViewport[2] >= 0)
    //    &&
    //        (this->overrideViewport[3] >= 0)) {
    //        currentViewport = glm::ivec4(this->overrideViewport[0], this->overrideViewport[1],
    //            this->overrideViewport[2], this->overrideViewport[3]);
    //    }
    //} else */{
    //    // this is correct in non-override mode,
    //    //  because then the tile will be whole viewport
    //    //auto camRes = this->cam.resolution_gate();
    //    //currentViewport = glm::ivec4(0, 0, camRes.width(), camRes.height());
    //}

    if (this->overrideCall != nullptr) {
        if (cr3d != nullptr) {
            view::RenderOutputOpenGL* ro = dynamic_cast<view::RenderOutputOpenGL*>(overrideCall);
            if (ro != nullptr) {
                *static_cast<view::RenderOutputOpenGL*>(cr3d) = *ro;
            }
        }
        this->overrideCall->EnableOutputBuffer();
    } else if (cr3d != nullptr) {
        cr3d->SetOutputBuffer(GL_BACK);
        cr3d->GetViewport(); // access the viewport to enforce evaluation TODO is this still necessary
    }

    const float* bkgndCol = (this->overrideBkgndCol != nullptr) ? this->overrideBkgndCol : this->BkgndColour();

    if (cr3d == NULL) {
        /*this->renderTitle(this->cam.Parameters()->TileRect().Left(), this->cam.Parameters()->TileRect().Bottom(),
            this->cam.Parameters()->TileRect().Width(), this->cam.Parameters()->TileRect().Height(),
            this->cam.Parameters()->VirtualViewSize().Width(), this->cam.Parameters()->VirtualViewSize().Height(),
            (this->cam.Parameters()->Projection() != vislib::graphics::CameraParameters::MONO_ORTHOGRAPHIC) &&
            (this->cam.Parameters()->Projection() != vislib::graphics::CameraParameters::MONO_PERSPECTIVE),
            this->cam.Parameters()->Eye() == vislib::graphics::CameraParameters::LEFT_EYE, instTime);*/
        this->endFrame(true);
        return; // empty enought
    } else {
        cr3d->SetGpuAffinity(context.GpuAffinity);
        this->removeTitleRenderer();
        cr3d->SetBackgroundColor(glm::vec4(bkgndCol[0], bkgndCol[1], bkgndCol[2], 0.0f));
    }

    // mueller: I moved the following code block before clearing the back buffer,
    // because in case the FBO is enabled here, the depth buffer is not cleared
    // (but the one of the previous buffer) and the renderer might not create any
    // fragment in this case - besides that the FBO content is not cleared,
    // which could be a problem if the FBO is reused.
    // if (this->overrideCall != NULL) {
    //    this->overrideCall->EnableOutputBuffer();
    //} else {
    //    cr3d->SetOutputBuffer(GL_BACK);
    //}

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

        this->timeCtrl.SetTimeExtend(cr3d->TimeFramesCount(), cr3d->IsInSituTime());
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

    // TODO
    // this->camParams->CalcClipping(this->bboxs.ClipBox(), 0.1f);
    // This is painfully wrong in the vislib camera, and is fixed here as sort of hotfix
    // float fc = this->camParams->FarClip();
    // float nc = this->camParams->NearClip();
    // float fnc = fc * 0.001f;
    // if (fnc > nc) {
    //     this->camParams->SetClip(fnc, fc);
    // }
    this->cam.CalcClipping(this->bboxs.ClipBox(), 0.1f);

    cam_type::snapshot_type camsnap;
    cam_type::matrix_type viewCam, projCam;
    this->cam.calc_matrices(camsnap, viewCam, projCam);

    glm::mat4 view = viewCam;
    glm::mat4 proj = projCam;
    glm::mat4 mvp = projCam * viewCam;

    /*glm::vec3 eyepos(cam.eye_position().x(), cam.eye_position().y(), cam.eye_position().z());
    glm::vec3 snappos(camsnap.position.x(), camsnap.position.y(), camsnap.position.z());
    printf("%u: %s %s\n", this->GetCoreInstance()->GetFrameID(), glm::to_string(view).c_str(),
        glm::to_string(proj).c_str());
    printf("%u: %s %s\n", this->GetCoreInstance()->GetFrameID(), glm::to_string(eyepos).c_str(),
        glm::to_string(snappos).c_str());*/

    if (cr3d != nullptr) {
        cr3d->SetCameraState(this->cam);
        (*cr3d)(view::AbstractCallRender::FnRender);
    }

    this->setCameraValues(this->cam);

    AbstractRenderingView::endFrame();

    // this->lastFrameParams->CopyFrom(this->OnGetCamParams, false);

    if (this->doHookCode() && frameIsNew) {
        this->doAfterRenderHook();
    }
}

/*
 * View3D_2::ResetView
 */
void View3D_2::ResetView(void) {
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
    double pseudoWidth = this->bboxs.BoundingBox().Width();
    double pseudoHeight = this->bboxs.BoundingBox().Height();
    double pseudoDepth = this->bboxs.BoundingBox().Depth();
    defaultview dv = static_cast<defaultview>(this->cameraSetViewChooserParam.Param<param::EnumParam>()->Value());
    switch (dv) {
    case DEFAULTVIEW_FRONT :
    case DEFAULTVIEW_BACK : // this is the init case above or equivalent
    break;
    case DEFAULTVIEW_RIGHT :
    case DEFAULTVIEW_LEFT : pseudoWidth = this->bboxs.BoundingBox().Depth();
    pseudoHeight = this->bboxs.BoundingBox().Height();
    pseudoDepth = this->bboxs.BoundingBox().Width();
    break;
    case DEFAULTVIEW_TOP :
    case DEFAULTVIEW_BOTTOM : pseudoWidth = this->bboxs.BoundingBox().Width();
    pseudoHeight = this->bboxs.BoundingBox().Depth();
    pseudoDepth = this->bboxs.BoundingBox().Height();
    break;
    default:;
      
    }
    auto dim = this->cam.resolution_gate();
    double halfFovX =
        (static_cast<double>(dim.width()) * static_cast<double>(this->cam.aperture_angle_radians() / 2.0f)) /
        static_cast<double>(dim.height());
    double distX = pseudoWidth / (2.0 * tan(halfFovX));
    double distY = pseudoHeight /
                   (2.0 * tan(static_cast<double>(this->cam.aperture_angle_radians() / 2.0f)));
    float dist = static_cast<float>((distX > distY) ? distX : distY);
    dist = dist + (pseudoDepth / 2.0f);
    auto bbc = this->bboxs.BoundingBox().CalcCenter();
    auto bbcglm = glm::vec4(bbc.GetX(), bbc.GetY(), bbc.GetZ(), 1.0f);
    const double cos0 = 0.0;
    const double cos45 = sqrt(2.0) / 2.0;
    const double cos90 = 1.0;
    const double sin0 = 1.0;
    const double sin45 = cos45;
    const double sin90 = 0.0;
    if (!this->valuesFromOutside) {
        // quat rot(theta) around axis(x,y,z) -> q = (sin(theta/2)*x, sin(theta/2)*y, sin(theta/2)*z, cos(theta/2))
        switch (dv) {
            case DEFAULTVIEW_FRONT : this->cam.orientation(cam_type::quaternion_type::create_identity());
            this->cam.position(bbcglm + glm::vec4(0.0f, 0.0f, dist, 0.0f));
            break;
            case DEFAULTVIEW_BACK : // 180 deg around y axis
            this->cam.orientation(cam_type::quaternion_type(0, 1.0, 0, 0.0f));
            this->cam.position(bbcglm + glm::vec4(0.0f, 0.0f, -dist, 0.0f));
            break;
            case DEFAULTVIEW_RIGHT : // 90 deg around y axis
            this->cam.orientation(cam_type::quaternion_type(0, sin45 * 1.0, 0, cos45));
            this->cam.position(bbcglm + glm::vec4(dist, 0.0f, 0.0f, 0.0f));
            break;
            case DEFAULTVIEW_LEFT : // 90 deg reverse around y axis
            this->cam.orientation(cam_type::quaternion_type(0, -sin45 * 1.0, 0, cos45));
            this->cam.position(bbcglm + glm::vec4(-dist, 0.0f, 0.0f, 0.0f));
            break;
            case DEFAULTVIEW_TOP : // 90 deg around x axis
            this->cam.orientation(cam_type::quaternion_type(-sin45 * 1.0, 0, 0, cos45));
            this->cam.position(bbcglm + glm::vec4(0.0f, dist, 0.0f, 0.0f));
            break;
            case DEFAULTVIEW_BOTTOM : // 90 deg reverse around x axis
            this->cam.orientation(cam_type::quaternion_type(sin45 * 1.0, 0, 0, cos45));
            this->cam.position(bbcglm + glm::vec4(0.0f, -dist, 0.0f, 0.0f));
            break;
            default:;
        }
    }

    this->rotCenter = glm::vec3(bbc.GetX(), bbc.GetY(), bbc.GetZ());

    // TODO Further manipulators? better value?
    this->valuesFromOutside = false;
}

/*
 * View3D_2::Resize
 */
void View3D_2::Resize(unsigned int width, unsigned int height) {
    if (this->cam.resolution_gate().width() != width || this->cam.resolution_gate().height() != height) {
        this->cam.resolution_gate(cam_type::screen_size_type(static_cast<LONG>(width), static_cast<LONG>(height)));
    }
    if (this->cam.image_tile().width() != width || this->cam.image_tile().height() != height) {
        this->cam.image_tile(cam_type::screen_rectangle_type(
            std::array<int, 4>({0, static_cast<int>(height), static_cast<int>(width), 0})));
    }
}

/*
 * View3D_2::OnRenderView
 */
bool View3D_2::OnRenderView(Call& call) {
    std::array<float, 3> overBC;
    // std::array<int, 4> overVP = {0, 0, 0, 0};
    view::CallRenderView* crv = dynamic_cast<view::CallRenderView*>(&call);
    if (crv == nullptr) return false;

    // this->overrideViewport = overVP.data();
    // if (crv->IsViewportSet()) {
    //    overVP[2] = crv->ViewportWidth();
    //    overVP[3] = crv->ViewportHeight();
    //    if (!crv->IsTileSet()) {
    //        // TODO
    //    }
    //}
    if (crv->IsBackgroundSet()) {
        overBC[0] = static_cast<float>(crv->BackgroundRed()) / 255.0f;
        overBC[1] = static_cast<float>(crv->BackgroundGreen()) / 255.0f;
        overBC[2] = static_cast<float>(crv->BackgroundBlue()) / 255.0f;
        this->overrideBkgndCol = overBC.data(); // hurk
    }

    this->overrideCall = dynamic_cast<view::AbstractCallRender*>(&call);

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

    if (this->overrideCall != nullptr) {
        this->overrideCall->DisableOutputBuffer();
        this->overrideCall = nullptr;
    }

    this->overrideBkgndCol = nullptr;
    this->overrideViewport = nullptr;

    return true;
}

/*
 * View3D_2::UpdateFreeze
 */
void View3D_2::UpdateFreeze(bool freeze) {
    // intentionally empty?
}

/*
 * View3D_2::OnKey
 */
bool view::View3D_2::OnKey(view::Key key, view::KeyAction action, view::Modifiers mods) {
    auto* cr = this->rendererSlot.CallAs<CallRender3D_2>();
    if (cr != nullptr) {
        view::InputEvent evt;
        evt.tag = view::InputEvent::Tag::Key;
        evt.keyData.key = key;
        evt.keyData.action = action;
        evt.keyData.mods = mods;
        cr->SetInputEvent(evt);
        if ((*cr)(CallRender3D_2::FnOnKey)) return true;
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
 * View3D_2::OnChar
 */
bool view::View3D_2::OnChar(unsigned int codePoint) {
    auto* cr = this->rendererSlot.CallAs<view::CallRender3D_2>();
    if (cr == NULL) return false;

    view::InputEvent evt;
    evt.tag = view::InputEvent::Tag::Char;
    evt.charData.codePoint = codePoint;
    cr->SetInputEvent(evt);
    if (!(*cr)(view::CallRender3D_2::FnOnChar)) return false;

    return true;
}

/*
 * View3D_2::OnMouseButton
 */
bool view::View3D_2::OnMouseButton(view::MouseButton button, view::MouseButtonAction action, view::Modifiers mods) {

    bool anyManipulatorActive = arcballManipulator.manipulating() || translateManipulator.manipulating() ||
                                rotateManipulator.manipulating() || turntableManipulator.manipulating() ||
                                orbitAltitudeManipulator.manipulating();

    if (!cameraControlOverrideActive && !anyManipulatorActive) {
        auto* cr = this->rendererSlot.CallAs<CallRender3D_2>();
        if (cr != nullptr) {
            view::InputEvent evt;
            evt.tag = view::InputEvent::Tag::MouseButton;
            evt.mouseButtonData.button = button;
            evt.mouseButtonData.action = action;
            evt.mouseButtonData.mods = mods;
            cr->SetInputEvent(evt);
            if ((*cr)(CallRender3D_2::FnOnMouseButton)) return true;
        }
    }

    if (action == view::MouseButtonAction::PRESS) {
        this->pressedMouseMap[button] = true;
    } else if (action == view::MouseButtonAction::RELEASE) {
        this->pressedMouseMap[button] = false;
    }

    // This mouse handling/mapping is so utterly weird and should die!
    auto down = action == view::MouseButtonAction::PRESS;
    bool altPressed = mods.test(view::Modifier::ALT); // this->modkeys.test(view::Modifier::ALT);
    bool ctrlPressed = mods.test(view::Modifier::CTRL); // this->modkeys.test(view::Modifier::CTRL);

    // get window resolution to help computing mouse coordinates
    auto wndSize = this->cam.resolution_gate();

    if (!this->toggleMouseSelection) {
        switch (button) {
        case megamol::core::view::MouseButton::BUTTON_LEFT:
            this->cursor2d.SetButtonState(0, down);

            if (!anyManipulatorActive) {
                if (altPressed ^
                    (this->arcballDefault &&
                        !ctrlPressed)) // Left mouse press + alt/arcDefault+noCtrl -> activate arcball manipluator
                {
                    this->arcballManipulator.setActive(
                        wndSize.width() - static_cast<int>(this->mouseX), static_cast<int>(this->mouseY));
                } else if (ctrlPressed) // Left mouse press + Ctrl -> activate orbital manipluator
                {
                    this->turntableManipulator.setActive(
                        wndSize.width() - static_cast<int>(this->mouseX), static_cast<int>(this->mouseY));
                }
            }

            break;
        case megamol::core::view::MouseButton::BUTTON_RIGHT:
            this->cursor2d.SetButtonState(1, down);

            if (!anyManipulatorActive) {
                if ((altPressed ^ this->arcballDefault) || ctrlPressed) {
                    this->orbitAltitudeManipulator.setActive(
                        wndSize.width() - static_cast<int>(this->mouseX), static_cast<int>(this->mouseY));
                } else {
                    this->rotateManipulator.setActive();
                    this->translateManipulator.setActive(
                        wndSize.width() - static_cast<int>(this->mouseX), static_cast<int>(this->mouseY));
                }
            }

            break;
        case megamol::core::view::MouseButton::BUTTON_MIDDLE:
            this->cursor2d.SetButtonState(2, down);

            if (!anyManipulatorActive) {
                this->translateManipulator.setActive(
                    wndSize.width() - static_cast<int>(this->mouseX), static_cast<int>(this->mouseY));
            }

            break;
        default:
            break;
        }


        if (action == view::MouseButtonAction::RELEASE) // Mouse release + no other mouse button pressed ->
                                                        // deactivate all mouse manipulators
        {
            if (!(this->cursor2d.GetButtonState(0) || this->cursor2d.GetButtonState(1) ||
                    this->cursor2d.GetButtonState(2))) {
                this->arcballManipulator.setInactive();
                this->orbitAltitudeManipulator.setInactive();
                this->rotateManipulator.setInactive();
                this->turntableManipulator.setInactive();
                this->translateManipulator.setInactive();
            }
        }
    }
    return true;
}

/*
 * View3D_2::OnMouseMove
 */
bool view::View3D_2::OnMouseMove(double x, double y) {
    this->mouseX = (float)static_cast<int>(x);
    this->mouseY = (float)static_cast<int>(y);

    bool anyManipulatorActive = arcballManipulator.manipulating() || translateManipulator.manipulating() ||
                                rotateManipulator.manipulating() || turntableManipulator.manipulating() ||
                                orbitAltitudeManipulator.manipulating();

    if (!anyManipulatorActive) {
        auto* cr = this->rendererSlot.CallAs<CallRender3D_2>();
        if (cr != nullptr) {
            view::InputEvent evt;
            evt.tag = view::InputEvent::Tag::MouseMove;
            evt.mouseMoveData.x = x;
            evt.mouseMoveData.y = y;
            cr->SetInputEvent(evt);
            if ((*cr)(CallRender3D_2::FnOnMouseMove)) return true;
        }
    }

    // This mouse handling/mapping is so utterly weird and should die!
    if (!this->toggleMouseSelection) {
        this->cursor2d.SetPosition(x, y, true);

        glm::vec3 newPos;

        auto wndSize = this->cam.resolution_gate();

        if (this->turntableManipulator.manipulating()) {
            this->turntableManipulator.on_drag(wndSize.width() - static_cast<int>(this->mouseX),
                static_cast<int>(this->mouseY), glm::vec4(rotCenter, 1.0));
        }

        if (this->arcballManipulator.manipulating()) {
            this->arcballManipulator.on_drag(wndSize.width() - static_cast<int>(this->mouseX),
                static_cast<int>(this->mouseY), glm::vec4(rotCenter, 1.0));
        }

        if (this->orbitAltitudeManipulator.manipulating()) {
            this->orbitAltitudeManipulator.on_drag(wndSize.width() - static_cast<int>(this->mouseX),
                static_cast<int>(this->mouseY), glm::vec4(rotCenter, 1.0));
        }

        if (this->translateManipulator.manipulating() && !this->rotateManipulator.manipulating() ) {

            // compute proper step size by computing pixel world size at distance to rotCenter
            glm::vec3 currCamPos(static_cast<glm::vec4>(this->cam.position()));
            float orbitalAltitude = glm::length(currCamPos - rotCenter);
            auto fovy = cam.half_aperture_angle_radians();
            auto vertical_height = 2.0f * std::tan(fovy) * orbitalAltitude;
            auto pixel_world_size = vertical_height / wndSize.height();

            this->translateManipulator.set_step_size(pixel_world_size);

            this->translateManipulator.move_horizontally(wndSize.width() - static_cast<int>(this->mouseX));
            this->translateManipulator.move_vertically(static_cast<int>(this->mouseY));
        }
    }

    return true;
}

/*
 * View3D_2::OnMouseScroll
 */
bool view::View3D_2::OnMouseScroll(double dx, double dy) {
    auto* cr = this->rendererSlot.CallAs<view::CallRender3D_2>();
    if (cr != NULL) {
        view::InputEvent evt;
        evt.tag = view::InputEvent::Tag::MouseScroll;
        evt.mouseScrollData.dx = dx;
        evt.mouseScrollData.dy = dy;
        cr->SetInputEvent(evt);
        if ((*cr)(view::CallRender3D_2::FnOnMouseScroll)) return true;
    }


    // This mouse handling/mapping is so utterly weird and should die!
    if (!this->toggleMouseSelection && (abs(dy) > 0.0)) {
        if (this->rotateManipulator.manipulating()) {
            this->viewKeyMoveStepSlot.Param<param::FloatParam>()->SetValue(
                this->viewKeyMoveStepSlot.Param<param::FloatParam>()->Value() + 
                (dy * 0.1f * this->viewKeyMoveStepSlot.Param<param::FloatParam>()->Value())
            ); 
        } else {
            auto cam_pos = this->cam.eye_position();
            auto rot_cntr = thecam::math::point<glm::vec4>(glm::vec4(this->rotCenter, 0.0f));

            cam_pos.w() = 0.0f;

            auto v = thecam::math::normalise(rot_cntr - cam_pos);

            auto altitude = thecam::math::length(rot_cntr - cam_pos);

            this->cam.position(cam_pos + (v * dy * (altitude / 50.0f)));
        }
    }

    return true;
}

/*
 * View3D_2::unpackMouseCoordinates
 */
void View3D_2::unpackMouseCoordinates(float& x, float& y) {
    // TODO is this correct?
    x *= static_cast<float>(this->cam.resolution_gate().width());
    y *= static_cast<float>(this->cam.resolution_gate().height());
    y -= 1.0f;
}

/*
 * View3D_2::create
 */
bool View3D_2::create(void) {
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
        } catch (...) {
        }
    }

    // TODO implement

    this->cursor2d.SetButtonCount(3);

    this->firstImg = true;

    return true;
}

/*
 * View3D_2::release
 */
void View3D_2::release(void) { this->removeTitleRenderer(); }

/*
 * View3D_2::mouseSensitivityChanged
 */
bool View3D_2::mouseSensitivityChanged(param::ParamSlot& p) { return true; }

/*
 * View3D_2::OnGetCamParams
 */
// bool View3D_2::OnGetCamParams(view::CallCamParamSync& c) {
//    // TODO implement
//    return true;
//}

/*
 * View3D_2::onStoreCamera
 */
bool View3D_2::onStoreCamera(param::ParamSlot& p) {
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
 * View3D_2::onRestoreCamera
 */
bool View3D_2::onRestoreCamera(param::ParamSlot& p) {
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
 * View3D_2::onResetView
 */
bool View3D_2::onResetView(param::ParamSlot& p) {
    this->ResetView();
    return true;
}

/*
 * View3D_2::onToggleButton
 */
bool View3D_2::onToggleButton(param::ParamSlot& p) {
    // TODO implement
    return true;
}

/*
 * View3D_2::determineCameraFilePath
 */
std::string View3D_2::determineCameraFilePath(void) const {
    auto path = this->GetCoreInstance()->GetLuaState()->GetScriptPath();
    if (path.empty()) return path; // early exit for mmprj projects
    auto dotpos = path.find_last_of('.');
    path = path.substr(0, dotpos);
    path.append("_cam.json");
    return path;
}

/*
 * View3D_2::handleCameraMovement
 */
void View3D_2::handleCameraMovement(void) {
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

    if (this->rotateManipulator.manipulating()) {
        auto resolution = this->cam.resolution_gate();
        glm::vec2 midpoint(resolution.width() / 2.0f, resolution.height() / 2.0f);
        glm::vec2 mouseDirection = glm::vec2(this->mouseX, this->mouseY) - midpoint;
        mouseDirection.x = -mouseDirection.x;
        mouseDirection /= midpoint;

        this->rotateManipulator.pitch(-mouseDirection.y * rotationStep);
        this->rotateManipulator.yaw(
            mouseDirection.x * rotationStep,
            this->viewKeyFixToWorldUpSlot.Param<param::BoolParam>()->Value()
        );
    }

    glm::vec3 newCamPos(static_cast<glm::vec4>(this->cam.eye_position()));
    glm::vec3 camDir(static_cast<glm::vec4>(this->cam.view_vector()));
    rotCenter = newCamPos + orbitalAltitude * glm::normalize(camDir);
}

/*
 * View3D_2::setCameraValues
 */
void View3D_2::setCameraValues(const view::Camera_2& cam) {
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
 * View3D_2::adaptCameraValues
 */
bool View3D_2::adaptCameraValues(view::Camera_2& cam) {
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


bool View3D_2::cameraOvrCallback(param::ParamSlot& p) {
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
