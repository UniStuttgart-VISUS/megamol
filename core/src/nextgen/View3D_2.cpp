/*
 * View3D_2.cpp
 *
 * Copyright (C) 2018 by VISUS (Universitaet Stuttgart).
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "mmcore/nextgen/View3D_2.h"
#include "vislib/graphics/gl/IncludeAllGL.h"
#ifdef _WIN32
#    include <windows.h>
#endif /* _WIN32 */
#include <chrono>
#include <fstream>
#include "mmcore/CoreInstance.h"
#include "mmcore/misc/PngBitmapCodec.h"
#include "mmcore/nextgen/CallRender3D_2.h"
#include "mmcore/param/BoolParam.h"
#include "mmcore/param/ButtonParam.h"
#include "mmcore/param/EnumParam.h"
#include "mmcore/param/FilePathParam.h"
#include "mmcore/param/FloatParam.h"
#include "mmcore/param/StringParam.h"
#include "mmcore/param/Vector3fParam.h"
#include "mmcore/utility/ColourParser.h"
#include "mmcore/view/AbstractCallRender.h"
#include "mmcore/view/CallRenderView.h" // TODO new call?
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
#include "vislib/sys/Log.h"
#include "vislib/sys/sysfunctions.h"

#include "mmcore/utility/gl/Texture2D.h"

using namespace megamol::core;
using namespace megamol::core::nextgen;

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
    , resetViewSlot("resetView", "Triggers the reset of the view")
    , firstImg(false)
    , stereoFocusDistSlot("stereo::focusDist", "focus distance for stereo projection")
    , stereoEyeDistSlot("stereo::eyeDist", "eye distance for stereo projection")
    , overrideCall(NULL)
    , viewKeyMoveStepSlot("viewKey::MoveStep", "The move step size in world coordinates")
    , viewKeyRunFactorSlot("viewKey::RunFactor", "The factor for step size multiplication when running (shift)")
    , viewKeyAngleStepSlot("viewKey::AngleStep", "The angle rotate step in degrees")
    , mouseSensitivitySlot("viewKey::MouseSensitivity", "used for WASD mode")
    , viewKeyRotPointSlot("viewKey::RotPoint", "The point around which the view will be roateted")
    , enableMouseSelectionSlot("enableMouseSelection", "Enable selecting and picking with the mouse")
    , showViewCubeSlot("viewcube::show", "Shows the view cube helper")
    , resetViewOnBBoxChangeSlot("resetViewOnBBoxChange", "whether to reset the view when the bounding boxes change")
    , mouseX(0.0f)
    , mouseY(0.0f)
    , mouseFlags(0)
    , timeCtrl()
    , toggleMouseSelection(false)
    , hookOnChangeOnlySlot("hookOnChange", "whether post-hooks are triggered when the frame would be identical") {

    using vislib::sys::KeyCode;

    // this->camParams = this->cam.Parameters();
    // this->camOverrides = new CameraParamOverride(this->camParams);

    // vislib::graphics::ImageSpaceType defWidth(static_cast<vislib::graphics::ImageSpaceType>(100));
    // vislib::graphics::ImageSpaceType defHeight(static_cast<vislib::graphics::ImageSpaceType>(100));

    // this->camParams->SetVirtualViewSize(defWidth, defHeight);
    // this->camParams->SetTileRect(vislib::math::Rectangle<float>(0.0f, 0.0f, defWidth, defHeight));

    this->cam.resolution_gate(cam_type::screen_size_type(100, 100));

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

    this->resetViewSlot.SetParameter(new param::ButtonParam(view::Key::KEY_HOME));
    this->resetViewSlot.SetUpdateCallback(&View3D_2::onResetView);
    this->MakeSlotAvailable(&this->resetViewSlot);

    this->ResetView();

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

    this->viewKeyAngleStepSlot.SetParameter(new param::FloatParam(1.0f, 0.001f, 360.0f));
    this->MakeSlotAvailable(&this->viewKeyAngleStepSlot);

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

    this->resetViewOnBBoxChangeSlot.SetParameter(new param::BoolParam(false));
    this->MakeSlotAvailable(&this->resetViewOnBBoxChangeSlot);

    this->showViewCubeSlot.SetParameter(new param::BoolParam(true));
    this->MakeSlotAvailable(&this->showViewCubeSlot);

    for (unsigned int i = 0; this->timeCtrl.GetSlot(i) != NULL; i++) {
        this->MakeSlotAvailable(this->timeCtrl.GetSlot(i));
    }

    // this->MakeSlotAvailable(&this->slotGetCamParams);
    // this->MakeSlotAvailable(&this->slotSetCamParams);

    this->hookOnChangeOnlySlot.SetParameter(new param::BoolParam(false));
    this->MakeSlotAvailable(&this->hookOnChangeOnlySlot);

    this->translateManipulator.set_target(this->cam);
    this->translateManipulator.enable();

    this->rotateManipulator.set_target(this->cam);
    this->rotateManipulator.enable();

    this->arcballManipulator.set_rotation_centre(glm::vec4(0.0f, 0.0f, 0.0f, 1.0f));
    this->arcballManipulator.set_target(this->cam);
    this->arcballManipulator.set_radius(1.0f);
    this->arcballManipulator.enable();
    this->arcballCenterDistance = 0.0f;

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
unsigned int nextgen::View3D_2::GetCameraSyncNumber(void) const {
    // TODO implement
    return 0;
}


/*
 * View3D_2::SerialiseCamera
 */
void nextgen::View3D_2::SerialiseCamera(vislib::Serialiser& serialiser) const {
    // TODO currently emtpy because the old serialization sucks
}


/*
 * View3D_2::DeserialiseCamera
 */
void nextgen::View3D_2::DeserialiseCamera(vislib::Serialiser& serialiser) {
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

    AbstractRenderingView::beginFrame();

    // TODO Conditionally synchronise camera from somewhere else.
    // this->SyncCamParams(this->cam.Parameters());

    // clear viewport
    if (this->overrideViewport != nullptr) {
        if ((this->overrideViewport[0] >= 0) && (this->overrideViewport[1] >= 0) && (this->overrideViewport[2] >= 0) &&
            (this->overrideViewport[3] >= 0)) {
            currentViewport = glm::ivec4(this->overrideViewport[0], this->overrideViewport[1],
                this->overrideViewport[2], this->overrideViewport[3]);
        }
    } else {
        // this is correct in non-override mode,
        //  because then the tile will be whole viewport
        auto camRes = this->cam.resolution_gate();
        currentViewport = glm::ivec4(0, 0, camRes.width(), camRes.height());
    }

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
        if (this->firstImg || (!(cr3d->AccessBoundingBoxes() == this->bboxs) &&
                                  !(!cr3d->AccessBoundingBoxes().IsAnyValid() && !this->bboxs.IsBoundingBoxValid() &&
                                      !this->bboxs.IsClipBoxValid()))) {
            this->bboxs = cr3d->AccessBoundingBoxes();
            glm::vec3 bbcenter = glm::make_vec3(this->bboxs.BoundingBox().CalcCenter().PeekCoordinates());
            this->arcballManipulator.set_rotation_centre(glm::vec4(bbcenter, 1.0f));


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

    if (cr3d != nullptr) {
        cr3d->SetCameraState(this->cam);
        (*cr3d)(view::AbstractCallRender::FnRender);
    }


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
    // TODO implement
    this->cam.near_clipping_plane(0.1f);
    this->cam.far_clipping_plane(100.0f);
    this->cam.aperture_angle(30.0f);
    this->cam.disparity(0.05f);
    this->cam.eye(thecam::Eye::mono);
    this->cam.projection_type(thecam::Projection_type::perspective);
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

    this->cam.position(bbcglm + glm::vec4(0.0f, 0.0f, dist, 0.0f));
    this->cam.orientation(cam_type::quaternion_type::create_identity());
    this->arcballCenterDistance = dist;

    glm::mat4 vm = this->cam.view_matrix();
    glm::mat4 pm = this->cam.projection_matrix();

    // TODO Further manipulators? better value?
    this->translateManipulator.set_step_size(dist);
}

/*
 * View3D_2::Resize
 */
void View3D_2::Resize(unsigned int width, unsigned int height) {
    this->cam.resolution_gate(cam_type::screen_size_type(
        static_cast<LONG>(width), static_cast<LONG>(height))); // TODO this is ugly and has to die...
}

/*
 * View3D_2::OnRenderView
 */
bool View3D_2::OnRenderView(Call& call) {
    std::array<float, 3> overBC;
    std::array<int, 4> overVP = {0, 0, 0, 0};
    view::CallRenderView* crv = dynamic_cast<view::CallRenderView*>(&call);
    if (crv == nullptr) return false;

    this->overrideViewport = overVP.data();
    if (crv->IsViewportSet()) {
        overVP[2] = crv->ViewportWidth();
        overVP[3] = crv->ViewportHeight();
        if (!crv->IsTileSet()) {
            // TODO
        }
    }
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
bool nextgen::View3D_2::OnKey(view::Key key, view::KeyAction action, view::Modifiers mods) {
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
        if (action == view::KeyAction::PRESS) {
            this->modkeys.set(view::Modifier::ALT);
        } else if (action == view::KeyAction::RELEASE) {
            this->modkeys.reset(view::Modifier::ALT);
        }
    }
    if (key == view::Key::KEY_LEFT_SHIFT || key == view::Key::KEY_RIGHT_SHIFT) {
        if (action == view::KeyAction::PRESS) {
            this->modkeys.set(view::Modifier::SHIFT);
        } else if (action == view::KeyAction::RELEASE) {
            this->modkeys.reset(view::Modifier::SHIFT);
        }
    }
    if (key == view::Key::KEY_LEFT_CONTROL || key == view::Key::KEY_RIGHT_CONTROL) {
        if (action == view::KeyAction::PRESS) {
            this->modkeys.set(view::Modifier::CTRL);
        } else if (action == view::KeyAction::RELEASE) {
            this->modkeys.reset(view::Modifier::CTRL);
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
bool nextgen::View3D_2::OnChar(unsigned int codePoint) {
    auto* cr = this->rendererSlot.CallAs<nextgen::CallRender3D_2>();
    if (cr == NULL) return false;

    view::InputEvent evt;
    evt.tag = view::InputEvent::Tag::Char;
    evt.charData.codePoint = codePoint;
    cr->SetInputEvent(evt);
    if (!(*cr)(nextgen::CallRender3D_2::FnOnChar)) return false;

    return true;
}

/*
 * View3D_2::OnMouseButton
 */
bool nextgen::View3D_2::OnMouseButton(view::MouseButton button, view::MouseButtonAction action, view::Modifiers mods) {
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

    if (action == view::MouseButtonAction::PRESS) {
        this->pressedMouseMap[button] = true;
    } else if (action == view::MouseButtonAction::RELEASE) {
        this->pressedMouseMap[button] = false;
    }

    // This mouse handling/mapping is so utterly weird and should die!
    auto down = action == view::MouseButtonAction::PRESS;
    bool altPressed = this->modkeys.test(view::Modifier::ALT);

    if (!this->toggleMouseSelection) {
        switch (button) {
        case megamol::core::view::MouseButton::BUTTON_LEFT:
            this->cursor2d.SetButtonState(0, down);
            if (action == view::MouseButtonAction::PRESS && (altPressed ^ this->arcballDefault)) {
                if (!this->arcballManipulator.manipulating()) {
                    glm::vec3 curPos = static_cast<glm::vec4>(this->cam.eye_position());
                    glm::vec3 camDir = static_cast<glm::vec4>(this->cam.view_vector());
                    // glm::vec3 rotCenter = curPos + this->arcballCenterDistance * glm::normalize(camDir);
                    // glm::vec3 rotCenter = static_cast<glm::vec4>(this->arcballManipulator.rotation_centre());
                    // this->arcballManipulator.set_rotation_centre(glm::vec4(rotCenter, 1.0f));
                    // this->arcballManipulator.set_radius(glm::distance(rotCenter, curPos));
                    this->arcballManipulator.set_radius(1.0f);
                    auto wndSize = this->cam.resolution_gate();
                    this->arcballManipulator.on_drag_start(
                        wndSize.width() - static_cast<int>(this->mouseX), static_cast<int>(this->mouseY));
                }
            } else if (action == view::MouseButtonAction::RELEASE && (altPressed ^ this->arcballDefault)) {
                this->arcballManipulator.on_drag_stop();
            }
            break;
        case megamol::core::view::MouseButton::BUTTON_RIGHT:
            this->cursor2d.SetButtonState(1, down);
            break;
        case megamol::core::view::MouseButton::BUTTON_MIDDLE:
            this->cursor2d.SetButtonState(2, down);
            break;
        default:
            break;
        }
    }
    return true;
}

/*
 * View3D_2::OnMouseMove
 */
bool nextgen::View3D_2::OnMouseMove(double x, double y) {
    this->mouseX = (float)static_cast<int>(x);
    this->mouseY = (float)static_cast<int>(y);

    auto* cr = this->rendererSlot.CallAs<CallRender3D_2>();
    if (cr != nullptr) {
        view::InputEvent evt;
        evt.tag = view::InputEvent::Tag::MouseMove;
        evt.mouseMoveData.x = x;
        evt.mouseMoveData.y = y;
        cr->SetInputEvent(evt);
        if ((*cr)(CallRender3D_2::FnOnMouseMove)) return true;
    }

    // This mouse handling/mapping is so utterly weird and should die!
    if (!this->toggleMouseSelection) {
        this->cursor2d.SetPosition(x, y, true);
        if (this->arcballManipulator.manipulating()) {
            auto wndSize = this->cam.resolution_gate();
            this->arcballManipulator.on_drag(
                wndSize.width() - static_cast<int>(this->mouseX), static_cast<int>(this->mouseY));
        }
    }

    return true;
}

/*
 * View3D_2::OnMouseScroll
 */
bool nextgen::View3D_2::OnMouseScroll(double dx, double dy) {
    auto* cr = this->rendererSlot.CallAs<nextgen::CallRender3D_2>();
    if (cr == NULL) return false;

    view::InputEvent evt;
    evt.tag = view::InputEvent::Tag::MouseScroll;
    evt.mouseScrollData.dx = dx;
    evt.mouseScrollData.dy = dy;
    cr->SetInputEvent(evt);
    if (!(*cr)(nextgen::CallRender3D_2::FnOnMouseScroll)) return false;

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
        vislib::sys::Log::DefaultLog.WriteWarn(
            "The camera output file path could not be determined. This is probably due to the usage of .mmprj project "
            "files. Please use a .lua project file instead");
        return false;
    }

    if (!this->overrideCamSettingsSlot.Param<param::BoolParam>()->Value()) {
        // check if the file already exists
        std::ifstream file(path);
        if (file.good()) {
            file.close();
            vislib::sys::Log::DefaultLog.WriteWarn(
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
        vislib::sys::Log::DefaultLog.WriteWarn(
            "The camera output file could not be written to '%s' because the file could not be opened.", path.c_str());
        return false;
    }

    vislib::sys::Log::DefaultLog.WriteInfo("Camera statistics successfully written to '%s'", path.c_str());
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
            vislib::sys::Log::DefaultLog.WriteWarn(
                "The entered camera string was not valid. No change of the camera has been performed");
        } else {
            this->cam = minstate;
            return true;
        }
    }

    auto path = this->determineCameraFilePath();
    if (path.empty()) {
        vislib::sys::Log::DefaultLog.WriteWarn(
            "The camera camera file path could not be determined. This is probably due to the usage of .mmprj project "
            "files. Please use a .lua project file instead");
        return false;
    }

    std::ifstream file(path);
    std::string text;
    if (file.is_open()) {
        text.assign(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
    } else {
        vislib::sys::Log::DefaultLog.WriteWarn("The camera output file at '%s' could not be opened.", path.c_str());
        return false;
    }
    auto copy = this->savedCameras;
    bool success = this->serializer.deserialize(copy, text);
    if (!success) {
        vislib::sys::Log::DefaultLog.WriteWarn(
            "The reading of the camera parameters did not work properly. No changes were made.");
        return false;
    }
    this->savedCameras = copy;
    if (this->savedCameras.back().second) {
        this->cam = this->savedCameras.back().first;
    } else {
        vislib::sys::Log::DefaultLog.WriteWarn("The stored default cam was not valid. The old default cam is used");
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
	// the default case is 60 fps therefore we calculate the multiples for the step factor using that
	auto constexpr micros = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::seconds(1)) / 60.0f;
	float factor = this->lastFrameDuration / micros;
	step *= factor;

    const float runFactor = this->viewKeyRunFactorSlot.Param<param::FloatParam>()->Value();
    if (this->modkeys.test(view::Modifier::SHIFT)) {
        step *= runFactor;
    }

    bool anymodpressed = !this->modkeys.none();
    float rotationStep = this->viewKeyAngleStepSlot.Param<param::FloatParam>()->Value();
	rotationStep *= factor;

    if (!(this->arcballDefault ^ this->modkeys.test(view::Modifier::ALT))) {
        auto resolution = this->cam.resolution_gate();
        glm::vec2 midpoint(resolution.width() / 2.0f, resolution.height() / 2.0f);
        glm::vec2 mouseDirection = glm::vec2(this->mouseX, this->mouseY) - midpoint;
        mouseDirection.x = -mouseDirection.x;
        mouseDirection /= midpoint;

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

        if (this->pressedMouseMap.count(view::MouseButton::BUTTON_LEFT) > 0 &&
            this->pressedMouseMap[view::MouseButton::BUTTON_LEFT]) {
            this->rotateManipulator.pitch(-mouseDirection.y * rotationStep);
            this->rotateManipulator.yaw(mouseDirection.x * rotationStep);
        }
    }
}
