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
    , lightDir(0.5f, -1.0f, -1.0f)
    , isCamLight(true)
    , bboxs()
    , showBBox("showBBox", "Bool parameter to show/hide the bounding box")
    , showLookAt("showLookAt", "Flag showing the look at point")
    , cameraSettingsSlot("camsettings", "The stored camera settings")
    , storeCameraSettingsSlot("storecam", "Triggers the storage of the camera settings")
    , restoreCameraSettingsSlot("restorecam", "Triggers the restore of the camera settings")
    , resetViewSlot("resetView", "Triggers the reset of the view")
    , firstImg(false)
    , isCamLightSlot(
          "light::isCamLight", "Flag whether the light is relative to the camera or to the world coordinate system")
    , lightDirSlot("light::direction", "Direction vector of the light")
    , lightColDifSlot("light::diffuseCol", "Diffuse light colour")
    , lightColAmbSlot("light::ambientCol", "Ambient light colour")
    , stereoFocusDistSlot("stereo::focusDist", "focus distance for stereo projection")
    , stereoEyeDistSlot("stereo::eyeDist", "eye distance for stereo projection")
    , overrideCall(NULL)
    , viewKeyMoveStepSlot("viewKey::MoveStep", "The move step size in world coordinates")
    , viewKeyRunFactorSlot("viewKey::RunFactor", "The factor for step size multiplication when running (shift)")
    , viewKeyAngleStepSlot("viewKey::AngleStep", "The angle rotate step in degrees")
    , mouseSensitivitySlot("viewKey::MouseSensitivity", "used for WASD mode")
    , viewKeyRotPointSlot("viewKey::RotPoint", "The point around which the view will be roateted")
    , viewKeyRotLeftSlot("viewKey::RotLeft", "Rotates the view to the left (around the up-axis)")
    , viewKeyRotRightSlot("viewKey::RotRight", "Rotates the view to the right (around the up-axis)")
    , viewKeyRotUpSlot("viewKey::RotUp", "Rotates the view to the top (around the right-axis)")
    , viewKeyRotDownSlot("viewKey::RotDown", "Rotates the view to the bottom (around the right-axis)")
    , viewKeyRollLeftSlot("viewKey::RollLeft", "Rotates the view counter-clockwise (around the view-axis)")
    , viewKeyRollRightSlot("viewKey::RollRight", "Rotates the view clockwise (around the view-axis)")
    , viewKeyZoomInSlot("viewKey::ZoomIn", "Zooms in (moves the camera)")
    , viewKeyZoomOutSlot("viewKey::ZoomOut", "Zooms out (moves the camera)")
    , viewKeyMoveLeftSlot("viewKey::MoveLeft", "Moves to the left")
    , viewKeyMoveRightSlot("viewKey::MoveRight", "Moves to the right")
    , viewKeyMoveUpSlot("viewKey::MoveUp", "Moves to the top")
    , viewKeyMoveDownSlot("viewKey::MoveDown", "Moves to the bottom")
    , toggleBBoxSlot("toggleBBox", "Button to toggle the bounding box")
    , bboxCol{1.0f, 1.0f, 1.0f, 0.625f}
    , bboxColSlot("bboxCol", "Sets the colour for the bounding box")
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

    this->showBBox.SetParameter(new param::BoolParam(true));
    this->MakeSlotAvailable(&this->showBBox);

    this->showLookAt.SetParameter(new param::BoolParam(false));
    this->MakeSlotAvailable(&this->showLookAt);

    this->cameraSettingsSlot.SetParameter(new param::StringParam(""));
    this->MakeSlotAvailable(&this->cameraSettingsSlot);

    this->storeCameraSettingsSlot.SetParameter(
        new param::ButtonParam(vislib::sys::KeyCode::KEY_MOD_ALT | vislib::sys::KeyCode::KEY_MOD_SHIFT | 'C'));
    this->storeCameraSettingsSlot.SetUpdateCallback(&View3D_2::onStoreCamera);
    this->MakeSlotAvailable(&this->storeCameraSettingsSlot);

    this->restoreCameraSettingsSlot.SetParameter(new param::ButtonParam(vislib::sys::KeyCode::KEY_MOD_ALT | 'c'));
    this->restoreCameraSettingsSlot.SetUpdateCallback(&View3D_2::onRestoreCamera);
    this->MakeSlotAvailable(&this->restoreCameraSettingsSlot);

    this->resetViewSlot.SetParameter(new param::ButtonParam(vislib::sys::KeyCode::KEY_HOME));
    this->resetViewSlot.SetUpdateCallback(&View3D_2::onResetView);
    this->MakeSlotAvailable(&this->resetViewSlot);

    /*this->isCamLightSlot << new param::BoolParam(this->isCamLight);
    this->MakeSlotAvailable(&this->isCamLightSlot);

    this->lightDirSlot << new param::Vector3fParam(this->lightDir);
    this->MakeSlotAvailable(&this->lightDirSlot);

    this->lightColDif[0] = this->lightColDif[1] = this->lightColDif[2] = 1.0f;
    this->lightColDif[3] = 1.0f;

    this->lightColAmb[0] = this->lightColAmb[1] = this->lightColAmb[2] = 0.2f;
    this->lightColAmb[3] = 1.0f;

    this->lightColDifSlot << new param::StringParam(
        utility::ColourParser::ToString(this->lightColDif[0], this->lightColDif[1], this->lightColDif[2]));
    this->MakeSlotAvailable(&this->lightColDifSlot);

    this->lightColAmbSlot << new param::StringParam(
        utility::ColourParser::ToString(this->lightColAmb[0], this->lightColAmb[1], this->lightColAmb[2]));
    this->MakeSlotAvailable(&this->lightColAmbSlot);
    */

    this->ResetView();

    // TODO
    // this->stereoEyeDistSlot << new param::FloatParam(this->camParams->StereoDisparity(), 0.0f);
    // this->MakeSlotAvailable(&this->stereoEyeDistSlot);

    // TODO
    // this->stereoFocusDistSlot << new param::FloatParam(this->camParams->FocalDistance(false), 0.0f);
    // this->MakeSlotAvailable(&this->stereoFocusDistSlot);

    this->viewKeyMoveStepSlot.SetParameter(new param::FloatParam(0.1f, 0.001f));
    this->MakeSlotAvailable(&this->viewKeyMoveStepSlot);

    this->viewKeyRunFactorSlot.SetParameter(new param::FloatParam(2.0f, 0.1f));
    this->MakeSlotAvailable(&this->viewKeyRunFactorSlot);

    this->viewKeyAngleStepSlot.SetParameter(new param::FloatParam(15.0f, 0.001f, 360.0f));
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

    this->viewKeyRotLeftSlot.SetParameter(new param::ButtonParam(KeyCode::KEY_LEFT | KeyCode::KEY_MOD_CTRL));
    this->viewKeyRotLeftSlot.SetUpdateCallback(&View3D_2::viewKeyPressed);
    this->MakeSlotAvailable(&this->viewKeyRotLeftSlot);

    this->viewKeyRotRightSlot.SetParameter(new param::ButtonParam(KeyCode::KEY_RIGHT | KeyCode::KEY_MOD_CTRL));
    this->viewKeyRotRightSlot.SetUpdateCallback(&View3D_2::viewKeyPressed);
    this->MakeSlotAvailable(&this->viewKeyRotRightSlot);

    this->viewKeyRotUpSlot.SetParameter(new param::ButtonParam(KeyCode::KEY_UP | KeyCode::KEY_MOD_CTRL));
    this->viewKeyRotUpSlot.SetUpdateCallback(&View3D_2::viewKeyPressed);
    this->MakeSlotAvailable(&this->viewKeyRotUpSlot);

    this->viewKeyRotDownSlot.SetParameter(new param::ButtonParam(KeyCode::KEY_DOWN | KeyCode::KEY_MOD_CTRL));
    this->viewKeyRotDownSlot.SetUpdateCallback(&View3D_2::viewKeyPressed);
    this->MakeSlotAvailable(&this->viewKeyRotDownSlot);

    this->viewKeyRollLeftSlot.SetParameter(
        new param::ButtonParam(KeyCode::KEY_LEFT | KeyCode::KEY_MOD_CTRL | KeyCode::KEY_MOD_SHIFT));
    this->viewKeyRollLeftSlot.SetUpdateCallback(&View3D_2::viewKeyPressed);
    this->MakeSlotAvailable(&this->viewKeyRollLeftSlot);

    this->viewKeyRollRightSlot.SetParameter(
        new param::ButtonParam(KeyCode::KEY_RIGHT | KeyCode::KEY_MOD_CTRL | KeyCode::KEY_MOD_SHIFT));
    this->viewKeyRollRightSlot.SetUpdateCallback(&View3D_2::viewKeyPressed);
    this->MakeSlotAvailable(&this->viewKeyRollRightSlot);

    this->viewKeyZoomInSlot.SetParameter(
        new param::ButtonParam(KeyCode::KEY_UP | KeyCode::KEY_MOD_CTRL | KeyCode::KEY_MOD_SHIFT));
    this->viewKeyZoomInSlot.SetUpdateCallback(&View3D_2::viewKeyPressed);
    this->MakeSlotAvailable(&this->viewKeyZoomInSlot);

    this->viewKeyZoomOutSlot.SetParameter(
        new param::ButtonParam(KeyCode::KEY_DOWN | KeyCode::KEY_MOD_CTRL | KeyCode::KEY_MOD_SHIFT));
    this->viewKeyZoomOutSlot.SetUpdateCallback(&View3D_2::viewKeyPressed);
    this->MakeSlotAvailable(&this->viewKeyZoomOutSlot);

    this->viewKeyMoveLeftSlot.SetParameter(
        new param::ButtonParam(KeyCode::KEY_LEFT | KeyCode::KEY_MOD_CTRL | KeyCode::KEY_MOD_ALT));
    this->viewKeyMoveLeftSlot.SetUpdateCallback(&View3D_2::viewKeyPressed);
    this->MakeSlotAvailable(&this->viewKeyMoveLeftSlot);

    this->viewKeyMoveRightSlot.SetParameter(
        new param::ButtonParam(KeyCode::KEY_RIGHT | KeyCode::KEY_MOD_CTRL | KeyCode::KEY_MOD_ALT));
    this->viewKeyMoveRightSlot.SetUpdateCallback(&View3D_2::viewKeyPressed);
    this->MakeSlotAvailable(&this->viewKeyMoveRightSlot);

    this->viewKeyMoveUpSlot.SetParameter(
        new param::ButtonParam(KeyCode::KEY_UP | KeyCode::KEY_MOD_CTRL | KeyCode::KEY_MOD_ALT));
    this->viewKeyMoveUpSlot.SetUpdateCallback(&View3D_2::viewKeyPressed);
    this->MakeSlotAvailable(&this->viewKeyMoveUpSlot);

    this->viewKeyMoveDownSlot.SetParameter(
        new param::ButtonParam(KeyCode::KEY_DOWN | KeyCode::KEY_MOD_CTRL | KeyCode::KEY_MOD_ALT));
    this->viewKeyMoveDownSlot.SetUpdateCallback(&View3D_2::viewKeyPressed);
    this->MakeSlotAvailable(&this->viewKeyMoveDownSlot);

    this->toggleBBoxSlot.SetParameter(new param::ButtonParam('i' | KeyCode::KEY_MOD_ALT));
    this->toggleBBoxSlot.SetUpdateCallback(&View3D_2::onToggleButton);
    this->MakeSlotAvailable(&this->toggleBBoxSlot);

    this->enableMouseSelectionSlot.SetParameter(new param::ButtonParam(KeyCode::KEY_TAB));
    this->enableMouseSelectionSlot.SetUpdateCallback(&View3D_2::onToggleButton);
    this->MakeSlotAvailable(&this->enableMouseSelectionSlot);

    this->resetViewOnBBoxChangeSlot.SetParameter(new param::BoolParam(false));
    this->MakeSlotAvailable(&this->resetViewOnBBoxChangeSlot);

    this->bboxColSlot.SetParameter(new param::StringParam(
        utility::ColourParser::ToString(this->bboxCol[0], this->bboxCol[1], this->bboxCol[2], this->bboxCol[3])));
    this->MakeSlotAvailable(&this->bboxColSlot);

    this->showViewCubeSlot.SetParameter(new param::BoolParam(true));
    this->MakeSlotAvailable(&this->showViewCubeSlot);

    for (unsigned int i = 0; this->timeCtrl.GetSlot(i) != NULL; i++) {
        this->MakeSlotAvailable(this->timeCtrl.GetSlot(i));
    }

    //this->MakeSlotAvailable(&this->slotGetCamParams);
    //this->MakeSlotAvailable(&this->slotSetCamParams);

    this->hookOnChangeOnlySlot.SetParameter(new param::BoolParam(false));
    this->MakeSlotAvailable(&this->hookOnChangeOnlySlot);
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
    // TODO implement
}


/*
 * View3D_2::DeserialiseCamera
 */
void nextgen::View3D_2::DeserialiseCamera(vislib::Serialiser& serialiser) {
    // TODO implement
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

    AbstractRenderingView::beginFrame();

    // TODO Conditionally synchronise camera from somewhere else.
    // this->SyncCamParams(this->cam.Parameters());

    // clear viewport
    if (this->overrideViewport != nullptr) {
        if ((this->overrideViewport[0] >= 0) && (this->overrideViewport[1] >= 0) && (this->overrideViewport[2] >= 0) &&
            (this->overrideViewport[3] >= 0)) {
            /*glViewport(this->overrideViewport[0], this->overrideViewport[1], this->overrideViewport[2],
                this->overrideViewport[3]);*/
            currentViewport = glm::ivec4(this->overrideViewport[0], this->overrideViewport[1], this->overrideViewport[2], this->overrideViewport[3]);
        }
    } else {
        // this is correct in non-override mode,
        //  because then the tile will be whole viewport
        // TODO
        // glViewport(0, 0, static_cast<GLsizei>(this->camParams->TileRect().Width()),
        // static_cast<GLsizei>(this->camParams->TileRect().Height()));
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

            if (this->firstImg) {
                this->ResetView();
                this->firstImg = false;

                if (!this->cameraSettingsSlot.Param<param::StringParam>()->Value().IsEmpty()) {
                    this->onRestoreCamera(this->restoreCameraSettingsSlot);
                }
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

    //cam_type::snapshot_type camsnap;
    //cam_type::matrix_type viewCam, projCam;
    //this->cam.calc_matrices(camsnap, viewCam, projCam);
    //
    //glm::mat4 view = viewCam;
    //glm::mat4 proj = projCam;
    //glm::mat4 mvp = projCam * viewCam;

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
    this->cam.eye(thecam::eye::mono);
    this->cam.projection_type(thecam::projection_type::perspective);
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

    // this->cam.look_at(bbcglm + glm::vec4(0.0f, 0.0f, dist, 0.0f), bbcglm);
    this->cam.position(bbcglm + glm::vec4(0.0f, 0.0f, dist, 0.0f));
    // TODO set up vector correctly
    // via cam.orientation quaternion
    this->cam.orientation(cam_type::quaternion_type::create_identity());

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
    // TODO implement
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
    auto* cr = this->rendererSlot.CallAs<nextgen::CallRender3D_2>();
    if (cr == NULL) return false;

    view::InputEvent evt;
    evt.tag = view::InputEvent::Tag::Key;
    evt.keyData.key = key;
    evt.keyData.action = action;
    evt.keyData.mods = mods;
    cr->SetInputEvent(evt);
    if (!(*cr)(nextgen::CallRender3D_2::FnOnKey)) return false;

    return true;
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
    // This mouse handling/mapping is so utterly weird and should die!
    auto down = action == view::MouseButtonAction::PRESS;
    if (mods.test(view::Modifier::SHIFT)) {
        this->modkeys.SetModifierState(vislib::graphics::InputModifiers::MODIFIER_SHIFT, down);
    } else if (mods.test(view::Modifier::CTRL)) {
        this->modkeys.SetModifierState(vislib::graphics::InputModifiers::MODIFIER_CTRL, down);
    } else if (mods.test(view::Modifier::ALT)) {
        this->modkeys.SetModifierState(vislib::graphics::InputModifiers::MODIFIER_ALT, down);
    }

    if (!this->toggleMouseSelection) {
        switch (button) {
        case megamol::core::view::MouseButton::BUTTON_LEFT:
            this->cursor2d.SetButtonState(0, down);
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
    } else {
        auto* cr = this->rendererSlot.CallAs<nextgen::CallRender3D_2>();
        if (cr == NULL) return false;

        view::InputEvent evt;
        evt.tag = view::InputEvent::Tag::MouseButton;
        evt.mouseButtonData.button = button;
        evt.mouseButtonData.action = action;
        evt.mouseButtonData.mods = mods;
        cr->SetInputEvent(evt);
        if (!(*cr)(nextgen::CallRender3D_2::FnOnMouseButton)) return false;
    }
    return true;
}

/*
 * View3D_2::OnMouseMove
 */
bool nextgen::View3D_2::OnMouseMove(double x, double y) {
    this->mouseX = (float)static_cast<int>(x);
    this->mouseY = (float)static_cast<int>(y);

    // This mouse handling/mapping is so utterly weird and should die!
    if (!this->toggleMouseSelection) {
        this->cursor2d.SetPosition(x, y, true);
    } else {
        auto* cr = this->rendererSlot.CallAs<nextgen::CallRender3D_2>();
        if (cr) {
            view::InputEvent evt;
            evt.tag = view::InputEvent::Tag::MouseMove;
            evt.mouseMoveData.x = x;
            evt.mouseMoveData.y = y;
            cr->SetInputEvent(evt);
            if (!(*cr)(nextgen::CallRender3D_2::FnOnMouseMove)) {
                return false;
            }
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
    // TODO the vislib shaders have to die a slow and painful death
    vislib::graphics::gl::ShaderSource lineVertSrc;
    vislib::graphics::gl::ShaderSource lineFragSrc;
    if (!this->GetCoreInstance()->ShaderSourceFactory().MakeShaderSource("lines::vertex", lineVertSrc)) {
        vislib::sys::Log::DefaultLog.WriteError("Unable to load vertex shader source for bounding box line shader");
    }
    if (!this->GetCoreInstance()->ShaderSourceFactory().MakeShaderSource("lines::fragment", lineFragSrc)) {
        vislib::sys::Log::DefaultLog.WriteError("Unable to load fragment shader source for bounding box line shader");
    }
    try {
        if (!this->lineShader.Create(
                lineVertSrc.Code(), lineVertSrc.Count(), lineFragSrc.Code(), lineFragSrc.Count())) {
            throw vislib::Exception("Shader creation failure", __FILE__, __LINE__);
        }
    } catch (vislib::Exception e) {
        vislib::sys::Log::DefaultLog.WriteError("Unable to create bounding box line shader: %s\n", e.GetMsgA());
        return false;
    }

    // TODO implement


    // TODO cursor stuff

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
//bool View3D_2::OnGetCamParams(view::CallCamParamSync& c) {
//    // TODO implement
//    return true;
//}

/*
 * View3D_2::onStoreCamera
 */
bool View3D_2::onStoreCamera(param::ParamSlot& p) {
    // TODO implement
    return true;
}

/*
 * View3D_2::onRestoreCamera
 */
bool View3D_2::onRestoreCamera(param::ParamSlot& p) {
    // TODO implement
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
 * View3D_2::viewKeyPressed
 */
bool View3D_2::viewKeyPressed(param::ParamSlot& p) {
    // TODO implement
    return true;
}

/*
 * View3D_2::onToggleButton
 */
bool View3D_2::onToggleButton(param::ParamSlot& p) {
    // TODO implement
    return true;
}
