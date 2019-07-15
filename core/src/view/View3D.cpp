/*
 * View3D.cpp
 *
 * Copyright (C) 2008 - 2010 by VISUS (Universitaet Stuttgart).
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "mmcore/view/View3D.h"
#include <GL/glu.h>
#include "mmcore/CoreInstance.h"
#include "mmcore/param/BoolParam.h"
#include "mmcore/param/ButtonParam.h"
#include "mmcore/param/ColorParam.h"
#include "mmcore/param/EnumParam.h"
#include "mmcore/param/FloatParam.h"
#include "mmcore/param/StringParam.h"
#include "mmcore/param/Vector3fParam.h"
#include "mmcore/utility/ColourParser.h"
#include "mmcore/view/CallRender3D.h"
#include "mmcore/view/CallRenderView.h"
#include "mmcore/view/CameraParamOverride.h"
#include "vislib/Exception.h"
#include "vislib/String.h"
#include "vislib/StringSerialiser.h"
#include "vislib/graphics/CameraParamsStore.h"
#include "vislib/graphics/gl/IncludeAllGL.h"
#include "vislib/math/Point.h"
#include "vislib/math/Quaternion.h"
#include "vislib/math/mathfunctions.h"
#include "vislib/sys/Log.h"
#include "vislib/sys/sysfunctions.h"
#ifdef ENABLE_KEYBOARD_VIEW_CONTROL
#    include "mmcore/view/Input.h"
#endif /* ENABLE_KEYBOARD_VIEW_CONTROL */
#include "vislib/Trace.h"
#include "vislib/math/Vector.h"
//#define ROTATOR_HACK
#ifdef ROTATOR_HACK
#    include "vislib/math/Matrix.h"
#    include "vislib/math/Quaternion.h"
#endif

using namespace megamol::core;


/*
 * view::View3D::View3D
 */
view::View3D::View3D(void)
    : view::AbstractRenderingView()
    , AbstractCamParamSync()
    , cam()
    , camParams()
    , camOverrides()
    , cursor2d()
    , modkeys()
    , rotator1()
    , rotator2()
    , zoomer1()
    , zoomer2()
    , mover()
    , lookAtDist()
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
    , frozenValues(NULL)
    , isCamLightSlot(
          "light::isCamLight", "Flag whether the light is relative to the camera or to the world coordinate system")
    , lightDirSlot("light::direction", "Direction vector of the light")
    , lightColDifSlot("light::diffuseCol", "Diffuse light colour")
    , lightColAmbSlot("light::ambientCol", "Ambient light colour")
    , stereoFocusDistSlot("stereo::focusDist", "focus distance for stereo projection")
    , stereoEyeDistSlot("stereo::eyeDist", "eye distance for stereo projection")
    , overrideCall(NULL)
    ,
#ifdef ENABLE_KEYBOARD_VIEW_CONTROL
    viewKeyMoveStepSlot("viewKey::MoveStep", "The move step size in world coordinates")
    , viewKeyRunFactorSlot("viewKey::RunFactor", "The factor for step size multiplication when running (shift)")
    , viewKeyAngleStepSlot("viewKey::AngleStep", "The angle rotate step in degrees")
    , mouseSensitivitySlot("viewKey::MouseSensitivity", "used for WASD mode")
    , viewKeyRotPointSlot("viewKey::RotPoint", "The point around which the view will be roateted")
    ,
#endif /* ENABLE_KEYBOARD_VIEW_CONTROL */
    toggleBBoxSlot("toggleBBox", "Button to toggle the bounding box")
    , toggleSoftCursorSlot("toggleSoftCursor", "Button to toggle the soft cursor")
    , bboxCol{1.0f, 1.0f, 1.0f, 0.625f}
    , bboxColSlot("bboxCol", "Sets the colour for the bounding box")
    , showViewCubeSlot("viewcube::show", "Shows the view cube helper")
    , resetViewOnBBoxChangeSlot("resetViewOnBBoxChange", "whether to reset the view when the bounding boxes change")
    , mouseX(0.0f)
    , mouseY(0.0f)
    , timeCtrl()
    , hookOnChangeOnlySlot("hookOnChange", "whether post-hooks are triggered when the frame would be identical") {

    this->camParams = this->cam.Parameters();
    this->camOverrides = new CameraParamOverride(this->camParams);

    vislib::graphics::ImageSpaceType defWidth(static_cast<vislib::graphics::ImageSpaceType>(100));
    vislib::graphics::ImageSpaceType defHeight(static_cast<vislib::graphics::ImageSpaceType>(100));

    this->camParams->SetVirtualViewSize(defWidth, defHeight);
    this->camParams->SetTileRect(vislib::math::Rectangle<float>(0.0f, 0.0f, defWidth, defHeight));

    this->rendererSlot.SetCompatibleCall<CallRender3DDescription>();
    this->MakeSlotAvailable(&this->rendererSlot);

    // empty bounding box will trigger initialisation
    this->bboxs.Clear();

    this->showBBox << new param::BoolParam(true);
    this->MakeSlotAvailable(&this->showBBox);
    this->showLookAt << new param::BoolParam(false);
    this->MakeSlotAvailable(&this->showLookAt);

    this->cameraSettingsSlot << new param::StringParam("");
    this->MakeSlotAvailable(&this->cameraSettingsSlot);

    this->storeCameraSettingsSlot << new param::ButtonParam(
        view::Key::KEY_C, (view::Modifier::ALT | view::Modifier::SHIFT));
    this->storeCameraSettingsSlot.SetUpdateCallback(&View3D::onStoreCamera);
    this->MakeSlotAvailable(&this->storeCameraSettingsSlot);

    this->restoreCameraSettingsSlot << new param::ButtonParam(view::Key::KEY_C, view::Modifier::ALT);
    this->restoreCameraSettingsSlot.SetUpdateCallback(&View3D::onRestoreCamera);
    this->MakeSlotAvailable(&this->restoreCameraSettingsSlot);

    this->resetViewSlot << new param::ButtonParam(view::Key::KEY_HOME);
    this->resetViewSlot.SetUpdateCallback(&View3D::onResetView);
    this->MakeSlotAvailable(&this->resetViewSlot);

    this->isCamLightSlot << new param::BoolParam(this->isCamLight);
    this->MakeSlotAvailable(&this->isCamLightSlot);

    this->lightDirSlot << new param::Vector3fParam(this->lightDir);
    this->MakeSlotAvailable(&this->lightDirSlot);

    this->lightColDif[0] = this->lightColDif[1] = this->lightColDif[2] = 1.0f;
    this->lightColDif[3] = 1.0f;

    this->lightColAmb[0] = this->lightColAmb[1] = this->lightColAmb[2] = 0.2f;
    this->lightColAmb[3] = 1.0f;

    this->lightColDifSlot << new param::ColorParam(
        this->lightColDif[0], this->lightColDif[1], this->lightColDif[2], 1.0f);
    this->MakeSlotAvailable(&this->lightColDifSlot);

    this->lightColAmbSlot << new param::ColorParam(
        this->lightColAmb[0], this->lightColAmb[1], this->lightColAmb[2], 1.0f);
    this->MakeSlotAvailable(&this->lightColAmbSlot);

    this->ResetView();

    this->stereoEyeDistSlot << new param::FloatParam(this->camParams->StereoDisparity(), 0.0f);
    this->MakeSlotAvailable(&this->stereoEyeDistSlot);

    this->stereoFocusDistSlot << new param::FloatParam(this->camParams->FocalDistance(false), 0.0f);
    this->MakeSlotAvailable(&this->stereoFocusDistSlot);

#ifdef ENABLE_KEYBOARD_VIEW_CONTROL
    this->viewKeyMoveStepSlot << new param::FloatParam(0.1f, 0.001f);
    this->MakeSlotAvailable(&this->viewKeyMoveStepSlot);

    this->viewKeyRunFactorSlot << new param::FloatParam(2.0f, 0.1f);
    this->MakeSlotAvailable(&this->viewKeyRunFactorSlot);

    this->viewKeyAngleStepSlot << new param::FloatParam(15.0f, 0.001f, 360.0f);
    this->MakeSlotAvailable(&this->viewKeyAngleStepSlot);

    this->mouseSensitivitySlot << new param::FloatParam(3.0f, 0.001f, 10.0f);
    this->mouseSensitivitySlot.SetUpdateCallback(&View3D::mouseSensitivityChanged);
    this->MakeSlotAvailable(&this->mouseSensitivitySlot);

    param::EnumParam* vrpsev = new param::EnumParam(1);
    vrpsev->SetTypePair(0, "Position");
    vrpsev->SetTypePair(1, "Look-At");
    this->viewKeyRotPointSlot << vrpsev;
    this->MakeSlotAvailable(&this->viewKeyRotPointSlot);

#endif /* ENABLE_KEYBOARD_VIEW_CONTROL */

    this->toggleSoftCursorSlot << new param::ButtonParam(view::Key::KEY_I, view::Modifier::CTRL);
    this->toggleSoftCursorSlot.SetUpdateCallback(&View3D::onToggleButton);
    this->MakeSlotAvailable(&this->toggleSoftCursorSlot);

    this->toggleBBoxSlot << new param::ButtonParam(view::Key::KEY_I, view::Modifier::ALT);
    this->toggleBBoxSlot.SetUpdateCallback(&View3D::onToggleButton);
    this->MakeSlotAvailable(&this->toggleBBoxSlot);

    this->resetViewOnBBoxChangeSlot << new param::BoolParam(false);
    this->MakeSlotAvailable(&this->resetViewOnBBoxChangeSlot);

    this->bboxColSlot << new param::ColorParam(this->bboxCol[0], this->bboxCol[1], this->bboxCol[2], this->bboxCol[3]);
    this->MakeSlotAvailable(&this->bboxColSlot);

    this->showViewCubeSlot << new param::BoolParam(true);
    this->MakeSlotAvailable(&this->showViewCubeSlot);

    for (unsigned int i = 0; this->timeCtrl.GetSlot(i) != NULL; i++) {
        this->MakeSlotAvailable(this->timeCtrl.GetSlot(i));
    }

    this->MakeSlotAvailable(&this->slotGetCamParams);
    this->MakeSlotAvailable(&this->slotSetCamParams);

    this->hookOnChangeOnlySlot.SetParameter(new param::BoolParam(false));
    this->MakeSlotAvailable(&this->hookOnChangeOnlySlot);
}


/*
 * view::View3D::~View3D
 */
view::View3D::~View3D(void) {
    this->Release();
    SAFE_DELETE(this->frozenValues);
    this->overrideCall = NULL; // DO NOT DELETE
}


/*
 * view::View3D::GetCameraSyncNumber
 */
unsigned int view::View3D::GetCameraSyncNumber(void) const { return this->camParams->SyncNumber(); }


/*
 * view::View3D::SerialiseCamera
 */
void view::View3D::SerialiseCamera(vislib::Serialiser& serialiser) const { this->camParams->Serialise(serialiser); }


/*
 * view::View3D::DeserialiseCamera
 */
void view::View3D::DeserialiseCamera(vislib::Serialiser& serialiser) { this->camParams->Deserialise(serialiser); }


/*
 * view::View3D::Render
 */
void view::View3D::Render(const mmcRenderViewContext& context) {
    float time = static_cast<float>(context.Time);
    float instTime = static_cast<float>(context.InstanceTime);

    if (this->doHookCode()) {
        this->doBeforeRenderHook();
    }

    CallRender3D* cr3d = this->rendererSlot.CallAs<CallRender3D>();

    if (this->viewKeyRotLeft || this->viewKeyRotRight || this->viewKeyRotUp || this->viewKeyRotDown ||
        this->viewKeyRollLeft || this->viewKeyRollRight) {
        // rotate
        float angle = vislib::math::AngleDeg2Rad(this->viewKeyAngleStepSlot.Param<param::FloatParam>()->Value());
        vislib::math::Quaternion<float> q;
        int ptIdx = this->viewKeyRotPointSlot.Param<param::EnumParam>()->Value();
        // ptIdx == 0 : Position
        // ptIdx == 1 : LookAt

        if (this->viewKeyRotLeft) {
            q.Set(angle, this->camParams->Up());
        } else if (this->viewKeyRotRight) {
            q.Set(-angle, this->camParams->Up());
        } else if (this->viewKeyRotUp) {
            q.Set(angle, this->camParams->Right());
        } else if (this->viewKeyRotDown) {
            q.Set(-angle, this->camParams->Right());
        } else if (this->viewKeyRollLeft) {
            q.Set(angle, this->camParams->Front());
        } else if (this->viewKeyRollRight) {
            q.Set(-angle, this->camParams->Front());
        }

        vislib::math::Vector<float, 3> pos(this->camParams->Position().PeekCoordinates());
        vislib::math::Vector<float, 3> lat(this->camParams->LookAt().PeekCoordinates());
        vislib::math::Vector<float, 3> up(this->camParams->Up());

        if (ptIdx == 0) {
            lat -= pos;
            lat = q * lat;
            up = q * up;
            lat += pos;

        } else if (ptIdx == 1) {
            pos -= lat;
            pos = q * pos;
            up = q * up;
            pos += lat;
        }

        this->camParams->SetView(vislib::math::Point<float, 3>(pos.PeekComponents()),
            vislib::math::Point<float, 3>(lat.PeekComponents()), up);

    } else if (this->viewKeyZoomIn || viewKeyZoomOut || this->viewKeyMoveLeft || this->viewKeyMoveRight ||
               this->viewKeyMoveUp || this->viewKeyMoveDown) {
        // move
        float step = this->viewKeyMoveStepSlot.Param<param::FloatParam>()->Value();
        const float runFactor = this->viewKeyRunFactorSlot.Param<param::FloatParam>()->Value();
        if (this->running) {
            step *= runFactor;
        }
        vislib::math::Vector<float, 3> move;

        if (this->viewKeyZoomIn) {
            move = this->camParams->Front();
            move *= step;
        } else if (this->viewKeyZoomOut) {
            move = this->camParams->Front();
            move *= -step;
        } else if (this->viewKeyMoveLeft) {
            move = this->camParams->Right();
            move *= -step;
        } else if (this->viewKeyMoveRight) {
            move = this->camParams->Right();
            move *= step;
        } else if (this->viewKeyMoveUp) {
            move = this->camParams->Up();
            move *= step;
        } else if (this->viewKeyMoveDown) {
            move = this->camParams->Up();
            move *= -step;
        }

        this->camParams->SetView(this->camParams->Position() + move, this->camParams->LookAt() + move,
            vislib::math::Vector<float, 3>(this->camParams->Up()));
    }

    AbstractRenderingView::beginFrame();

    // Conditionally synchronise camera from somewhere else.
    this->SyncCamParams(this->cam.Parameters());

    // clear viewport
    if (this->overrideViewport != NULL) {
        if ((this->overrideViewport[0] >= 0) && (this->overrideViewport[1] >= 0) && (this->overrideViewport[2] > 0) &&
            (this->overrideViewport[3] > 0)) {
            ::glViewport(this->overrideViewport[0], this->overrideViewport[1], this->overrideViewport[2],
                this->overrideViewport[3]);
        }
    } else {
        // this is correct in non-override mode,
        //  because then the tile will be whole viewport
        ::glViewport(0, 0, static_cast<GLsizei>(this->camParams->TileRect().Width()),
            static_cast<GLsizei>(this->camParams->TileRect().Height()));
    }

    if (this->overrideCall != NULL) {
        if (cr3d != nullptr) {
            RenderOutputOpenGL* ro = dynamic_cast<RenderOutputOpenGL*>(overrideCall);
            if (ro != nullptr) {
                *static_cast<RenderOutputOpenGL*>(cr3d) = *ro;
            }
        }
        this->overrideCall->EnableOutputBuffer();
    } else if (cr3d != nullptr) {
        cr3d->SetOutputBuffer(GL_BACK);
        if (cr3d != nullptr) cr3d->GetViewport(); // access the viewport to enforce evaluation
    }

    const float* bkgndCol = (this->overrideBkgndCol != NULL) ? this->overrideBkgndCol : this->BkgndColour();
    ::glClearColor(bkgndCol[0], bkgndCol[1], bkgndCol[2], 0.0f);
    ::glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    if (cr3d == NULL) {
        this->renderTitle(this->cam.Parameters()->TileRect().Left(), this->cam.Parameters()->TileRect().Bottom(),
            this->cam.Parameters()->TileRect().Width(), this->cam.Parameters()->TileRect().Height(),
            this->cam.Parameters()->VirtualViewSize().Width(), this->cam.Parameters()->VirtualViewSize().Height(),
            (this->cam.Parameters()->Projection() != vislib::graphics::CameraParameters::MONO_ORTHOGRAPHIC) &&
                (this->cam.Parameters()->Projection() != vislib::graphics::CameraParameters::MONO_PERSPECTIVE),
            this->cam.Parameters()->Eye() == vislib::graphics::CameraParameters::LEFT_EYE, instTime);
        this->endFrame(true);
        return; // empty enought
    } else {
        cr3d->SetGpuAffinity(context.GpuAffinity);
        this->removeTitleRenderer();
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
        param::FloatParam* fp = this->stereoEyeDistSlot.Param<param::FloatParam>();
        this->camParams->SetStereoDisparity(fp->Value());
        fp->SetValue(this->camParams->StereoDisparity());
        this->stereoEyeDistSlot.ResetDirty();
    }
    if (this->stereoFocusDistSlot.IsDirty()) {
        param::FloatParam* fp = this->stereoFocusDistSlot.Param<param::FloatParam>();
        this->camParams->SetFocalDistance(fp->Value());
        fp->SetValue(this->camParams->FocalDistance(false));
        this->stereoFocusDistSlot.ResetDirty();
    }
    if (cr3d != NULL) {
        (*cr3d)(AbstractCallRender::FnGetExtents);
        if (this->firstImg ||
            (!(cr3d->AccessBoundingBoxes() == this->bboxs) &&
                !(!cr3d->AccessBoundingBoxes().IsAnyValid() && !this->bboxs.IsObjectSpaceBBoxValid() &&
                    !this->bboxs.IsObjectSpaceClipBoxValid() && this->bboxs.IsWorldSpaceBBoxValid() &&
                    !this->bboxs.IsWorldSpaceClipBoxValid()))) {
            this->bboxs = cr3d->AccessBoundingBoxes();

            if (this->firstImg) {
                this->ResetView();
                this->firstImg = false;
                if (!this->cameraSettingsSlot.Param<param::StringParam>()->Value().IsEmpty()) {
                    this->onRestoreCamera(this->restoreCameraSettingsSlot);
                }
            } else if (resetViewOnBBoxChangeSlot.Param<param::BoolParam>()->Value())
                this->ResetView();
        }

#ifdef ROTATOR_HACK
        cr3d->SetTimeFramesCount(360);
#endif

        this->timeCtrl.SetTimeExtend(cr3d->TimeFramesCount(), cr3d->IsInSituTime());
        if (time > static_cast<float>(cr3d->TimeFramesCount())) {
            time = static_cast<float>(cr3d->TimeFramesCount());
        }

#ifndef ROTATOR_HACK
        cr3d->SetTime(this->frozenValues ? this->frozenValues->time : time);
#else
        cr3d->SetTime(0.0f);
#endif
        cr3d->SetCameraParameters(this->cam.Parameters()); // < here we use the 'active' parameters!
        cr3d->SetLastFrameTime(AbstractRenderingView::lastFrameTime());
    }
    this->camParams->CalcClipping(this->bboxs.ClipBox(), 0.1f);
    // This is painfully wrong in the vislib camera, and is fixed here as sort of hotfix
    float fc = this->camParams->FarClip();
    float nc = this->camParams->NearClip();
    float fnc = fc * 0.001f;
    if (fnc > nc) {
        this->camParams->SetClip(fnc, fc);
    }

    if (!(*this->lastFrameParams == *(this->camParams.DynamicCast<vislib::graphics::CameraParamsStore>())) ||
        !this->hookOnChangeOnlySlot.Param<param::BoolParam>()->Value()) {
        // vislib::sys::Log::DefaultLog.WriteInfo("view %s: camera has changed, the frame has sensible information.",
        // this->FullName().PeekBuffer());
        frameIsNew = true;
    } else {
        frameIsNew = false;
    }

    if (this->bboxColSlot.IsDirty()) {
        this->bboxColSlot.Param<param::ColorParam>()->Value(
            this->bboxCol[0], this->bboxCol[1], this->bboxCol[2], this->bboxCol[3]);
        this->bboxColSlot.ResetDirty();
    }

    // set light parameters
    if (this->isCamLightSlot.IsDirty()) {
        this->isCamLightSlot.ResetDirty();
        this->isCamLight = this->isCamLightSlot.Param<param::BoolParam>()->Value();
    }
    if (this->lightDirSlot.IsDirty()) {
        this->lightDirSlot.ResetDirty();
        this->lightDir = this->lightDirSlot.Param<param::Vector3fParam>()->Value();
    }
    if (this->lightColAmbSlot.IsDirty()) {
        this->lightColAmbSlot.ResetDirty();
        this->lightColAmbSlot.Param<param::ColorParam>()->Value(
            this->lightColAmb[0], this->lightColAmb[1], this->lightColAmb[2]);
    }
    if (this->lightColDifSlot.IsDirty()) {
        this->lightColDifSlot.ResetDirty();
        this->lightColDifSlot.Param<param::ColorParam>()->Value(
            this->lightColDif[0], this->lightColDif[1], this->lightColDif[2]);
    }
    ::glEnable(GL_LIGHTING); // TODO: check renderer capabilities
    ::glEnable(GL_LIGHT0);
    const float lp[4] = {-this->lightDir.X(), -this->lightDir.Y(), -this->lightDir.Z(), 0.0f};
    ::glLightfv(GL_LIGHT0, GL_AMBIENT, this->lightColAmb);
    ::glLightfv(GL_LIGHT0, GL_DIFFUSE, this->lightColDif);
    const float zeros[4] = {0.0f, 0.0f, 0.0f, 1.0f};
    const float ones[4] = {1.0f, 1.0f, 1.0f, 1.0f};
    ::glLightfv(GL_LIGHT0, GL_SPECULAR, ones);
    ::glLightModelfv(GL_LIGHT_MODEL_AMBIENT, zeros);

    // setup matrices

#ifdef ROTATOR_HACK
    ::vislib::math::Point<float, 3> c_e_p = this->cam.Parameters()->EyePosition();
    ::vislib::math::Point<float, 3> c_l_p = this->cam.Parameters()->LookAt();
    ::vislib::math::Vector<float, 3> c_u_v = this->cam.Parameters()->Up();

    ::vislib::math::Vector<float, 3> c_l_v = c_e_p - c_l_p;
    ::vislib::math::Quaternion<float> c_l_q(time * M_PI / 180.0f, c_u_v);
    c_l_v = c_l_q * c_l_v;
    this->cam.Parameters()->SetView(c_l_p + c_l_v, c_l_p, c_u_v);
    this->cam.Parameters()->CalcClipping(this->bboxs.ClipBox(), 0.1f);
#endif

    ::glMatrixMode(GL_PROJECTION);
    ::glLoadIdentity();
    this->cam.glMultProjectionMatrix();

    ::glMatrixMode(GL_MODELVIEW);
    ::glLoadIdentity();

    if (this->isCamLight) glLightfv(GL_LIGHT0, GL_POSITION, lp);

    this->cam.glMultViewMatrix();

    if (!this->isCamLight) glLightfv(GL_LIGHT0, GL_POSITION, lp);

    // render bounding box backside
    if (this->showBBox.Param<param::BoolParam>()->Value()) {
        this->renderBBoxBackside();
    }
    if (this->showLookAt.Param<param::BoolParam>()->Value()) {
        this->renderLookAt();
    }
    ::glPushMatrix();

    if (cr3d != NULL) {
        cr3d->SetInstanceTime(instTime);
#ifndef ROTATOR_HACK
        cr3d->SetTime(time);
#else
        cr3d->SetTime(0);
#endif
    }

    // call for render
    if (cr3d != NULL) {
        (*cr3d)(AbstractCallRender::FnRender);
    }

    // render bounding box front
    ::glPopMatrix();
    if (this->showBBox.Param<param::BoolParam>()->Value()) {
        this->renderBBoxFrontside();
    }

    if (this->showViewCubeSlot.Param<param::BoolParam>()->Value()) {
        this->renderViewCube();
    }

    if (this->showSoftCursor()) {
        this->renderSoftCursor();
    }

#ifdef ROTATOR_HACK
    this->cam.Parameters()->SetView(c_e_p, c_l_p, c_u_v);
#endif

    AbstractRenderingView::endFrame();

    this->lastFrameParams->CopyFrom(this->camParams, false);

    if (this->doHookCode() && frameIsNew) {
        this->doAfterRenderHook();
    }
}


/*
 * view::View3D::ResetView
 */
void view::View3D::ResetView(void) {
    using namespace vislib::graphics;
    VLTRACE(VISLIB_TRCELVL_INFO, "View3D::ResetView\n");

    this->camParams->SetClip(0.1f, 100.0f);
    this->camParams->SetApertureAngle(30.0f);
    this->camParams->SetProjection(vislib::graphics::CameraParameters::MONO_PERSPECTIVE);
    this->camParams->SetStereoParameters(0.3f, /* this is not so clear! */
        vislib::graphics::CameraParameters::LEFT_EYE, 0.0f /* this is autofocus */);
    this->camParams->Limits()->LimitClippingDistances(0.01f, 0.1f);

    if (!this->bboxs.IsWorldSpaceBBoxValid()) {
        this->bboxs.SetWorldSpaceBBox(-1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0);
    }
    float dist = (0.5f * sqrtf((this->bboxs.WorldSpaceBBox().Width() * this->bboxs.WorldSpaceBBox().Width()) +
                               (this->bboxs.WorldSpaceBBox().Depth() * this->bboxs.WorldSpaceBBox().Depth()) +
                               (this->bboxs.WorldSpaceBBox().Height() * this->bboxs.WorldSpaceBBox().Height()))) /
                 tanf(this->cam.Parameters()->HalfApertureAngle());

    ImageSpaceDimension dim = this->camParams->VirtualViewSize();
    double halfFovX =
        (static_cast<double>(dim.Width()) * static_cast<double>(this->cam.Parameters()->HalfApertureAngle())) /
        static_cast<double>(dim.Height());
    double distX = static_cast<double>(this->bboxs.WorldSpaceBBox().Width()) / (2.0 * tan(halfFovX));
    double distY = static_cast<double>(this->bboxs.WorldSpaceBBox().Height()) /
                   (2.0 * tan(static_cast<double>(this->cam.Parameters()->HalfApertureAngle())));
    dist = static_cast<float>((distX > distY) ? distX : distY);
    dist = dist + (this->bboxs.WorldSpaceBBox().Depth() / 2.0f);
    SceneSpacePoint3D bbc = this->bboxs.WorldSpaceBBox().CalcCenter();

    this->camParams->SetView(bbc + SceneSpaceVector3D(0.0f, 0.0f, dist), bbc, SceneSpaceVector3D(0.0f, 1.0f, 0.0f));

    this->zoomer1.SetSpeed(dist);
    this->lookAtDist.SetSpeed(dist);
}


/*
 * view::View3D::Resize
 */
void view::View3D::Resize(unsigned int width, unsigned int height) {
    this->camParams->SetVirtualViewSize(
        static_cast<vislib::graphics::ImageSpaceType>(width), static_cast<vislib::graphics::ImageSpaceType>(height));
}


/*
 * view::View3D::OnRenderView
 */
bool view::View3D::OnRenderView(Call& call) {
    float overBC[3];
    int overVP[4] = {0, 0, 0, 0};
    view::CallRenderView* crv = dynamic_cast<view::CallRenderView*>(&call);
    if (crv == NULL) return false;

    this->overrideViewport = overVP;
    if (crv->IsProjectionSet() || crv->IsTileSet() || crv->IsViewportSet()) {
        this->cam.SetParameters(this->camOverrides);
        this->camOverrides.DynamicCast<CameraParamOverride>()->SetOverrides(*crv);
        if (crv->IsViewportSet()) {
            overVP[2] = crv->ViewportWidth();
            overVP[3] = crv->ViewportHeight();
            if (!crv->IsTileSet()) {
                this->camOverrides->SetVirtualViewSize(
                    static_cast<vislib::graphics::ImageSpaceType>(crv->ViewportWidth()),
                    static_cast<vislib::graphics::ImageSpaceType>(crv->ViewportHeight()));
                this->camOverrides->ResetTileRect();
            }
        }
    }
    if (crv->IsBackgroundSet()) {
        overBC[0] = static_cast<float>(crv->BackgroundRed()) / 255.0f;
        overBC[1] = static_cast<float>(crv->BackgroundGreen()) / 255.0f;
        overBC[2] = static_cast<float>(crv->BackgroundBlue()) / 255.0f;
        this->overrideBkgndCol = overBC; // hurk
    }

    this->overrideCall = dynamic_cast<AbstractCallRender*>(&call);

    float time = crv->Time();
    if (time < 0.0f) time = this->DefaultTime(crv->InstanceTime());
    mmcRenderViewContext context;
    ::ZeroMemory(&context, sizeof(context));
    context.Time = time;
    context.InstanceTime = crv->InstanceTime();
    // TODO: Affinity
    this->Render(context);

    if (this->overrideCall != NULL) {
        // mueller: I added the DisableOutputBuffer (resetting the override was here
        // before) in order to make sure that the viewport is reset that has been
        // set by an override call.
        this->overrideCall->DisableOutputBuffer();
        this->overrideCall = NULL;
    }

    if (crv->IsProjectionSet() || crv->IsTileSet() || crv->IsViewportSet()) {
        this->cam.SetParameters(this->frozenValues ? this->frozenValues->camParams : this->camParams);
    }
    this->overrideBkgndCol = NULL;
    this->overrideViewport = NULL;

    return true;
}


/*
 * view::View3D::UpdateFreeze
 */
void view::View3D::UpdateFreeze(bool freeze) {
    // printf("%s view\n", freeze ? "Freezing" : "Unfreezing");
    if (freeze) {
        if (this->frozenValues == NULL) {
            this->frozenValues = new FrozenValues();
            this->camOverrides.DynamicCast<CameraParamOverride>()->SetParametersBase(this->frozenValues->camParams);
            this->cam.SetParameters(this->frozenValues->camParams);
        }
        *(this->frozenValues->camParams) = *this->camParams;
        this->frozenValues->time = 0.0f;
    } else {
        this->camOverrides.DynamicCast<CameraParamOverride>()->SetParametersBase(this->camParams);
        this->cam.SetParameters(this->camParams);
        SAFE_DELETE(this->frozenValues);
    }
}


bool view::View3D::OnKey(Key key, KeyAction action, Modifiers mods) {
    auto* cr = this->rendererSlot.CallAs<view::CallRender3D>();
    if (cr == NULL) return false;

    running = mods.test(Modifier::SHIFT);
    bool down = (action == KeyAction::PRESS || action == KeyAction::REPEAT) && (action != KeyAction::RELEASE);
    bool ret = true;
    if (wasd) {
        switch (key) {
        case Key::KEY_W:
            this->viewKeyZoomIn = down;
            break;
        case Key::KEY_S:
            this->viewKeyZoomOut = down;
            break;
        case Key::KEY_A:
            this->viewKeyMoveLeft = down;
            break;
        case Key::KEY_D:
            this->viewKeyMoveRight = down;
            break;
        case Key::KEY_Q:
            this->viewKeyRollLeft = down;
            break;
        case Key::KEY_E:
            this->viewKeyRollRight = down;
            break;
        case Key::KEY_R:
            this->viewKeyMoveUp = down;
            break;
        case Key::KEY_F:
            this->viewKeyMoveDown = down;
            break;
        case Key::KEY_UP:
            if (invertY)
                this->viewKeyRotDown = down;
            else
                this->viewKeyRotUp = down;
            break;
        case Key::KEY_DOWN:
            if (invertY)
                this->viewKeyRotUp = down;
            else
                this->viewKeyRotDown = down;
            break;
        case Key::KEY_LEFT:
            if (invertX)
                this->viewKeyRotRight = down;
            else
                this->viewKeyRotLeft = down;
            break;
        case Key::KEY_RIGHT:
            if (invertX)
                this->viewKeyRotLeft = down;
            else
                this->viewKeyRotRight = down;
            break;
        default:
            ret = false;
        }
    } else {
        auto ctrlshift = mods.test(Modifier::CTRL) && mods.test(Modifier::SHIFT) && !mods.test(Modifier::ALT);
        auto ctrlalt = mods.test(Modifier::CTRL) && mods.test(Modifier::ALT) && !mods.test(Modifier::SHIFT);
        auto ctrl = mods.test(Modifier::CTRL) && !mods.test(Modifier::ALT) && !mods.test(Modifier::SHIFT);
        switch (key) {
        case Key::KEY_UP:
            this->viewKeyZoomIn = ctrlshift && down;
            this->viewKeyMoveUp = ctrlalt && down;
            if (ctrl) {
                if (invertY)
                    this->viewKeyRotDown = down;
                else
                    this->viewKeyRotUp = down;
            }
            break;
        case Key::KEY_DOWN:
            this->viewKeyZoomOut = ctrlshift && down;
            this->viewKeyMoveDown = ctrlalt && down;
            if (ctrl) {
                if (!invertY)
                    this->viewKeyRotDown = down;
                else
                    this->viewKeyRotUp = down;
            }
            break;
        case Key::KEY_LEFT:
            this->viewKeyMoveLeft = ctrlshift && down;
            this->viewKeyRollLeft = ctrlalt && down;
            if (ctrl) {
                if (invertX)
                    this->viewKeyRotRight = down;
                else
                    this->viewKeyRotLeft = down;
            }
            break;
        case Key::KEY_RIGHT:
            this->viewKeyMoveRight = ctrlshift && down;
            this->viewKeyRollRight = ctrlalt && down;
            if (ctrl) {
                if (!invertX)
                    this->viewKeyRotRight = down;
                else
                    this->viewKeyRotLeft = down;
            }
            break;
        default:
            ret = false;
        }
    }

    if (!ret) {
        InputEvent evt;
        evt.tag = InputEvent::Tag::Key;
        evt.keyData.key = key;
        evt.keyData.action = action;
        evt.keyData.mods = mods;
        cr->SetInputEvent(evt);
        if (!(*cr)(view::CallRender3D::FnOnKey)) return false;
    }

    return ret;
}


bool view::View3D::OnChar(unsigned int codePoint) {
    auto* cr = this->rendererSlot.CallAs<view::CallRender3D>();
    if (cr == NULL) return false;

    InputEvent evt;
    evt.tag = InputEvent::Tag::Char;
    evt.charData.codePoint = codePoint;
    cr->SetInputEvent(evt);
    if (!(*cr)(view::CallRender3D::FnOnChar)) return false;

    return true;
}


bool view::View3D::OnMouseButton(MouseButton button, MouseButtonAction action, Modifiers mods) {
    auto* cr = this->rendererSlot.CallAs<view::CallRender3D>();
    if (cr) {
        InputEvent evt;
        evt.tag = InputEvent::Tag::MouseButton;
        evt.mouseButtonData.button = button;
        evt.mouseButtonData.action = action;
        evt.mouseButtonData.mods = mods;
        cr->SetInputEvent(evt);
        if ((*cr)(view::CallRender3D::FnOnMouseButton)) return true;
    }

    auto down = action == MouseButtonAction::PRESS;
    if (mods.test(view::Modifier::SHIFT)) {
        this->modkeys.SetModifierState(vislib::graphics::InputModifiers::MODIFIER_SHIFT, down);
    } else if (mods.test(view::Modifier::CTRL)) {
        this->modkeys.SetModifierState(vislib::graphics::InputModifiers::MODIFIER_CTRL, down);
    } else if (mods.test(view::Modifier::ALT)) {
        this->modkeys.SetModifierState(vislib::graphics::InputModifiers::MODIFIER_ALT, down);
    }

    switch (button) {
    case view::MouseButton::BUTTON_LEFT:
        this->cursor2d.SetButtonState(0, down);
        break;
    case view::MouseButton::BUTTON_RIGHT:
        this->cursor2d.SetButtonState(1, down);
        break;
    case view::MouseButton::BUTTON_MIDDLE:
        this->cursor2d.SetButtonState(2, down);
        break;
    default:
        break;
    }

    return true;
}


bool view::View3D::OnMouseMove(double x, double y) {
    auto* cr = this->rendererSlot.CallAs<view::CallRender3D>();
    if (cr) {
        InputEvent evt;
        evt.tag = InputEvent::Tag::MouseMove;
        evt.mouseMoveData.x = x;
        evt.mouseMoveData.y = y;
        cr->SetInputEvent(evt);
        if ((*cr)(view::CallRender3D::FnOnMouseMove)) return true;
    }

    this->mouseX = (float)static_cast<int>(x);
    this->mouseY = (float)static_cast<int>(y);

    this->cursor2d.SetPosition(x, y, true);

    return true;
}


bool view::View3D::OnMouseScroll(double dx, double dy) {
    auto* cr = this->rendererSlot.CallAs<view::CallRender3D>();
    if (cr == NULL) return false;

    InputEvent evt;
    evt.tag = InputEvent::Tag::MouseScroll;
    evt.mouseScrollData.dx = dx;
    evt.mouseScrollData.dy = dy;
    cr->SetInputEvent(evt);
    if (!(*cr)(view::CallRender3D::FnOnMouseScroll)) return false;

    return true;
}


/*
 * view::View3D::unpackMouseCoordinates
 */
void view::View3D::unpackMouseCoordinates(float& x, float& y) {
    x *= this->camParams->VirtualViewSize().Width();
    y *= this->camParams->VirtualViewSize().Height();
    y -= 1.0f;
}


/*
 * view::View3D::create
 */
bool view::View3D::create(void) {

    try {
        wasd = vislib::CharTraitsW::ParseBool(this->GetCoreInstance()->Configuration().ConfigValue("wasd"));
    } catch (...) {
    }
    try {
        invertX = vislib::CharTraitsW::ParseBool(this->GetCoreInstance()->Configuration().ConfigValue("invertX"));
    } catch (...) {
    }
    try {
        invertY = vislib::CharTraitsW::ParseBool(this->GetCoreInstance()->Configuration().ConfigValue("invertY"));
    } catch (...) {
    }

    this->cursor2d.SetButtonCount(3); /* This could be configurable. */
    this->modkeys.SetModifierCount(3);

    this->rotator1.SetCameraParams(this->camParams);
    this->rotator1.SetTestButton(0 /* left mouse button */);
    this->rotator1.SetAltModifier(vislib::graphics::InputModifiers::MODIFIER_SHIFT);
    this->rotator1.SetModifierTestCount(2);
    this->rotator1.SetModifierTest(0, vislib::graphics::InputModifiers::MODIFIER_CTRL, wasd);
    this->rotator1.SetModifierTest(1, vislib::graphics::InputModifiers::MODIFIER_ALT, false);

    if (wasd) {
        this->rotator2.SetInvertX(invertX);
        this->rotator2.SetInvertY(invertY);
    }
    this->rotator2.SetCameraParams(this->camParams);
    this->rotator2.SetTestButton(0 /* left mouse button */);
    this->rotator2.SetAltModifier(vislib::graphics::InputModifiers::MODIFIER_SHIFT);
    this->rotator2.SetModifierTestCount(2);
    this->rotator2.SetModifierTest(0, vislib::graphics::InputModifiers::MODIFIER_CTRL, !wasd);
    this->rotator2.SetModifierTest(1, vislib::graphics::InputModifiers::MODIFIER_ALT, false);

    this->zoomer1.SetCameraParams(this->camParams);
    this->zoomer1.SetTestButton(2 /* mid mouse button */);
    this->zoomer1.SetModifierTestCount(2);
    this->zoomer1.SetModifierTest(0, vislib::graphics::InputModifiers::MODIFIER_ALT, false);
    this->zoomer1.SetModifierTest(1, vislib::graphics::InputModifiers::MODIFIER_CTRL, false);
    this->zoomer1.SetZoomBehaviour(vislib::graphics::CameraZoom2DMove::FIX_LOOK_AT);

    this->zoomer2.SetCameraParams(this->camParams);
    this->zoomer2.SetTestButton(2 /* mid mouse button */);
    this->zoomer2.SetModifierTestCount(2);
    this->zoomer2.SetModifierTest(0, vislib::graphics::InputModifiers::MODIFIER_ALT, true);
    this->zoomer2.SetModifierTest(1, vislib::graphics::InputModifiers::MODIFIER_CTRL, false);

    this->mover.SetCameraParams(this->camParams);
    this->mover.SetTestButton(0 /* left mouse button */);
    this->mover.SetModifierTestCount(1);
    this->mover.SetModifierTest(0, vislib::graphics::InputModifiers::MODIFIER_ALT, true);

    this->lookAtDist.SetCameraParams(this->camParams);
    this->lookAtDist.SetTestButton(2 /* mid mouse button */);
    this->lookAtDist.SetModifierTestCount(1);
    this->lookAtDist.SetModifierTest(0, vislib::graphics::InputModifiers::MODIFIER_CTRL, true);

    this->cursor2d.SetCameraParams(this->camParams);
    this->cursor2d.RegisterCursorEvent(&this->rotator1);
    this->cursor2d.RegisterCursorEvent(&this->rotator2);
    this->cursor2d.RegisterCursorEvent(&this->zoomer1);
    this->cursor2d.RegisterCursorEvent(&this->zoomer2);
    this->cursor2d.RegisterCursorEvent(&this->mover);
    this->cursor2d.RegisterCursorEvent(&this->lookAtDist);
    this->cursor2d.SetInputModifiers(&this->modkeys);

    this->modkeys.RegisterObserver(&this->cursor2d);

    this->firstImg = true;

    return true;
}


/*
 * view::View3D::release
 */
void view::View3D::release(void) {
    this->removeTitleRenderer();
    this->cursor2d.UnregisterCursorEvent(&this->rotator1);
    this->cursor2d.UnregisterCursorEvent(&this->rotator2);
    this->cursor2d.UnregisterCursorEvent(&this->zoomer1);
    this->cursor2d.UnregisterCursorEvent(&this->zoomer2);
    this->cursor2d.UnregisterCursorEvent(&this->mover);
    SAFE_DELETE(this->frozenValues);
}


bool view::View3D::mouseSensitivityChanged(param::ParamSlot& p) {
    this->rotator2.SetMouseSensitivity(p.Param<param::FloatParam>()->Value());
    return true;
}


/*
 * view::View3D::renderBBox
 */
void view::View3D::renderBBox(void) {
    const vislib::math::Cuboid<float>& boundingBox = this->bboxs.WorldSpaceBBox();
    if (!this->bboxs.IsWorldSpaceBBoxValid()) {
        this->bboxs.SetWorldSpaceBBox(-1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f);
    }

    ::glBegin(GL_QUADS);

    ::glEdgeFlag(true);

    ::glVertex3f(boundingBox.Left(), boundingBox.Bottom(), boundingBox.Back());
    ::glVertex3f(boundingBox.Left(), boundingBox.Top(), boundingBox.Back());
    ::glVertex3f(boundingBox.Right(), boundingBox.Top(), boundingBox.Back());
    ::glVertex3f(boundingBox.Right(), boundingBox.Bottom(), boundingBox.Back());

    ::glVertex3f(boundingBox.Left(), boundingBox.Bottom(), boundingBox.Front());
    ::glVertex3f(boundingBox.Right(), boundingBox.Bottom(), boundingBox.Front());
    ::glVertex3f(boundingBox.Right(), boundingBox.Top(), boundingBox.Front());
    ::glVertex3f(boundingBox.Left(), boundingBox.Top(), boundingBox.Front());

    ::glVertex3f(boundingBox.Left(), boundingBox.Top(), boundingBox.Back());
    ::glVertex3f(boundingBox.Left(), boundingBox.Top(), boundingBox.Front());
    ::glVertex3f(boundingBox.Right(), boundingBox.Top(), boundingBox.Front());
    ::glVertex3f(boundingBox.Right(), boundingBox.Top(), boundingBox.Back());

    ::glVertex3f(boundingBox.Left(), boundingBox.Bottom(), boundingBox.Back());
    ::glVertex3f(boundingBox.Right(), boundingBox.Bottom(), boundingBox.Back());
    ::glVertex3f(boundingBox.Right(), boundingBox.Bottom(), boundingBox.Front());
    ::glVertex3f(boundingBox.Left(), boundingBox.Bottom(), boundingBox.Front());

    ::glVertex3f(boundingBox.Left(), boundingBox.Bottom(), boundingBox.Back());
    ::glVertex3f(boundingBox.Left(), boundingBox.Bottom(), boundingBox.Front());
    ::glVertex3f(boundingBox.Left(), boundingBox.Top(), boundingBox.Front());
    ::glVertex3f(boundingBox.Left(), boundingBox.Top(), boundingBox.Back());

    ::glVertex3f(boundingBox.Right(), boundingBox.Bottom(), boundingBox.Back());
    ::glVertex3f(boundingBox.Right(), boundingBox.Top(), boundingBox.Back());
    ::glVertex3f(boundingBox.Right(), boundingBox.Top(), boundingBox.Front());
    ::glVertex3f(boundingBox.Right(), boundingBox.Bottom(), boundingBox.Front());

    ::glEnd();

//#define _SHOW_CLIPBOX
#ifdef _SHOW_CLIPBOX
    {
        ::glColor4ub(255, 0, 0, 128);
        const vislib::math::Cuboid<float>& boundingBox = this->bboxs.WorldSpaceClipBox();
        ::glBegin(GL_QUADS);

        ::glEdgeFlag(true);

        ::glVertex3f(boundingBox.Left(), boundingBox.Bottom(), boundingBox.Back());
        ::glVertex3f(boundingBox.Left(), boundingBox.Top(), boundingBox.Back());
        ::glVertex3f(boundingBox.Right(), boundingBox.Top(), boundingBox.Back());
        ::glVertex3f(boundingBox.Right(), boundingBox.Bottom(), boundingBox.Back());

        ::glVertex3f(boundingBox.Left(), boundingBox.Bottom(), boundingBox.Front());
        ::glVertex3f(boundingBox.Right(), boundingBox.Bottom(), boundingBox.Front());
        ::glVertex3f(boundingBox.Right(), boundingBox.Top(), boundingBox.Front());
        ::glVertex3f(boundingBox.Left(), boundingBox.Top(), boundingBox.Front());

        ::glVertex3f(boundingBox.Left(), boundingBox.Top(), boundingBox.Back());
        ::glVertex3f(boundingBox.Left(), boundingBox.Top(), boundingBox.Front());
        ::glVertex3f(boundingBox.Right(), boundingBox.Top(), boundingBox.Front());
        ::glVertex3f(boundingBox.Right(), boundingBox.Top(), boundingBox.Back());

        ::glVertex3f(boundingBox.Left(), boundingBox.Bottom(), boundingBox.Back());
        ::glVertex3f(boundingBox.Right(), boundingBox.Bottom(), boundingBox.Back());
        ::glVertex3f(boundingBox.Right(), boundingBox.Bottom(), boundingBox.Front());
        ::glVertex3f(boundingBox.Left(), boundingBox.Bottom(), boundingBox.Front());

        ::glVertex3f(boundingBox.Left(), boundingBox.Bottom(), boundingBox.Back());
        ::glVertex3f(boundingBox.Left(), boundingBox.Bottom(), boundingBox.Front());
        ::glVertex3f(boundingBox.Left(), boundingBox.Top(), boundingBox.Front());
        ::glVertex3f(boundingBox.Left(), boundingBox.Top(), boundingBox.Back());

        ::glVertex3f(boundingBox.Right(), boundingBox.Bottom(), boundingBox.Back());
        ::glVertex3f(boundingBox.Right(), boundingBox.Top(), boundingBox.Back());
        ::glVertex3f(boundingBox.Right(), boundingBox.Top(), boundingBox.Front());
        ::glVertex3f(boundingBox.Right(), boundingBox.Bottom(), boundingBox.Front());

        ::glEnd();
    }
#endif /* _SHOW_CLIPBOX */
}


/*
 * view::View3D::renderBBoxBackside
 */
void view::View3D::renderBBoxBackside(void) {
    ::glDisable(GL_LIGHTING);
    ::glEnable(GL_BLEND);
    ::glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    ::glEnable(GL_DEPTH_TEST);
    ::glEnable(GL_CULL_FACE);
    ::glCullFace(GL_FRONT);
    ::glEnable(GL_LINE_SMOOTH);
    ::glLineWidth(1.25f);
    //::glDisable(GL_TEXTURE_2D);
    ::glPolygonMode(GL_BACK, GL_LINE);

    // XXX: Note that historically, we had a hard-coded alpha of 0.625f, but just for the backside.
    ::glColor4fv(this->bboxCol);
    this->renderBBox();

    //::glPolygonMode(GL_BACK, GL_FILL);
    //::glDisable(GL_DEPTH_TEST);

    //::glColor4ub(this->bboxCol.R(), this->bboxCol.G(), this->bboxCol.B(), 16);
    // this->renderBBox();

    ::glCullFace(GL_BACK);

    glEnable(GL_LIGHTING);
    glDisable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE);
    glDisable(GL_DEPTH_TEST);
    glDisable(GL_LINE_SMOOTH);
    glLineWidth(1.0f);
    glDisable(GL_CULL_FACE);
    glPolygonMode(GL_BACK, GL_FILL);
}


/*
 * view::View3D::renderBBoxFrontside
 */
void view::View3D::renderBBoxFrontside(void) {
    ::glDisable(GL_LIGHTING);
    ::glEnable(GL_BLEND);
    ::glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    ::glEnable(GL_DEPTH_TEST);
    ::glDepthFunc(GL_LEQUAL);
    ::glEnable(GL_CULL_FACE);
    ::glCullFace(GL_BACK);
    ::glEnable(GL_LINE_SMOOTH);
    ::glLineWidth(1.75f);
    //::glDisable(GL_TEXTURE_2D);
    ::glPolygonMode(GL_FRONT, GL_LINE);

    ::glColor4fv(this->bboxCol);
    this->renderBBox();

    ::glDepthFunc(GL_LESS);
    ::glPolygonMode(GL_FRONT, GL_FILL);

    glEnable(GL_LIGHTING);
    glDisable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE);
    glDisable(GL_DEPTH_TEST);
    glDisable(GL_LINE_SMOOTH);
    glLineWidth(1.0f);
    glDisable(GL_CULL_FACE);
}


/*
 * view::View3D::renderLookAt
 */
void view::View3D::renderLookAt(void) {
    const vislib::math::Cuboid<float>& boundingBox = this->bboxs.WorldSpaceBBox();
    vislib::math::Point<float, 3> minp(vislib::math::Min(boundingBox.Left(), boundingBox.Right()),
        vislib::math::Min(boundingBox.Bottom(), boundingBox.Top()),
        vislib::math::Min(boundingBox.Back(), boundingBox.Front()));
    vislib::math::Point<float, 3> maxp(vislib::math::Max(boundingBox.Left(), boundingBox.Right()),
        vislib::math::Max(boundingBox.Bottom(), boundingBox.Top()),
        vislib::math::Max(boundingBox.Back(), boundingBox.Front()));
    vislib::math::Point<float, 3> lap = this->cam.Parameters()->LookAt();
    bool xin = true;
    if (lap.X() < minp.X()) {
        lap.SetX(minp.X());
        xin = false;
    } else if (lap.X() > maxp.X()) {
        lap.SetX(maxp.X());
        xin = false;
    }
    bool yin = true;
    if (lap.Y() < minp.Y()) {
        lap.SetY(minp.Y());
        yin = false;
    } else if (lap.Y() > maxp.Y()) {
        lap.SetY(maxp.Y());
        yin = false;
    }
    bool zin = true;
    if (lap.Z() < minp.Z()) {
        lap.SetZ(minp.Z());
        zin = false;
    } else if (lap.Z() > maxp.Z()) {
        lap.SetZ(maxp.Z());
        zin = false;
    }

    ::glDisable(GL_LIGHTING);
    ::glLineWidth(1.4f);
    ::glEnable(GL_BLEND);
    ::glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    ::glEnable(GL_DEPTH_TEST);

    ::glBegin(GL_LINES);

    ::glColor3ub(255, 0, 0);

    ::glVertex3f(lap.X(), lap.Y(), lap.Z());
    ::glVertex3f(maxp.X(), lap.Y(), lap.Z());
    ::glColor3ub(192, 192, 192);
    ::glVertex3f(lap.X(), lap.Y(), lap.Z());
    ::glVertex3f(minp.X(), lap.Y(), lap.Z());

    ::glColor3ub(0, 255, 0);
    ::glVertex3f(lap.X(), lap.Y(), lap.Z());
    ::glVertex3f(lap.X(), maxp.Y(), lap.Z());
    ::glColor3ub(192, 192, 192);
    ::glVertex3f(lap.X(), lap.Y(), lap.Z());
    ::glVertex3f(lap.X(), minp.Y(), lap.Z());

    ::glColor3ub(0, 0, 255);
    ::glVertex3f(lap.X(), lap.Y(), lap.Z());
    ::glVertex3f(lap.X(), lap.Y(), maxp.Z());
    ::glColor3ub(192, 192, 192);
    ::glVertex3f(lap.X(), lap.Y(), lap.Z());
    ::glVertex3f(lap.X(), lap.Y(), minp.Z());

    ::glEnd();

    ::glEnable(GL_LIGHTING);
}


/*
 * view::View3D::renderSoftCursor
 */
void view::View3D::renderSoftCursor(void) {
    ::glMatrixMode(GL_PROJECTION);
    ::glLoadIdentity();
    ::glMatrixMode(GL_MODELVIEW);
    ::glLoadIdentity();

    vislib::SmartPtr<vislib::graphics::CameraParameters> params = this->cam.Parameters();

    const float cursorScale = 1.0f;

    ::glTranslatef(-1.0f, -1.0f, 0.0f);
    ::glScalef(2.0f / params->TileRect().Width(), 2.0f / params->TileRect().Height(), 1.0f);
    ::glTranslatef(-params->TileRect().Left(), -params->TileRect().Bottom(), 0.0f);
    ::glScalef(params->VirtualViewSize().Width() / this->camParams->VirtualViewSize().Width(),
        params->VirtualViewSize().Height() / this->camParams->VirtualViewSize().Height(), 1.0f);
    ::glTranslatef(this->cursor2d.X(), this->cursor2d.Y(), 0.0f);
    ::glScalef(cursorScale * this->camParams->VirtualViewSize().Width() / params->VirtualViewSize().Width(),
        -cursorScale * this->camParams->VirtualViewSize().Height() / params->VirtualViewSize().Height(), 1.0f);

    ::glEnable(GL_BLEND);
    ::glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    ::glEnable(GL_LINE_SMOOTH);
    ::glLineWidth(1.5f);
    ::glBindTexture(GL_TEXTURE_2D, 0);
    ::glDisable(GL_TEXTURE_2D);
    ::glDisable(GL_LIGHTING);
    ::glDisable(GL_DEPTH_TEST);

    ::glBegin(GL_TRIANGLE_FAN);
    ::glColor3ub(255, 255, 255);
    ::glVertex2i(0, 0);
    ::glColor3ub(245, 245, 245);
    ::glVertex2i(0, 17);
    ::glColor3ub(238, 238, 238);
    ::glVertex2i(4, 13);
    ::glColor3ub(211, 211, 211);
    ::glVertex2i(7, 18);
    ::glVertex2i(9, 18);
    ::glVertex2i(9, 16);
    ::glColor3ub(234, 234, 234);
    ::glVertex2i(7, 12);
    ::glColor3ub(226, 226, 226);
    ::glVertex2i(12, 12);
    ::glEnd();
    ::glBegin(GL_LINE_LOOP);
    ::glColor3ub(0, 0, 0);
    ::glVertex2i(0, 0);
    ::glVertex2i(0, 17);
    ::glVertex2i(4, 13);
    ::glVertex2i(7, 18);
    ::glVertex2i(9, 18);
    ::glVertex2i(9, 16);
    ::glVertex2i(7, 12);
    ::glVertex2i(12, 12);
    ::glEnd();

    ::glDisable(GL_LINE_SMOOTH);
    ::glDisable(GL_BLEND);
}


/*
 * view::View3D::OnGetCamParams
 */
bool view::View3D::OnGetCamParams(CallCamParamSync& c) {
    c.SetCamParams(this->cam.Parameters());
    return true;
}


/*
 * view::View3D::onStoreCamera
 */
bool view::View3D::onStoreCamera(param::ParamSlot& p) {
    vislib::TStringSerialiser strser;
    strser.ClearData();
    this->camParams->Serialise(strser);
    vislib::TString str(strser.GetString());
    str.EscapeCharacters(_T('\\'), _T("\n\r\t"), _T("nrt"));
    str.Append(_T("}"));
    str.Prepend(_T("{"));
    this->cameraSettingsSlot.Param<param::StringParam>()->SetValue(str);

    vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_INFO, "Camera parameters stored in \"%s\"",
        this->cameraSettingsSlot.FullName().PeekBuffer());
    return true;
}


/*
 * view::View3D::onRestoreCamera
 */
bool view::View3D::onRestoreCamera(param::ParamSlot& p) {
    vislib::TString str = this->cameraSettingsSlot.Param<param::StringParam>()->Value();
    try {
        if ((str[0] != _T('{')) || (str[str.Length() - 1] != _T('}'))) {
            throw vislib::Exception("invalid string: surrounding brackets missing", __FILE__, __LINE__);
        }
        str = str.Substring(1, str.Length() - 2);
        if (!str.UnescapeCharacters(_T('\\'), _T("\n\r\t"), _T("nrt"))) {
            throw vislib::Exception("unrecognised escape sequence", __FILE__, __LINE__);
        }
        vislib::TStringSerialiser strser(str);
        vislib::graphics::CameraParamsStore cps;
        cps = *this->camParams.operator->();
        cps.Deserialise(strser);
        cps.SetVirtualViewSize(this->camParams->VirtualViewSize());
        cps.SetTileRect(this->camParams->TileRect());
        *this->camParams.operator->() = cps;
        // now avoid resetting the camera by the initialisation
        if (!this->bboxs.IsWorldSpaceBBoxValid()) {
            this->bboxs.SetWorldSpaceBBox(-1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f);
        }

        vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_INFO, "Camera parameters restored from \"%s\"",
            this->cameraSettingsSlot.FullName().PeekBuffer());
    } catch (vislib::Exception ex) {
        vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
            "Cannot restore camera parameters from \"%s\": %s (%s; %d)",
            this->cameraSettingsSlot.FullName().PeekBuffer(), ex.GetMsgA(), ex.GetFile(), ex.GetLine());
    } catch (...) {
        vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
            "Cannot restore camera parameters from \"%s\": unexpected exception",
            this->cameraSettingsSlot.FullName().PeekBuffer());
    }
    return true;
}


/*
 * view::View3D::onResetView
 */
bool view::View3D::onResetView(param::ParamSlot& p) {
    this->ResetView();
    return true;
}


/*
 * view::View3D::onToggleButton
 */
bool view::View3D::onToggleButton(param::ParamSlot& p) {
    param::BoolParam* bp = NULL;

    if (&p == &this->toggleSoftCursorSlot) {
        this->toggleSoftCurse();
        return true;
    } else if (&p == &this->toggleBBoxSlot) {
        bp = this->showBBox.Param<param::BoolParam>();
    }

    if (bp != NULL) {
        bp->SetValue(!bp->Value());
    }
    return true;
}


/*
 * view::View3D::renderViewCube
 */
void view::View3D::renderViewCube(void) {
    ::glEnable(GL_BLEND);
    ::glBlendFunc(GL_SRC_ALPHA, GL_ONE);
    ::glDisable(GL_LIGHTING);
    //::glDisable(GL_TEXTURE_2D);
    ::glEnable(GL_LINE_SMOOTH);
    ::glLineWidth(1.5f);
    ::glDisable(GL_DEPTH_TEST);

    float pmf[16];
    float vmf[16];
    double pm[16];
    double vm[16];
    int vp[4], tvp[4];
    this->cam.ProjectionMatrix(pmf);
    this->cam.ViewMatrix(vmf);
    for (unsigned int i = 0; i < 16; i++) {
        pm[i] = static_cast<double>(pmf[i]);
        vm[i] = static_cast<double>(vmf[i]);
    }
    const float viewportScale = 100.0f;
    vp[0] = vp[1] = 0;
    vp[2] = static_cast<int>(this->cam.Parameters()->VirtualViewSize()[0] * viewportScale);
    vp[3] = static_cast<int>(this->cam.Parameters()->VirtualViewSize()[1] * viewportScale);
    tvp[0] = static_cast<int>(this->cam.Parameters()->TileRect().Left() * viewportScale);
    tvp[1] = static_cast<int>(this->cam.Parameters()->TileRect().Bottom() * viewportScale);
    tvp[2] = static_cast<int>(this->cam.Parameters()->TileRect().Width() * viewportScale);
    tvp[3] = static_cast<int>(this->cam.Parameters()->TileRect().Height() * viewportScale);

    double wx, wy, wz, sx1, sy1, sz1, sx2, sy2, sz2;
    double size = vislib::math::Min(static_cast<double>(vp[2]), static_cast<double>(vp[3])) * 0.1f;
    wx = static_cast<double>(vp[2]) - size;
    wy = static_cast<double>(vp[3]) - size;
    wz = 0.5;
    ::gluUnProject(wx, wy, wz, vm, pm, tvp, &sx1, &sy1, &sz1);
    size *= 0.5;
    wx = static_cast<double>(vp[2]) - size;
    wy = static_cast<double>(vp[3]) - size;
    wz = 0.5;
    ::gluUnProject(wx, wy, wz, vm, pm, tvp, &sx2, &sy2, &sz2);
    sx2 -= sx1;
    sy2 -= sy1;
    sz2 -= sz1;
    size = vislib::math::Sqrt(sx2 * sx2 + sy2 * sy2 + sz2 * sz2);
    size *= 0.5;

    ::glTranslated(sx1, sy1, sz1);
    ::glScaled(size, size, size);

    ::glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
    ::glEnable(GL_CULL_FACE);
    ::glCullFace(GL_FRONT);

    ::glBegin(GL_QUADS);
    ::glColor4ub(255, 0, 0, 32);
    ::glVertex3i(1, 1, 1);
    ::glVertex3i(1, -1, 1);
    ::glVertex3i(1, -1, -1);
    ::glVertex3i(1, 1, -1);
    ::glColor4ub(255, 255, 255, 32);
    ::glVertex3i(-1, 1, 1);
    ::glVertex3i(-1, 1, -1);
    ::glVertex3i(-1, -1, -1);
    ::glVertex3i(-1, -1, 1);
    ::glColor4ub(0, 255, 0, 32);
    ::glVertex3i(1, 1, 1);
    ::glVertex3i(1, 1, -1);
    ::glVertex3i(-1, 1, -1);
    ::glVertex3i(-1, 1, 1);
    ::glColor4ub(255, 255, 255, 32);
    ::glVertex3i(1, -1, 1);
    ::glVertex3i(-1, -1, 1);
    ::glVertex3i(-1, -1, -1);
    ::glVertex3i(1, -1, -1);
    ::glColor4ub(0, 0, 255, 32);
    ::glVertex3i(1, 1, 1);
    ::glVertex3i(-1, 1, 1);
    ::glVertex3i(-1, -1, 1);
    ::glVertex3i(1, -1, 1);
    ::glColor4ub(255, 255, 255, 32);
    ::glVertex3i(1, 1, -1);
    ::glVertex3i(1, -1, -1);
    ::glVertex3i(-1, -1, -1);
    ::glVertex3i(-1, 1, -1);
    ::glEnd();

    ::glCullFace(GL_BACK);
    ::glBegin(GL_QUADS);
    ::glColor4ub(255, 0, 0, 127);
    ::glVertex3i(1, 1, 1);
    ::glVertex3i(1, -1, 1);
    ::glVertex3i(1, -1, -1);
    ::glVertex3i(1, 1, -1);
    ::glColor4ub(255, 255, 255, 127);
    ::glVertex3i(-1, 1, 1);
    ::glVertex3i(-1, 1, -1);
    ::glVertex3i(-1, -1, -1);
    ::glVertex3i(-1, -1, 1);
    ::glColor4ub(0, 255, 0, 127);
    ::glVertex3i(1, 1, 1);
    ::glVertex3i(1, 1, -1);
    ::glVertex3i(-1, 1, -1);
    ::glVertex3i(-1, 1, 1);
    ::glColor4ub(255, 255, 255, 127);
    ::glVertex3i(1, -1, 1);
    ::glVertex3i(-1, -1, 1);
    ::glVertex3i(-1, -1, -1);
    ::glVertex3i(1, -1, -1);
    ::glColor4ub(0, 0, 255, 127);
    ::glVertex3i(1, 1, 1);
    ::glVertex3i(-1, 1, 1);
    ::glVertex3i(-1, -1, 1);
    ::glVertex3i(1, -1, 1);
    ::glColor4ub(255, 255, 255, 127);
    ::glVertex3i(1, 1, -1);
    ::glVertex3i(1, -1, -1);
    ::glVertex3i(-1, -1, -1);
    ::glVertex3i(-1, 1, -1);
    ::glEnd();

    ::glCullFace(GL_BACK);
    ::glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

    ::glBegin(GL_LINES);
    ::glColor4ub(192, 192, 192, 192);
    ::glVertex3i(0, 0, 0);
    ::glVertex3i(-1, 0, 0);
    ::glVertex3i(0, 0, 0);
    ::glVertex3i(0, -1, 0);
    ::glVertex3i(0, 0, 0);
    ::glVertex3i(0, 0, -1);
    ::glColor4ub(255, 0, 0, 192);
    ::glVertex3i(0, 0, 0);
    ::glVertex3i(1, 0, 0);
    ::glColor4ub(0, 255, 0, 192);
    ::glVertex3i(0, 0, 0);
    ::glVertex3i(0, 1, 0);
    ::glColor4ub(0, 0, 255, 192);
    ::glVertex3i(0, 0, 0);
    ::glVertex3i(0, 0, 1);
    ::glEnd();

    glDisable(GL_BLEND);
    glEnable(GL_LIGHTING);
    glDisable(GL_LINE_SMOOTH);
    glLineWidth(1.0f);
    glDisable(GL_CULL_FACE);
}
