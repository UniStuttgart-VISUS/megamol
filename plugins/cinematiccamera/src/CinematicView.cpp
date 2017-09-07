/**
* CinematicView.cpp
*/

#include "stdafx.h"

#include "mmcore/param/BoolParam.h"
#include "mmcore/param/EnumParam.h"
#include "mmcore/param/IntParam.h"
#include "mmcore/param/ButtonParam.h"
#include "mmcore/view/CallRender3D.h"
#include "mmcore/param/FloatParam.h"
#include "mmcore/view/CallRenderView.h"

#include "vislib/math/Rectangle.h"
#include "vislib/Trace.h"

#include "CinematicView.h"
#include "CallCinematicCamera.h"


using namespace megamol;
using namespace megamol::core;
using namespace cinematiccamera;


/*
* CinematicView::CinematicView
*/
CinematicView::CinematicView(void) : View3D(),
	keyframeKeeperSlot("keyframeKeeper", "Connects to the Keyframe Keeper."),
	selectedSkyboxSideParam(  "Cinematic::01 Skybox side", "Select the skybox side rendering"),
    cinematicHeightParam("Cinematic::02 Cinematic width","Set resolution of cineamtic view in hotzontal direction."),
    cinematicWidthParam("Cinematic::03 Cinematic height", "Set resolution of cineamtic view in vertical direction"),
    shownKeyframe()
    {

    this->keyframeKeeperSlot.SetCompatibleCall<CallCinematicCameraDescription>();
    this->MakeSlotAvailable(&this->keyframeKeeperSlot);

	param::EnumParam *sbs = new param::EnumParam(SKYBOX_NONE);
	sbs->SetTypePair(SKYBOX_NONE,   "None");
	sbs->SetTypePair(SKYBOX_FRONT,	"Front");
	sbs->SetTypePair(SKYBOX_BACK,	"Back");
	sbs->SetTypePair(SKYBOX_LEFT,	"Left");
	sbs->SetTypePair(SKYBOX_RIGHT,	"Right");
	sbs->SetTypePair(SKYBOX_UP,		"Up");
	sbs->SetTypePair(SKYBOX_DOWN,	"Down");
	this->selectedSkyboxSideParam << sbs;
	this->MakeSlotAvailable(&this->selectedSkyboxSideParam);

    // init variables
    this->currentViewTime = 0.0f;     
    this->maxAnimTime     = 1.0f;
    this->bboxCenter      = vislib::math::Point<float, 3>(0.0f, 0.0f, 0.0f);
    this->cineWidth       = 1920;
    this->cineHeight      = 1080;

    // init parameters
    this->cinematicHeightParam.SetParameter(new param::IntParam(this->cineHeight, 1));
    this->MakeSlotAvailable(&this->cinematicHeightParam);

    this->cinematicWidthParam.SetParameter(new param::IntParam(this->cineWidth, 1));
    this->MakeSlotAvailable(&this->cinematicWidthParam);

    // TEMPORARY HACK #########################################################
    // Disable parameter slot -> 'TAB'-key is needed in cinematic renderer to enable mouse selection
    this->enableMouseSelectionSlot.MakeUnavailable();
}


/*
* CinematicView::~CinematicView
*/
CinematicView::~CinematicView(void) {

    //this->fbo.Disable();
    //this->fbo.Release();
}


/*
* CinematicView::Render
*/
void CinematicView::Render(const mmcRenderViewContext& context) {

    view::CallRender3D *cr3d = this->rendererSlot.CallAs<core::view::CallRender3D>();
    if (!(*cr3d)(1)) return; // get extents

    CallCinematicCamera *ccc = this->keyframeKeeperSlot.CallAs<CallCinematicCamera>();
    if (!ccc) return;
    // Updated data from cinematic camera call
    if (!(*ccc)(CallCinematicCamera::CallForGetUpdatedKeyframeData)) return;
    
    // Update paramters
    if (this->cinematicHeightParam.IsDirty()) {
        this->cinematicHeightParam.ResetDirty();
        this->cineWidth = this->cinematicHeightParam.Param<param::IntParam>()->Value();
    }
    if (this->cinematicWidthParam.IsDirty()) {
        this->cinematicWidthParam.ResetDirty();
        this->cineHeight = this->cinematicWidthParam.Param<param::IntParam>()->Value();
    }

    // Check for new max anim time
    this->maxAnimTime = static_cast<float>(cr3d->TimeFramesCount());
    if(this->maxAnimTime != ccc->getMaxAnimTime()){
        ccc->setMaxAnimTime(this->maxAnimTime);
        if (!(*ccc)(CallCinematicCamera::CallForSetAnimationData)) return;
    }

    // Check for new bounding box center
    this->bboxCenter = cr3d->AccessBoundingBoxes().WorldSpaceBBox().CalcCenter();
    if (this->bboxCenter != ccc->getBboxCenter()) {
        ccc->setBboxCenter(this->bboxCenter);
        if (!(*ccc)(CallCinematicCamera::CallForSetAnimationData)) return;
    }

    // Time is set by running ANIMATION from view (e.g. anim::play parameter)
    float animTime    = static_cast<float>(context.Time);
    float repeat      = floorf(this->currentViewTime / this->maxAnimTime) * this->maxAnimTime;
    if (this->currentViewTime != (animTime + repeat)) {
        // Select the keyframe based on the current animation time.
        if (this->currentViewTime < (animTime + repeat)) {
            this->currentViewTime = repeat + animTime;
        }
        else { // animTime < tmpAnimTime -> animation time restarts from 0
            this->currentViewTime = repeat + this->maxAnimTime + animTime;
        }

        // Reset view time and animation time to beginning if total time is reached
        if (this->currentViewTime > ccc->getTotalTime()) {
            param::ParamSlot* animTimeParam = static_cast<param::ParamSlot*>(this->timeCtrl.GetSlot(2));
            animTimeParam->Param<param::FloatParam>()->SetValue(0.0f, true);
            this->currentViewTime = 0.0f;
        }

        ccc->setSelectedKeyframeTime(this->currentViewTime);
        // Update selected keyframe
        if (!(*ccc)(CallCinematicCamera::CallForSetSelectedKeyframe)) return;
        // Updated call data 
        if (!(*ccc)(CallCinematicCamera::CallForGetUpdatedKeyframeData)) return;
    }
    else { // Time is set by SELECTED KEYFRAME
        // wrap time if total time exceeds animation time of data set
        float selectTime     = ccc->getSelectedKeyframe().getTime();
        float selectAnimTime = selectTime - (floorf(selectTime / this->maxAnimTime) * this->maxAnimTime);
        // Set animation time based on selected keyframe (GetSlot(2)= this->animTimeSlot)
        param::ParamSlot* animTimeParam = static_cast<param::ParamSlot*>(this->timeCtrl.GetSlot(2));
        animTimeParam->Param<param::FloatParam>()->SetValue(selectAnimTime, true);
        this->currentViewTime = selectTime;
    }

    // Set camera parameters of selected keyframe for this view
    // but ONLY if selected keyframe differs to last locally stored and shown keyframe
    Keyframe s = ccc->getSelectedKeyframe(); // maybe updated
    vislib::SmartPtr<vislib::graphics::CameraParameters> p = s.getCamParameters();
    // Every camera parameter has be compared separatly because euqality of cameras only checks pointer equality
    if ((this->shownKeyframe.getTime()             != s.getTime())   ||
        (this->shownKeyframe.getCamPosition()      != p->Position()) ||
        (this->shownKeyframe.getCamLookAt()        != p->LookAt())   ||
        (this->shownKeyframe.getCamUp()            != p->Up())       ||
        (this->shownKeyframe.getCamApertureAngle() != p->ApertureAngle())) {

        this->cam.Parameters()->SetView(p->Position(), p->LookAt(), p->Up());
        this->cam.Parameters()->SetApertureAngle(p->ApertureAngle());

        vislib::graphics::Camera c;
        if (!p.IsNull()) {
            c.Parameters()->SetPosition(p->Position());
            c.Parameters()->SetLookAt(p->LookAt());
            c.Parameters()->SetUp(p->Up());
            c.Parameters()->SetApertureAngle(p->ApertureAngle());
        }
        this->shownKeyframe.setTime(s.getTime());
        this->shownKeyframe.setCamera(c);
    }

    // Propagate camera parameter to keyframe keeper
    ccc->setCameraParameter(this->cam.Parameters());
    if (!(*ccc)(CallCinematicCamera::CallForSetCameraForKeyframe)) return;

    // Get camera parameters
    vislib::SmartPtr<vislib::graphics::CameraParameters> cp = this->camParams;
    vislib::math::Point<float, 3>  camPos      = cp->Position();
    vislib::math::Vector<float, 3> camRight    = cp->Right();
    vislib::math::Vector<float, 3> camUp       = cp->Up();
    vislib::math::Vector<float, 3> camFront    = cp->Front();
    vislib::math::Point<float, 3>  camLookAt   = cp->LookAt();
    float                          camAperture = cp->ApertureAngle();
    float                          tmpDist     = cp->FocalDistance();

    // Adjust cam to selected skybox side
    SkyboxSides side = static_cast<SkyboxSides>(this->selectedSkyboxSideParam.Param<param::EnumParam>()->Value());
    if (side != SKYBOX_NONE) {
        // set aperture angle to 90 deg
        cp->SetApertureAngle(90.0f);
        if (side == SKYBOX_BACK) {
            cp->SetView(camPos, camPos - camFront * tmpDist, camUp);
        }
        else if (side == SKYBOX_RIGHT) {
            cp->SetView(camPos, camPos + camRight * tmpDist, camUp);
        }
        else if (side == SKYBOX_LEFT) {
            cp->SetView(camPos, camPos - camRight * tmpDist, camUp);
        }
        else if (side == SKYBOX_UP) {
            cp->SetView(camPos, camPos + camUp * tmpDist, -camFront);
        }
        else if (side == SKYBOX_DOWN) {
            cp->SetView(camPos, camPos - camUp * tmpDist, camFront);
        }
    }   

    // Viewport stuff ---------------------------------------------------------
    int vp[4];
    glGetIntegerv(GL_VIEWPORT, vp);
    int  vpWidth   = vp[2] - vp[0]; // or  cr3d->GetViewport().GetSize().GetHeight();
    int  vpHeight   = vp[3] - vp[1]; // or  cr3d->GetViewport().GetSize().GetWidth();
    float vpRatio   = static_cast<float>(vpWidth) / static_cast<float>(vpHeight);
    float cineRatio = static_cast<float>(this->cineWidth) / static_cast<float>(this->cineHeight);
    float fboWidth  = vpWidth;
    float fboHeight = vpHeight;
    if (cineRatio > vpRatio) {
        fboHeight = (static_cast<int>(static_cast<float>(vpWidth) / cineRatio));
    }
    else if (cineRatio < vpRatio) {
        fboWidth = (static_cast<int>(static_cast<float>(vpHeight) * cineRatio));
    }

    // Render to texture ------------------------------------------------------------
    // Suppress TRACE output of fbo.Enable() and >fbo.Create()
#if defined(DEBUG) || defined(_DEBUG)
    unsigned int otl = vislib::Trace::GetInstance().GetLevel();
    vislib::Trace::GetInstance().SetLevel(0);
#endif /* DEBUG || _DEBUG */
    if (!this->fbo.IsValid()) {
        if (!this->fbo.Create(fboWidth, fboHeight, GL_RGBA32F, GL_RGBA, GL_UNSIGNED_BYTE, vislib::graphics::gl::FramebufferObject::ATTACHMENT_TEXTURE, GL_DEPTH_COMPONENT24)) {
            throw vislib::Exception("[CINEMATIC VIEW] Unable to create image framebuffer object.", __FILE__, __LINE__);
            return;
        }
    }
    if (this->fbo.IsValid()) {
        if (this->fbo.Enable() != GL_NO_ERROR) {
            throw vislib::Exception("[CINEMATIC VIEW] Cannot enable Framebuffer object.", __FILE__, __LINE__);
            return;
        }
    }
    else {
        throw vislib::Exception("[CINEMATIC VIEW] Framebuffer object is not valid.", __FILE__, __LINE__);
        return;
    }
    // Reset TRACE output level
#if defined(DEBUG) || defined(_DEBUG)
    vislib::Trace::GetInstance().SetLevel(otl);
#endif /* DEBUG || _DEBUG */

    // Set override viewport of view (otherwise viewport is overwritten in Base::Render(context))
    int fboVp[4] = { 0, 0, fboWidth, fboHeight };
    this->overrideViewport = fboVp;
    // Set new viewport settings for camera
    this->camParams->SetVirtualViewSize(static_cast<vislib::graphics::ImageSpaceType>(fboWidth), static_cast<vislib::graphics::ImageSpaceType>(fboHeight));
    this->camParams->SetTileRect(vislib::math::Rectangle<float>(0.0f, 0.0f, static_cast<float>(fboWidth), static_cast<float>(fboHeight)));

    // Set output buffer for override call (otherwise render call is overwritten in Base::Render(context))
    GLenum callOutBuffer = cr3d->OutputBuffer();
    cr3d->SetOutputBuffer(this->fbo.GetID());
    this->overrideCall = cr3d;

    // Call Render-Function of parent View3D
    Base::Render(context);

    // Reset override render call
    cr3d->SetOutputBuffer(callOutBuffer);
    this->overrideCall = NULL;
    // Reset override viewport
    this->overrideViewport = NULL;

    this->fbo.Disable();

    // Draw final image -------------------------------------------------------
    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glLoadIdentity();
    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glLoadIdentity();
    // Reset viewport
    glViewport(vp[0], vp[1], vp[2], vp[3]);
    glOrtho(0.0f, static_cast<float>(vpWidth), 0.0f, static_cast<float>(vpHeight), -1.0, 1.0);

    //glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
    glDisable(GL_LIGHTING);
    glDisable(GL_DEPTH_TEST);
    glDisable(GL_BLEND);
    glDisable(GL_CULL_FACE);

    // Draw texture
    float wHalfVp    = static_cast<float>(vpWidth) / 2.0f;
    float hHalfVp    = static_cast<float>(vpHeight) / 2.0f;
    float wHalfFbo   = static_cast<float>(fboWidth) / 2.0f;
    float hHalfFbo   = static_cast<float>(fboHeight) / 2.0f;
    glActiveTexture(GL_TEXTURE0);
    glEnable(GL_TEXTURE_2D);
    this->fbo.BindColourTexture();
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glColor4f(1.0f, 1.0f, 1.0f, 1.0f);
    glBegin(GL_QUADS);
        glTexCoord2f(0.0f, 0.0f); glVertex2f(wHalfVp - wHalfFbo, hHalfVp - hHalfFbo);
        glTexCoord2f(1.0f, 0.0f); glVertex2f(wHalfVp + wHalfFbo, hHalfVp - hHalfFbo);
        glTexCoord2f(1.0f, 1.0f); glVertex2f(wHalfVp + wHalfFbo, hHalfVp + hHalfFbo);
        glTexCoord2f(0.0f, 1.0f); glVertex2f(wHalfVp - wHalfFbo, hHalfVp + hHalfFbo);
    glEnd();
    glBindTexture(GL_TEXTURE_2D, 0);
    glDisable(GL_TEXTURE_2D);

    // Draw letter box quads in fgColor
    // Get the foreground color (inverse background color)
    float bgColor[4];
    float fgColor[4] = { 1.0f, 1.0f, 1.0f, 1.0f };
    glGetFloatv(GL_COLOR_CLEAR_VALUE, bgColor);
    for (unsigned int i = 0; i < 4; i++) {
        fgColor[i] -= bgColor[i];
    }
    int x = 0;
    int y = 0;
    if (fboWidth < vpWidth) {
        x = (static_cast<int>(static_cast<float>(vpWidth - fboWidth) / 2.0f));
        y = vpHeight;
    }
    else if (fboHeight < vpHeight) {
        x = vpWidth;
        y = (static_cast<int>(static_cast<float>(vpHeight - fboHeight) / 2.0f));
    }
    glColor3fv(fgColor);
    glBegin(GL_QUADS);
        glVertex2i(0, 0);
        glVertex2i(x, 0);
        glVertex2i(x, y);
        glVertex2i(0, y);
        glVertex2i(vpWidth,     vpHeight);
        glVertex2i(vpWidth - x, vpHeight);
        glVertex2i(vpWidth - x, vpHeight - y);
        glVertex2i(vpWidth,     vpHeight - y);
    glEnd();

    glMatrixMode(GL_PROJECTION);
    glPopMatrix();
    glMatrixMode(GL_MODELVIEW);
    glPopMatrix();

    this->fbo.Release();
}