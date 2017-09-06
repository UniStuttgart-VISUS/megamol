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

    this->cineWidth    = -1.0f;
    this->cineHeight   = -1.0f;
    this->vpWidth      = -1.0f;
    this->vpHeight     = -1.0f;
    this->fboWidth     = -1.0f;
    this->fboHeight    = -1.0f;

    // init parameters
    this->cinematicHeightParam.SetParameter(new param::IntParam(1, 1));
    this->MakeSlotAvailable(&this->cinematicHeightParam);

    this->cinematicWidthParam.SetParameter(new param::IntParam(1, 1));
    this->MakeSlotAvailable(&this->cinematicWidthParam);

    // TEMPORARY HACK #########################################################
    // Disable parameter slot -> 'TAB'-key is needed in cinematic renderer to enable mouse selection
    this->enableMouseSelectionSlot.MakeUnavailable();
}


/*
* CinematicView::~CinematicView
*/
CinematicView::~CinematicView(void) {

    glDeleteTextures(1, &this->colorBuffer);
    glDeleteTextures(1, &this->depthBuffer);
    glDeleteFramebuffers(1, &this->frameBuffer);

}


/*
* CinematicView::setupRenderToTexture
*/
void CinematicView::setupRenderToTexture() {

    if (!areExtsAvailable("GL_EXT_framebuffer_object GL_ARB_draw_buffers"))
        return;

    // Delete textures + fbo if necessary
    if (glIsFramebuffer(this->frameBuffer)) {
        glDeleteTextures(1, &this->colorBuffer);
        glDeleteTextures(1, &this->depthBuffer);
        glDeleteFramebuffers(1, &this->frameBuffer);
    }

    float vpRatio    = static_cast<float>(this->vpWidth)   / static_cast<float>(this->vpHeight);
    float cineRatio  = static_cast<float>(this->cineWidth) / static_cast<float>(this->cineHeight);
    this->fboWidth   = this->vpWidth;
    this->fboHeight  = this->vpHeight;
    if (cineRatio > vpRatio) {
        this->fboHeight = (static_cast<int>(static_cast<float>(this->vpWidth) / cineRatio));
    }
    else if (cineRatio < vpRatio) {
        this->fboWidth = (static_cast<int>(static_cast<float>(this->vpHeight) * cineRatio));
    }

    // Create color texture
    glGenTextures(1, &this->colorBuffer);
    glBindTexture(GL_TEXTURE_2D, this->colorBuffer);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB32F, this->fboWidth, this->fboHeight, 0, GL_RGBA, GL_UNSIGNED_BYTE, 0);
    glBindTexture(GL_TEXTURE_2D, 0);

    // Create depth texture
    glGenTextures(1, &this->depthBuffer);
    glBindTexture(GL_TEXTURE_2D, this->depthBuffer);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT16, this->fboWidth, this->fboHeight, 0, GL_DEPTH_COMPONENT, GL_UNSIGNED_BYTE, 0);
    glBindTexture(GL_TEXTURE_2D, 0);

    // Create FramebufferObject
    glGenFramebuffersEXT(1, &this->frameBuffer);
    glBindFramebufferEXT(GL_FRAMEBUFFER, this->frameBuffer);
    glFramebufferTexture2DEXT(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, this->colorBuffer, 0);
    glFramebufferTexture2DEXT(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, this->depthBuffer, 0);

    // Check status of fbo
    if (glCheckFramebufferStatusEXT(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
        sys::Log::DefaultLog.WriteError("[CINEMATIC VIEW] [setupRenderToTexture] Could not create framebuffer object.");
    }
    else {
        sys::Log::DefaultLog.WriteInfo("[CINEMATIC VIEW] [setupRenderToTexture] Creating framebuffer object was successful.");
    }

    glFramebufferTexture2DEXT(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, 0, 0);
    glFramebufferTexture2DEXT(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, 0, 0);
    glBindFramebufferEXT(GL_FRAMEBUFFER, 0);
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

    // get viewport
    int vp[4];
    glGetIntegerv(GL_VIEWPORT, vp);
    int  vpW = vp[2] - vp[0];
    int  vpH = vp[3] - vp[1];

    bool resetFramebuffer = false;
    if ((this->vpHeight != vpH) || (this->vpWidth != vpW)) {
        this->vpHeight = vpH;
        this->vpWidth  = vpW;
        resetFramebuffer = true;
    }
    if ((this->cineHeight < 0) || (this->cineWidth < 0)) { // first time
        this->cineWidth  = vpWidth;
        this->cineHeight = vpHeight;
        this->cinematicHeightParam.Param<param::IntParam>()->SetValue(this->cineWidth, false);
        this->cinematicWidthParam.Param<param::IntParam>()->SetValue(this->cineHeight, false);
        resetFramebuffer = true;
    }
    // Update paramters
    if (this->cinematicHeightParam.IsDirty()) {
        this->cinematicHeightParam.ResetDirty();
        this->cineWidth = this->cinematicHeightParam.Param<param::IntParam>()->Value();
        resetFramebuffer = true;
    }
    if (this->cinematicWidthParam.IsDirty()) {
        this->cinematicWidthParam.ResetDirty();
        this->cineHeight = this->cinematicWidthParam.Param<param::IntParam>()->Value();
        resetFramebuffer = true;
    }
    // Update framebuffer size
    if (resetFramebuffer) {
        this->setupRenderToTexture();
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

    // ------------------------------------------------------------------------
    // Render to texture

    //cr3d->DisableOutputBuffer();
    if (cr3d->FrameBufferObject() != NULL)
        cr3d->FrameBufferObject()->Disable();


    glBindFramebufferEXT(GL_FRAMEBUFFER, 0);

    glBindFramebufferEXT(GL_FRAMEBUFFER, this->frameBuffer);
    GLenum DrawBuffers[1] = { GL_COLOR_ATTACHMENT0 };
    glDrawBuffers(1, DrawBuffers);
    glFramebufferTexture2DEXT(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, this->colorBuffer, 0);
    glFramebufferTexture2DEXT(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, this->depthBuffer, 0);

    // Set override viewport of view (otherwise viewport is overwritten in Base::Render(context))
    int fboVp[4] = { 0, 0, this->fboWidth, this->fboHeight };
    this->overrideViewport = fboVp;
    // Set new viewport settings for camera
    this->camParams->SetVirtualViewSize(static_cast<vislib::graphics::ImageSpaceType>(this->fboWidth), static_cast<vislib::graphics::ImageSpaceType>(this->fboHeight));
    this->camParams->SetTileRect(vislib::math::Rectangle<float>(0.0f, 0.0f, static_cast<float>(this->fboWidth), static_cast<float>(this->fboHeight)));

    // Call Render-Function of parent View3D
    Base::Render(context);

    this->overrideViewport = NULL;
    glFramebufferTexture2DEXT(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, 0, 0);
    glFramebufferTexture2DEXT(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, 0, 0);
    glBindFramebufferEXT(GL_FRAMEBUFFER, 0);

    //cr3d->EnableOutputBuffer();
    if (cr3d->FrameBufferObject() != NULL)
        cr3d->FrameBufferObject()->Enable();

    // ------------------------------------------------------------------------
    // Draw final image
    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glLoadIdentity();
    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glLoadIdentity();

    // Reset viewport
    glViewport(vp[0], vp[1], vp[2], vp[3]);
    glOrtho(0.0f, static_cast<float>(this->vpWidth), 0.0f, static_cast<float>(this->vpHeight), -1.0, 1.0);

    //glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
    glDisable(GL_LIGHTING);
    glDisable(GL_DEPTH_TEST);
    glDisable(GL_BLEND);
    glDisable(GL_CULL_FACE);

    // Get the foreground color (inverse background color)
    float bgColor[4];
    float fgColor[4] = { 1.0f, 1.0f, 1.0f, 1.0f };
    glGetFloatv(GL_COLOR_CLEAR_VALUE, bgColor);
    for (unsigned int i = 0; i < 4; i++) {
        fgColor[i] -= bgColor[i];
    }

    // Draw white background quad
    glColor3f(1.0f, 1.0f, 1.0f);
    glBegin(GL_QUADS);
        glVertex2i(0, 0);
        glVertex2i(this->vpWidth, 0);
        glVertex2i(this->vpWidth, this->vpHeight);
        glVertex2i(0, this->vpHeight);
    glEnd();

    // Draw texture
    float wHalfVp    = static_cast<float>(this->vpWidth) / 2.0f;
    float hHalfVp    = static_cast<float>(this->vpHeight) / 2.0f;
    float wHalfFbo   = static_cast<float>(this->fboWidth) / 2.0f;
    float hHalfFbo   = static_cast<float>(this->fboHeight) / 2.0f;
    glActiveTexture(GL_TEXTURE0);
    glEnable(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D, this->colorBuffer);
    glBegin(GL_QUADS);
        glTexCoord2f(0.0f, 0.0f); glVertex2f(wHalfVp - wHalfFbo, hHalfVp - hHalfFbo);
        glTexCoord2f(1.0f, 0.0f); glVertex2f(wHalfVp + wHalfFbo, hHalfVp - hHalfFbo);
        glTexCoord2f(1.0f, 1.0f); glVertex2f(wHalfVp + wHalfFbo, hHalfVp + hHalfFbo);
        glTexCoord2f(0.0f, 1.0f); glVertex2f(wHalfVp - wHalfFbo, hHalfVp + hHalfFbo);
    glEnd();
    glBindTexture(GL_TEXTURE_2D, 0);
    glDisable(GL_TEXTURE_2D);

    // Draw letter box quads
    int x = 0;
    int y = 0;
    if (this->fboWidth < this->vpWidth) {
        x = (static_cast<int>(static_cast<float>(this->vpWidth - this->fboWidth) / 2.0f));
        y = this->vpHeight;
    }
    else if (this->fboHeight < this->vpHeight) {
        x = this->vpWidth;
        y = (static_cast<int>(static_cast<float>(this->vpHeight - this->fboHeight) / 2.0f));
    }
    glColor3fv(fgColor);
    glBegin(GL_QUADS);
        glVertex2i(0, 0);
        glVertex2i(x, 0);
        glVertex2i(x, y);
        glVertex2i(0, y);
        glVertex2i(this->vpWidth,     this->vpHeight);
        glVertex2i(this->vpWidth - x, this->vpHeight);
        glVertex2i(this->vpWidth - x, this->vpHeight - y);
        glVertex2i(this->vpWidth,     this->vpHeight - y);
    glEnd();


    glMatrixMode(GL_PROJECTION);
    glPopMatrix();
    glMatrixMode(GL_MODELVIEW);
    glPopMatrix();

}