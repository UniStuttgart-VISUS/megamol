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
    cinematicResolutionXParam("Cinematic::02 Cinematic resolution X","Set resolution of cineamtic view in hotzontal direction."),
    cinematicResolutionYParam("Cinematic::03 Cinematic resolution Y", "Set resolution of cineamtic view in vertical direction"),
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
    this->cineXRes        = 1920;
    this->cineYRes        = 1080;
    this->maxAnimTime     = 1.0f;
    this->bboxCenter      = vislib::math::Point<float, 3>(0.0f, 0.0f, 0.0f);

    // init parameters
    this->cinematicResolutionXParam.SetParameter(new param::IntParam(this->cineXRes, 1));
    this->MakeSlotAvailable(&this->cinematicResolutionXParam);

    this->cinematicResolutionYParam.SetParameter(new param::IntParam(this->cineYRes, 1));
    this->MakeSlotAvailable(&this->cinematicResolutionYParam);


    // TEMPORARY HACK #########################################################
    // Disable parameter slot -> 'TAB'-key is needed in cinematic renderer to enable mouse selection
    this->enableMouseSelectionSlot.MakeUnavailable();

    this->setupRenderToTexture();
}


/*
* CinematicView::~CinematicView
*/
CinematicView::~CinematicView(void) {
    // intentionally empty
}


/*
* CinematicView::setupRenderToTexture
*/
void CinematicView::setupRenderToTexture() {











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
    if (this->cinematicResolutionXParam.IsDirty()) {
        this->cinematicResolutionXParam.ResetDirty();
        this->cineXRes = this->cinematicResolutionXParam.Param<param::IntParam>()->Value();
    }
    if (this->cinematicResolutionYParam.IsDirty()) {
        this->cinematicResolutionYParam.ResetDirty();
        this->cineYRes = this->cinematicResolutionYParam.Param<param::IntParam>()->Value();
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
    vislib::SmartPtr<vislib::graphics::CameraParameters> cp = this->cam.Parameters();
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

    /*
    GLint viewportStuff[4];
    glGetIntegerv(GL_VIEWPORT, viewportStuff);

    // Create color texture
    GLuint colBuf;
    glGenTextures(1, &colBuf);
    glBindTexture(GL_TEXTURE_2D, colBuf);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB32F, this->cineXRes, this->cineYRes, 0, GL_RGBA, GL_UNSIGNED_BYTE, 0);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glBindTexture(GL_TEXTURE_2D, 0);

    // Create depth texture
    GLuint depBuf;
    glGenTextures(1, &depBuf);
    glBindTexture(GL_TEXTURE_2D, depBuf);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT16, this->cineXRes, this->cineYRes, 0, GL_DEPTH_COMPONENT, GL_UNSIGNED_BYTE, 0);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glBindTexture(GL_TEXTURE_2D, 0);

    // Create FramebufferObject
    GLuint fbo;
    glGenFramebuffers(1, &fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, colBuf, 0);
    glFramebufferTexture2DEXT(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, depBuf, 0);

    // Check status of fbo
    if (glCheckFramebufferStatusEXT(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
        vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR, "Could not create FBO");
        return;
    }

    GLenum DrawBuffers[1] = {GL_COLOR_ATTACHMENT0};
    glDrawBuffers(1, DrawBuffers);



    // Render to framebuffer
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glViewport(0, 0, this->cineXRes, this->cineYRes); // Render on the whole framebuffer, complete from the lower left corner to the upper right
    */



    // Call Renderer
    Base::Render(context);



    /*
    // Unbind buffers 
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, 0, 0);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, 0, 0);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    // Reset viewport
    glViewport(viewportStuff[0], viewportStuff[1], viewportStuff[2], viewportStuff[3]);

    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glLoadIdentity();
    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glLoadIdentity();

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, colBuf);
   
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_MIRRORED_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_MIRRORED_REPEAT);

    // Draw
    glBegin(GL_QUADS);
        glTexCoord2f(0, 0); glVertex2f(-1.0f, -1.0f);
        glTexCoord2f(1, 0); glVertex2f(1.0f, -1.0f);
        glTexCoord2f(1, 1); glVertex2f(1.0f, 1.0f);
        glTexCoord2f(0, 1); glVertex2f(-1.0f, 1.0f);
    glEnd();

    glBindTexture(GL_TEXTURE_2D, 0);

    glMatrixMode(GL_PROJECTION);
    glPopMatrix();
    glMatrixMode(GL_MODELVIEW);
    glPopMatrix();

    glDisable(GL_TEXTURE_2D);

    glDeleteTextures(1, &colBuf);
    glDeleteTextures(1, &depBuf);
    glDeleteFramebuffersEXT(1, &fbo);
    */

    /*
    // Get current viewport
    int    viewportHeight  = cr3d->GetViewport().GetSize().GetHeight();
    int    viewportWidth   = cr3d->GetViewport().GetSize().GetWidth();
    int    letterboxHeight = 0;
    int    letterboxWidth  = 0;
    float  viewportRatio   = (float)viewportWidth / (float)viewportHeight;
    float  cinematicRatio  = (float)this->cineXRes / (float)this->cineYRes;

    if (cinematicRatio < viewportRatio) { // cinematic view is higher than current viewport
        letterboxHeight = viewportHeight;
        letterboxWidth = (int)(((float)viewportWidth - ((float)viewportHeight * cinematicRatio)) / 2.0f);
    }
    else if (cinematicRatio > viewportRatio) { // cinematic view is wider than current viewport
        letterboxHeight = (int)(((float)viewportHeight - ((float)viewportWidth / cinematicRatio)) / 2.0f);
        letterboxWidth = viewportWidth;
    }
    // Draw 2D Letterbox
    if (cinematicRatio != viewportRatio) {
        glMatrixMode(GL_PROJECTION);
        glPushMatrix();
        glLoadIdentity();
        glOrtho(0.0, (GLdouble)viewportWidth, 0.0, (GLdouble)viewportHeight, -1.0, 1.0);

        glMatrixMode(GL_MODELVIEW);
        glPushMatrix();
        glLoadIdentity();

        glDisable(GL_CULL_FACE);
        glClear(GL_DEPTH_BUFFER_BIT);
        glDisable(GL_LIGHTING);

        glColor3f(0.5f, 0.5f, 0.5f);

        // Bottom or left bar
        glBegin(GL_QUADS);
        glVertex2i(0, 0);
        glVertex2i(letterboxWidth, 0);
        glVertex2i(letterboxWidth, letterboxHeight);
        glVertex2i(0, letterboxHeight);
        glEnd();

        // Top or right bar
        glBegin(GL_QUADS);
        glVertex2i(viewportWidth,                  viewportHeight);
        glVertex2i(viewportWidth - letterboxWidth, viewportHeight);
        glVertex2i(viewportWidth - letterboxWidth, viewportHeight - letterboxHeight);
        glVertex2i(viewportWidth,                  viewportHeight - letterboxHeight);
        glEnd();

        glPopMatrix();

        glMatrixMode(GL_PROJECTION);
        glPopMatrix();
    }
    */
}