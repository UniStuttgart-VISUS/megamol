/**
* CinematicView.cpp
*
*/

#include "stdafx.h"

#include "mmcore/param/BoolParam.h"
#include "mmcore/param/EnumParam.h"
#include "mmcore/param/IntParam.h"
#include "mmcore/param/ButtonParam.h"
#include "mmcore/view/CallRender3D.h"
#include "mmcore/param/FloatParam.h"
#include "mmcore/view/CallRenderView.h"
#include "mmcore/param/FilePathParam.h"

#include "vislib/math/Rectangle.h"
#include "vislib/Trace.h"
#include "vislib/sys/Path.h"

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
    renderParam(              "01_renderAnim", "Toggle rendering of complete animation to PNG files."),
    toggleAnimPlayParam(      "02_playPreview", "Toggle playing animation as preview"),
	selectedSkyboxSideParam(  "03_skyboxSide", "Select the skybox side."),
    resHeightParam(           "04_cinematicHeight", "The height resolution of the cineamtic view to render."), 
    resWidthParam(            "05_cinematicWidth","The width resolution of the cineamtic view to render."),
    fpsParam(                 "06_fps", "Frames per second the animation should be rendered."),
    shownKeyframe()
    {

    this->keyframeKeeperSlot.SetCompatibleCall<CallCinematicCameraDescription>();
    this->MakeSlotAvailable(&this->keyframeKeeperSlot);

    // init variables

    this->deltaAnimTime   = clock();
    this->playAnim        = false;
    this->sbSide          = CinematicView::SkyboxSides::SKYBOX_NONE;
    this->cineWidth       = 1920;
    this->cineHeight      = 1080;
    this->vpW             = 0;
    this->vpH             = 0;
    this->rendering       = false;
    this->resetFbo        = true;
    this->fps             = 24;
    this->expFrameCnt     = 1;

    // init png data struct
    this->pngdata.bpp      = 3;
    this->pngdata.width    = static_cast<unsigned int>(this->cineWidth);
    this->pngdata.height   = static_cast<unsigned int>(this->cineHeight);
    this->pngdata.filename = "";
    this->pngdata.cnt      = 0;
    this->pngdata.animTime = 0.0f;
    this->pngdata.buffer   = NULL;
    this->pngdata.ptr      = NULL;
    this->pngdata.infoptr  = NULL;
    this->pngdata.filename = "";
    this->pngdata.lock     = true;
    //this->pngdata.file;

    // init parameters
    param::EnumParam *sbs = new param::EnumParam(this->sbSide);
    sbs->SetTypePair(CinematicView::SkyboxSides::SKYBOX_NONE,  "None");
    sbs->SetTypePair(CinematicView::SkyboxSides::SKYBOX_FRONT, "Front");
    sbs->SetTypePair(CinematicView::SkyboxSides::SKYBOX_BACK,  "Back");
    sbs->SetTypePair(CinematicView::SkyboxSides::SKYBOX_LEFT,  "Left");
    sbs->SetTypePair(CinematicView::SkyboxSides::SKYBOX_RIGHT, "Right");
    sbs->SetTypePair(CinematicView::SkyboxSides::SKYBOX_UP,    "Up");
    sbs->SetTypePair(CinematicView::SkyboxSides::SKYBOX_DOWN,  "Down");
    this->selectedSkyboxSideParam << sbs;
    this->MakeSlotAvailable(&this->selectedSkyboxSideParam);

    this->resHeightParam.SetParameter(new param::IntParam(this->cineHeight, 1));
    this->MakeSlotAvailable(&this->resHeightParam);

    this->resWidthParam.SetParameter(new param::IntParam(this->cineWidth, 1));
    this->MakeSlotAvailable(&this->resWidthParam);

    this->fpsParam.SetParameter(new param::IntParam(static_cast<int>(this->fps), 1));
    this->MakeSlotAvailable(&this->fpsParam);

    this->renderParam.SetParameter(new param::ButtonParam('r'));
    this->MakeSlotAvailable(&this->renderParam);

    this->toggleAnimPlayParam.SetParameter(new param::ButtonParam(' '));
    this->MakeSlotAvailable(&this->toggleAnimPlayParam);

    // TEMPORARY HACK #########################################################
    // Disable enableMouseSelectionSlot for this view
    // -> 'TAB'-key is needed in view3D of cinematic renderer to enable 
    //    mouse selection for manipulatiors
    this->enableMouseSelectionSlot.MakeUnavailable();
    // Disable toggleAnimPlaySlot for this view
    // -> 'SPACE'-key is needed to be asigned to own animation param
    param::ParamSlot* toggleAnimPlaySlot = static_cast<param::ParamSlot*>(this->timeCtrl.GetSlot(3)); // toggleAnimPlaySlot
    toggleAnimPlaySlot->MakeUnavailable();
}


/*
* CinematicView::~CinematicView
*/
CinematicView::~CinematicView(void) {

    if (this->pngdata.ptr != NULL) {
        if (this->pngdata.infoptr != NULL) {
            png_destroy_write_struct(&this->pngdata.ptr, &this->pngdata.infoptr);
        }
        else {
            png_destroy_write_struct(&this->pngdata.ptr, (png_infopp)NULL);
        }
    }

    try { this->pngdata.file.Flush(); } catch (...) {}
    try { this->pngdata.file.Close(); } catch (...) {}

    ARY_SAFE_DELETE(this->pngdata.buffer);

    if (this->pngdata.buffer != NULL) {
        vislib::sys::Log::DefaultLog.WriteError("[CINEMATIC VIEW] [render] pngdata.buffer is not NULL.");
    }
    if (this->pngdata.ptr != NULL) {
        vislib::sys::Log::DefaultLog.WriteError("[CINEMATIC VIEW] [render] pngdata.ptr is not NULL.");
    }
    if (this->pngdata.infoptr != NULL) {
        vislib::sys::Log::DefaultLog.WriteError("[CINEMATIC VIEW] [render] pngdata.infoptr is not NULL.");
    }

    if (this->fbo.IsValid()) {
        if (this->fbo.IsEnabled()) {
            this->fbo.Disable();
        }
    }
    this->fbo.Release();
}


/*
* CinematicView::Render
*/
void CinematicView::Render(const mmcRenderViewContext& context) {

    view::CallRender3D *cr3d = this->rendererSlot.CallAs<core::view::CallRender3D>();
    if (cr3d == NULL) return;
    if (!(*cr3d)(1)) return; // get extents

    CallCinematicCamera *ccc = this->keyframeKeeperSlot.CallAs<CallCinematicCamera>();
    if (ccc == NULL) return;
    // Updated data from cinematic camera call
    if (!(*ccc)(CallCinematicCamera::CallForGetUpdatedKeyframeData)) return;

    // Set new bounding box center
    ccc->setBboxCenter(cr3d->AccessBoundingBoxes().WorldSpaceBBox().CalcCenter());
    // Set total simulation time of call
    ccc->setTotalSimTime(static_cast<float>(cr3d->TimeFramesCount()));
    // Set FPS to call
    ccc->setFps(this->fps);
    if (!(*ccc)(CallCinematicCamera::CallForSetSimulationData)) return;

    bool loadNewCamParams = false;

    // Update parameters ------------------------------------------------------
    if (this->toggleAnimPlayParam.IsDirty()) {
        this->toggleAnimPlayParam.ResetDirty();
        this->playAnim = !this->playAnim;
        if (this->playAnim) {
            this->deltaAnimTime = clock();
        }
    }
    if (this->selectedSkyboxSideParam.IsDirty()) {
        this->selectedSkyboxSideParam.ResetDirty();
        if (this->rendering) {
            this->selectedSkyboxSideParam.Param<param::EnumParam>()->SetValue(static_cast<int>(this->sbSide), false);
            vislib::sys::Log::DefaultLog.WriteWarn("[CINEMATIC VIEW] [resHeightParam] Changes are not applied while rendering is running.");
        }
        else {
            this->sbSide = static_cast<CinematicView::SkyboxSides>(this->selectedSkyboxSideParam.Param<param::EnumParam>()->Value());
            loadNewCamParams = true;
        }
    }
    if (this->resHeightParam.IsDirty()) {
        this->resHeightParam.ResetDirty();
        if (this->rendering) {
            this->resHeightParam.Param<param::IntParam>()->SetValue(this->cineHeight, false);
            vislib::sys::Log::DefaultLog.WriteWarn("[CINEMATIC VIEW] [resHeightParam] Changes are not applied while rendering is running.");
        }
        else {
            this->cineHeight = this->resHeightParam.Param<param::IntParam>()->Value();
            this->resetFbo = true;
        }
    }
    if (this->resWidthParam.IsDirty()) {
        this->resWidthParam.ResetDirty();
        if (this->rendering) {
            this->resWidthParam.Param<param::IntParam>()->SetValue(this->cineWidth, false);
            vislib::sys::Log::DefaultLog.WriteWarn("[CINEMATIC VIEW] [resWidthParam] Changes are not applied while rendering is running.");
        }
        else {
            this->cineWidth = this->resWidthParam.Param<param::IntParam>()->Value();
            this->resetFbo = true;
        }
    }
    if (this->fpsParam.IsDirty()) {
        this->fpsParam.ResetDirty();
        if (this->rendering) {
            this->fpsParam.Param<param::IntParam>()->SetValue(static_cast<int>(this->fps), false);
            vislib::sys::Log::DefaultLog.WriteWarn("[CINEMATIC VIEW] [fpsParam] Changes are not applied while rendering is running.");
        }
        else {
            this->fps = static_cast<unsigned int>(this->fpsParam.Param<param::IntParam>()->Value());
        }
    }
    if (this->renderParam.IsDirty()) {
        this->renderParam.ResetDirty();
        if (!this->rendering) {
            this->rtf_setup();
        }
        else {
            this->rtf_finish();
        }
    }


    // Time settings -----------------------------------------------
    if (this->rendering) {
        this->rtf_set_time_and_camera();
        loadNewCamParams = true;
    }
    else {
        // If time is set by running ANIMATION 
        if (this->playAnim) {
            clock_t tmpTime     = clock();
            clock_t cTime       = tmpTime - this->deltaAnimTime;
            this->deltaAnimTime = tmpTime;

            float animTime = ccc->getSelectedKeyframe().getAnimTime() + ((float)cTime) / (float)(CLOCKS_PER_SEC);
            if (animTime > ccc->getTotalAnimTime()) { // Reset time if max animation time is reached
                animTime = 0.0f;
            }
            ccc->setSelectedKeyframeTime(animTime);
            // Update selected keyframe
            if (!(*ccc)(CallCinematicCamera::CallForGetSelectedKeyframeAtTime)) return;
            
            loadNewCamParams = true;
        }
        this->setSimTime(ccc->getSelectedKeyframe().getSimTime());
    }

    // Set camera parameters of selected keyframe for this view.
    // But only if selected keyframe differs to last locally stored and shown keyframe.
    // Load new camera setting from selected keyframe when skybox side changes or rendering 
    // or animation loaded new slected keyframe.
    Keyframe skf = ccc->getSelectedKeyframe();
    if ((this->shownKeyframe != skf) || loadNewCamParams) {
        this->shownKeyframe = skf;

        this->cam.Parameters()->SetView(skf.getCamPosition(), skf.getCamLookAt(), skf.getCamUp());
        this->cam.Parameters()->SetApertureAngle(skf.getCamApertureAngle());
        loadNewCamParams = true;
    }

    // Apply showing skybox side ONLY if new camera parameters are set
    if (loadNewCamParams) {

        // Adjust cam to selected skybox side
        if (this->sbSide != CinematicView::SkyboxSides::SKYBOX_NONE) {
            // Get camera parameters
            vislib::SmartPtr<vislib::graphics::CameraParameters> cp = this->cam.Parameters();
            vislib::math::Point<float, 3>  camPos    = cp->Position();
            vislib::math::Vector<float, 3> camRight  = cp->Right();
            vislib::math::Vector<float, 3> camUp     = cp->Up();
            vislib::math::Vector<float, 3> camFront  = cp->Front();
            vislib::math::Point<float, 3>  camLookAt = cp->LookAt();
            float                          tmpDist   = cp->FocalDistance();

            // set aperture angle to 90 deg
            cp->SetApertureAngle(90.0f);
            if (this->sbSide == CinematicView::SkyboxSides::SKYBOX_BACK) {
                cp->SetView(camPos, camPos - camFront * tmpDist, camUp);
            }
            else if (this->sbSide == CinematicView::SkyboxSides::SKYBOX_RIGHT) {
                cp->SetView(camPos, camPos + camRight * tmpDist, camUp);
            }
            else if (this->sbSide == CinematicView::SkyboxSides::SKYBOX_LEFT) {
                cp->SetView(camPos, camPos - camRight * tmpDist, camUp);
            }
            else if (this->sbSide == CinematicView::SkyboxSides::SKYBOX_UP) {
                cp->SetView(camPos, camPos + camUp * tmpDist, -camFront);
            }
            else if (this->sbSide == CinematicView::SkyboxSides::SKYBOX_DOWN) {
                cp->SetView(camPos, camPos - camUp * tmpDist, camFront);
            }
        }
    }

    // Propagate camera parameters to keyframe keeper (sky box camera params are propageted too!)
    ccc->setCameraParameters(this->cam.Parameters());
    if (!(*ccc)(CallCinematicCamera::CallForSetCameraForKeyframe)) return;

    // Viewport stuff ---------------------------------------------------------
    int vp[4];
    glGetIntegerv(GL_VIEWPORT, vp);
    int   vpWidth = vp[2] - vp[0];  
    int   vpHeight = vp[3] - vp[1]; 

    float vpRatio = static_cast<float>(vpWidth) / static_cast<float>(vpHeight);
    float cineRatio = static_cast<float>(this->cineWidth) / static_cast<float>(this->cineHeight);
    int   fboWidth = vpWidth;
    int   fboHeight = vpHeight;

    if (this->rendering) {
        fboWidth = this->cineWidth;
        fboHeight = this->cineHeight;
    }
    else {
        // Calculate reduced fbo width and height
        fboWidth = vpWidth;
        fboHeight = vpHeight;
        if (cineRatio > vpRatio) {
            fboHeight = (static_cast<int>(static_cast<float>(vpWidth) / cineRatio));
        }
        else if (cineRatio < vpRatio) {
            fboWidth = (static_cast<int>(static_cast<float>(vpHeight) * cineRatio));
        }
        // Check for viewport changes
        if ((this->vpW != vpWidth) || (this->vpH != vpHeight)) {
            this->vpW = vpWidth;
            this->vpH = vpHeight;
            this->resetFbo = true;
        }
    }
    // Set override viewport of view (otherwise viewport is overwritten in Base::Render(context))
    int fboVp[4] = { 0, 0, fboWidth, fboHeight };
    this->overrideViewport = fboVp;
    // Set new viewport settings for camera
    this->cam.Parameters()->SetVirtualViewSize(static_cast<vislib::graphics::ImageSpaceType>(fboWidth), static_cast<vislib::graphics::ImageSpaceType>(fboHeight));
   

    // Render to texture ------------------------------------------------------------

    // Create new frame for file
    if (this->rendering) {
        this->rtf_create_frame();
    }

// Suppress TRACE output of fbo.Enable() and fbo.Create()
#if defined(DEBUG) || defined(_DEBUG)
    unsigned int otl = vislib::Trace::GetInstance().GetLevel();
    vislib::Trace::GetInstance().SetLevel(0);
#endif // DEBUG || _DEBUG 

    if (this->resetFbo || (!this->fbo.IsValid())) {
        if (this->fbo.IsValid()) {
            this->fbo.Release();
        }
        if (!this->fbo.Create(fboWidth, fboHeight, GL_RGBA8, GL_RGBA, GL_UNSIGNED_BYTE, vislib::graphics::gl::FramebufferObject::ATTACHMENT_TEXTURE, GL_DEPTH_COMPONENT24)) {
            throw vislib::Exception("[CINEMATIC VIEW] [render] Unable to create image framebuffer object.", __FILE__, __LINE__);
            return;
        }
        this->resetFbo = false;
    }
    if (this->fbo.Enable() != GL_NO_ERROR) {
        throw vislib::Exception("[CINEMATIC VIEW] [render] Cannot enable Framebuffer object.", __FILE__, __LINE__);
        return;
    }
// Reset TRACE output level
#if defined(DEBUG) || defined(_DEBUG)
    vislib::Trace::GetInstance().SetLevel(otl);
#endif // DEBUG || _DEBUG 

    // Set output buffer for override call (otherwise render call is overwritten in Base::Render(context))
    //GLenum callOutBuffer = cr3d->OutputBuffer();
    cr3d->SetOutputBuffer(this->fbo.GetID());

    this->overrideCall = cr3d;

    // Call Render-Function of parent View3D
    Base::Render(context);

    this->fbo.Disable();

    //cr3d->SetOutputBuffer(callOutBuffer);

    // Reset override render call
    this->overrideCall = NULL;
    // Reset override viewport
    this->overrideViewport = NULL; 

    // Write frame to file
    if (this->rendering) {
        this->rtf_write_frame();
    }

    // Draw final image -------------------------------------------------------
    glDisable(GL_LIGHTING);
    glDisable(GL_DEPTH_TEST);
    glDisable(GL_BLEND);
    glDisable(GL_CULL_FACE);

    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glLoadIdentity();
    // Reset viewport
    glViewport(vp[0], vp[1], vp[2], vp[3]);
    glOrtho(0.0f, static_cast<float>(vpWidth), 0.0f, static_cast<float>(vpHeight), -1.0, 1.0);

    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glLoadIdentity();

    // Adjust fbo viewport size if it is greater than viewport
    int fboVpWidth  = fboWidth;
    int fboVpHeight = fboHeight;
    if ((fboVpWidth > vpWidth) || ((fboVpWidth < vpWidth) && (fboVpHeight < vpHeight))) {
        fboVpWidth  = vpWidth;
        fboVpHeight = static_cast<int>(static_cast<float>(vpWidth) / cineRatio);
    }
    if ((fboVpHeight > vpHeight) || ((fboVpWidth < vpWidth) && (fboVpHeight < vpHeight))) {
        fboVpHeight = vpHeight;
        fboVpWidth  = static_cast<int>(static_cast<float>(vpHeight) * cineRatio);
    }
    float right  = static_cast<float>(vpWidth + fboVpWidth) / 2.0f;
    float left   = static_cast<float>(vpWidth - fboVpWidth) / 2.0f;
    float bottom = static_cast<float>(vpHeight - fboVpHeight) / 2.0f;
    float up     = static_cast<float>(vpHeight + fboVpHeight) / 2.0f;

    // Draw texture
    glActiveTexture(GL_TEXTURE0);
    glEnable(GL_TEXTURE_2D);
    this->fbo.BindColourTexture();
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glColor4f(1.0f, 1.0f, 1.0f, 1.0f);
    glBegin(GL_QUADS);
        glTexCoord2f(0.0f, 0.0f); glVertex2f(left,  bottom);
        glTexCoord2f(1.0f, 0.0f); glVertex2f(right, bottom);
        glTexCoord2f(1.0f, 1.0f); glVertex2f(right, up);
        glTexCoord2f(0.0f, 1.0f); glVertex2f(left,  up);
    glEnd();
    glBindTexture(GL_TEXTURE_2D, 0);
    glDisable(GL_TEXTURE_2D);

    // Draw letter box  -------------------------------------------------------

    // Color stuff ------------------------------------------------------------
    const float *bgColor = this->bkgndColour();
    // COLORS
    float lbColor[4] = { 1.0f, 1.0f, 1.0f, 1.0f };
    // Adapt colors depending on  Lightness
    float L = (vislib::math::Max(bgColor[0], vislib::math::Max(bgColor[1], bgColor[2])) + vislib::math::Min(bgColor[0], vislib::math::Min(bgColor[1], bgColor[2]))) / 2.0f;
    if (L > 0.5f) {
        for (unsigned int i = 0; i < 3; i++) {
            lbColor[i] = 0.0f;
        }
    }

    // Calculate position of texture
    int x = 0;
    int y = 0;
    if (fboVpWidth < vpWidth) {
        x = (static_cast<int>(static_cast<float>(vpWidth - fboVpWidth) / 2.0f));
        y = vpHeight;
    }
    else if (fboVpHeight < vpHeight) {
        x = vpWidth;
        y = (static_cast<int>(static_cast<float>(vpHeight - fboVpHeight) / 2.0f));
    }
    // Draw letter box quads in letter box Color
    glColor4fv(lbColor);
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
    
    // Reset opengl
    glDepthFunc(GL_LEQUAL);
    glEnable(GL_DEPTH_TEST);

    // Unlock renderer after first frame
    if (this->rendering && this->pngdata.lock) {
        this->pngdata.lock = false;
    }
}


/*
* CinematicView::rtf_setup
*/
bool CinematicView::rtf_setup() {

    CallCinematicCamera *ccc = this->keyframeKeeperSlot.CallAs<CallCinematicCamera>();
    if (ccc == NULL) return false;

    // init png data struct
    this->pngdata.bpp       = 3;
    this->pngdata.width     = static_cast<unsigned int>(this->cineWidth);
    this->pngdata.height    = static_cast<unsigned int>(this->cineHeight);
    this->pngdata.cnt       = 0;
    this->pngdata.animTime  = 0.0f;
    this->pngdata.buffer    = NULL;
    this->pngdata.ptr       = NULL;
    this->pngdata.infoptr   = NULL;
    //this->pngdata.file;

    // Calculate pre-decimal point positions for frame counter in filename
    this->expFrameCnt = 1;
    float frameCnt = (float)(this->fps) * ccc->getTotalAnimTime();
    while (frameCnt > 1.0f) {
        frameCnt /= 10.0f;
        this->expFrameCnt++;
    }

    // Create new folder
    vislib::StringA frameFolder;
    time_t t = std::time(0); // get time now
    struct tm *now = std::localtime(&t);
    frameFolder.Format("frames_%i%02i%02i-%02i%02i%02i_%ifps",  (now->tm_year + 1900), (now->tm_mon + 1), now->tm_mday, now->tm_hour, now->tm_min, now->tm_sec, this->fps);
    this->pngdata.path = vislib::sys::Path::GetCurrentDirectoryA();
    this->pngdata.path = vislib::sys::Path::Concatenate(this->pngdata.path, frameFolder);
    vislib::sys::Path::MakeDirectory(this->pngdata.path);

    // Set current time stamp to file name
    this->pngdata.filename = "frames";
    // this->pngdata.filename.Format("frame_%i%i%i-%i%i%i_-_", (now->tm_year + 1900), (now->tm_mon + 1), now->tm_mday, now->tm_hour, now->tm_min, now->tm_sec);

    // Create new byte buffer
    this->pngdata.buffer = new BYTE[this->pngdata.width * this->pngdata.height * this->pngdata.bpp];
    if (this->pngdata.buffer == NULL) {
        throw vislib::Exception("[CINEMATIC VIEW] [startAnimRendering] Cannot allocate image buffer.", __FILE__, __LINE__);
    }

    // Reset animation time to zero
    ccc->setSelectedKeyframeTime(0.0f);
    // Update selected keyframe
    if (!(*ccc)(CallCinematicCamera::CallForGetSelectedKeyframeAtTime)) return false;
    // Set current simulation time
    this->setSimTime(ccc->getSelectedKeyframe().getSimTime());
    
    // Lock rendering and wait for one frame to get the new animation time applied. 
    // Otherwise first frame is not set right -> just for high resolutions ?
    this->pngdata.lock = true;

    this->rendering = true;
    this->resetFbo  = true;

    vislib::sys::Log::DefaultLog.WriteInfo("[CINEMATIC VIEW] STARTED rendnering of complete animation.");
    return true;
}


/*
* CinematicView::rtf_set_time_and_camera
*/
bool CinematicView::rtf_set_time_and_camera() {

    if (!this->pngdata.lock) {
        CallCinematicCamera *ccc = this->keyframeKeeperSlot.CallAs<CallCinematicCamera>();
        if (ccc == NULL) return false;

        if ((this->pngdata.animTime < 0.0f) || (this->pngdata.animTime > ccc->getTotalAnimTime())) {
            throw vislib::Exception("[CINEMATIC VIEW] [rtf_set_time_and_camera] Invalid animation time.", __FILE__, __LINE__);
        }

        // Get selected keyframe for current animation time
        ccc->setSelectedKeyframeTime(this->pngdata.animTime);
        // Update selected keyframe
        if (!(*ccc)(CallCinematicCamera::CallForGetSelectedKeyframeAtTime)) return false;

        // Set current simulation time
        this->setSimTime(ccc->getSelectedKeyframe().getSimTime());

        // Increase to next time step
        float fpsFrac = (1.0f / static_cast<float>(this->fps));
        this->pngdata.animTime += fpsFrac;

        // Finish rendering if max anim time is reached
        if (this->pngdata.animTime > ccc->getTotalAnimTime()) {
            this->rtf_finish();
        }
    }
    return true;
}


/*
* CinematicView::rtf_create_frame
*/
bool CinematicView::rtf_create_frame() {

    if (!this->pngdata.lock) {
        vislib::StringA tmpFilename, tmpStr;
        tmpStr.Format(".%i", this->expFrameCnt);
        tmpStr.Prepend("%0");
        tmpStr.Append("i.png");
        tmpFilename.Format(tmpStr.PeekBuffer(), this->pngdata.cnt);
        tmpFilename.Prepend(this->pngdata.filename);

        // open final image file
        if (!this->pngdata.file.Open(vislib::sys::Path::Concatenate(this->pngdata.path, tmpFilename),
            vislib::sys::File::WRITE_ONLY, vislib::sys::File::SHARE_EXCLUSIVE, vislib::sys::File::CREATE_OVERWRITE)) {
            throw vislib::Exception("[CINEMATIC VIEW] [startAnimRendering] Cannot open output file", __FILE__, __LINE__);
        }

        // init png lib
        this->pngdata.ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, &this->pngError, &this->pngWarn);
        if (this->pngdata.ptr == NULL) {
            throw vislib::Exception("[CINEMATIC VIEW] [startAnimRendering] Cannot create png structure", __FILE__, __LINE__);
        }
        this->pngdata.infoptr = png_create_info_struct(this->pngdata.ptr);
        if (this->pngdata.infoptr == NULL) {
            throw vislib::Exception("[CINEMATIC VIEW] [startAnimRendering] Cannot create png info", __FILE__, __LINE__);
        }
        png_set_write_fn(this->pngdata.ptr, static_cast<void*>(&this->pngdata.file), &this->pngWrite, &this->pngFlush);
        png_set_IHDR(this->pngdata.ptr, this->pngdata.infoptr, this->pngdata.width, this->pngdata.height, 8, PNG_COLOR_TYPE_RGB, PNG_INTERLACE_NONE, PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);

        this->pngdata.cnt++;
    }

    return true;
}


/*
* CinematicView::rtf_write_frame
*/
bool CinematicView::rtf_write_frame() {

    if (!this->pngdata.lock) {
        if (fbo.GetColourTexture(this->pngdata.buffer, 0, GL_RGB, GL_UNSIGNED_BYTE) != GL_NO_ERROR) {
            throw vislib::Exception("[CINEMATIC VIEW] [writeTextureToPng] Failed to create Screenshot: Cannot read image data", __FILE__, __LINE__);
        }

        BYTE** rows = NULL;
        try {
            rows = new BYTE*[this->cineHeight];
            for (UINT i = 0; i < this->pngdata.height; i++) {
                rows[this->pngdata.height - (1 + i)] = this->pngdata.buffer + this->pngdata.bpp * i * this->pngdata.width;
            }
            png_set_rows(this->pngdata.ptr, this->pngdata.infoptr, rows);

            png_write_png(this->pngdata.ptr, this->pngdata.infoptr, PNG_TRANSFORM_IDENTITY, NULL);

            ARY_SAFE_DELETE(rows);
        }
        catch (...) {
            if (rows != NULL) {
                ARY_SAFE_DELETE(rows);
            }
            throw;
        }

        if (this->pngdata.ptr != NULL) {
            if (this->pngdata.infoptr != NULL) {
                png_destroy_write_struct(&this->pngdata.ptr, &this->pngdata.infoptr);
            }
            else {
                png_destroy_write_struct(&this->pngdata.ptr, (png_infopp)NULL);
            }
        }

        try { this->pngdata.file.Flush(); }  catch (...) {}
        try { this->pngdata.file.Close(); }  catch (...) {}
    }
    return true;
}


/*
* CinematicView::rtf_finish
*/
bool CinematicView::rtf_finish() {

    if (this->pngdata.ptr != NULL) {
        if (this->pngdata.infoptr != NULL) {
            png_destroy_write_struct(&this->pngdata.ptr, &this->pngdata.infoptr);
        }
        else {
            png_destroy_write_struct(&this->pngdata.ptr, (png_infopp)NULL);
        }
    }

    try { this->pngdata.file.Flush(); } catch (...) {}
    try { this->pngdata.file.Close(); } catch (...) {}

    ARY_SAFE_DELETE(this->pngdata.buffer);

    if (this->fbo.IsValid()) {
        if (this->fbo.IsEnabled()) {
            this->fbo.Disable();
        }
    }
    this->fbo.Release();

    this->resetFbo = true;
    this->rendering = false;

    vislib::sys::Log::DefaultLog.WriteInfo("[CINEMATIC VIEW] STOPPED rendering.");
    return true;
}


/*
* CinematicView::setSimTime
*/
bool CinematicView::setSimTime(float st) {

    view::CallRender3D *cr3d = this->rendererSlot.CallAs<core::view::CallRender3D>();
    if (cr3d == NULL) {
        vislib::sys::Log::DefaultLog.WriteError("[CINEMATIC VIEW] [setSimTime] cr3d is NULL.");
        return false;
    }

    float simTime = st;
    param::ParamSlot* animTimeParam = static_cast<param::ParamSlot*>(this->timeCtrl.GetSlot(2));
    animTimeParam->Param<param::FloatParam>()->SetValue(simTime * static_cast<float>(cr3d->TimeFramesCount()), true);

    return true;
}