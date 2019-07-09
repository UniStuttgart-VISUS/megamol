/**
 * CinematicView.cpp
 *
 * Copyright (C) 2017 by VISUS (Universitaet Stuttgart).
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"

#include "CinematicView.h"

using namespace megamol;
using namespace megamol::core;
using namespace megamol::core::view;
using namespace megamol::core::utility;
using namespace megamol::cinematic;

using namespace vislib;

CinematicView::CinematicView(void)
    : View3D()
    , keyframeKeeperSlot("keyframeKeeper", "Connects to the Keyframe Keeper.")
    , renderParam(               "renderAnim", "Toggle rendering of complete animation to PNG files.")
    , toggleAnimPlayParam(       "playPreview", "Toggle playing animation as preview")
    , selectedSkyboxSideParam(   "skyboxSide", "Select the skybox side.")
    , cubeModeRenderParam(       "cubeMode", "Render cube around dataset with skyboxSide as side selector.")
    , resWidthParam(             "cinematicWidth", "The width resolution of the cineamtic view to render.")
    , resHeightParam(            "cinematicHeight", "The height resolution of the cineamtic view to render.")
    , fpsParam(                  "fps", "Frames per second the animation should be rendered.")
    , startRenderFrameParam(     "firstRenderFrame", "Set first frame number to start rendering with " 
                                                     "(allows continuing aborted rendering without starting "
                                                     "from the beginning).")
    , delayFirstRenderFrameParam("delayFirstRenderFrame", "Delay (in seconds) to wait until first frame "
                                                          "for rendering is written (needed to get right "
                                                          "first frame especially for high resolutions and "
                                                          "for distributed rendering).")
    , frameFolderParam(          "frameFolder", "Specify folder where the frame files should be stored.")
    , addSBSideToNameParam(      "addSBSideToName", "Toggle whether skybox side should be added to output filename")
    , eyeParam(                  "stereo::eye", "Select eye position (for stereo view).")
    , projectionParam(           "stereo::projection", "Select camera projection.")
    , theFont(megamol::core::utility::SDFFont::FontName::ROBOTO_SANS)
    , deltaAnimTime(clock())
    , shownKeyframe()
    , playAnim(false)
    , cineWidth(1920)
    , cineHeight(1080)
    , vpWLast(0)
    , vpHLast(0)
    , sbSide(CinematicView::SkyboxSides::SKYBOX_NONE)
    , fbo()
    , rendering(false)
    , fps(24)
    , pngdata() {

    // init callback
    this->keyframeKeeperSlot.SetCompatibleCall<CallKeyframeKeeperDescription>();
    this->MakeSlotAvailable(&this->keyframeKeeperSlot);

    // init parameters
    this->renderParam.SetParameter(new param::ButtonParam(core::view::Key::KEY_R, core::view::Modifier::CTRL));
    this->MakeSlotAvailable(&this->renderParam);

    this->toggleAnimPlayParam.SetParameter(new param::ButtonParam(core::view::Key::KEY_SPACE, core::view::Modifier::CTRL));
    this->MakeSlotAvailable(&this->toggleAnimPlayParam);

    param::EnumParam* sbs = new param::EnumParam(this->sbSide);
    sbs->SetTypePair(CinematicView::SkyboxSides::SKYBOX_NONE, "None");
    sbs->SetTypePair(CinematicView::SkyboxSides::SKYBOX_FRONT, "Front");
    sbs->SetTypePair(CinematicView::SkyboxSides::SKYBOX_BACK, "Back");
    sbs->SetTypePair(CinematicView::SkyboxSides::SKYBOX_LEFT, "Left");
    sbs->SetTypePair(CinematicView::SkyboxSides::SKYBOX_RIGHT, "Right");
    sbs->SetTypePair(CinematicView::SkyboxSides::SKYBOX_UP, "Up");
    sbs->SetTypePair(CinematicView::SkyboxSides::SKYBOX_DOWN, "Down");
    this->selectedSkyboxSideParam << sbs;
    this->MakeSlotAvailable(&this->selectedSkyboxSideParam);

    this->cubeModeRenderParam << new param::BoolParam(false);
    this->MakeSlotAvailable(&this->cubeModeRenderParam);

    this->resWidthParam.SetParameter(new param::IntParam(this->cineWidth, 1));
    this->MakeSlotAvailable(&this->resWidthParam);

    this->resHeightParam.SetParameter(new param::IntParam(this->cineHeight, 1));
    this->MakeSlotAvailable(&this->resHeightParam);

    this->fpsParam.SetParameter(new param::IntParam(static_cast<int>(this->fps), 1));
    this->MakeSlotAvailable(&this->fpsParam);

    this->startRenderFrameParam.SetParameter(new param::IntParam(0, 0));
    this->MakeSlotAvailable(&this->startRenderFrameParam);

    this->delayFirstRenderFrameParam.SetParameter(new param::FloatParam(1.0f));
    this->MakeSlotAvailable(&this->delayFirstRenderFrameParam);

    this->frameFolderParam.SetParameter(new param::FilePathParam(""));
    this->MakeSlotAvailable(&this->frameFolderParam);

    this->addSBSideToNameParam << new param::BoolParam(false);
    this->MakeSlotAvailable(&this->addSBSideToNameParam);

    param::EnumParam* enp = new param::EnumParam(vislib::graphics::CameraParameters::StereoEye::LEFT_EYE);
    enp->SetTypePair(vislib::graphics::CameraParameters::StereoEye::LEFT_EYE, "Left");
    enp->SetTypePair(vislib::graphics::CameraParameters::StereoEye::RIGHT_EYE, "Right");
    this->eyeParam << enp;
    this->MakeSlotAvailable(&this->eyeParam);

    param::EnumParam* pep = new param::EnumParam(vislib::graphics::CameraParameters::ProjectionType::MONO_PERSPECTIVE);
    pep->SetTypePair(vislib::graphics::CameraParameters::ProjectionType::MONO_ORTHOGRAPHIC, "Mono Orthographic");
    pep->SetTypePair(vislib::graphics::CameraParameters::ProjectionType::MONO_PERSPECTIVE, "Mono Perspective");
    pep->SetTypePair(vislib::graphics::CameraParameters::ProjectionType::STEREO_OFF_AXIS, "Stereo Off-Axis");
    pep->SetTypePair(vislib::graphics::CameraParameters::ProjectionType::STEREO_PARALLEL, "Stereo Parallel");
    pep->SetTypePair(vislib::graphics::CameraParameters::ProjectionType::STEREO_TOE_IN, "Stereo ToeIn");
    this->projectionParam << pep;
    this->MakeSlotAvailable(&this->projectionParam);
}


CinematicView::~CinematicView(void) {

    if (this->pngdata.ptr != nullptr) {
        if (this->pngdata.infoptr != nullptr) {
            png_destroy_write_struct(&this->pngdata.ptr, &this->pngdata.infoptr);
        } else {
            png_destroy_write_struct(&this->pngdata.ptr, (png_infopp) nullptr);
        }
    }

    try {
        this->pngdata.file.Flush();
    } catch (...) {
    }
    try {
        this->pngdata.file.Close();
    } catch (...) {
    }

    ARY_SAFE_DELETE(this->pngdata.buffer);

    if (this->pngdata.buffer != nullptr) {
        vislib::sys::Log::DefaultLog.WriteError("[CINEMATIC VIEW] [render] pngdata.buffer is not nullptr.");
    }
    if (this->pngdata.ptr != nullptr) {
        vislib::sys::Log::DefaultLog.WriteError("[CINEMATIC VIEW] [render] pngdata.ptr is not nullptr.");
    }
    if (this->pngdata.infoptr != nullptr) {
        vislib::sys::Log::DefaultLog.WriteError("[CINEMATIC VIEW] [render] pngdata.infoptr is not nullptr.");
    }

    this->fbo.Release();
}


void CinematicView::Render(const mmcRenderViewContext& context) {

    view::CallRender3D* cr3d = this->rendererSlot.CallAs<core::view::CallRender3D>();
    if (cr3d == nullptr) return;
    if (!(*cr3d)(view::AbstractCallRender::FnGetExtents)) return;

    CallKeyframeKeeper* ccc = this->keyframeKeeperSlot.CallAs<CallKeyframeKeeper>();
    if (ccc == nullptr) return;
    // Updated data from cinematic camera call
    if (!(*ccc)(CallKeyframeKeeper::CallForGetUpdatedKeyframeData)) return;

    // Set bounding box center of model
    ccc->setBboxCenter(cr3d->AccessBoundingBoxes().WorldSpaceBBox().CalcCenter());
    // Set total simulation time of call
    ccc->setTotalSimTime(static_cast<float>(cr3d->TimeFramesCount()));
    // Set FPS to call
    ccc->setFps(this->fps);
    if (!(*ccc)(CallKeyframeKeeper::CallForSetSimulationData)) return;

    bool loadNewCamParams = false;

    // Disabling toggleAnimPlaySlot in the parent view3d
    // -> 'SPACE'-key is needed to be available here for own control of animation time
    this->SetSlotUnavailable(static_cast<AbstractSlot*>(this->timeCtrl.GetSlot(3)));

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
            vislib::sys::Log::DefaultLog.WriteWarn(
                "[CINEMATIC VIEW] [resHeightParam] Changes are not applied while rendering is running.");
        } else {
            this->sbSide = static_cast<CinematicView::SkyboxSides>(
                this->selectedSkyboxSideParam.Param<param::EnumParam>()->Value());
            loadNewCamParams = true;
        }
    }

    if (this->resHeightParam.IsDirty()) {
        this->resHeightParam.ResetDirty();
        if (this->rendering) {
            this->resHeightParam.Param<param::IntParam>()->SetValue(this->cineHeight, false);
            vislib::sys::Log::DefaultLog.WriteWarn(
                "[CINEMATIC VIEW] [resHeightParam] Changes are not applied while rendering is running.");
        } else {
            this->cineHeight = this->resHeightParam.Param<param::IntParam>()->Value();
        }
    }
    if (this->resWidthParam.IsDirty()) {
        this->resWidthParam.ResetDirty();
        if (this->rendering) {
            this->resWidthParam.Param<param::IntParam>()->SetValue(this->cineWidth, false);
            vislib::sys::Log::DefaultLog.WriteWarn(
                "[CINEMATIC VIEW] [resWidthParam] Changes are not applied while rendering is running.");
        } else {
            this->cineWidth = this->resWidthParam.Param<param::IntParam>()->Value();
        }
    }
    if (this->fpsParam.IsDirty()) {
        this->fpsParam.ResetDirty();
        if (this->rendering) {
            this->fpsParam.Param<param::IntParam>()->SetValue(static_cast<int>(this->fps), false);
            vislib::sys::Log::DefaultLog.WriteWarn(
                "[CINEMATIC VIEW] [fpsParam] Changes are not applied while rendering is running.");
        } else {
            this->fps = static_cast<unsigned int>(this->fpsParam.Param<param::IntParam>()->Value());
        }
    }
    if (this->renderParam.IsDirty()) {
        this->renderParam.ResetDirty();
        this->rendering = !this->rendering;
        if (this->rendering) {
            this->render2file_setup();
        } else {
            this->render2file_finish();
        }
    }
    // Set (mono/stereo) projection for camera
    if (this->projectionParam.IsDirty()) {
        this->projectionParam.ResetDirty();
        this->cam.Parameters()->SetProjection(static_cast<vislib::graphics::CameraParameters::ProjectionType>(
            this->projectionParam.Param<param::EnumParam>()->Value()));
    }
    // Set eye position for camera
    if (this->eyeParam.IsDirty()) {
        this->eyeParam.ResetDirty();
        this->cam.Parameters()->SetEye(static_cast<vislib::graphics::CameraParameters::StereoEye>(
            this->eyeParam.Param<param::EnumParam>()->Value()));
    }

    // Time settings ----------------------------------------------------------
    // Load animation time
    if (this->rendering) {
        if ((this->pngdata.animTime < 0.0f) || (this->pngdata.animTime > ccc->getTotalAnimTime())) {
            throw vislib::Exception("[CINEMATIC VIEW] Invalid animation time.", __FILE__, __LINE__);
        }
        ccc->setSelectedKeyframeTime(this->pngdata.animTime);
        if (!(*ccc)(CallKeyframeKeeper::CallForGetSelectedKeyframeAtTime)) return;
        loadNewCamParams = true;
    } else {
        // If time is set by running ANIMATION
        if (this->playAnim) {
            clock_t tmpTime = clock();
            clock_t cTime = tmpTime - this->deltaAnimTime;
            this->deltaAnimTime = tmpTime;

            float animTime = ccc->getSelectedKeyframe().GetAnimTime() + ((float)cTime) / (float)(CLOCKS_PER_SEC);
            if ((animTime < 0.0f) ||
                (animTime > ccc->getTotalAnimTime())) { // Reset time if max animation time is reached
                animTime = 0.0f;
            }
            ccc->setSelectedKeyframeTime(animTime);
            if (!(*ccc)(CallKeyframeKeeper::CallForGetSelectedKeyframeAtTime)) return;
            loadNewCamParams = true;
        }
    }
    // Load simulation time
    this->setSimTime(ccc->getSelectedKeyframe().GetSimTime());

    // Viewport stuff ---------------------------------------------------------
    int vp[4];
    glGetIntegerv(GL_VIEWPORT, vp);
    int vpW_int = (vp[2] - vp[0]);
    int vpH_int = (vp[3] - vp[1]) - static_cast<int>(CC_MENU_HEIGHT); // Reduced by menue height
    float vpH_flt = static_cast<float>(vpH_int);
    float vpW_flt = static_cast<float>(vpW_int);
    int fboWidth = vpW_int;
    int fboHeight = vpH_int;
    float cineRatio = static_cast<float>(this->cineWidth) / static_cast<float>(this->cineHeight);

    if (this->rendering) {
        fboWidth = this->cineWidth;
        fboHeight = this->cineHeight;
    } else {
        float vpRatio = vpW_flt / vpH_flt;

        // Check for viewport changes
        if ((this->vpWLast != vpW_int) || (this->vpHLast != vpH_int)) {
            this->vpWLast = vpW_int;
            this->vpHLast = vpH_int;
        }

        // Calculate reduced fbo width and height
        if ((this->cineWidth < vpW_int) && (this->cineHeight < vpH_int)) {
            fboWidth = this->cineWidth;
            fboHeight = this->cineHeight;
        } else {
            fboWidth = vpW_int;
            fboHeight = vpH_int;

            if (cineRatio > vpRatio) {
                fboHeight = (static_cast<int>(vpW_flt / cineRatio));
            } else if (cineRatio < vpRatio) {
                fboWidth = (static_cast<int>(vpH_flt * cineRatio));
            }
        }
    }

    // Camera settings --------------------------------------------------------

    // Set new viewport settings for camera
    this->cam.Parameters()->SetVirtualViewSize(static_cast<vislib::graphics::ImageSpaceType>(fboWidth),
        static_cast<vislib::graphics::ImageSpaceType>(fboHeight));

    // Set camera parameters of selected keyframe for this view.
    // But only if selected keyframe differs to last locally stored and shown keyframe.
    // Load new camera setting from selected keyframe when skybox side changes or rendering
    // or animation loaded new slected keyframe.
    Keyframe skf = ccc->getSelectedKeyframe();
    if (loadNewCamParams || (this->shownKeyframe != skf)) {

        this->shownKeyframe = skf;

        // Apply selected keyframe parameters only, if at least one valid keyframe exists.
        if (!ccc->getKeyframes()->IsEmpty()) {
            this->cam.Parameters()->SetView(skf.GetCamPosition(), skf.GetCamLookAt(), skf.GetCamUp());
            this->cam.Parameters()->SetApertureAngle(skf.GetCamApertureAngle());
        } else {
            this->ResetView();
        }
        // Apply showing skybox side ONLY if new camera parameters are set
        if (this->sbSide != CinematicView::SkyboxSides::SKYBOX_NONE) {
            // Get camera parameters
            vislib::SmartPtr<vislib::graphics::CameraParameters> cp = this->cam.Parameters();
            vislib::math::Point<float, 3> camPos = cp->Position();
            vislib::math::Vector<float, 3> camRight = cp->Right();
            vislib::math::Vector<float, 3> camUp = cp->Up();
            vislib::math::Vector<float, 3> camFront = cp->Front();
            vislib::math::Point<float, 3> camLookAt = cp->LookAt();
            float tmpDist = cp->FocalDistance();

            // Adjust cam to selected skybox side
            // set aperture angle to 90 deg
            cp->SetApertureAngle(90.0f);
            if (!this->cubeModeRenderParam.Param<param::BoolParam>()->Value()) {
                if (this->sbSide == CinematicView::SkyboxSides::SKYBOX_BACK) {
                    cp->SetView(camPos, camPos - camFront * tmpDist, camUp);
                } else if (this->sbSide == CinematicView::SkyboxSides::SKYBOX_RIGHT) {
                    cp->SetView(camPos, camPos + camRight * tmpDist, camUp);
                } else if (this->sbSide == CinematicView::SkyboxSides::SKYBOX_LEFT) {
                    cp->SetView(camPos, camPos - camRight * tmpDist, camUp);
                } else if (this->sbSide == CinematicView::SkyboxSides::SKYBOX_UP) {
                    cp->SetView(camPos, camPos + camUp * tmpDist, -camFront);
                } else if (this->sbSide == CinematicView::SkyboxSides::SKYBOX_DOWN) {
                    cp->SetView(camPos, camPos - camUp * tmpDist, camFront);
                }
            } else {
                auto const center = cr3d->AccessBoundingBoxes().WorldSpaceBBox().CalcCenter();
                auto const width = cr3d->AccessBoundingBoxes().WorldSpaceBBox().Width();
                auto const height = cr3d->AccessBoundingBoxes().WorldSpaceBBox().Height();
                auto const depth = cr3d->AccessBoundingBoxes().WorldSpaceBBox().Depth();
                if (this->sbSide == CinematicView::SkyboxSides::SKYBOX_FRONT) {
                    auto const tmpCamPos = center - 0.5f * (depth + width) * camFront;
                    cp->SetView(tmpCamPos, tmpCamPos + camFront * 0.5f * (depth + width), camUp);
                } else if (this->sbSide == CinematicView::SkyboxSides::SKYBOX_BACK) {
                    auto const tmpCamPos = center + 0.5f * (depth + width) * camFront;
                    cp->SetView(tmpCamPos, tmpCamPos - camFront * 0.5f * (depth + width), camUp);
                } else if (this->sbSide == CinematicView::SkyboxSides::SKYBOX_RIGHT) {
                    auto const tmpCamPos = center + 0.5f * (depth + width) * camRight;
                    cp->SetView(tmpCamPos, tmpCamPos - camRight * 0.5f * (depth + width), camUp);
                } else if (this->sbSide == CinematicView::SkyboxSides::SKYBOX_LEFT) {
                    auto const tmpCamPos = center - 0.5f * (depth + width) * camRight;
                    cp->SetView(tmpCamPos, tmpCamPos + camRight * 0.5f * (depth + width), camUp);
                } else if (this->sbSide == CinematicView::SkyboxSides::SKYBOX_UP) {
                    auto const tmpCamPos = center + 0.5f * (height + width) * camUp;
                    cp->SetView(tmpCamPos, tmpCamPos - camUp * 0.5f * (height + width), -camFront);
                } else if (this->sbSide == CinematicView::SkyboxSides::SKYBOX_DOWN) {
                    auto const tmpCamPos = center - 0.5f * (height + width) * camUp;
                    cp->SetView(tmpCamPos, tmpCamPos + camUp * 0.5f * (height + width), camFront);
                }
            }
        }
    }

    // Propagate camera parameters to keyframe keeper (sky box camera params are propageted too!)
    ccc->setCameraParameters(this->cam.Parameters());
    if (!(*ccc)(CallKeyframeKeeper::CallForSetCameraForKeyframe)) return;

        // Render to texture ------------------------------------------------------------

        // Suppress TRACE output of fbo.Enable() and fbo.Create()
#if defined(DEBUG) || defined(_DEBUG)
    unsigned int otl = vislib::Trace::GetInstance().GetLevel();
    vislib::Trace::GetInstance().SetLevel(0);
#endif // DEBUG || _DEBUG

    if (this->fbo.IsValid()) {
        if ((this->fbo.GetWidth() != fboWidth) || (this->fbo.GetHeight() != fboHeight)) {
            this->fbo.Release();
        }
    }
    if (!this->fbo.IsValid()) {
        if (!this->fbo.Create(fboWidth, fboHeight, GL_RGBA8, GL_RGBA, GL_UNSIGNED_BYTE,
                vislib::graphics::gl::FramebufferObject::ATTACHMENT_TEXTURE, GL_DEPTH_COMPONENT24)) {
            throw vislib::Exception(
                "[CINEMATIC VIEW] [render] Unable to create image framebuffer object.", __FILE__, __LINE__);
            return;
        }
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
    cr3d->SetOutputBuffer(&this->fbo);
    Base::overrideCall = cr3d;

    // Set override viewport of view (otherwise viewport is overwritten in Base::Render(context))
    int fboVp[4] = {0, 0, fboWidth, fboHeight};
    Base::overrideViewport = fboVp;

    // Call Render-Function of parent View3D
    Base::Render(context);

    // Reset override render call
    Base::overrideCall = nullptr;
    // Reset override viewport
    Base::overrideViewport = nullptr;

    if (this->fbo.IsEnabled()) {
        this->fbo.Disable();
    }

    // Write frame to file
    if (this->rendering) {

        // Check if fbo in cr3d was reset by renderer
        this->pngdata.write_lock = ((cr3d->FrameBufferObject() != nullptr) ? (0) : (1));
        if (this->pngdata.write_lock > 0) {
            // vislib::sys::Log::DefaultLog.WriteInfo("[CINEMATIC RENDERING] Received empty FBO ...\n");
        }

        // Lock writing frame to file for specific tim
        std::chrono::duration<double> diff = (std::chrono::system_clock::now() - this->pngdata.start_time);
        if (diff.count() < static_cast<double>(this->delayFirstRenderFrameParam.Param<param::FloatParam>()->Value())) {
            this->pngdata.write_lock = 1;
        }

        // Write frame to PNG file
        this->render2file_write_png();
    }

    // Draw final image -------------------------------------------------------
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
    glDisable(GL_LIGHTING);
    glDisable(GL_CULL_FACE);

    glDisable(GL_BLEND);

    glEnable(GL_DEPTH_TEST);
    glDisable(GL_DEPTH_TEST);

    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glLoadIdentity();
    // Reset viewport
    glViewport(vp[0], vp[1], vp[2], vp[3]);
    glOrtho(0.0f, vpW_flt, 0.0f, (vpH_flt + (CC_MENU_HEIGHT)), -1.0,
        1.0); // Reset to true viewport size including menue height

    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glLoadIdentity();

    // Background color
    const float* bgColor = Base::BkgndColour();
    // COLORS
    float lbColor[4] = {1.0f, 1.0f, 1.0f, 1.0f};
    float white[4]   = {1.0f, 1.0f, 1.0f, 1.0f};
    float yellow[4]  = {1.0f, 1.0f, 0.0f, 1.0f};
    float menu[4]    = {0.0f, 0.0f, 0.3f, 1.0f};
    // Adapt colors depending on lightness
    float L = (vislib::math::Max(bgColor[0], vislib::math::Max(bgColor[1], bgColor[2])) +
                  vislib::math::Min(bgColor[0], vislib::math::Min(bgColor[1], bgColor[2]))) /
              2.0f;
    if (L > 0.5f) {
        for (unsigned int i = 0; i < 3; i++) {
            lbColor[i] = 0.0f;
        }
    }
    float fgColor[4] = {1.0f, 1.0f, 1.0f, 1.0f};
    for (unsigned int i = 0; i < 3; i++) {
        fgColor[i] -= lbColor[i];
    }

    // Adjust fbo viewport size if it is greater than viewport
    int fbovpW_int = fboWidth;
    int fbovpH_int = fboHeight;
    if ((fbovpW_int > vpW_int) || ((fbovpW_int < vpW_int) && (fbovpH_int < vpH_int))) {
        fbovpW_int = vpW_int;
        fbovpH_int = static_cast<int>(vpW_flt / cineRatio);
    }
    if ((fbovpH_int > vpH_int) || ((fbovpW_int < vpW_int) && (fbovpH_int < vpH_int))) {
        fbovpH_int = vpH_int;
        fbovpW_int = static_cast<int>(vpH_flt * cineRatio);
    }
    float right = (vpW_int + static_cast<float>(fbovpW_int)) / 2.0f;
    float left = (vpW_int - static_cast<float>(fbovpW_int)) / 2.0f;
    float bottom = (vpH_int - static_cast<float>(fbovpH_int)) / 2.0f;
    float up = (vpH_int + static_cast<float>(fbovpH_int)) / 2.0f;

    // Draw texture
    glActiveTexture(GL_TEXTURE0);
    glEnable(GL_TEXTURE_2D);
    this->fbo.BindColourTexture();
    glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glBegin(GL_QUADS);
        glTexCoord2f(0.0f, 0.0f);
        glVertex3f(left, bottom, 0.0f);
        glTexCoord2f(1.0f, 0.0f);
        glVertex3f(right, bottom, 0.0f);
        glTexCoord2f(1.0f, 1.0f);
        glVertex3f(right, up, 0.0f);
        glTexCoord2f(0.0f, 1.0f);
        glVertex3f(left, up, 0.0f);
    glEnd();
    glBindTexture(GL_TEXTURE_2D, 0);
    glDisable(GL_TEXTURE_2D);

    // Draw letter box  -------------------------------------------------------
    glDisable(GL_DEPTH_TEST);

    // Calculate position of texture
    int x = 0;
    int y = 0;
    if (fbovpW_int < vpW_int) {
        x = (static_cast<int>((vpW_int - static_cast<float>(fbovpW_int)) / 2.0f));
        y = vpH_int;
    } else if (fbovpH_int < vpH_int) {
        x = vpW_int;
        y = (static_cast<int>((vpH_int - static_cast<float>(fbovpH_int)) / 2.0f));
    }
    // Draw letter box quads in letter box Color
    glColor4fv(lbColor);
    glBegin(GL_QUADS);
        glVertex2i(0, 0);
        glVertex2i(x, 0);
        glVertex2i(x, y);
        glVertex2i(0, y);
        glVertex2i(vpW_int, vpH_int);
        glVertex2i(vpW_int - x, vpH_int);
        glVertex2i(vpW_int - x, vpH_int - y);
        glVertex2i(vpW_int, vpH_int - y);
    glEnd();

    // DRAW MENU --------------------------------------------------------------
    if (!this->theFont.Initialise(this->GetCoreInstance())) {
        return;
    }

    vislib::StringA leftLabel = " CINEMATIC ";
    vislib::StringA midLabel = "";
    if (this->rendering) {
        midLabel = "! RENDERING IN PROGRESS !";
    } else if (this->playAnim) {
        midLabel = " Playing Animation ";
    }
    vislib::StringA rightLabel = "";

    float lbFontSize = (CC_MENU_HEIGHT);
    float leftLabelWidth = this->theFont.LineWidth(lbFontSize, leftLabel);
    float midleftLabelWidth = this->theFont.LineWidth(lbFontSize, midLabel);
    float rightLabelWidth = this->theFont.LineWidth(lbFontSize, rightLabel);

    // Adapt font size if height of menu text is greater than menu height
    float vpWhalf = vpW_flt / 2.0f;
    while (((leftLabelWidth + midleftLabelWidth / 2.0f) > vpWhalf) ||
           ((rightLabelWidth + midleftLabelWidth / 2.0f) > vpWhalf)) {
        lbFontSize -= 0.5f;
        leftLabelWidth = this->theFont.LineWidth(lbFontSize, leftLabel);
        midleftLabelWidth = this->theFont.LineWidth(lbFontSize, midLabel);
        rightLabelWidth = this->theFont.LineWidth(lbFontSize, rightLabel);
    }

    // Draw menu background
    glColor4fv(menu);
    glBegin(GL_QUADS);
    glVertex2f(0.0f, vpH_flt + (CC_MENU_HEIGHT));
    glVertex2f(0.0f, vpH_flt);
    glVertex2f(vpW_flt, vpH_flt);
    glVertex2f(vpW_flt, vpH_flt + (CC_MENU_HEIGHT));
    glEnd();

    // Draw menu labels
    float labelPosY = vpH_flt + (CC_MENU_HEIGHT) / 2.0f + lbFontSize / 2.0f;
    this->theFont.DrawString(
        white, 0.0f, labelPosY, lbFontSize, false, leftLabel, megamol::core::utility::AbstractFont::ALIGN_LEFT_TOP);
    this->theFont.DrawString(yellow, (vpW_flt - midleftLabelWidth) / 2.0f, labelPosY, lbFontSize, false, midLabel,
        megamol::core::utility::AbstractFont::ALIGN_LEFT_TOP);
    this->theFont.DrawString(white, (vpW_flt - rightLabelWidth), labelPosY, lbFontSize, false, rightLabel,
        megamol::core::utility::AbstractFont::ALIGN_LEFT_TOP);

    // ------------------------------------------------------------------------
    glMatrixMode(GL_PROJECTION);
    glPopMatrix();
    glMatrixMode(GL_MODELVIEW);
    glPopMatrix();

    // Reset opengl
    glDisable(GL_BLEND);
    glEnable(GL_DEPTH_TEST);
}


bool CinematicView::render2file_setup() {

    CallKeyframeKeeper* ccc = this->keyframeKeeperSlot.CallAs<CallKeyframeKeeper>();
    if (ccc == nullptr) return false;

    // init png data struct
    this->pngdata.bpp = 3;
    this->pngdata.width = static_cast<unsigned int>(this->cineWidth);
    this->pngdata.height = static_cast<unsigned int>(this->cineHeight);
    this->pngdata.buffer = nullptr;
    this->pngdata.ptr = nullptr;
    this->pngdata.infoptr = nullptr;
    this->pngdata.write_lock = 1;
    this->pngdata.start_time = std::chrono::system_clock::now();

    unsigned int startFrameCnt =
        static_cast<unsigned int>(this->startRenderFrameParam.Param<param::IntParam>()->Value());
    unsigned int maxFrameCnt = (unsigned int)(this->pngdata.animTime * (float)this->fps);
    if (startFrameCnt > maxFrameCnt) {
        startFrameCnt = maxFrameCnt;
        vislib::sys::Log::DefaultLog.WriteWarn(
            "[CINEMATIC VIEW] Max frame count %d exceeded: %d", maxFrameCnt, startFrameCnt);
    }
    this->pngdata.cnt = startFrameCnt;
    this->pngdata.animTime = (float)this->pngdata.cnt / (float)this->fps;

    // Calculate pre-decimal point positions for frame counter in filename
    this->pngdata.exp_frame_cnt = 1;
    float frameCnt = (float)(this->fps) * ccc->getTotalAnimTime();
    while (frameCnt > 1.0f) {
        frameCnt /= 10.0f;
        this->pngdata.exp_frame_cnt++;
    }

    // Creating new folder
    vislib::StringA frameFolder;
    time_t t = std::time(0); // get time now
    struct tm* now = nullptr;
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
    struct tm nowdata;
    now = &nowdata;
    localtime_s(now, &t);
#else  /* defined(_WIN32) && (_MSC_VER >= 1400) */
    now = localtime(&t);
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */
    frameFolder.Format("frames_%i%02i%02i-%02i%02i%02i_%02ifps", (now->tm_year + 1900), (now->tm_mon + 1), now->tm_mday,
        now->tm_hour, now->tm_min, now->tm_sec, this->fps);
    this->pngdata.path = static_cast<vislib::StringA>(this->frameFolderParam.Param<param::FilePathParam>()->Value());
    if (this->pngdata.path.IsEmpty()) {
        this->pngdata.path = vislib::sys::Path::GetCurrentDirectoryA();
        this->pngdata.path = vislib::sys::Path::Concatenate(this->pngdata.path, frameFolder);
    }
    vislib::sys::Path::MakeDirectory(this->pngdata.path);

    // Set current time stamp to file name
    this->pngdata.filename = "frames";

    // Create new byte buffer
    this->pngdata.buffer = new BYTE[this->pngdata.width * this->pngdata.height * this->pngdata.bpp];
    if (this->pngdata.buffer == nullptr) {
        throw vislib::Exception(
            "[CINEMATIC VIEW] [startAnimRendering] Cannot allocate image buffer.", __FILE__, __LINE__);
    }

    // Disable showing BBOX and CUBE (Uniform backCol is needed for being able to detect changes written to fbo.)
    Base::showViewCubeSlot.Param<param::BoolParam>()->SetValue(false);
    Base::showBBox.Param<param::BoolParam>()->SetValue(false);
    vislib::sys::Log::DefaultLog.WriteInfo("[CINEMATIC VIEW] Hiding Bbox and Cube rendering of complete animation.");

    vislib::sys::Log::DefaultLog.WriteInfo("[CINEMATIC VIEW] STARTED rendering of complete animation.");

    return true;
}


bool CinematicView::render2file_write_png() {

    if (this->pngdata.write_lock == 0) {

        CallKeyframeKeeper* ccc = this->keyframeKeeperSlot.CallAs<CallKeyframeKeeper>();
        if (ccc == nullptr) return false;

        vislib::StringA tmpFilename, tmpStr;
        tmpStr.Format(".%i", this->pngdata.exp_frame_cnt);
        tmpStr.Prepend("%0");
        tmpStr.Append("i.png");
        tmpFilename.Format(tmpStr.PeekBuffer(), this->pngdata.cnt);
        if (this->sbSide != CinematicView::SKYBOX_NONE &&
            this->addSBSideToNameParam.Param<core::param::BoolParam>()->Value()) {
            if (this->sbSide == CinematicView::SkyboxSides::SKYBOX_FRONT) {
                tmpFilename.Prepend("_front.");
            } else if (this->sbSide == CinematicView::SkyboxSides::SKYBOX_BACK) {
                tmpFilename.Prepend("_back.");
            } else if (this->sbSide == CinematicView::SkyboxSides::SKYBOX_RIGHT) {
                tmpFilename.Prepend("_right.");
            } else if (this->sbSide == CinematicView::SkyboxSides::SKYBOX_LEFT) {
                tmpFilename.Prepend("_left.");
            } else if (this->sbSide == CinematicView::SkyboxSides::SKYBOX_UP) {
                tmpFilename.Prepend("_up.");
            } else if (this->sbSide == CinematicView::SkyboxSides::SKYBOX_DOWN) {
                tmpFilename.Prepend("_down.");
            }
        }
        tmpFilename.Prepend(this->pngdata.filename);

        // open final image file
        if (!this->pngdata.file.Open(vislib::sys::Path::Concatenate(this->pngdata.path, tmpFilename),
                vislib::sys::File::WRITE_ONLY, vislib::sys::File::SHARE_EXCLUSIVE,
                vislib::sys::File::CREATE_OVERWRITE)) {
            throw vislib::Exception(
                "[CINEMATIC VIEW] [startAnimRendering] Cannot open output file", __FILE__, __LINE__);
        }

        // init png lib
        this->pngdata.ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, nullptr, &this->pngError, &this->pngWarn);
        if (this->pngdata.ptr == nullptr) {
            throw vislib::Exception(
                "[CINEMATIC VIEW] [startAnimRendering] Cannot create png structure", __FILE__, __LINE__);
        }
        this->pngdata.infoptr = png_create_info_struct(this->pngdata.ptr);
        if (this->pngdata.infoptr == nullptr) {
            throw vislib::Exception("[CINEMATIC VIEW] [startAnimRendering] Cannot create png info", __FILE__, __LINE__);
        }
        png_set_write_fn(this->pngdata.ptr, static_cast<void*>(&this->pngdata.file), &this->pngWrite, &this->pngFlush);
        png_set_IHDR(this->pngdata.ptr, this->pngdata.infoptr, this->pngdata.width, this->pngdata.height, 8,
            PNG_COLOR_TYPE_RGB, PNG_INTERLACE_NONE, PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);

        if (fbo.GetColourTexture(this->pngdata.buffer, 0, GL_RGB, GL_UNSIGNED_BYTE) != GL_NO_ERROR) {
            throw vislib::Exception(
                "[CINEMATIC VIEW] [writeTextureToPng] Failed to create Screenshot: Cannot read image data", __FILE__,
                __LINE__);
        }

        BYTE** rows = nullptr;
        try {
            rows = new BYTE*[this->cineHeight];
            for (UINT i = 0; i < this->pngdata.height; i++) {
                rows[this->pngdata.height - (1 + i)] =
                    this->pngdata.buffer + this->pngdata.bpp * i * this->pngdata.width;
            }
            png_set_rows(this->pngdata.ptr, this->pngdata.infoptr, rows);

            png_write_png(this->pngdata.ptr, this->pngdata.infoptr, PNG_TRANSFORM_IDENTITY, nullptr);

            ARY_SAFE_DELETE(rows);
        } catch (...) {
            if (rows != nullptr) {
                ARY_SAFE_DELETE(rows);
            }
            throw;
        }

        if (this->pngdata.ptr != nullptr) {
            if (this->pngdata.infoptr != nullptr) {
                png_destroy_write_struct(&this->pngdata.ptr, &this->pngdata.infoptr);
            } else {
                png_destroy_write_struct(&this->pngdata.ptr, (png_infopp) nullptr);
            }
        }

        try {
            this->pngdata.file.Flush();
        } catch (...) {
        }
        try {
            this->pngdata.file.Close();
        } catch (...) {
        }

        vislib::sys::Log::DefaultLog.WriteWarn(
            "[CINEMATIC VIEW] [render2file_write_png] Wrote png file %d for animation time %f ...\n", this->pngdata.cnt,
            this->pngdata.animTime);

        // --------------------------------------------------------------------

        // Increase to next time step
        float fpsFrac = (1.0f / static_cast<float>(this->fps));
        this->pngdata.animTime += fpsFrac;

        // Fit animTime to exact full seconds (removing rounding error)
        if (std::abs(this->pngdata.animTime - std::round(this->pngdata.animTime)) < (fpsFrac / 2.0)) {
            this->pngdata.animTime = std::round(this->pngdata.animTime);
        }

        if (this->pngdata.animTime == ccc->getTotalAnimTime()) {
            /// Handling this case is actually only necessary when rendering is done via FBOCompositor => Rndering
            /// crashes > WHY ? Rendering last frame with animation time = total animation time is otherwise no problem.
            this->pngdata.animTime -= 0.000005f;
        } else if (this->pngdata.animTime >= ccc->getTotalAnimTime()) {
            // Stop rendering if max anim time is reached
            this->render2file_finish();
            return false;
        }

        this->pngdata.cnt++;
    }

    return true;
}


bool CinematicView::render2file_finish() {

    if (this->pngdata.ptr != nullptr) {
        if (this->pngdata.infoptr != nullptr) {
            png_destroy_write_struct(&this->pngdata.ptr, &this->pngdata.infoptr);
        } else {
            png_destroy_write_struct(&this->pngdata.ptr, (png_infopp) nullptr);
        }
    }

    try {
        this->pngdata.file.Flush();
    } catch (...) {
    }
    try {
        this->pngdata.file.Close();
    } catch (...) {
    }

    ARY_SAFE_DELETE(this->pngdata.buffer);

    this->rendering = false;

    vislib::sys::Log::DefaultLog.WriteInfo("[CINEMATIC VIEW] STOPPED rendering.");
    return true;
}


bool CinematicView::setSimTime(float st) {

    view::CallRender3D* cr3d = this->rendererSlot.CallAs<core::view::CallRender3D>();
    if (cr3d == nullptr) {
        vislib::sys::Log::DefaultLog.WriteError("[CINEMATIC VIEW] [setSimTime] cr3d is nullptr.");
        return false;
    }

    float simTime = st;

    param::ParamSlot* animTimeParam = static_cast<param::ParamSlot*>(this->timeCtrl.GetSlot(2));
    animTimeParam->Param<param::FloatParam>()->SetValue(simTime * static_cast<float>(cr3d->TimeFramesCount()), true);

    return true;
}
