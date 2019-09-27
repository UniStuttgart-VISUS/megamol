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


CinematicView::CinematicView(void) : View3D_2()
    , keyframeKeeperSlot("keyframeKeeper", "Connects to the Keyframe Keeper.")
    , renderParam("renderAnim", "Toggle rendering of complete animation to PNG files.")
    , toggleAnimPlayParam("playPreview", "Toggle playing animation as preview")
    , selectedSkyboxSideParam("skyboxSide", "Select the skybox side.")
    , cubeModeRenderParam("cubeMode", "Render cube around dataset with skyboxSide as side selector.")
    , resWidthParam("cinematicWidth", "The width resolution of the cineamtic view to render.")
    , resHeightParam("cinematicHeight", "The height resolution of the cineamtic view to render.")
    , fpsParam("fps", "Frames per second the animation should be rendered.")
    , startRenderFrameParam("firstRenderFrame", "Set first frame number to start rendering with (allows continuing aborted rendering without starting from the beginning).")
    , delayFirstRenderFrameParam("delayFirstRenderFrame", "Delay (in seconds) to wait until first frame for rendering is written (needed to get right first frame especially for high resolutions and for distributed rendering).")
    , frameFolderParam( "frameFolder", "Specify folder where the frame files should be stored.")
    , addSBSideToNameParam( "addSBSideToName", "Toggle whether skybox side should be added to output filename")
    , eyeParam("stereo::eye", "Select eye position (for stereo view).")
    , projectionParam("stereo::projection", "Select camera projection.")
    , fbo()
    , png_data()
    , utils()
    , deltaAnimTime(clock())
    , shownKeyframe()
    , playAnim(false)
    , cineWidth(1920)
    , cineHeight(1080)
    , vpWLast(0)
    , vpHLast(0)
    , sbSide(CinematicView::SkyboxSides::SKYBOX_NONE)
    , rendering(false)
    , fps(24) {

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

    if (this->png_data.ptr != nullptr) {
        if (this->png_data.infoptr != nullptr) {
            png_destroy_write_struct(&this->png_data.ptr, &this->png_data.infoptr);
        } else {
            png_destroy_write_struct(&this->png_data.ptr, (png_infopp) nullptr);
        }
    }

    try {
        this->png_data.file.Flush();
    } catch (...) {
    }
    try {
        this->png_data.file.Close();
    } catch (...) {
    }

    ARY_SAFE_DELETE(this->png_data.buffer);

    if (this->png_data.buffer != nullptr) {
        vislib::sys::Log::DefaultLog.WriteError("[CINEMATIC VIEW] [render] pngdata.buffer is not nullptr.");
    }
    if (this->png_data.ptr != nullptr) {
        vislib::sys::Log::DefaultLog.WriteError("[CINEMATIC VIEW] [render] pngdata.ptr is not nullptr.");
    }
    if (this->png_data.infoptr != nullptr) {
        vislib::sys::Log::DefaultLog.WriteError("[CINEMATIC VIEW] [render] pngdata.infoptr is not nullptr.");
    }

    this->fbo.Release();
}


void CinematicView::Render(const mmcRenderViewContext& context) {

    auto cr3d = this->rendererSlot.CallAs<core::view::CallRender3D_2>();
    if (cr3d == nullptr) return;
    if (!(*cr3d)(view::AbstractCallRender::FnGetExtents)) return;

    // Get update data from keyframe keeper
    auto ccc = this->keyframeKeeperSlot.CallAs<CallKeyframeKeeper>();
    if (ccc == nullptr) return;
    if (!(*ccc)(CallKeyframeKeeper::CallForGetUpdatedKeyframeData)) return;
    ccc->setBboxCenter(P2G(cr3d->AccessBoundingBoxes().BoundingBox().CalcCenter()));
    ccc->setTotalSimTime(static_cast<float>(cr3d->TimeFramesCount()));
    ccc->setFps(this->fps);
    if (!(*ccc)(CallKeyframeKeeper::CallForSetSimulationData)) return;

    // Initialise render utils once
    if (!this->utils.Initialized()) {
        if (!this->utils.Initialise(this->GetCoreInstance())) {
            vislib::sys::Log::DefaultLog.WriteError("[TRACKINGSHOT RENDERER] [create] Couldn't initialize render utils.");
            return;
        }
    }

    // Update parameters 
    bool loadNewCamParams = false;
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
            this->render2file_cleanup();
        }
    }
    // Set (mono/stereo) projection for camera
    if (this->projectionParam.IsDirty()) {
        this->projectionParam.ResetDirty();
        vislib::sys::Log::DefaultLog.WriteWarn("[CINEMATIC VIEW] Selecting Stereo Projection is currently not supported.");
        ///XXX this->cam.SetProjection(static_cast<vislib::graphics::CameraParameters::ProjectionType>(this->projectionParam.Param<param::EnumParam>()->Value()));
    }
    // Set eye position for camera
    if (this->eyeParam.IsDirty()) {
        this->eyeParam.ResetDirty();
        vislib::sys::Log::DefaultLog.WriteWarn("[CINEMATIC VIEW] Selecting Stereo Eye is currently not supported.");
        ///XXX this->cam.Parameters()->SetEye(static_cast<vislib::graphics::CameraParameters::StereoEye>(this->eyeParam.Param<param::EnumParam>()->Value()));
    }

    // Time settings ----------------------------------------------------------
    // Load animation time
    if (this->rendering) {
        if ((this->png_data.animTime < 0.0f) || (this->png_data.animTime > ccc->getTotalAnimTime())) {
            throw vislib::Exception("[CINEMATIC VIEW] Invalid animation time.", __FILE__, __LINE__);
        }
        ccc->setSelectedKeyframeTime(this->png_data.animTime);
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
    // Load current simulation time to parameter
    this->setSimulationTimeParameter(ccc->getSelectedKeyframe().GetSimTime());

    // Viewport ---------------------------------------------------------------
    int vp[4];
    glGetIntegerv(GL_VIEWPORT, vp);
    int vpW_int = (vp[2] - vp[0]);
    int vpH_int = (vp[3] - vp[1]) - static_cast<int>(CC_MENU_HEIGHT); // Reduced by menue height
    float vpW_flt = static_cast<float>(vpW_int);
    float vpH_flt = static_cast<float>(vpH_int);
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
    ///OLD this->cam.Parameters()->SetVirtualViewSize(static_cast<vislib::graphics::ImageSpaceType>(fboWidth), static_cast<vislib::graphics::ImageSpaceType>(fboHeight));
    auto res = cam_type::screen_size_type(glm::ivec2(fboWidth, fboHeight));
    this->cam.resolution_gate(res);
    auto tile = cam_type::screen_rectangle_type(std::array<int, 4>{0, 0, fboWidth, fboHeight});
    this->cam.image_tile(tile);
 
    // Set camera parameters of selected keyframe for this view.
    // But only if selected keyframe differs to last locally stored and shown keyframe.
    // Load new camera setting from selected keyframe when skybox side changes or rendering
    // or animation loaded new slected keyframe.
    Keyframe skf = ccc->getSelectedKeyframe();
    if (loadNewCamParams || (this->shownKeyframe != skf)) {
        this->shownKeyframe = skf;

        // Apply selected keyframe parameters only, if at least one valid keyframe exists.
        if (!ccc->getKeyframes()->empty()) {
            ///OLD this->cam.Parameters()->SetView(skf.GetCamPosition(), skf.GetCamLookAt(), skf.GetCamUp());
            ///OLD this->cam.aperture_angle = skf.GetCamApertureAngle();
            ///---
            ///NEW this->cam.position() = skf.GetCamPosition();
            ///NEW this->cam.view_vector() = skf.GetCamLookAt();
            ///NEW this->cam.up_vector() = skf.GetCamUp();
            ///NEW this->cam.aperture_angle() = skf.GetCamApertureAngle();
        } else {
            this->ResetView();
        }

        // Apply showing skybox side ONLY if new camera parameters are set
        //if (this->sbSide != CinematicView::SkyboxSides::SKYBOX_NONE) {
        //    // Get camera parameters
        //    vislib::SmartPtr<vislib::graphics::CameraParameters> cp = this->cam.Parameters();
        //    vislib::math::Point<float, 3> camPos = cp->Position();
        //    vislib::math::Vector<float, 3> camRight = cp->Right();
        //    vislib::math::Vector<float, 3> camUp = cp->Up();
        //    vislib::math::Vector<float, 3> camFront = cp->Front();
        //    vislib::math::Point<float, 3> camLookAt = cp->LookAt();
        //    float tmpDist = cp->FocalDistance();

        //    // Adjust cam to selected skybox side
        //    // set aperture angle to 90 deg
        //    cp->SetApertureAngle(90.0f);
        //    if (!this->cubeModeRenderParam.Param<param::BoolParam>()->Value()) {
        //        if (this->sbSide == CinematicView::SkyboxSides::SKYBOX_BACK) {
        //            cp->SetView(camPos, camPos - camFront * tmpDist, camUp);
        //        } else if (this->sbSide == CinematicView::SkyboxSides::SKYBOX_RIGHT) {
        //            cp->SetView(camPos, camPos + camRight * tmpDist, camUp);
        //        } else if (this->sbSide == CinematicView::SkyboxSides::SKYBOX_LEFT) {
        //            cp->SetView(camPos, camPos - camRight * tmpDist, camUp);
        //        } else if (this->sbSide == CinematicView::SkyboxSides::SKYBOX_UP) {
        //            cp->SetView(camPos, camPos + camUp * tmpDist, -camFront);
        //        } else if (this->sbSide == CinematicView::SkyboxSides::SKYBOX_DOWN) {
        //            cp->SetView(camPos, camPos - camUp * tmpDist, camFront);
        //        }
        //    } else {
        //        auto const center = cr3d->AccessBoundingBoxes().BoundingBox().CalcCenter();
        //        auto const width = cr3d->AccessBoundingBoxes().BoundingBox().Width();
        //        auto const height = cr3d->AccessBoundingBoxes().BoundingBox().Height();
        //        auto const depth = cr3d->AccessBoundingBoxes().BoundingBox().Depth();
        //        if (this->sbSide == CinematicView::SkyboxSides::SKYBOX_FRONT) {
        //            auto const tmpCamPos = center - 0.5f * (depth + width) * camFront;
        //            cp->SetView(tmpCamPos, tmpCamPos + camFront * 0.5f * (depth + width), camUp);
        //        } else if (this->sbSide == CinematicView::SkyboxSides::SKYBOX_BACK) {
        //            auto const tmpCamPos = center + 0.5f * (depth + width) * camFront;
        //            cp->SetView(tmpCamPos, tmpCamPos - camFront * 0.5f * (depth + width), camUp);
        //        } else if (this->sbSide == CinematicView::SkyboxSides::SKYBOX_RIGHT) {
        //            auto const tmpCamPos = center + 0.5f * (depth + width) * camRight;
        //            cp->SetView(tmpCamPos, tmpCamPos - camRight * 0.5f * (depth + width), camUp);
        //        } else if (this->sbSide == CinematicView::SkyboxSides::SKYBOX_LEFT) {
        //            auto const tmpCamPos = center - 0.5f * (depth + width) * camRight;
        //            cp->SetView(tmpCamPos, tmpCamPos + camRight * 0.5f * (depth + width), camUp);
        //        } else if (this->sbSide == CinematicView::SkyboxSides::SKYBOX_UP) {
        //            auto const tmpCamPos = center + 0.5f * (height + width) * camUp;
        //            cp->SetView(tmpCamPos, tmpCamPos - camUp * 0.5f * (height + width), -camFront);
        //        } else if (this->sbSide == CinematicView::SkyboxSides::SKYBOX_DOWN) {
        //            auto const tmpCamPos = center - 0.5f * (height + width) * camUp;
        //            cp->SetView(tmpCamPos, tmpCamPos + camUp * 0.5f * (height + width), camFront);
        //        }
        //    }
        //}
    }

    // Propagate camera parameters to keyframe keeper (sky box camera params are propageted too!)
    auto cam_ptr = std::make_shared<megamol::core::view::Camera_2>(this->cam);
    ccc->setCameraParameters(cam_ptr);
    if (!(*ccc)(CallKeyframeKeeper::CallForSetCameraForKeyframe)) return;

    // Render to texture ------------------------------------------------------------

/// Suppress TRACE output of fbo.Enable() and fbo.Create()
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
                "[CINEMATIC VIEW] [Render] Unable to create image framebuffer object.", __FILE__, __LINE__);
            return;
        }
    }
    if (this->fbo.Enable() != GL_NO_ERROR) {
        throw vislib::Exception("[CINEMATIC VIEW] [Render] Cannot enable Framebuffer object.", __FILE__, __LINE__);
        return;
    }

/// Reset TRACE output level
#if defined(DEBUG) || defined(_DEBUG)
    vislib::Trace::GetInstance().SetLevel(otl);
#endif // DEBUG || _DEBUG

    // Set output buffer for override call (otherwise render call is overwritten in Base::Render(context))
    cr3d->SetOutputBuffer(&this->fbo);
    Base::overrideCall = cr3d;

    // Set override viewport of view (otherwise viewport is overwritten in Base::Render(context))
    int fboVp[4] = {0, 0, fboWidth, fboHeight};
    Base::overrideViewport = fboVp;

    // Call Render-Function of parent View3D_2
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
        this->png_data.write_lock = ((cr3d->FrameBufferObject() != nullptr) ? (0) : (1));
        if (this->png_data.write_lock > 0) {
            // vislib::sys::Log::DefaultLog.WriteInfo("[CINEMATIC RENDERING] Received empty FBO ...\n");
        }
        // Lock writing frame to file for specific tim
        std::chrono::duration<double> diff = (std::chrono::system_clock::now() - this->png_data.start_time);
        if (diff.count() < static_cast<double>(this->delayFirstRenderFrameParam.Param<param::FloatParam>()->Value())) {
            this->png_data.write_lock = 1;
        }
        this->render2file_write();
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
    //this->fbo.GetColourTextureID();
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

    // Push menu --------------------------------------------------------------

    std::string leftLabel = " CINEMATIC ";
    std::string midLabel = "";
    if (this->rendering) {
        midLabel = " ! RENDERING IN PROGRESS ! ";
    } else if (this->playAnim) {
        midLabel = " Playing Animation ";
    }
    std::string rightLabel = "";
    //this->utils.PushMenu(leftLabel, midLabel, rightLabel, vpW_flt, vpH_flt);

    //float lbFontSize = (CC_MENU_HEIGHT);
    //float leftLabelWidth = this->theFont.LineWidth(lbFontSize, leftLabel);
    //float midleftLabelWidth = this->theFont.LineWidth(lbFontSize, midLabel);
    //float rightLabelWidth = this->theFont.LineWidth(lbFontSize, rightLabel);

    //// Adapt font size if height of menu text is greater than menu height
    //float vpWhalf = vpW_flt / 2.0f;
    //while (((leftLabelWidth + midleftLabelWidth / 2.0f) > vpWhalf) ||
    //       ((rightLabelWidth + midleftLabelWidth / 2.0f) > vpWhalf)) {
    //    lbFontSize -= 0.5f;
    //    leftLabelWidth = this->theFont.LineWidth(lbFontSize, leftLabel);
    //    midleftLabelWidth = this->theFont.LineWidth(lbFontSize, midLabel);
    //    rightLabelWidth = this->theFont.LineWidth(lbFontSize, rightLabel);
    //}

    //// Draw menu background
    //glColor4fv(menu);
    //glBegin(GL_QUADS);
    //glVertex2f(0.0f, vpH_flt + (CC_MENU_HEIGHT));
    //glVertex2f(0.0f, vpH_flt);
    //glVertex2f(vpW_flt, vpH_flt);
    //glVertex2f(vpW_flt, vpH_flt + (CC_MENU_HEIGHT));
    //glEnd();

    //// Draw menu labels
    //float labelPosY = vpH_flt + (CC_MENU_HEIGHT) / 2.0f + lbFontSize / 2.0f;
    //this->theFont.DrawString(
    //    white, 0.0f, labelPosY, lbFontSize, false, leftLabel, megamol::core::utility::AbstractFont::ALIGN_LEFT_TOP);
    //this->theFont.DrawString(yellow, (vpW_flt - midleftLabelWidth) / 2.0f, labelPosY, lbFontSize, false, midLabel,
    //    megamol::core::utility::AbstractFont::ALIGN_LEFT_TOP);
    //this->theFont.DrawString(white, (vpW_flt - rightLabelWidth), labelPosY, lbFontSize, false, rightLabel,
    //    megamol::core::utility::AbstractFont::ALIGN_LEFT_TOP);

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

    auto ccc = this->keyframeKeeperSlot.CallAs<CallKeyframeKeeper>();
    if (ccc == nullptr) return false;

    // init png data struct
    this->png_data.bpp = 3;
    this->png_data.width = static_cast<unsigned int>(this->cineWidth);
    this->png_data.height = static_cast<unsigned int>(this->cineHeight);
    this->png_data.buffer = nullptr;
    this->png_data.ptr = nullptr;
    this->png_data.infoptr = nullptr;
    this->png_data.write_lock = 1;
    this->png_data.start_time = std::chrono::system_clock::now();

    unsigned int startFrameCnt =
        static_cast<unsigned int>(this->startRenderFrameParam.Param<param::IntParam>()->Value());
    unsigned int maxFrameCnt = (unsigned int)(this->png_data.animTime * (float)this->fps);
    if (startFrameCnt > maxFrameCnt) {
        startFrameCnt = maxFrameCnt;
        vislib::sys::Log::DefaultLog.WriteWarn(
            "[CINEMATIC VIEW] [render2file_setup] Max frame count %d exceeded: %d", maxFrameCnt, startFrameCnt);
    }
    this->png_data.cnt = startFrameCnt;
    this->png_data.animTime = (float)this->png_data.cnt / (float)this->fps;

    // Calculate pre-decimal point positions for frame counter in filename
    this->png_data.exp_frame_cnt = 1;
    float frameCnt = (float)(this->fps) * ccc->getTotalAnimTime();
    while (frameCnt > 1.0f) {
        frameCnt /= 10.0f;
        this->png_data.exp_frame_cnt++;
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
    this->png_data.path = static_cast<vislib::StringA>(this->frameFolderParam.Param<param::FilePathParam>()->Value());
    if (this->png_data.path.IsEmpty()) {
        this->png_data.path = vislib::sys::Path::GetCurrentDirectoryA();
        this->png_data.path = vislib::sys::Path::Concatenate(this->png_data.path, frameFolder);
    }
    vislib::sys::Path::MakeDirectory(this->png_data.path);

    // Set current time stamp to file name
    this->png_data.filename = "frames";

    // Create new byte buffer
    this->png_data.buffer = new BYTE[this->png_data.width * this->png_data.height * this->png_data.bpp];
    if (this->png_data.buffer == nullptr) {
        throw vislib::Exception(
            "[CINEMATIC VIEW] [render2file_setup] Cannot allocate image buffer.", __FILE__, __LINE__);
    }

    // Disable showing BBOX and CUBE (Uniform backCol is needed for being able to detect changes written to fbo.)
    ///XXX Base::showViewCubeSlot.Param<param::BoolParam>()->SetValue(false);
    ///XXX Base::showBBox.Param<param::BoolParam>()->SetValue(false);
    ///XXX vislib::sys::Log::DefaultLog.WriteInfo("[CINEMATIC VIEW] Hiding Bbox and Cube rendering of complete animation.");

    vislib::sys::Log::DefaultLog.WriteInfo("[CINEMATIC VIEW] STARTED rendering of complete animation.");

    return true;
}


bool CinematicView::render2file_write() {

    if (this->png_data.write_lock == 0) {
        auto ccc = this->keyframeKeeperSlot.CallAs<CallKeyframeKeeper>();
        if (ccc == nullptr) return false;

        vislib::StringA tmpFilename, tmpStr;
        tmpStr.Format(".%i", this->png_data.exp_frame_cnt);
        tmpStr.Prepend("%0");
        tmpStr.Append("i.png");
        tmpFilename.Format(tmpStr.PeekBuffer(), this->png_data.cnt);
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
        tmpFilename.Prepend(this->png_data.filename);

        // open final image file
        if (!this->png_data.file.Open(vislib::sys::Path::Concatenate(this->png_data.path, tmpFilename),
                vislib::sys::File::WRITE_ONLY, vislib::sys::File::SHARE_EXCLUSIVE,
                vislib::sys::File::CREATE_OVERWRITE)) {
            throw vislib::Exception(
                "[CINEMATIC VIEW] [startAnimRendering] Cannot open output file", __FILE__, __LINE__);
        }

        // init png lib
        this->png_data.ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, nullptr, &this->pngError, &this->pngWarn);
        if (this->png_data.ptr == nullptr) {
            throw vislib::Exception(
                "[CINEMATIC VIEW] [startAnimRendering] Cannot create png structure", __FILE__, __LINE__);
        }
        this->png_data.infoptr = png_create_info_struct(this->png_data.ptr);
        if (this->png_data.infoptr == nullptr) {
            throw vislib::Exception("[CINEMATIC VIEW] [startAnimRendering] Cannot create png info", __FILE__, __LINE__);
        }
        png_set_write_fn(this->png_data.ptr, static_cast<void*>(&this->png_data.file), &this->pngWrite, &this->pngFlush);
        png_set_IHDR(this->png_data.ptr, this->png_data.infoptr, this->png_data.width, this->png_data.height, 8,
            PNG_COLOR_TYPE_RGB, PNG_INTERLACE_NONE, PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);

        if (fbo.GetColourTexture(this->png_data.buffer, 0, GL_RGB, GL_UNSIGNED_BYTE) != GL_NO_ERROR) {
            throw vislib::Exception(
                "[CINEMATIC VIEW] [writeTextureToPng] Failed to create Screenshot: Cannot read image data", __FILE__,
                __LINE__);
        }

        BYTE** rows = nullptr;
        try {
            rows = new BYTE*[this->cineHeight];
            for (UINT i = 0; i < this->png_data.height; i++) {
                rows[this->png_data.height - (1 + i)] =
                    this->png_data.buffer + this->png_data.bpp * i * this->png_data.width;
            }
            png_set_rows(this->png_data.ptr, this->png_data.infoptr, rows);

            png_write_png(this->png_data.ptr, this->png_data.infoptr, PNG_TRANSFORM_IDENTITY, nullptr);

            ARY_SAFE_DELETE(rows);
        } catch (...) {
            if (rows != nullptr) {
                ARY_SAFE_DELETE(rows);
            }
            throw;
        }
        if (this->png_data.ptr != nullptr) {
            if (this->png_data.infoptr != nullptr) {
                png_destroy_write_struct(&this->png_data.ptr, &this->png_data.infoptr);
            } else {
                png_destroy_write_struct(&this->png_data.ptr, (png_infopp) nullptr);
            }
        }
        try {
            this->png_data.file.Flush();
        } catch (...) {
        }
        try {
            this->png_data.file.Close();
        } catch (...) {
        }
        vislib::sys::Log::DefaultLog.WriteWarn(
            "[CINEMATIC VIEW] [render2file_write] Wrote png file %d for animation time %f ...\n", this->png_data.cnt,
            this->png_data.animTime);

        // --------------------------------------------------------------------

        // Next time step
        float fpsFrac = (1.0f / static_cast<float>(this->fps));
        this->png_data.animTime += fpsFrac;

        // Fit animTime to exact full seconds (removing rounding error)
        if (std::abs(this->png_data.animTime - std::round(this->png_data.animTime)) < (fpsFrac / 2.0)) {
            this->png_data.animTime = std::round(this->png_data.animTime);
        }

        if (this->png_data.animTime == ccc->getTotalAnimTime()) {
            ///XXX Handling this case is actually only necessary when rendering is done via FBOCompositor 
            ///XXX => Rendering crashes (WHY?) Rendering last frame with animation time = total animation time is otherwise no problem.
            this->png_data.animTime -= 0.000005f;
        } else if (this->png_data.animTime >= ccc->getTotalAnimTime()) {
            this->render2file_cleanup();
            return false;
        }

        this->png_data.cnt++;
    }

    return true;
}


bool CinematicView::render2file_cleanup() {

    if (this->png_data.ptr != nullptr) {
        if (this->png_data.infoptr != nullptr) {
            png_destroy_write_struct(&this->png_data.ptr, &this->png_data.infoptr);
        } else {
            png_destroy_write_struct(&this->png_data.ptr, (png_infopp) nullptr);
        }
    }
    try {
        this->png_data.file.Flush();
    } catch (...) {
    }
    try {
        this->png_data.file.Close();
    } catch (...) {
    }
    ARY_SAFE_DELETE(this->png_data.buffer);
    this->rendering = false;
    vislib::sys::Log::DefaultLog.WriteInfo("[CINEMATIC VIEW] STOPPED rendering.");

    return true;
}


bool CinematicView::setSimulationTimeParameter(float st) {

    auto cr3d = this->rendererSlot.CallAs<core::view::CallRender3D_2>();
    if (cr3d == nullptr)  return false;

    float simTime = st;
    param::ParamSlot* animTimeParam = static_cast<param::ParamSlot*>(this->timeCtrl.GetSlot(2));
    animTimeParam->Param<param::FloatParam>()->SetValue(simTime * static_cast<float>(cr3d->TimeFramesCount()), true);

    return true;
}
