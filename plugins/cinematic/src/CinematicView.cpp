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
    , vp_lastw(0)
    , vp_lasth(0)
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

    this->render2file_cleanup();
    this->fbo.Release();
}


void CinematicView::Render(const mmcRenderViewContext& context) {

    auto cr3d = this->rendererSlot.CallAs<core::view::CallRender3D_2>();
    if (cr3d == nullptr) return;
    if (!(*cr3d)(view::AbstractCallRender::FnGetExtents)) return;

    // Get update data from keyframe keeper -----------------------------------
    auto ccc = this->keyframeKeeperSlot.CallAs<CallKeyframeKeeper>();
    if (ccc == nullptr) return;
    if (!(*ccc)(CallKeyframeKeeper::CallForGetUpdatedKeyframeData)) return;
    ccc->SetBboxCenter(P2G(cr3d->AccessBoundingBoxes().BoundingBox().CalcCenter()));
    ccc->SetTotalSimTime(static_cast<float>(cr3d->TimeFramesCount()));
    ccc->SetFps(this->fps);
    if (!(*ccc)(CallKeyframeKeeper::CallForSetSimulationData)) return;

    // Initialise render utils once
    if (!this->utils.Initialized()) {
        if (!this->utils.Initialise(this->GetCoreInstance())) {
            vislib::sys::Log::DefaultLog.WriteError("[TRACKINGSHOT RENDERER] [create] Couldn't initialize render utils.");
            return;
        }
    }

    // Update parameters ------------------------------------------------------
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
            vislib::sys::Log::DefaultLog.WriteInfo("[CINEMATIC VIEW] STARTED rendering of complete animation.");
        } else {
            this->render2file_cleanup();
            vislib::sys::Log::DefaultLog.WriteInfo("[CINEMATIC VIEW] STOPPED rendering.");
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
        if ((this->png_data.animTime < 0.0f) || (this->png_data.animTime > ccc->GetTotalAnimTime())) {
            throw vislib::Exception("[CINEMATIC VIEW] Invalid animation time.", __FILE__, __LINE__);
        }
        ccc->SetSelectedKeyframeTime(this->png_data.animTime);
        if (!(*ccc)(CallKeyframeKeeper::CallForGetSelectedKeyframeAtTime)) return;
        loadNewCamParams = true;
    } else {
        // If time is set by running ANIMATION
        if (this->playAnim) {
            clock_t tmpTime = clock();
            clock_t cTime = tmpTime - this->deltaAnimTime;
            this->deltaAnimTime = tmpTime;
            float animTime = ccc->GetSelectedKeyframe().GetAnimTime() + ((float)cTime) / (float)(CLOCKS_PER_SEC);
            if ((animTime < 0.0f) ||
                (animTime > ccc->GetTotalAnimTime())) { // Reset time if max animation time is reached
                animTime = 0.0f;
            }
            ccc->SetSelectedKeyframeTime(animTime);
            if (!(*ccc)(CallKeyframeKeeper::CallForGetSelectedKeyframeAtTime)) return;
            loadNewCamParams = true;
        }
    }
    // Load current simulation time to parameter
    float simTime = ccc->GetSelectedKeyframe().GetSimTime();
    param::ParamSlot* animTimeParam = static_cast<param::ParamSlot*>(this->timeCtrl.GetSlot(2));
    animTimeParam->Param<param::FloatParam>()->SetValue(simTime * static_cast<float>(cr3d->TimeFramesCount()), true);

    // Viewport ---------------------------------------------------------------
    glm::ivec4 viewport;
    glGetIntegerv(GL_VIEWPORT, glm::value_ptr(viewport));
    const int vp_iw = viewport[2];
    const int vp_ih = viewport[3]; ;
    const int vp_ih_reduced = vp_ih - static_cast<int>(this->utils.GetTextLineHeight());
    const float vp_fw = static_cast<float>(vp_iw);
    const float vp_fh = static_cast<float>(vp_ih);
    const float vp_fh_reduced = static_cast<float>(vp_ih_reduced);
    const float cineRatio = static_cast<float>(this->cineWidth) / static_cast<float>(this->cineHeight);

    int fboWidth = vp_iw;
    int fboHeight = vp_ih_reduced;
    if (this->rendering) {
        fboWidth = this->cineWidth;
        fboHeight = this->cineHeight;
    } else {
        float vpRatio = vp_fw / vp_fh_reduced;
        // Check for viewport changes
        if ((this->vp_lastw != vp_iw) || (this->vp_lasth != vp_ih_reduced)) {
            this->vp_lastw = vp_iw;
            this->vp_lasth = vp_ih_reduced;
        }
        // Calculate reduced fbo width and height
        if ((this->cineWidth < vp_iw) && (this->cineHeight < vp_ih_reduced)) {
            fboWidth = this->cineWidth;
            fboHeight = this->cineHeight;
        } else {
            fboWidth = vp_iw;
            fboHeight = vp_ih_reduced;

            if (cineRatio > vpRatio) {
                fboHeight = (static_cast<int>(vp_fw / cineRatio));
            } else if (cineRatio < vpRatio) {
                fboWidth = (static_cast<int>(vp_fh_reduced * cineRatio));
            }
        }
    }

    int texWidth = fboWidth;
    int texHeight = fboHeight;
    if ((texWidth > vp_iw) || ((texWidth < vp_iw) && (texHeight < vp_ih_reduced))) {
        texWidth = vp_iw;
        texHeight = static_cast<int>(vp_fw / cineRatio);
    }
    if ((texHeight > vp_ih_reduced) || ((texWidth < vp_iw) && (texHeight < vp_ih_reduced))) {
        texHeight = vp_ih_reduced;
        texWidth = static_cast<int>(vp_fh_reduced * cineRatio);
    }

    // Camera settings --------------------------------------------------------
    // Set fbo viewport of camera
    auto res = cam_type::screen_size_type(glm::ivec2(fboWidth, fboHeight));
    this->cam.resolution_gate(res);
    auto tile = cam_type::screen_rectangle_type(std::array<int, 4>{0, 0, fboWidth, fboHeight});
    this->cam.image_tile(tile);
    // Set camera parameters of selected keyframe for this view.
    /// But only if selected keyframe differs to last locally stored and shown keyframe.
    /// Load new camera setting from selected keyframe when skybox side changes or rendering
    /// of animation loaded new slected keyframe.
    Keyframe skf = ccc->GetSelectedKeyframe();
    if (loadNewCamParams || (this->shownKeyframe != skf)) {
        this->shownKeyframe = skf;
        // Apply selected keyframe parameters only, if at least one valid keyframe exists.
        if (!ccc->GetKeyframes()->empty()) {
            this->cam = skf.GetCameraState();
        } else {
            this->ResetView();
        }
        // Apply showing skybox side ONLY if new camera parameters are set
        if (this->sbSide != CinematicView::SkyboxSides::SKYBOX_NONE) {
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
        }
    }
    // Propagate camera parameters to keyframe keeper (sky box camera params are propageted too!)
    cam_type::minimal_state_type camera_state;
    this->cam.get_minimal_state(camera_state);
    ccc->SetCameraState(std::make_shared<camera_state_type>(camera_state));
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

    auto bc = Base::BkgndColour();
    glClearColor(bc[0], bc[1], bc[2], 0.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

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

    // Init rendering ---------------------------------------------------------
    this->utils.SetBackgroundColor(glm::vec4(bc[0], bc[1], bc[2], 1.0f));
    glm::mat4 ortho = glm::ortho(0.0f, vp_fw, 0.0f, vp_fh, -1.0f, 1.0f);
    glClearColor(bc[0], bc[1], bc[2], 0.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // Push texture -----------------------------------------------------------
    float right = (vp_fw + static_cast<float>(texWidth)) / 2.0f;
    float left = (vp_fw - static_cast<float>(texWidth)) / 2.0f;
    float bottom = (vp_fh_reduced + static_cast<float>(texHeight)) / 2.0f;
    float up = (vp_fh_reduced - static_cast<float>(texHeight)) / 2.0f;
    glm::vec3 pos_bottom_left = { left, bottom, 0.0f };
    glm::vec3 pos_upper_left = { left, up, 0.0f };
    glm::vec3 pos_upper_right = { right, up, 0.0f };
    glm::vec3 pos_bottom_right = { right, bottom, 0.0f };
    this->utils.Push2DColorTexture(this->fbo.GetColourTextureID(), pos_bottom_left, pos_upper_left, pos_upper_right, pos_bottom_right);

    // Push letter box --------------------------------------------------------
    float letter_x = 0;
    float letter_y = 0;
    if (texWidth < vp_iw) {
        letter_x = (vp_fw - static_cast<float>(texWidth)) / 2.0f;
        letter_y = vp_fh_reduced;
    } else if (texHeight < vp_ih_reduced) {
        letter_x = vp_fw;
        letter_y = (vp_fh_reduced - static_cast<float>(texHeight)) / 2.0f;
    }
    pos_bottom_left = { vp_fw - letter_x, vp_fh_reduced - letter_y, 0.0f};
    pos_upper_left = { vp_fw - letter_x, vp_fh_reduced, 0.0f };
    pos_upper_right = { vp_fw, vp_fh_reduced, 0.0f };
    pos_bottom_right = { vp_fw, vp_fh_reduced - letter_y, 0.0f };
    this->utils.PushQuadPrimitive(pos_bottom_left, pos_upper_left, pos_upper_right, pos_bottom_right, this->utils.Color(CinematicUtils::Colors::LETTER_BOX));
    pos_bottom_left = { 0.0f, 0.0f, 0.0f };
    pos_upper_left = { 0.0f, letter_y, 0.0f };
    pos_upper_right = { letter_x, letter_y, 0.0f };
    pos_bottom_right = { letter_x, 0.0f, 0.0f };
    this->utils.PushQuadPrimitive(pos_bottom_left, pos_upper_left, pos_upper_right, pos_bottom_right, this->utils.Color(CinematicUtils::Colors::LETTER_BOX));

    // Push menu --------------------------------------------------------------
    std::string leftLabel = " CINEMATIC ";
    std::string midLabel = "";
    if (this->rendering) {
        midLabel = " ! RENDERING IN PROGRESS ! ";
    } else if (this->playAnim) {
        midLabel = " Playing Animation ";
    }
    std::string rightLabel = "";
    this->utils.PushMenu(leftLabel, midLabel, rightLabel, vp_fw, vp_fh);

    // Draw 2D ----------------------------------------------------------------
    this->utils.DrawAll(ortho, glm::vec2(vp_fw, vp_fh));
}


bool CinematicView::render2file_setup() {

    auto ccc = this->keyframeKeeperSlot.CallAs<CallKeyframeKeeper>();
    if (ccc == nullptr) return false;

    // init png data struct
    this->png_data.bpp = 3;
    this->png_data.width = static_cast<unsigned int>(this->cineWidth);
    this->png_data.height = static_cast<unsigned int>(this->cineHeight);
    this->png_data.buffer = nullptr;
    this->png_data.structptr = nullptr;
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
    float frameCnt = (float)(this->fps) * ccc->GetTotalAnimTime();
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
                "[CINEMATIC VIEW] [render2file_write] Cannot open output file", __FILE__, __LINE__);
        }

        // init png lib
        this->png_data.structptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, nullptr, &this->pngError, &this->pngWarn);
        if (this->png_data.structptr == nullptr) {
            throw vislib::Exception(
                "[CINEMATIC VIEW] [render2file_write] Cannot create png structure", __FILE__, __LINE__);
        }
        this->png_data.infoptr = png_create_info_struct(this->png_data.structptr);
        if (this->png_data.infoptr == nullptr) {
            throw vislib::Exception("[CINEMATIC VIEW] [render2file_write] Cannot create png info", __FILE__, __LINE__);
        }
        png_set_write_fn(this->png_data.structptr, static_cast<void*>(&this->png_data.file), &this->pngWrite, &this->pngFlush);
        png_set_IHDR(this->png_data.structptr, this->png_data.infoptr, this->png_data.width, this->png_data.height, 8,
            PNG_COLOR_TYPE_RGB, PNG_INTERLACE_NONE, PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);

        if (this->fbo.GetColourTexture(this->png_data.buffer, 0, GL_RGB, GL_UNSIGNED_BYTE) != GL_NO_ERROR) {
            throw vislib::Exception(
                "[CINEMATIC VIEW] [render2file_write] Failed to create Screenshot: Cannot read image data", __FILE__,
                __LINE__);
        }

        BYTE** rows = nullptr;
        try {
            rows = new BYTE*[this->cineHeight];
            for (UINT i = 0; i < this->png_data.height; i++) {
                rows[this->png_data.height - (1 + i)] =
                    this->png_data.buffer + this->png_data.bpp * i * this->png_data.width;
            }
            png_set_rows(this->png_data.structptr, this->png_data.infoptr, rows);

            png_write_png(this->png_data.structptr, this->png_data.infoptr, PNG_TRANSFORM_IDENTITY, nullptr);

            ARY_SAFE_DELETE(rows);
        } catch (...) {
            if (rows != nullptr) {
                ARY_SAFE_DELETE(rows);
            }
            throw;
        }
        if (this->png_data.structptr != nullptr) {
            if (this->png_data.infoptr != nullptr) {
                png_destroy_write_struct(&this->png_data.structptr, &this->png_data.infoptr);
            } else {
                png_destroy_write_struct(&this->png_data.structptr, (png_infopp) nullptr);
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

        if (this->png_data.animTime == ccc->GetTotalAnimTime()) {
            ///XXX Handling this case is actually only necessary when rendering is done via FBOCompositor 
            ///XXX => Rendering crashes (WHY? - Rendering last frame with animation time = total animation time is otherwise no problem)
            this->png_data.animTime -= 0.000005f;
        } else if (this->png_data.animTime >= ccc->GetTotalAnimTime()) {
            this->render2file_cleanup();
            return false;
        }

        this->png_data.cnt++;
    }

    return true;
}


bool CinematicView::render2file_cleanup() {

    this->rendering = false;

    if (this->png_data.structptr != nullptr) {
        if (this->png_data.infoptr != nullptr) {
            png_destroy_write_struct(&this->png_data.structptr, &this->png_data.infoptr);
        }
        else {
            png_destroy_write_struct(&this->png_data.structptr, (png_infopp) nullptr);
        }
    }

    try {
        this->png_data.file.Flush();
    }
    catch (...) {
    }

    try {
        this->png_data.file.Close();
    }
    catch (...) {
    }

    ARY_SAFE_DELETE(this->png_data.buffer);

    if (this->png_data.buffer != nullptr) {
        vislib::sys::Log::DefaultLog.WriteError("[CINEMATIC VIEW] [render2file_cleanup] pngdata.buffer is not nullptr.");
    }
    if (this->png_data.structptr != nullptr) {
        vislib::sys::Log::DefaultLog.WriteError("[CINEMATIC VIEW] [render2file_cleanup] pngdata.structptr is not nullptr.");
    }
    if (this->png_data.infoptr != nullptr) {
        vislib::sys::Log::DefaultLog.WriteError("[CINEMATIC VIEW] [render2file_cleanup] pngdata.infoptr is not nullptr.");
    }

    return true;
}

