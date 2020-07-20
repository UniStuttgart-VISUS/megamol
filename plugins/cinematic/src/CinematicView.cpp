/**
 * CinematicView.cpp
 *
 * Copyright (C) 2017 by VISUS (Universitaet Stuttgart).
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "CinematicView.h"

#include "mmcore/thecam/utility/types.h"


using namespace megamol;
using namespace megamol::core;
using namespace megamol::core::view;
using namespace megamol::core::utility;
using namespace megamol::cinematic;

using namespace vislib;


CinematicView::CinematicView(void) : View3D_2()
    , keyframeKeeperSlot("keyframeData", "Connects to the Keyframe Keeper.")
    , renderParam("cinematic::renderAnim", "Toggle rendering of complete animation to PNG files.")
    , toggleAnimPlayParam("cinematic::playPreview", "Toggle playing animation as preview")
    , selectedSkyboxSideParam("cinematic::skyboxSide", "Select the skybox side.")
    , skyboxCubeModeParam("cinematic::skyboxCubeMode", "Render cube around dataset with skybox side as side selector.")
    , resWidthParam("cinematic::cinematicWidth", "The width resolution of the cineamtic view to render.")
    , resHeightParam("cinematic::cinematicHeight", "The height resolution of the cineamtic view to render.")
    , fpsParam("cinematic::fps", "Frames per second the animation should be rendered.")
    , firstRenderFrameParam("cinematic::firstFrame", "Set first frame number to start rendering with.")
    , lastRenderFrameParam("cinematic::lastFrame", "Set last frame number to end rendering with.")
    , delayFirstRenderFrameParam("cinematic::delayFirstFrame", "Delay (in seconds) to wait until first frame for rendering is written (needed to get right first frame especially for high resolutions and for distributed rendering).")
    , frameFolderParam( "cinematic::frameFolder", "Specify folder where the frame files should be stored.")
    , addSBSideToNameParam( "cinematic::addSBSideToName", "Toggle whether skybox side should be added to output filename")
    , eyeParam("cinematic::stereo_eye", "Select eye position (for stereo view).")
    , projectionParam("cinematic::stereo_projection", "Select camera projection.")
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
    , fps(24)
    , skyboxCubeMode(false) {

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

    this->skyboxCubeModeParam << new param::BoolParam(this->skyboxCubeMode);
    this->MakeSlotAvailable(&this->skyboxCubeModeParam);

    this->resWidthParam.SetParameter(new param::IntParam(this->cineWidth, 1));
    this->MakeSlotAvailable(&this->resWidthParam);

    this->resHeightParam.SetParameter(new param::IntParam(this->cineHeight, 1));
    this->MakeSlotAvailable(&this->resHeightParam);

    this->fpsParam.SetParameter(new param::IntParam(static_cast<int>(this->fps), 1));
    this->MakeSlotAvailable(&this->fpsParam);

    this->firstRenderFrameParam.SetParameter(new param::IntParam(0, 0));
    this->MakeSlotAvailable(&this->firstRenderFrameParam);

    this->lastRenderFrameParam.SetParameter(new param::IntParam((std::numeric_limits<int>::max)(), 1));
    this->MakeSlotAvailable(&this->lastRenderFrameParam);

    this->delayFirstRenderFrameParam.SetParameter(new param::FloatParam(1.0f));
    this->MakeSlotAvailable(&this->delayFirstRenderFrameParam);

    this->frameFolderParam.SetParameter(new param::FilePathParam(""));
    this->MakeSlotAvailable(&this->frameFolderParam);

    this->addSBSideToNameParam << new param::BoolParam(false);
    this->MakeSlotAvailable(&this->addSBSideToNameParam);

    param::EnumParam* enp = new param::EnumParam(static_cast<int>(megamol::core::thecam::Eye::mono));
    enp->SetTypePair(static_cast<int>(megamol::core::thecam::Eye::mono), "Mono");
    enp->SetTypePair(static_cast<int>(megamol::core::thecam::Eye::left), "Left");
    enp->SetTypePair(static_cast<int>(megamol::core::thecam::Eye::centre), "Center");
    enp->SetTypePair(static_cast<int>(megamol::core::thecam::Eye::right), "Right");
    this->eyeParam << enp;
    this->MakeSlotAvailable(&this->eyeParam);

    param::EnumParam* pep = new param::EnumParam(static_cast<int>(megamol::core::thecam::Projection_type::perspective));
    pep->SetTypePair(static_cast<int>(megamol::core::thecam::Projection_type::perspective), "Mono Perspective");
    pep->SetTypePair(static_cast<int>(megamol::core::thecam::Projection_type::orthographic), "Mono Orthographic");
    pep->SetTypePair(static_cast<int>(megamol::core::thecam::Projection_type::off_axis), "Stereo Off-Axis");
    pep->SetTypePair(static_cast<int>(megamol::core::thecam::Projection_type::parallel), "Stereo Prallel");
    pep->SetTypePair(static_cast<int>(megamol::core::thecam::Projection_type::toe_in), "Stereo Toe-In");
    pep->SetTypePair(static_cast<int>(megamol::core::thecam::Projection_type::converged), "Converged");
    this->projectionParam << pep;
    this->MakeSlotAvailable(&this->projectionParam);
}


CinematicView::~CinematicView(void) {

    this->render_to_file_cleanup();
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
    ccc->SetBboxCenter(vislib_point_to_glm(cr3d->AccessBoundingBoxes().BoundingBox().CalcCenter()));
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
        }
    }
    if (this->skyboxCubeModeParam.IsDirty()) {
        this->skyboxCubeModeParam.ResetDirty();
        this->skyboxCubeMode = this->skyboxCubeModeParam.Param<param::BoolParam>()->Value();
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
            this->render_to_file_setup();
        } else {
            this->render_to_file_cleanup();
        }
    }
    // Set (mono/stereo) projection for camera
    if (this->projectionParam.IsDirty()) {
        this->projectionParam.ResetDirty();
        this->cam.projection_type(static_cast<megamol::core::thecam::Projection_type>(this->projectionParam.Param<param::EnumParam>()->Value()));
    }
    // Set eye position for camera
    if (this->eyeParam.IsDirty()) {
        this->eyeParam.ResetDirty();
        this->cam.eye(static_cast<megamol::core::thecam::Eye>(this->eyeParam.Param<param::EnumParam>()->Value()));
    }

    // Time settings ----------------------------------------------------------
    // Load animation time
    if (this->rendering) {
        if ((this->png_data.animTime < 0.0f) || (this->png_data.animTime > ccc->GetTotalAnimTime())) {
            throw vislib::Exception("[CINEMATIC VIEW] Invalid animation time.", __FILE__, __LINE__);
        }
        ccc->SetSelectedKeyframeTime(this->png_data.animTime);
        if (!(*ccc)(CallKeyframeKeeper::CallForGetSelectedKeyframeAtTime)) return;
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
        }
    }
    Keyframe skf = ccc->GetSelectedKeyframe();

    // Load current simulation time to parameter
    float simTime = skf.GetSimTime();
    param::ParamSlot* animTimeParam = static_cast<param::ParamSlot*>(this->timeCtrl.GetSlot(2));
    animTimeParam->Param<param::FloatParam>()->SetValue(simTime * static_cast<float>(cr3d->TimeFramesCount()), true);

    // Viewport ---------------------------------------------------------------
    /// Viewport of camera will only be set when Base::Render(context) was called, so we have tot grab it from OpenGL (?)
    glm::ivec4 viewport;
    glGetIntegerv(GL_VIEWPORT, glm::value_ptr(viewport));
    const int vp_iw = viewport[2];
    const int vp_ih = viewport[3]; ;
    const int vp_ih_reduced = vp_ih - static_cast<int>(this->utils.GetTextLineHeight());
    const float vp_fw = static_cast<float>(vp_iw);
    const float vp_fh = static_cast<float>(vp_ih);
    const float vp_fh_reduced = static_cast<float>(vp_ih_reduced);
    const float cineRatio = static_cast<float>(this->cineWidth) / static_cast<float>(this->cineHeight);
    // FBO viewport
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
    // Texture viewport
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

    // Set camera settings ----------------------------------------------------
    auto res = cam_type::screen_size_type(glm::ivec2(fboWidth, fboHeight));
    this->cam.resolution_gate(res);
    auto tile = cam_type::screen_rectangle_type(std::array<int, 4>{0, 0, fboWidth, fboHeight});
    this->cam.image_tile(tile);

    // Set camera parameters of selected keyframe for this view.
    // But only if selected keyframe differs to last locally stored and shown keyframe.
    if (this->shownKeyframe != skf) {
        this->shownKeyframe = skf;

        // Apply selected keyframe parameters only, if at least one valid keyframe exists.
        if (!ccc->GetKeyframes()->empty()) {
            // ! Using only a subset of the keyframe camera state !
            auto pos = skf.GetCameraState().position;
            this->cam.position(glm::vec4(pos[0], pos[1], pos[2], 1.0f));
            auto ori = skf.GetCameraState().orientation;
            this->cam.orientation(glm::quat(ori[3], ori[0], ori[1], ori[2]));
            auto aper = skf.GetCameraState().half_aperture_angle_radians;
            this->cam.half_aperture_angle_radians(aper);
        }
        else {
            this->ResetView();
        }
    }

    // Propagate current camera state to keyframe keeper (before applying following skybox side settings).
    cam_type::minimal_state_type camera_state;
    this->cam.get_minimal_state(camera_state);
    ccc->SetCameraState(std::make_shared<camera_state_type>(camera_state));
    if (!(*ccc)(CallKeyframeKeeper::CallForSetCameraForKeyframe)) return;

    // Apply showing skybox side ONLY if new camera parameters are set.
    // Non-permanent overwrite of selected keyframe camera by skybox camera settings.
    if (this->sbSide != CinematicView::SkyboxSides::SKYBOX_NONE) {
        cam_type::snapshot_type snapshot;
        this->cam.take_snapshot(snapshot, thecam::snapshot_content::all);

        glm::vec4 snap_pos = snapshot.position;
        glm::vec4 snap_right = snapshot.right_vector;
        glm::vec4 snap_up = snapshot.up_vector;
        glm::vec4 snap_view = snapshot.view_vector;

        glm::vec3 cam_right = static_cast<glm::vec3>(snap_right);
        glm::vec3 cam_up = static_cast<glm::vec3>(snap_up);
        glm::vec3 cam_view = static_cast<glm::vec3>(snap_view);

        glm::vec4 new_pos = snap_pos;
        glm::quat new_orientation;
        const float Rad180Degrees = glm::radians(180.0f);
        const float Rad90Degrees = glm::radians(90.0f);

        if (!this->skyboxCubeMode) {
            if (this->sbSide == CinematicView::SkyboxSides::SKYBOX_BACK) {
                new_orientation = glm::rotate(new_orientation, Rad180Degrees, cam_right);
                new_orientation = glm::rotate(new_orientation, Rad180Degrees, cam_view);
            }
            else if (this->sbSide == CinematicView::SkyboxSides::SKYBOX_RIGHT) {
                new_orientation = glm::rotate(new_orientation, Rad90Degrees, cam_up);
            }
            else if (this->sbSide == CinematicView::SkyboxSides::SKYBOX_LEFT) {
                new_orientation = glm::rotate(new_orientation, -Rad90Degrees, cam_up);
            }
            else if (this->sbSide == CinematicView::SkyboxSides::SKYBOX_UP) {
                new_orientation = glm::rotate(new_orientation, -Rad90Degrees, cam_right);
            }
            else if (this->sbSide == CinematicView::SkyboxSides::SKYBOX_DOWN) {
                new_orientation = glm::rotate(new_orientation, Rad90Degrees, cam_right);
            }
        }
        else {
            auto const center_ = cr3d->AccessBoundingBoxes().BoundingBox().CalcCenter();
            const glm::vec4 center = glm::vec4(center_.X(), center_.Y(), center_.Z(), 1.0f);
            const float width = cr3d->AccessBoundingBoxes().BoundingBox().Width();
            const float height = cr3d->AccessBoundingBoxes().BoundingBox().Height();
            const float depth = cr3d->AccessBoundingBoxes().BoundingBox().Depth();  
            if (this->sbSide == CinematicView::SkyboxSides::SKYBOX_FRONT) {
                new_pos = center - depth * snap_view;
            }
            else if (this->sbSide == CinematicView::SkyboxSides::SKYBOX_BACK) {
                new_pos = center + depth * snap_view;
                new_orientation = glm::rotate(new_orientation, Rad180Degrees, cam_right);
                new_orientation = glm::rotate(new_orientation, Rad180Degrees, cam_view);
            }
            else if (this->sbSide == CinematicView::SkyboxSides::SKYBOX_RIGHT) {
                new_pos = center + width * snap_right;
                new_orientation = glm::rotate(new_orientation, Rad90Degrees, cam_up);
            }
            else if (this->sbSide == CinematicView::SkyboxSides::SKYBOX_LEFT) {
                new_pos = center - width * snap_right;
                new_orientation = glm::rotate(new_orientation, -Rad90Degrees, cam_up);
            }
            else if (this->sbSide == CinematicView::SkyboxSides::SKYBOX_UP) {
                new_pos = center + height * snap_up;
                new_orientation = glm::rotate(new_orientation, -Rad90Degrees, cam_right);
            }
            else if (this->sbSide == CinematicView::SkyboxSides::SKYBOX_DOWN) {
                new_pos = center - height * snap_up;
                new_orientation = glm::rotate(new_orientation, Rad90Degrees, cam_right);
            }
        }
        // Apply new position, orientation and aperture angle to current camera.
        this->cam.position(new_pos);
        this->cam.orientation(new_orientation);
        this->cam.aperture_angle(90.0f);
    }

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
        // Check if fbo in cr3d was reset by renderer to indicate that no new frame is available (e.g. see remote/FBOCompositor2 Render())
        this->png_data.write_lock = ((cr3d->FrameBufferObject() != nullptr) ? (0) : (1));
        if (this->png_data.write_lock > 0) {
            vislib::sys::Log::DefaultLog.WriteInfo("[CINEMATIC RENDERING] Waiting for next frame (Received empty FBO from renderer) ...\n");
        }
        // Lock writing frame to file for specific tim
        std::chrono::duration<double> diff = (std::chrono::system_clock::now() - this->png_data.start_time);
        if (diff.count() < static_cast<double>(this->delayFirstRenderFrameParam.Param<param::FloatParam>()->Value())) {
            this->png_data.write_lock = 1;
        }
        this->render_to_file_write();
    }

    // Init rendering ---------------------------------------------------------
    this->utils.SetBackgroundColor(glm::vec4(bc[0], bc[1], bc[2], 1.0f));
    glm::mat4 ortho = glm::ortho(0.0f, vp_fw, 0.0f, vp_fh, -1.0f, 1.0f);
    glClearColor(bc[0], bc[1], bc[2], 0.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // Push fbo texture -------------------------------------------------------
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


bool CinematicView::render_to_file_setup() {

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

    unsigned int firstFrame =
        static_cast<unsigned int>(this->firstRenderFrameParam.Param<param::IntParam>()->Value());
    unsigned int lastFrame =
        static_cast<unsigned int>(this->lastRenderFrameParam.Param<param::IntParam>()->Value());

    unsigned int maxFrame = (unsigned int)(ccc->GetTotalAnimTime() * (float)this->fps);
    if (firstFrame > maxFrame) {
        vislib::sys::Log::DefaultLog.WriteWarn(
            "[CINEMATIC VIEW] [render_to_file_setup] Max frame count exceeded. Limiting first frame to maximum frame %d", maxFrame);
        firstFrame = maxFrame;
    }
    if (firstFrame > lastFrame) {
        vislib::sys::Log::DefaultLog.WriteWarn(
            "[CINEMATIC VIEW] [render_to_file_setup] First frame exceeds last frame. Limiting first frame to last frame %d", lastFrame);
        firstFrame = lastFrame;
    }

    this->png_data.cnt = firstFrame;
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
        this->png_data.path = vislib::sys::Path::Concatenate(vislib::sys::Path::GetCurrentDirectoryA(), frameFolder);
    }
    vislib::sys::Path::MakeDirectory(this->png_data.path);

    this->png_data.filename = "frames";

    // Create new byte buffer
    this->png_data.buffer = new BYTE[this->png_data.width * this->png_data.height * this->png_data.bpp];
    if (this->png_data.buffer == nullptr) {
        throw vislib::Exception(
            "[CINEMATIC VIEW] [render_to_file_setup] Cannot allocate image buffer.", __FILE__, __LINE__);
    }

    vislib::sys::Log::DefaultLog.WriteInfo("[CINEMATIC VIEW] STARTED rendering of complete animation.");

    return true;
}


bool CinematicView::render_to_file_write() {

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

        // Open final image file
        if (!this->png_data.file.Open(vislib::sys::Path::Concatenate(this->png_data.path, tmpFilename),
                vislib::sys::File::WRITE_ONLY, vislib::sys::File::SHARE_EXCLUSIVE,
                vislib::sys::File::CREATE_OVERWRITE)) {
            throw vislib::Exception(
                "[CINEMATIC VIEW] [render_to_file_write] Cannot open output file", __FILE__, __LINE__);
        }

        // Init png lib
        this->png_data.structptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, nullptr, &this->pngError, &this->pngWarn);
        if (this->png_data.structptr == nullptr) {
            throw vislib::Exception(
                "[CINEMATIC VIEW] [render_to_file_write] Unable to create png structure. ", __FILE__, __LINE__);
        }
        this->png_data.infoptr = png_create_info_struct(this->png_data.structptr);
        if (this->png_data.infoptr == nullptr) {
            throw vislib::Exception("[CINEMATIC VIEW] [render_to_file_write] Unable to create png info. ", __FILE__, __LINE__);
        }
        png_set_write_fn(this->png_data.structptr, static_cast<void*>(&this->png_data.file), &this->pngWrite, &this->pngFlush);
        png_set_IHDR(this->png_data.structptr, this->png_data.infoptr, this->png_data.width, this->png_data.height, 8,
            PNG_COLOR_TYPE_RGB, PNG_INTERLACE_NONE, PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);

        // Serialise current project into png header (see ScreenShooter.cpp, line 452)
        std::string serInstances, serModules, serCalls, serParams;
        this->GetCoreInstance()->SerializeGraph(serInstances, serModules, serCalls, serParams);
        auto confstr = serInstances + "\n" + serModules + "\n" + serCalls + "\n" + serParams;
        std::vector<png_byte> tempvec(confstr.begin(), confstr.end());
        tempvec.push_back('\0');
        png_set_eXIf_1(this->png_data.structptr, this->png_data.infoptr, tempvec.size(), tempvec.data());

        if (this->fbo.GetColourTexture(this->png_data.buffer, 0, GL_RGB, GL_UNSIGNED_BYTE) != GL_NO_ERROR) {
            throw vislib::Exception(
                "[CINEMATIC VIEW] [render_to_file_write] Unable to read color texture. ", __FILE__,
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
            "[CINEMATIC VIEW] [render_to_file_write] Wrote png file %d for animation time %f ...\n", this->png_data.cnt,
            this->png_data.animTime);

        // --------------------------------------------------------------------

        // Next frame/time step
        this->png_data.cnt++;
        this->png_data.animTime = (float)this->png_data.cnt / (float)this->fps;
        float fpsFrac = (1.0f / static_cast<float>(this->fps));
        // Fit animTime to exact full seconds (removing rounding error)
        if (std::abs(this->png_data.animTime - std::round(this->png_data.animTime)) < (fpsFrac / 2.0)) {
            this->png_data.animTime = std::round(this->png_data.animTime);
        }

        ///XXX Handling this case is actually only necessary when rendering is done via FBOCompositor 
        ///XXX Rendering crashes - WHY? 
        // XXX Rendering last frame with animation time = total animation time is otherwise no problem
        //if (this->png_data.animTime == ccc->GetTotalAnimTime()) {
        //    this->png_data.animTime -= 0.000005f;
        //} else 

        // Check condition for finishing rendering
        auto lastFrame = static_cast<unsigned int>(this->lastRenderFrameParam.Param<param::IntParam>()->Value());
        if ((this->png_data.animTime >= ccc->GetTotalAnimTime()) ||
            (this->png_data.cnt > lastFrame)) {
            this->render_to_file_cleanup();
            return false;
        }
    }

    return true;
}


bool CinematicView::render_to_file_cleanup() {

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
        vislib::sys::Log::DefaultLog.WriteError("[CINEMATIC VIEW] [render_to_file_cleanup] pngdata.buffer is not nullptr.");
    }
    if (this->png_data.structptr != nullptr) {
        vislib::sys::Log::DefaultLog.WriteError("[CINEMATIC VIEW] [render_to_file_cleanup] pngdata.structptr is not nullptr.");
    }
    if (this->png_data.infoptr != nullptr) {
        vislib::sys::Log::DefaultLog.WriteError("[CINEMATIC VIEW] [render_to_file_cleanup] pngdata.infoptr is not nullptr.");
    }

    vislib::sys::Log::DefaultLog.WriteInfo("[CINEMATIC VIEW] STOPPED rendering.");

    return true;
}

