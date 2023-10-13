/**
 * CinematicView.cpp
 *
 * Copyright (C) 2017 by VISUS (Universitaet Stuttgart).
 * Alle Rechte vorbehalten.
 */

#include "CinematicView.h"
#include "cinematic/CallKeyframeKeeper.h"
#include "mmcore/MegaMolGraph.h"
#include "mmcore/param/BoolParam.h"
#include "mmcore/param/ButtonParam.h"
#include "mmcore/param/EnumParam.h"
#include "mmcore/param/FilePathParam.h"
#include "mmcore/param/FloatParam.h"
#include "mmcore/param/IntParam.h"
#include "mmcore/utility/graphics/ScreenShotComments.h"
#include "vislib/sys/Path.h"


using namespace megamol;
using namespace megamol::core;
using namespace megamol::core::view;
using namespace megamol::core::utility;
using namespace megamol::cinematic_gl;

using namespace vislib;


CinematicView::CinematicView()
        : mmstd_gl::view::View3DGL()
        , keyframeKeeperSlot("keyframeData", "Connects to the Keyframe Keeper.")
        , renderParam("cinematic::renderAnim", "Toggle rendering of complete animation to PNG files.")
        , toggleAnimPlayParam("cinematic::playPreview", "Toggle playing animation as preview")
        , selectedSkyboxSideParam("cinematic::skyboxSide", "Select the skybox side.")
        , skyboxCubeModeParam(
              "cinematic::skyboxCubeMode", "Render cube around dataset with skybox side as side selector.")
        , resWidthParam("cinematic::cinematicWidth", "The width resolution of the cineamtic view to render.")
        , resHeightParam("cinematic::cinematicHeight", "The height resolution of the cineamtic view to render.")
        , fpsParam("cinematic::fps", "Frames per second the animation should be rendered.")
        , firstRenderFrameParam("cinematic::firstFrame", "Set first frame number to start rendering with.")
        , lastRenderFrameParam("cinematic::lastFrame", "Set last frame number to end rendering with.")
        , delayFirstRenderFrameParam("cinematic::delayFirstFrame",
              "Delay (in seconds) to wait until first frame for rendering is written (needed to get right first frame "
              "especially for high resolutions and for distributed rendering).")
        , frameFolderParam("cinematic::frameFolder", "Specify folder where the frame files should be stored.")
        , addSBSideToNameParam(
              "cinematic::addSBSideToName", "Toggle whether skybox side should be added to output filename")
        , png_data()
        , utils()
        , deltaAnimTime(clock())
        , shownKeyframe()
        , playAnim(false)
        , cineWidth(1920)
        , cineHeight(1080)
        , sbSide(CinematicView::SkyboxSides::SKYBOX_NONE)
        , rendering(false)
        , fps(24)
        , skyboxCubeMode(false)
        , cinematicFbo(nullptr) {

    // init callback
    this->keyframeKeeperSlot.SetCompatibleCall<cinematic::CallKeyframeKeeperDescription>();
    this->MakeSlotAvailable(&this->keyframeKeeperSlot);

    // init parameters
    this->renderParam.SetParameter(new param::ButtonParam(core::view::Key::KEY_R, core::view::Modifier::SHIFT));
    this->MakeSlotAvailable(&this->renderParam);

    this->toggleAnimPlayParam.SetParameter(
        new param::ButtonParam(core::view::Key::KEY_SPACE, core::view::Modifier::SHIFT));
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
    sbs = nullptr;

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

    this->frameFolderParam.SetParameter(new param::FilePathParam("", param::FilePathParam::Flag_Directory_ToBeCreated));
    this->MakeSlotAvailable(&this->frameFolderParam);

    this->addSBSideToNameParam << new param::BoolParam(false);
    this->MakeSlotAvailable(&this->addSBSideToNameParam);
}


CinematicView::~CinematicView() {

    this->cinematicFbo.reset();
    this->render_to_file_cleanup();
    this->Release();
}


ImageWrapper CinematicView::Render(double time, double instanceTime) {

    // Get update data from keyframe keeper -----------------------------------
    auto cr3d = this->_rhsRenderSlot.CallAs<mmstd_gl::CallRender3DGL>();
    auto ccc = this->keyframeKeeperSlot.CallAs<cinematic::CallKeyframeKeeper>();

    // init camera
    if ((this->_camera.get<view::Camera::ProjectionType>() != view::Camera::PERSPECTIVE) &&
        (this->_camera.get<view::Camera::ProjectionType>() != view::Camera::ORTHOGRAPHIC)) {
        auto intrinsics = core::view::Camera::PerspectiveParameters();
        intrinsics.fovy = 0.5f;
        intrinsics.aspect = static_cast<float>(this->_fbo->getWidth()) / static_cast<float>(this->_fbo->getHeight());
        intrinsics.near_plane = 0.01f;
        intrinsics.far_plane = 100.0f;
        /// intrinsics.image_plane_tile = ;
        this->_camera.setPerspectiveProjection(intrinsics);
    }

    if ((cr3d != nullptr) && (ccc != nullptr)) {

        bool ccc_success = (*ccc)(cinematic::CallKeyframeKeeper::CallForGetUpdatedKeyframeData);
        ccc_success &= (*ccc)(cinematic::CallKeyframeKeeper::CallForSetSimulationData);

        // Initialise render utils once
        bool utils_success = this->utils.Initialized();
        if (!utils_success) {
            if (this->utils.Initialise(frontend_resources.get<megamol::frontend_resources::RuntimeConfig>())) {
                utils_success = true;
            } else {
                megamol::core::utility::log::Log::DefaultLog.WriteError(
                    "[CINEMATIC VIEW] [Render] Couldn't initialize render utils. [%s, %s, line %d]\n", __FILE__,
                    __FUNCTION__, __LINE__);
            }
        }

        if (ccc_success && utils_success) {

            ccc->SetBboxCenter(
                core_gl::utility::vislib_point_to_glm(cr3d->AccessBoundingBoxes().BoundingBox().CalcCenter()));
            ccc->SetTotalSimTime(static_cast<float>(cr3d->TimeFramesCount()));
            ccc->SetFps(this->fps);

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
                    this->selectedSkyboxSideParam.Param<param::EnumParam>()->SetValue(
                        static_cast<int>(this->sbSide), false);
                    megamol::core::utility::log::Log::DefaultLog.WriteWarn(
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
                    megamol::core::utility::log::Log::DefaultLog.WriteWarn(
                        "[CINEMATIC VIEW] [resHeightParam] Changes are not applied while rendering is running.");
                } else {
                    this->cineHeight = this->resHeightParam.Param<param::IntParam>()->Value();
                }
            }
            if (this->resWidthParam.IsDirty()) {
                this->resWidthParam.ResetDirty();
                if (this->rendering) {
                    this->resWidthParam.Param<param::IntParam>()->SetValue(this->cineWidth, false);
                    megamol::core::utility::log::Log::DefaultLog.WriteWarn(
                        "[CINEMATIC VIEW] [resWidthParam] Changes are not applied while rendering is running.");
                } else {
                    this->cineWidth = this->resWidthParam.Param<param::IntParam>()->Value();
                }
            }
            if (this->fpsParam.IsDirty()) {
                this->fpsParam.ResetDirty();
                if (this->rendering) {
                    this->fpsParam.Param<param::IntParam>()->SetValue(static_cast<int>(this->fps), false);
                    megamol::core::utility::log::Log::DefaultLog.WriteWarn(
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
                    megamol::core::utility::log::Log::DefaultLog.WriteInfo("[CINEMATIC VIEW] Finished rendering.");
                }
            }

            // Time settings ----------------------------------------------------------
            // Load animation time
            if (this->rendering) {
                if ((this->png_data.animTime < 0.0f) || (this->png_data.animTime > ccc->GetTotalAnimTime())) {
                    throw vislib::Exception("[CINEMATIC VIEW] Invalid animation time.", __FILE__, __LINE__);
                }
                ccc->SetSelectedKeyframeTime(this->png_data.animTime);
            } else {
                // If time is set by running ANIMATION
                if (this->playAnim) {
                    clock_t tmpTime = clock();
                    clock_t cTime = tmpTime - this->deltaAnimTime;
                    this->deltaAnimTime = tmpTime;
                    float animTime = ccc->GetSelectedKeyframe().GetAnimTime() +
                                     static_cast<float>(cTime) / static_cast<float>(CLOCKS_PER_SEC);
                    if ((animTime < 0.0f) ||
                        (animTime > ccc->GetTotalAnimTime())) { // Reset time if max animation time is reached
                        animTime = 0.0f;
                    }
                    ccc->SetSelectedKeyframeTime(animTime);
                }
            }

            if ((*ccc)(cinematic::CallKeyframeKeeper::CallForGetSelectedKeyframeAtTime)) {
                cinematic::Keyframe skf = ccc->GetSelectedKeyframe();

                // Propagate current camera state to keyframe keeper (before applying following skybox side settings).
                ccc->SetCameraState(std::make_shared<Camera>(this->_camera));
                if (!(*ccc)(cinematic::CallKeyframeKeeper::CallForSetCameraForKeyframe)) {
                    throw vislib::Exception(
                        "[CINEMATIC VIEW] Could not propagate current camera to keyframe keeper.", __FILE__, __LINE__);
                }

                // Load current simulation time to this views animTimeParam = simulation time.
                /// TODO XXX One frame dealy due to propagation in GetExtends in following frame?!
                float simTime = skf.GetSimTime();
                param::ParamSlot* animTimeParam =
                    static_cast<param::ParamSlot*>(this->_timeCtrl.GetSlot(2)); // animTimeSlot
                auto frame_count = cr3d->TimeFramesCount();
                animTimeParam->Param<param::FloatParam>()->SetValue(simTime * static_cast<float>(frame_count), true);

                // Set camera parameters of selected keyframe for this view.
                // But only if selected keyframe differs to last locally stored and shown keyframe.
                if (this->shownKeyframe != skf) {
                    this->shownKeyframe = skf;

                    // Apply selected keyframe parameters only, if at least one valid keyframe exists.
                    if (!ccc->GetKeyframes()->empty()) {
                        // ! Using only a subset of the keyframe camera state (pose and fov/frustrum height)!
                        auto pose = skf.GetCamera().get<Camera::Pose>();
                        if (skf.GetCamera().get<Camera::ProjectionType>() == Camera::PERSPECTIVE) {
                            auto skf_intrinsics = skf.GetCamera().get<Camera::PerspectiveParameters>();
                            if (this->_camera.get<Camera::ProjectionType>() == Camera::PERSPECTIVE) {
                                auto cam_intrinsics = this->_camera.get<Camera::PerspectiveParameters>();
                                cam_intrinsics.fovy = skf_intrinsics.fovy;
                                this->_camera = Camera(pose, cam_intrinsics);
                            } else if (this->_camera.get<Camera::ProjectionType>() == Camera::ORTHOGRAPHIC) {
                                auto cam_intrinsics = this->_camera.get<Camera::OrthographicParameters>();
                                Camera::PerspectiveParameters pers_intrinsics;
                                pers_intrinsics.aspect = cam_intrinsics.aspect;
                                pers_intrinsics.far_plane = cam_intrinsics.far_plane;
                                pers_intrinsics.image_plane_tile = cam_intrinsics.image_plane_tile;
                                pers_intrinsics.near_plane = cam_intrinsics.near_plane;
                                pers_intrinsics.fovy = skf_intrinsics.fovy;
                                this->_camera = Camera(pose, pers_intrinsics);
                            } else {
                                megamol::core::utility::log::Log::DefaultLog.WriteError(
                                    "[Camera] Found no valid projection. [%s, %s, line %d]\n", __FILE__, __FUNCTION__,
                                    __LINE__);
                            }
                        } else if (skf.GetCamera().get<Camera::ProjectionType>() == Camera::ORTHOGRAPHIC) {
                            auto skf_intrinsics = skf.GetCamera().get<Camera::OrthographicParameters>();
                            if (this->_camera.get<Camera::ProjectionType>() == Camera::PERSPECTIVE) {
                                auto cam_intrinsics = this->_camera.get<Camera::PerspectiveParameters>();
                                Camera::OrthographicParameters orth_intrinsics;
                                orth_intrinsics.aspect = cam_intrinsics.aspect;
                                orth_intrinsics.far_plane = cam_intrinsics.far_plane;
                                orth_intrinsics.image_plane_tile = cam_intrinsics.image_plane_tile;
                                orth_intrinsics.near_plane = cam_intrinsics.near_plane;
                                orth_intrinsics.frustrum_height = skf_intrinsics.frustrum_height;
                                this->_camera = Camera(pose, orth_intrinsics);
                            } else if (this->_camera.get<Camera::ProjectionType>() == Camera::ORTHOGRAPHIC) {
                                auto cam_intrinsics = this->_camera.get<Camera::OrthographicParameters>();
                                cam_intrinsics.frustrum_height = skf_intrinsics.frustrum_height;
                                this->_camera = Camera(pose, cam_intrinsics);
                            } else {
                                megamol::core::utility::log::Log::DefaultLog.WriteError(
                                    "[Camera] Found no valid projection. [%s, %s, line %d]\n", __FILE__, __FUNCTION__,
                                    __LINE__);
                            }
                        } else {
                            megamol::core::utility::log::Log::DefaultLog.WriteError(
                                "[Camera] Found no valid projection. [%s, %s, line %d]\n", __FILE__, __FUNCTION__,
                                __LINE__);
                        }
                    }
                }

                // Apply showing skybox side ONLY if new camera parameters are set.
                // Non-permanent overwrite of selected keyframe camera by skybox camera settings.
                if (this->sbSide != CinematicView::SkyboxSides::SKYBOX_NONE) {
                    auto pose = this->_camera.get<Camera::Pose>();
                    const glm::vec3 cam_pos = pose.position;
                    const glm::vec3 cam_right = pose.right;
                    const glm::vec3 cam_up = pose.up;
                    const glm::vec3 cam_view = pose.direction;

                    glm::vec3 new_pos = cam_pos;
                    glm::quat new_orientation;
                    const float Rad180Degrees = glm::radians(180.0f);
                    const float Rad90Degrees = glm::radians(90.0f);

                    if (!this->skyboxCubeMode) {
                        if (this->sbSide == CinematicView::SkyboxSides::SKYBOX_BACK) {
                            new_orientation = glm::rotate(new_orientation, Rad180Degrees, cam_right);
                            new_orientation = glm::rotate(new_orientation, Rad180Degrees, cam_view);
                        } else if (this->sbSide == CinematicView::SkyboxSides::SKYBOX_RIGHT) {
                            new_orientation = glm::rotate(new_orientation, Rad90Degrees, cam_up);
                        } else if (this->sbSide == CinematicView::SkyboxSides::SKYBOX_LEFT) {
                            new_orientation = glm::rotate(new_orientation, -Rad90Degrees, cam_up);
                        } else if (this->sbSide == CinematicView::SkyboxSides::SKYBOX_UP) {
                            new_orientation = glm::rotate(new_orientation, -Rad90Degrees, cam_right);
                        } else if (this->sbSide == CinematicView::SkyboxSides::SKYBOX_DOWN) {
                            new_orientation = glm::rotate(new_orientation, Rad90Degrees, cam_right);
                        }
                    } else {
                        auto const center_ = cr3d->AccessBoundingBoxes().BoundingBox().CalcCenter();
                        const glm::vec3 center = glm::vec4(center_.X(), center_.Y(), center_.Z(), 1.0f);
                        const float width = cr3d->AccessBoundingBoxes().BoundingBox().Width();
                        const float height = cr3d->AccessBoundingBoxes().BoundingBox().Height();
                        const float depth = cr3d->AccessBoundingBoxes().BoundingBox().Depth();
                        if (this->sbSide == CinematicView::SkyboxSides::SKYBOX_FRONT) {
                            new_pos = center - depth * cam_view;
                        } else if (this->sbSide == CinematicView::SkyboxSides::SKYBOX_BACK) {
                            new_pos = center + depth * cam_view;
                            new_orientation = glm::rotate(new_orientation, Rad180Degrees, cam_right);
                            new_orientation = glm::rotate(new_orientation, Rad180Degrees, cam_view);
                        } else if (this->sbSide == CinematicView::SkyboxSides::SKYBOX_RIGHT) {
                            new_pos = center + width * cam_right;
                            new_orientation = glm::rotate(new_orientation, Rad90Degrees, cam_up);
                        } else if (this->sbSide == CinematicView::SkyboxSides::SKYBOX_LEFT) {
                            new_pos = center - width * cam_right;
                            new_orientation = glm::rotate(new_orientation, -Rad90Degrees, cam_up);
                        } else if (this->sbSide == CinematicView::SkyboxSides::SKYBOX_UP) {
                            new_pos = center + height * cam_up;
                            new_orientation = glm::rotate(new_orientation, -Rad90Degrees, cam_right);
                        } else if (this->sbSide == CinematicView::SkyboxSides::SKYBOX_DOWN) {
                            new_pos = center - height * cam_up;
                            new_orientation = glm::rotate(new_orientation, Rad90Degrees, cam_right);
                        }
                    }
                    // Apply new position, orientation and aperture angle to current camera.
                    if (this->_camera.get<Camera::ProjectionType>() == Camera::PERSPECTIVE) {
                        auto cam_intrinsics = this->_camera.get<Camera::PerspectiveParameters>();
                        cam_intrinsics.fovy = 90.0f / 180.0f * 3.14f; /// TODO proper conversion deg to rad
                        this->_camera = Camera(Camera::Pose(new_pos, new_orientation), cam_intrinsics);
                    } else {

                        /// TODO ?

                        megamol::core::utility::log::Log::DefaultLog.WriteError(
                            "[Camera] Found no valid projection. [%s, %s, line %d]\n", __FILE__, __FUNCTION__,
                            __LINE__);
                    }
                }

                // Viewport ---------------------------------------------------------------
                /// Assume current framebuffer resolution to be used as viewport resolution
                const int vp_iw = this->_fbo->getWidth();
                const int vp_ih = this->_fbo->getHeight();
                const int vp_ih_reduced = vp_ih - static_cast<int>(this->utils.GetTextLineHeight());
                const float vp_fw = static_cast<float>(vp_iw);
                const float vp_fh = static_cast<float>(vp_ih);
                const float vp_fh_reduced = static_cast<float>(vp_ih_reduced);
                const float cineRatio = static_cast<float>(this->cineWidth) / static_cast<float>(this->cineHeight);
                glm::mat4 ortho = glm::ortho(0.0f, vp_fw, 0.0f, vp_fh, -1.0f, 1.0f);

                // FBO viewport
                int fboWidth = vp_iw;
                int fboHeight = vp_ih_reduced;
                if (this->rendering) {
                    fboWidth = this->cineWidth;
                    fboHeight = this->cineHeight;
                } else {
                    float vpRatio = vp_fw / vp_fh_reduced;
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

                // Render to fbo ----------------------------------------------
                const auto tmp_fbo = this->_fbo;
                const auto tmp_cam = this->_camera;

                if (this->cinematicFbo == nullptr) {
                    this->cinematicFbo = std::make_shared<glowl::FramebufferObject>(fboWidth, fboHeight);
                    this->cinematicFbo->createColorAttachment(GL_RGB8, GL_RGB, GL_UNSIGNED_BYTE);
                } else {
                    if ((this->cinematicFbo->getWidth() != fboWidth) ||
                        (this->cinematicFbo->getHeight() != fboHeight)) {
                        this->cinematicFbo->resize(fboWidth, fboHeight);
                    }
                }
                this->_fbo = this->cinematicFbo;

                auto pose = this->_camera.get<Camera::Pose>();
                if (this->_camera.get<Camera::ProjectionType>() == Camera::PERSPECTIVE) {
                    auto cam_intrinsics = this->_camera.get<Camera::PerspectiveParameters>();
                    cam_intrinsics.aspect = static_cast<float>(fboWidth) / static_cast<float>(fboHeight);
                    this->_camera = Camera(pose, cam_intrinsics);
                } else if (this->_camera.get<Camera::ProjectionType>() == Camera::ORTHOGRAPHIC) {
                    auto cam_intrinsics = this->_camera.get<Camera::OrthographicParameters>();
                    cam_intrinsics.aspect = static_cast<float>(fboWidth) / static_cast<float>(fboHeight);
                    this->_camera = Camera(pose, cam_intrinsics);
                } else {
                    megamol::core::utility::log::Log::DefaultLog.WriteError(
                        "[Camera] Found no valid projection. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
                }

                Base::Render(time, instanceTime);

                this->_fbo = tmp_fbo;
                this->_camera = tmp_cam;

                // Write frame to file
                if (this->rendering) {
                    // Lock writing frame to file for specific time
                    std::chrono::duration<double> diff = (std::chrono::system_clock::now() - this->png_data.start_time);
                    if (diff.count() >
                        static_cast<double>(this->delayFirstRenderFrameParam.Param<param::FloatParam>()->Value())) {
                        this->png_data.write_lock = 0;
                    }
                    this->render_to_file_write();
                }

                // Render Decoration ------------------------------------------
                this->_fbo->bind();

                // Set letter box background
                auto bgcol = this->BackgroundColor();
                this->utils.SetBackgroundColor(bgcol);
                bgcol = this->utils.Color(CinematicUtils::Colors::LETTER_BOX);
                this->utils.SetBackgroundColor(bgcol);
                glClearColor(bgcol.r, bgcol.g, bgcol.b, bgcol.a);
                glClearDepth(1.0f);
                glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
                glViewport(0, 0, vp_fw, vp_fh);

                // Push fbo texture
                float right = (vp_fw + static_cast<float>(texWidth)) / 2.0f;
                float left = (vp_fw - static_cast<float>(texWidth)) / 2.0f;
                float bottom = (vp_fh_reduced + static_cast<float>(texHeight)) / 2.0f;
                float up = (vp_fh_reduced - static_cast<float>(texHeight)) / 2.0f;
                glm::vec3 pos_bottom_left = {left, bottom, 1.0f};
                glm::vec3 pos_upper_left = {left, up, 1.0f};
                glm::vec3 pos_upper_right = {right, up, 1.0f};
                glm::vec3 pos_bottom_right = {right, bottom, 1.0f};
                this->utils.Push2DColorTexture(this->cinematicFbo->getColorAttachment(0)->getName(), pos_bottom_left,
                    pos_upper_left, pos_upper_right, pos_bottom_right, true, glm::vec4(0.0f), true);

                // Push menu
                std::string leftLabel = " CINEMATIC ";
                std::string midLabel = "";
                if (this->rendering) {
                    midLabel = " ! RENDERING IN PROGRESS ! ";
                } else if (this->playAnim) {
                    midLabel = " Playing Animation ";
                }
                std::string rightLabel = "";
                this->utils.PushMenu(ortho, leftLabel, midLabel, rightLabel, glm::vec2(vp_fw, vp_fh), 1.0f);

                // Draw 2D
                this->utils.DrawAll(ortho, glm::vec2(vp_fw, vp_fh));

                glBindFramebuffer(GL_FRAMEBUFFER, 0);
            }
        }
    }

    return GetRenderingResult();
}


bool CinematicView::render_to_file_setup() {

    auto ccc = this->keyframeKeeperSlot.CallAs<cinematic::CallKeyframeKeeper>();
    if (ccc == nullptr) {
        return false;
    }

    // init png data struct
    if (this->cinematicFbo->getColorAttachment(0)->getFormat() == GL_RGB) {
        this->png_data.bpp = 3;
    } else if (this->cinematicFbo->getColorAttachment(0)->getFormat() == GL_RGBA) {
        this->png_data.bpp = 4;
    } else {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "[CINEMATIC VIEW] [render_to_file_setup] Unknown color attachment format. [%s, %s, line %d]\n", __FILE__,
            __FUNCTION__, __LINE__);
        this->rendering = false;
        return false;
    }
    this->png_data.width = static_cast<unsigned int>(this->cineWidth);
    this->png_data.height = static_cast<unsigned int>(this->cineHeight);
    this->png_data.buffer = nullptr;
    this->png_data.structptr = nullptr;
    this->png_data.infoptr = nullptr;
    this->png_data.write_lock = 1;
    this->png_data.start_time = std::chrono::system_clock::now();

    unsigned int firstFrame = static_cast<unsigned int>(this->firstRenderFrameParam.Param<param::IntParam>()->Value());
    unsigned int lastFrame = static_cast<unsigned int>(this->lastRenderFrameParam.Param<param::IntParam>()->Value());

    unsigned int maxFrame = (unsigned int) (ccc->GetTotalAnimTime() * (float) this->fps);
    if (firstFrame > maxFrame) {
        megamol::core::utility::log::Log::DefaultLog.WriteWarn(
            "[CINEMATIC VIEW] [render_to_file_setup] Max frame count exceeded. Limiting first frame to maximum frame "
            "%d",
            maxFrame);
        firstFrame = maxFrame;
    }
    if (firstFrame > lastFrame) {
        megamol::core::utility::log::Log::DefaultLog.WriteWarn(
            "[CINEMATIC VIEW] [render_to_file_setup] First frame exceeds last frame. Limiting first frame to last "
            "frame %d",
            lastFrame);
        firstFrame = lastFrame;
    }

    this->png_data.cnt = firstFrame;
    this->png_data.animTime = (float) this->png_data.cnt / (float) this->fps;

    // Calculate pre-decimal point positions for frame counter in filename
    this->png_data.exp_frame_cnt = 1;
    float frameCnt = (float) (this->fps) * ccc->GetTotalAnimTime();
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
    this->png_data.path = static_cast<vislib::StringA>(
        this->frameFolderParam.Param<param::FilePathParam>()->Value().generic_u8string().c_str());
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

    // Stop accidentially running animation in view
    param::ParamSlot* animParam = static_cast<param::ParamSlot*>(this->_timeCtrl.GetSlot(0)); // animPlaySlot
    animParam->Param<param::BoolParam>()->SetValue(false);

    megamol::core::utility::log::Log::DefaultLog.WriteInfo(
        "[CINEMATIC VIEW] Started rendering of complete animation...");

    return true;
}


bool CinematicView::render_to_file_write() {

    if (this->png_data.write_lock == 0) {
        auto ccc = this->keyframeKeeperSlot.CallAs<cinematic::CallKeyframeKeeper>();
        if (ccc == nullptr)
            return false;

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
        this->png_data.structptr =
            png_create_write_struct(PNG_LIBPNG_VER_STRING, nullptr, &this->pngError, &this->pngWarn);
        if (this->png_data.structptr == nullptr) {
            throw vislib::Exception(
                "[CINEMATIC VIEW] [render_to_file_write] Unable to create png structure. ", __FILE__, __LINE__);
        }
        this->png_data.infoptr = png_create_info_struct(this->png_data.structptr);
        if (this->png_data.infoptr == nullptr) {
            throw vislib::Exception(
                "[CINEMATIC VIEW] [render_to_file_write] Unable to create png info. ", __FILE__, __LINE__);
        }
        png_set_write_fn(
            this->png_data.structptr, static_cast<void*>(&this->png_data.file), &this->pngWrite, &this->pngFlush);

        std::string project;
        auto& megamolgraph = frontend_resources.get<megamol::core::MegaMolGraph>();
        project = const_cast<megamol::core::MegaMolGraph&>(megamolgraph).Convenience().SerializeGraph();

        megamol::core::utility::graphics::ScreenShotComments ssc(project);
        png_set_text(
            this->png_data.structptr, this->png_data.infoptr, ssc.GetComments().data(), ssc.GetComments().size());
        png_set_IHDR(this->png_data.structptr, this->png_data.infoptr, this->png_data.width, this->png_data.height, 8,
            PNG_COLOR_TYPE_RGB, PNG_INTERLACE_NONE, PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);

        {
            /// XXX Throws OpenGL error 1282 (Invalid Operation) - only available since OpenGL 4.5 ...
            /// glGetTextureImage(this->cinematicFbo->getColorAttachment(0)->getName(), 0, GL_RGBA, GL_UNSIGNED_BYTE,
            /// this->png_data.width * this->png_data.height, this->png_data.buffer);
            auto err = glGetError();
            glEnable(GL_TEXTURE_2D);
            glBindTexture(GL_TEXTURE_2D, this->cinematicFbo->getColorAttachment(0)->getName());
            if (glGetError() == GL_NO_ERROR) {
                glGetTexImage(GL_TEXTURE_2D, 0, this->cinematicFbo->getColorAttachment(0)->getFormat(),
                    this->cinematicFbo->getColorAttachment(0)->getType(), this->png_data.buffer);
            }
            glBindTexture(GL_TEXTURE_2D, 0);
            glDisable(GL_TEXTURE_2D);
            err = glGetError();
            if (err != GL_NO_ERROR) {
                throw vislib::Exception(
                    "[CINEMATIC VIEW] [render_to_file_write] Unable to read color texture. ", __FILE__, __LINE__);
            }
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
        } catch (...) {}
        try {
            this->png_data.file.Close();
        } catch (...) {}
        megamol::core::utility::log::Log::DefaultLog.WriteInfo(
            "[CINEMATIC VIEW] [render_to_file_write] Wrote png file %d for animation time %f ...\n", this->png_data.cnt,
            this->png_data.animTime);

        // --------------------------------------------------------------------

        // Next frame/time step
        this->png_data.cnt++;
        this->png_data.animTime = (float) this->png_data.cnt / (float) this->fps;
        float fpsFrac = (1.0f / static_cast<float>(this->fps));
        // Fit animTime to exact full seconds (removing rounding error)
        if (std::abs(this->png_data.animTime - std::round(this->png_data.animTime)) < (fpsFrac / 2.0)) {
            this->png_data.animTime = std::round(this->png_data.animTime);
        }

        /// XXX Handling this case is actually only necessary when rendering is done via FBOCompositor
        /// XXX Rendering crashes - WHY?
        /// XXX Rendering last frame with animation time = total animation time is otherwise no problem
        // if (this->png_data.animTime == ccc->GetTotalAnimTime()) {
        //    this->png_data.animTime -= 0.000005f;
        //} else

        // Check condition for finishing rendering
        auto lastFrame = static_cast<unsigned int>(this->lastRenderFrameParam.Param<param::IntParam>()->Value());
        if ((this->png_data.animTime > ccc->GetTotalAnimTime()) || (this->png_data.cnt > lastFrame)) {
            this->render_to_file_cleanup();
            megamol::core::utility::log::Log::DefaultLog.WriteInfo("[CINEMATIC VIEW] Finished rendering.");
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
        } else {
            png_destroy_write_struct(&this->png_data.structptr, (png_infopp) nullptr);
        }
    }

    try {
        this->png_data.file.Flush();
    } catch (...) {}

    try {
        this->png_data.file.Close();
    } catch (...) {}

    ARY_SAFE_DELETE(this->png_data.buffer);

    if (this->png_data.buffer != nullptr) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "[CINEMATIC VIEW] [render_to_file_cleanup] pngdata.buffer is not nullptr. [%s, %s, line %d]\n", __FILE__,
            __FUNCTION__, __LINE__);
    }
    if (this->png_data.structptr != nullptr) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "[CINEMATIC VIEW] [render_to_file_cleanup] pngdata.structptr is not nullptr. [%s, %s, line %d]\n", __FILE__,
            __FUNCTION__, __LINE__);
    }
    if (this->png_data.infoptr != nullptr) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "[CINEMATIC VIEW] [render_to_file_cleanup] pngdata.infoptr is not nullptr. [%s, %s, line %d]\n", __FILE__,
            __FUNCTION__, __LINE__);
    }

    return true;
}
