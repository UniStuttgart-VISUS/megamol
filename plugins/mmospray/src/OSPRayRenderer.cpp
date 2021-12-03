/*
 * OSPRayRenderer.cpp
 * Copyright (C) 2009-2017 by MegaMol Team
 * Alle Rechte vorbehalten.
 */

#include "OSPRayRenderer.h"
#include "mmcore/param/BoolParam.h"
#include "mmcore/param/EnumParam.h"
#include "mmcore/utility/log/Log.h"
#include "ospray/ospray_cpp.h"
#include "stdafx.h"
#include <chrono>

#include <sstream>
#include <stdint.h>

using namespace megamol::ospray;

/*
ospray::OSPRayRenderer::OSPRaySphereRenderer
*/
OSPRayRenderer::OSPRayRenderer(void)
        : AbstractOSPRayRenderer()
        , _cam()
        , _getStructureSlot("getStructure", "Connects to an OSPRay structure")
        , _enablePickingSlot("enable picking", "")

{
    this->_getStructureSlot.SetCompatibleCall<CallOSPRayStructureDescription>();
    this->MakeSlotAvailable(&this->_getStructureSlot);

    _imgSize = {0, 0};
    _time = 0;
    _framebuffer = nullptr;
    _renderer = nullptr;
    _camera = nullptr;
    _world = nullptr;

    _accum_time.count = 0;
    _accum_time.amount = 0;

    _enablePickingSlot << new core::param::BoolParam(false);
    MakeSlotAvailable(&_enablePickingSlot);
}


/*
ospray::OSPRayRenderer::~OSPRaySphereRenderer
*/
OSPRayRenderer::~OSPRayRenderer(void) {
    this->Release();
}


/*
ospray::OSPRayRenderer::create
*/
bool OSPRayRenderer::create() {
    return true;
}

/*
ospray::OSPRayRenderer::release
*/
void OSPRayRenderer::release() {
    this->clearOSPRayStuff();
    ospShutdown();
}

/*
ospray::OSPRayRenderer::Render
*/
bool OSPRayRenderer::Render(megamol::core::view::CallRender3D& cr) {
    this->initOSPRay();

    // if user wants to switch renderer
    if (this->_rd_type.IsDirty()) {
        //ospRelease(_camera);
        //ospRelease(_world);
        //ospRelease(_renderer);
        switch (this->_rd_type.Param<core::param::EnumParam>()->Value()) {
        case PATHTRACER:
            this->setupOSPRay("pathtracer");
            this->_rd_type_string = "pathtracer";
            break;
        case MPI_RAYCAST: //< TODO: Probably only valid if device is a "mpi_distributed" device
            this->setupOSPRay("mpi_raycast");
            this->_rd_type_string = "mpi_raycast";
            break;
        default:
            this->setupOSPRay("scivis");
            this->_rd_type_string = "scivis";
        }
        _renderer_has_changed = true;
        this->_rd_type.ResetDirty();
    }

    if (&cr == nullptr)
        return false;


    CallOSPRayStructure* os = this->_getStructureSlot.CallAs<CallOSPRayStructure>();
    if (os == nullptr)
        return false;
    // read data
    os->setStructureMap(&_structureMap);
    os->setTime(cr.Time());
    if (!os->fillStructureMap())
        return false;
    // check if data has changed
    _data_has_changed = false;
    _material_has_changed = false;
    _transformation_has_changed = false;
    _clipping_geo_changed = false;
    for (auto element : this->_structureMap) {
        auto structure = element.second;
        if (structure.dataChanged) {
            _data_has_changed = true;
        }
        if (structure.materialChanged) {
            _material_has_changed = true;
        }
        if (structure.transformationChanged) {
            _transformation_has_changed = true;
        }
        if (structure.clippingPlaneChanged) {
            _clipping_geo_changed = true;
        }
    }

    // Light setup
    auto call_light = _lightSlot.CallAs<core::view::light::CallLight>();
    if (call_light != nullptr) {
        if (!(*call_light)(0)) {
            return false;
        }
        _light_has_changed = call_light->hasUpdate();
    }

    using Camera = core::view::Camera;
    Camera cam = cr.GetCamera();

    // check data and camera hash
    if (_cam == cam) {
        _cam_has_changed = false;
    } else {
        _cam_has_changed = true;
    }

    // Generate complete snapshot and calculate matrices
    _cam = cam;

    // glDisable(GL_CULL_FACE);

    // new framebuffer at resize action
    auto fbo = cr.GetFramebuffer();
    if (fbo == nullptr) {
        return false;
    }
    if (fbo->width == 0 && fbo->height == 0)
        return false;

    // bool triggered = false;
    if (_imgSize[0] != fbo->width || _imgSize[1] != fbo->height || _accumulateSlot.IsDirty()) {
        // triggered = true;
        // Breakpoint for Screenshooter debugging
        // if (framebuffer != NULL) ospFreeFrameBuffer(framebuffer);
        _imgSize[0] = fbo->width;
        _imgSize[1] = fbo->height;
        _framebuffer = std::make_shared<::ospray::cpp::FrameBuffer>(
            _imgSize[0], _imgSize[1], OSP_FB_RGBA8, OSP_FB_COLOR | OSP_FB_DEPTH | OSP_FB_ACCUM);
        _db.resize(_imgSize[0] * _imgSize[1]);
        _framebuffer->commit();
    }

    //// if user wants to switch renderer
    // if (this->rd_type.IsDirty()) {
    //    ospRelease(camera);
    //    ospRelease(world);
    //    ospRelease(renderer);
    //    switch (this->rd_type.Param<core::param::EnumParam>()->Value()) {
    //    case PATHTRACER:
    //        this->setupOSPRay(renderer, camera, world, "pathtracer");
    //        break;
    //    case MPI_RAYCAST: //< TODO: Probably only valid if device is a "mpi_distributed" device
    //        this->setupOSPRay(renderer, camera, world, "mpi_raycast");
    //        break;
    //    default:
    //        this->setupOSPRay(renderer, camera, world, "scivis");
    //    }
    //    renderer_has_changed = true;
    //}
    setupOSPRayCamera(_cam);
    _camera->commit();

    // if nothing changes, the image is rendered multiple times
    if (_data_has_changed || _material_has_changed || _light_has_changed || _cam_has_changed || _renderer_has_changed ||
        _transformation_has_changed || _clipping_geo_changed ||
        !(this->_accumulateSlot.Param<core::param::BoolParam>()->Value()) ||
        _frameID != static_cast<size_t>(cr.Time()) || this->InterfaceIsDirty()) {


        auto cam_pose = _cam.get<Camera::Pose>();
        std::array<float, 3> eyeDir = {cam_pose.direction.x, cam_pose.direction.y, cam_pose.direction.z};
        if (_data_has_changed || _frameID != static_cast<size_t>(cr.Time()) || _renderer_has_changed) {
            // || this->InterfaceIsDirty()) {
            if (!this->generateRepresentations())
                return false;
            this->createInstances();
            std::vector<::ospray::cpp::Instance> instanceArray;
            std::transform(_instances.begin(), _instances.end(), std::back_inserter(instanceArray), second(_instances));
            _world->setParam("instance", ::ospray::cpp::CopiedData(instanceArray));

            // Enable Lights
            this->fillLightArray(eyeDir);
            _world->setParam("light", ::ospray::cpp::CopiedData(_lightArray));

            // Commiting world and measuring time
            auto t1 = std::chrono::high_resolution_clock::now();
            _world->commit();
            auto t2 = std::chrono::high_resolution_clock::now();
            const auto duration = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
            megamol::core::utility::log::Log::DefaultLog.WriteMsg(
                242, "[OSPRayRenderer] Commiting World took: %d microseconds", duration);
        }
        if (_material_has_changed && !_data_has_changed) {
            this->changeMaterial();
        }
        if (_transformation_has_changed || _material_has_changed && !_data_has_changed) {
            this->changeTransformation();
            std::vector<::ospray::cpp::Instance> instanceArray;
            std::transform(_instances.begin(), _instances.end(), std::back_inserter(instanceArray), second(_instances));
            _world->setParam("instance", ::ospray::cpp::CopiedData(instanceArray));
            _world->commit();
        }
        if (_light_has_changed && !_data_has_changed) {
            this->fillLightArray(eyeDir);
            _world->setParam("light", ::ospray::cpp::CopiedData(_lightArray));
            _world->commit();
        }


        this->InterfaceResetDirty();
        _time = cr.Time();
        _frameID = static_cast<size_t>(cr.Time());
        _renderer_has_changed = false;

        /*
            if (this->maxDepthTexture) {
                ospRelease(this->maxDepthTexture);
            }
            this->maxDepthTexture = getOSPDepthTextureFromOpenGLPerspective(*cr);
        */
        RendererSettings(cr.BackgroundColor());

        // Only usefull if dephbuffer is used as input
        //if (this->_useDB.Param<core::param::BoolParam>()->Value()) {
        //    // far distance
        //    float far_clip = _cam.far_clipping_plane();
        //    std::vector<float> far_dist(_imgSize[0] * _imgSize[1], far_clip);
        //    rkcommon::math::vec2i imgSize = {
        //        _imgSize[0],
        //        _imgSize[1]
        //    };

        //    auto depth_texture_data = ::ospray::cpp::CopiedData(far_dist.data(), OSP_FLOAT, imgSize);
        //    depth_texture_data.commit();
        //    auto depth_texture = ::ospray::cpp::Texture("texture2d");
        //    depth_texture.setParam("format", OSP_TEXTURE_R32F);
        //    depth_texture.setParam("filter", OSP_TEXTURE_FILTER_NEAREST);
        //    depth_texture.setParam("data", depth_texture_data);
        //    depth_texture.commit();

        //    _renderer->setParam("map_maxDepth", depth_texture);
        //} else {
        //    _renderer->setParam("map_maxDepth", NULL);
        //}

        _renderer->commit();

        // setup framebuffer and measure time
        auto t1 = std::chrono::high_resolution_clock::now();

        _framebuffer->clear(); //(OSP_FB_COLOR | OSP_FB_DEPTH | OSP_FB_ACCUM);
        _framebuffer->renderFrame(*_renderer, *_camera, *_world);

        // get the texture from the framebuffer
        auto fb = reinterpret_cast<uint32_t*>(_framebuffer->map(OSP_FB_COLOR));
        _fb = std::vector<uint32_t>(fb, fb + _imgSize[0] * _imgSize[1]);

        auto t2 = std::chrono::high_resolution_clock::now();
        const auto duration = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1);

        _accum_time.amount += duration.count();
        _accum_time.count += 1;
        if (_accum_time.amount >= static_cast<unsigned long long int>(1e6)) {
            const unsigned long long int mean_rendertime = _accum_time.amount / _accum_time.count;
            megamol::core::utility::log::Log::DefaultLog.WriteMsg(
                242, "[OSPRayRenderer] Rendering took: %d microseconds", mean_rendertime);
            _accum_time.count = 0;
            _accum_time.amount = 0;
        }


        if (this->_useDB.Param<core::param::BoolParam>()->Value()) {
            getOpenGLDepthFromOSPPerspective(_db);
        }

        // write a sequence of single pictures while the screenshooter is running
        // only for debugging
        // if (triggered) {
        //    std::ostringstream oss;
        //    oss << "ospframe" << this->number << ".ppm";
        //    std::string bla = oss.str();
        //    const char* fname = bla.c_str();
        //    osp::vec2i isize;
        //    isize.x = cr->GetCameraParameters()->TileRect().GetSize().GetWidth();
        //    isize.y = cr->GetCameraParameters()->TileRect().GetSize().GetHeight();
        //    writePPM(fname, isize, fb);
        //    this->number++;
        //}

        //std::string fname("blub.ppm");
        //writePPM(fname, _imgSize, fb);

        auto frmbuffer = cr.GetFramebuffer();
        frmbuffer->width = _imgSize[0];
        frmbuffer->height = _imgSize[1];
        frmbuffer->depthBuffer = _db;
        frmbuffer->colorBuffer = _fb;
        frmbuffer->depthBufferActive = this->_useDB.Param<core::param::BoolParam>()->Value();

        // clear stuff
        _framebuffer->unmap(fb);

        //auto dvce_ = ospGetCurrentDevice();
        //auto error_ = std::string(ospDeviceGetLastErrorMsg(dvce_));
        //megamol::core::utility::log::Log::DefaultLog.WriteError(std::string("OSPRAY last ERROR: " + error_).c_str());

    } else {
        // setup framebuffer and measure time
        auto t1 = std::chrono::high_resolution_clock::now();

        _framebuffer->renderFrame(*_renderer, *_camera, *_world);
        auto fb = reinterpret_cast<uint32_t*>(_framebuffer->map(OSP_FB_COLOR));
        _fb = std::vector<uint32_t>(fb, fb + _imgSize[0] * _imgSize[1]);

        auto frmbuffer = cr.GetFramebuffer();
        frmbuffer->width = _imgSize[0];
        frmbuffer->height = _imgSize[1];
        frmbuffer->depthBuffer = _db;
        frmbuffer->colorBuffer = _fb;

        _framebuffer->unmap(fb);

        auto t2 = std::chrono::high_resolution_clock::now();
        const auto duration = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1);

        _accum_time.amount += duration.count();
        _accum_time.count += 1;
        if (_accum_time.amount >= static_cast<unsigned long long int>(1e6)) {
            const unsigned long long int mean_rendertime = _accum_time.amount / _accum_time.count;
            megamol::core::utility::log::Log::DefaultLog.WriteMsg(
                242, "[OSPRayRenderer] Rendering took: %d microseconds", mean_rendertime);
            _accum_time.count = 0;
            _accum_time.amount = 0;
        }
    }

    return true;
}

bool OSPRayRenderer::OnMouseButton(
    core::view::MouseButton button, core::view::MouseButtonAction action, core::view::Modifiers mods) {
    if (mods.test(core::view::Modifier::SHIFT) && action == core::view::MouseButtonAction::PRESS &&
        _enablePickingSlot.Param<core::param::BoolParam>()->Value()) {
        auto const screenX = _mouse_x / _imgSize[0];
        auto const screenY = 1.f - (_mouse_y / _imgSize[1]);
        auto const pick_res = _framebuffer->pick(*_renderer, *_camera, *_world, screenX, screenY);

        for (auto const& entry : _geometricModels) {
            entry.first->setPickResult(-1, -1);
            if (pick_res.hasHit) {
                auto const fit = std::find(entry.second.begin(), entry.second.end(), pick_res.model);
                if (fit != entry.second.end()) {
                    entry.first->setPickResult(std::distance(entry.second.begin(), fit), pick_res.primID);
                }
                //core::utility::log::Log::DefaultLog.WriteInfo("[OSPRayRenderer] Pick result %d", pick_res.primID);
            }
        }

        return true;
    }

    return false;
}

bool OSPRayRenderer::OnMouseMove(double x, double y) {
    this->_mouse_x = static_cast<float>(x);
    this->_mouse_y = static_cast<float>(y);
    return false;
}

/*
ospray::OSPRayRenderer::InterfaceIsDirty()
*/
bool OSPRayRenderer::InterfaceIsDirty() {
    if (this->AbstractIsDirty()) {
        return true;
    } else {
        return false;
    }
}

/*
ospray::OSPRayRenderer::InterfaceResetDirty()
*/
void OSPRayRenderer::InterfaceResetDirty() {
    this->AbstractResetDirty();
}


/*
 * ospray::OSPRayRenderer::GetExtents
 */
bool OSPRayRenderer::GetExtents(megamol::core::view::CallRender3D& cr) {

    if (&cr == NULL)
        return false;
    CallOSPRayStructure* os = this->_getStructureSlot.CallAs<CallOSPRayStructure>();
    if (os == NULL)
        return false;
    os->setTime(static_cast<int>(cr.Time()));
    os->setExtendMap(&(this->_extendMap));
    if (!os->fillExtendMap())
        return false;

    megamol::core::BoundingBoxes_2 finalBox;
    unsigned int frameCnt = 0;
    for (auto pair : this->_extendMap) {
        auto element = pair.second;

        if (frameCnt == 0) {
            if (element.boundingBox->IsBoundingBoxValid()) {
                finalBox.SetBoundingBox(element.boundingBox->BoundingBox());
            } else if (element.boundingBox->IsClipBoxValid()) {
                finalBox.SetBoundingBox(element.boundingBox->ClipBox());
            } else {
                finalBox.SetBoundingBox(vislib::math::Cuboid<float>(-1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f));
            }
            if (element.boundingBox->IsClipBoxValid()) {
                finalBox.SetClipBox(element.boundingBox->ClipBox());
            } else if (element.boundingBox->IsBoundingBoxValid()) {
                finalBox.SetClipBox(element.boundingBox->BoundingBox());
            } else {
                finalBox.SetClipBox(vislib::math::Cuboid<float>(-1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f));
            }

        } else {
            if (element.boundingBox->IsBoundingBoxValid()) {
                vislib::math::Cuboid<float> box(finalBox.BoundingBox());
                box.Union(element.boundingBox->BoundingBox());
                finalBox.SetBoundingBox(box);
            } else if (element.boundingBox->IsClipBoxValid()) {
                vislib::math::Cuboid<float> box(finalBox.BoundingBox());
                box.Union(element.boundingBox->BoundingBox());
                finalBox.SetBoundingBox(box);
            }
            if (element.boundingBox->IsClipBoxValid()) {
                vislib::math::Cuboid<float> box(finalBox.ClipBox());
                box.Union(element.boundingBox->ClipBox());
                finalBox.SetClipBox(box);
            } else if (element.boundingBox->IsBoundingBoxValid()) {
                vislib::math::Cuboid<float> box(finalBox.ClipBox());
                box.Union(element.boundingBox->BoundingBox());
                finalBox.SetClipBox(box);
            }
        }
        frameCnt = vislib::math::Max(frameCnt, element.timeFramesCount);
    }
    cr.SetTimeFramesCount(frameCnt);

    cr.AccessBoundingBoxes().SetBoundingBox(finalBox.BoundingBox());
    cr.AccessBoundingBoxes().SetBoundingBox(finalBox.ClipBox());

    return true;
}

void OSPRayRenderer::getOpenGLDepthFromOSPPerspective(std::vector<float>& db) {

    auto proj_matrix = _cam.getProjectionMatrix();
    auto cam_pose = _cam.get<core::view::Camera::Pose>();
    auto cam_intrinsics = _cam.get<core::view::Camera::PerspectiveParameters>();

    const float fovy = cam_intrinsics.fovy;
    const float aspect = cam_intrinsics.aspect;

    const glm::vec3 cameraUp = cam_pose.up;
    const glm::vec3 cameraDir = cam_pose.direction;

    // map OSPRay depth buffer from provided frame buffer
    auto ospDepthBuffer = static_cast<float*>(_framebuffer->map(OSP_FB_DEPTH));

    const auto ospDepthBufferWidth = static_cast<const size_t>(_imgSize[0]);
    const auto ospDepthBufferHeight = static_cast<const size_t>(_imgSize[1]);

    db.resize(ospDepthBufferWidth * ospDepthBufferHeight);

    // transform from ray distance t to orthogonal Z depth
    auto dir_du = glm::normalize(glm::cross(cameraDir, cameraUp));
    auto dir_dv = glm::normalize(glm::cross(dir_du, cameraDir));

    const float imagePlaneSizeY = 2.f * tanf(fovy / 2.f * M_PI / 180.f);
    const float imagePlaneSizeX = imagePlaneSizeY * aspect;

    dir_du *= imagePlaneSizeX;
    dir_dv *= imagePlaneSizeY;

    const auto dir_00 = cameraDir - .5f * dir_du - .5f * dir_dv;

    // transform from linear to nonlinear OpenGL depth
    const auto A = proj_matrix[2][2];
    const auto B = proj_matrix[3][2];

    int j, i;
#pragma omp parallel for private(i)
    for (j = 0; j < ospDepthBufferHeight; j++) {
        for (i = 0; i < ospDepthBufferWidth; i++) {
            const auto dir_ij = glm::normalize(dir_00 + float(i) / float(ospDepthBufferWidth - 1) * dir_du +
                                               float(j) / float(ospDepthBufferHeight - 1) * dir_dv);

            const float tmp = ospDepthBuffer[j * ospDepthBufferWidth + i];
            float res = 0.5 * (-A * tmp + B) / tmp + 0.5;
            if (!std::isfinite(res))
                res = 1.0f;
            db[j * ospDepthBufferWidth + i] = res;
        }
    }
    // unmap OSPRay depth buffer
    _framebuffer->unmap(ospDepthBuffer);
}
