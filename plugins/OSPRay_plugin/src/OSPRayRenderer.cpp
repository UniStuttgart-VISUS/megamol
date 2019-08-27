/*
 * OSPRayRenderer.cpp
 * Copyright (C) 2009-2017 by MegaMol Team
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "OSPRayRenderer.h"
#include "mmcore/param/BoolParam.h"
#include "mmcore/param/EnumParam.h"
#include "vislib/graphics/CameraParamsStore.h"
#include "vislib/graphics/gl/IncludeAllGL.h"
#include "vislib/graphics/gl/ShaderSource.h"
#include "vislib/math/Vector.h"
#include "vislib/sys/Log.h"

#include "mmcore/CoreInstance.h"

#include <chrono>
#include <functional>

#include "ospcommon/vec.h"
#include "ospray/ospray.h"

#include <sstream>
#include <stdint.h>

using namespace megamol::ospray;

/*
ospray::OSPRayRenderer::OSPRaySphereRenderer
*/
OSPRayRenderer::OSPRayRenderer(void)
    : AbstractOSPRayRenderer()
	, cam()
    , osprayShader()
    , getStructureSlot("getStructure", "Connects to an OSPRay structure")

{
    this->getStructureSlot.SetCompatibleCall<CallOSPRayStructureDescription>();
    this->MakeSlotAvailable(&this->getStructureSlot);


    imgSize.x = 0;
    imgSize.y = 0;
    time = 0;
    framebuffer = NULL;
    renderer = NULL;
    camera = NULL;
    world = NULL;

    accum_time.count = 0;
    accum_time.amount = 0;
}


/*
ospray::OSPRayRenderer::~OSPRaySphereRenderer
*/
OSPRayRenderer::~OSPRayRenderer(void) {
    this->osprayShader.Release();
    this->Release();
}


/*
ospray::OSPRayRenderer::create
*/
bool OSPRayRenderer::create() {
    ASSERT(IsAvailable());

    vislib::graphics::gl::ShaderSource vert, frag;

    if (!instance()->ShaderSourceFactory().MakeShaderSource("ospray::vertex", vert)) {
        return false;
    }
    if (!instance()->ShaderSourceFactory().MakeShaderSource("ospray::fragment", frag)) {
        return false;
    }

    try {
        if (!this->osprayShader.Create(vert.Code(), vert.Count(), frag.Code(), frag.Count())) {
            vislib::sys::Log::DefaultLog.WriteMsg(
                vislib::sys::Log::LEVEL_ERROR, "Unable to compile ospray shader: Unknown error\n");
            return false;
        }
    } catch (vislib::graphics::gl::AbstractOpenGLShader::CompileException ce) {
        vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
            "Unable to compile ospray shader: (@%s): %s\n",
            vislib::graphics::gl::AbstractOpenGLShader::CompileException::CompileActionName(ce.FailedAction()),
            ce.GetMsgA());
        return false;
    } catch (vislib::Exception e) {
        vislib::sys::Log::DefaultLog.WriteMsg(
            vislib::sys::Log::LEVEL_ERROR, "Unable to compile ospray shader: %s\n", e.GetMsgA());
        return false;
    } catch (...) {
        vislib::sys::Log::DefaultLog.WriteMsg(
            vislib::sys::Log::LEVEL_ERROR, "Unable to compile ospray shader: Unknown exception\n");
        return false;
    }

    // this->initOSPRay(device);
    this->setupTextureScreen();
    // this->setupOSPRay(renderer, camera, world, "scivis");

    return true;
}

/*
ospray::OSPRayRenderer::release
*/
void OSPRayRenderer::release() {
    if (camera != NULL) ospRelease(camera);
    if (world != NULL) ospRelease(world);
    if (renderer != NULL) ospRelease(renderer);
    releaseTextureScreen();
}

/*
ospray::OSPRayRenderer::Render
*/
bool OSPRayRenderer::Render(megamol::core::view::CallRender3D_2& cr) {
    this->initOSPRay(device);

    if (device != ospGetCurrentDevice()) {
        ospSetCurrentDevice(device);
    }

    // if user wants to switch renderer
    if (this->rd_type.IsDirty()) {
        ospRelease(camera);
        ospRelease(world);
        ospRelease(renderer);
        switch (this->rd_type.Param<core::param::EnumParam>()->Value()) {
        case PATHTRACER:
            this->setupOSPRay(renderer, camera, world, "pathtracer");
            this->rd_type_string = "pathtracer";
            break;
        case MPI_RAYCAST: //< TODO: Probably only valid if device is a "mpi_distributed" device
            this->setupOSPRay(renderer, camera, world, "mpi_raycast");
            this->rd_type_string = "mpi_raycast";
            break;
        default:
            this->setupOSPRay(renderer, camera, world, "scivis");
            this->rd_type_string = "scivis";
        }
        renderer_has_changed = true;
        this->rd_type.ResetDirty();
    }

    if (&cr == nullptr) return false;


    CallOSPRayStructure* os = this->getStructureSlot.CallAs<CallOSPRayStructure>();
    if (os == nullptr) return false;
    // read data
    os->setStructureMap(&structureMap);
    os->setTime(cr.Time());
    if (!os->fillStructureMap()) return false;
    // check if data has changed
    data_has_changed = false;
    material_has_changed = false;
    for (auto element : this->structureMap) {
        auto structure = element.second;
        if (structure.dataChanged) {
            data_has_changed = true;
        }
        if (structure.materialChanged) {
            material_has_changed = true;
        }
    }

    // Light setup
	this->light_has_changed = this->GetLights();

	core::view::Camera_2 tmp_newcam;
	cr.GetCamera(tmp_newcam);
	cam_type::snapshot_type snapshot;
	cam_type::matrix_type viewTemp, projTemp;

	// Generate complete snapshot and calculate matrices
	tmp_newcam.calc_matrices(snapshot, viewTemp, projTemp, core::thecam::snapshot_content::all);

    // check data and camera hash
	if (cam.eye_position().x() != tmp_newcam.eye_position().x() ||
		cam.eye_position().y() != tmp_newcam.eye_position().y() ||
		cam.eye_position().z() != tmp_newcam.eye_position().z() ||
		cam.view_vector() != tmp_newcam.view_vector()
		) {
		cam_has_changed = true;
	} else {
		cam_has_changed = false;
	}

	// Generate complete snapshot and calculate matrices
	cam = tmp_newcam;

    // glDisable(GL_CULL_FACE);

    // new framebuffer at resize action
    // bool triggered = false;
    if (imgSize.x != cam.resolution_gate().width() ||
        imgSize.y != cam.resolution_gate().height() || accumulateSlot.IsDirty()) {
        // triggered = true;
        // Breakpoint for Screenshooter debugging
        if (framebuffer != NULL) ospFreeFrameBuffer(framebuffer);
        imgSize.x = cam.resolution_gate().width();
        imgSize.y = cam.resolution_gate().height();
        framebuffer = newFrameBuffer(imgSize, OSP_FB_RGBA8, OSP_FB_COLOR | OSP_FB_DEPTH | OSP_FB_ACCUM);
        db.resize(imgSize.x * imgSize.y);
        ospCommit(framebuffer);
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
    setupOSPRayCamera(camera, cam);
    ospCommit(camera);

    osprayShader.Enable();
    // if nothing changes, the image is rendered multiple times
    if (data_has_changed || material_has_changed || light_has_changed || cam_has_changed || renderer_has_changed ||
        !(this->accumulateSlot.Param<core::param::BoolParam>()->Value()) ||
        frameID != static_cast<size_t>(cr.Time()) || this->InterfaceIsDirty()) {

        if (data_has_changed || frameID != static_cast<size_t>(cr.Time()) || renderer_has_changed) {
            // || this->InterfaceIsDirty()) {
            if (!this->fillWorld()) return false;

            // Commiting world and measuring time
            auto t1 = std::chrono::high_resolution_clock::now();
            ospCommit(world);
            auto t2 = std::chrono::high_resolution_clock::now();
            const auto duration = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
            vislib::sys::Log::DefaultLog.WriteMsg(242, "OSPRayRenderer: Commiting World took: %d microseconds", duration);
        }
        if (material_has_changed && !data_has_changed) {
            this->changeMaterial();
        }
        this->InterfaceResetDirty();
        time = cr.Time();
        frameID = static_cast<size_t>(cr.Time());
        renderer_has_changed = false;

        /*
            if (this->maxDepthTexture) {
                ospRelease(this->maxDepthTexture);
            }
            this->maxDepthTexture = getOSPDepthTextureFromOpenGLPerspective(*cr);
        */
        RendererSettings(renderer);


        // Enable Lights
        auto eyeDir = cam.view_vector();
        this->fillLightArray(eyeDir);
        lightsToRender = ospNewData(this->lightArray.size(), OSP_OBJECT, lightArray.data(), 0);
        ospCommit(lightsToRender);
        ospSetData(renderer, "lights", lightsToRender);
        
        ospSetObject(renderer, "model", world);
        ospCommit(renderer);

        // setup framebuffer and measure time
        auto t1 = std::chrono::high_resolution_clock::now();

        ospFrameBufferClear(framebuffer, OSP_FB_COLOR | OSP_FB_DEPTH | OSP_FB_ACCUM);
        ospRenderFrame(framebuffer, renderer, OSP_FB_COLOR | OSP_FB_DEPTH | OSP_FB_ACCUM);
        // get the texture from the framebuffer
        fb = (uint32_t*)ospMapFrameBuffer(framebuffer, OSP_FB_COLOR);
        auto t2 = std::chrono::high_resolution_clock::now();
        const auto duration = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1);

        accum_time.amount += duration.count();
        accum_time.count += 1;
        if (accum_time.amount >= static_cast<unsigned long long int>(1e6)) {
            const unsigned long long int mean_rendertime = accum_time.amount / accum_time.count;
            vislib::sys::Log::DefaultLog.WriteMsg(242, "OSPRayRenderer: Rendering took: %d microseconds", mean_rendertime);
            accum_time.count = 0;
            accum_time.amount = 0;
        }

        if (this->useDB.Param<core::param::BoolParam>()->Value()) {
            getOpenGLDepthFromOSPPerspective(db.data());
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

        this->renderTexture2D(osprayShader, fb, db.data(), imgSize.x, imgSize.y, cr);

        // clear stuff
        ospUnmapFrameBuffer(fb, framebuffer);


        this->releaseOSPRayStuff();


    } else {
        ospRenderFrame(framebuffer, renderer, OSP_FB_COLOR | OSP_FB_DEPTH | OSP_FB_ACCUM);
        fb = (uint32_t*)ospMapFrameBuffer(framebuffer, OSP_FB_COLOR);

        this->renderTexture2D(osprayShader, fb, db.data(), imgSize.x, imgSize.y, cr);
        ospUnmapFrameBuffer(fb, framebuffer);
    }

    osprayShader.Disable();

    return true;
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
void OSPRayRenderer::InterfaceResetDirty() { this->AbstractResetDirty(); }


/*
 * ospray::OSPRayRenderer::GetExtents
 */
bool OSPRayRenderer::GetExtents(megamol::core::view::CallRender3D_2& cr) {

    if (&cr == NULL) return false;
    CallOSPRayStructure* os = this->getStructureSlot.CallAs<CallOSPRayStructure>();
    if (os == NULL) return false;
    os->setTime(static_cast<int>(cr.Time()));
    os->setExtendMap(&(this->extendMap));
    if (!os->fillExtendMap()) return false;

    megamol::core::BoundingBoxes_2 finalBox;
    unsigned int frameCnt = 0;
    for (auto pair : this->extendMap) {
        auto element = pair.second;

        if (frameCnt == 0) {
            if (element.boundingBox->IsObjectSpaceBBoxValid()) {
                finalBox.SetBoundingBox(element.boundingBox->ObjectSpaceBBox());
            } else if (element.boundingBox->IsObjectSpaceClipBoxValid()) {
                finalBox.SetBoundingBox(element.boundingBox->ObjectSpaceClipBox());
            } else {
                finalBox.SetBoundingBox(vislib::math::Cuboid<float>(-1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f));
            }
            if (element.boundingBox->IsObjectSpaceClipBoxValid()) {
                finalBox.SetClipBox(element.boundingBox->ObjectSpaceClipBox());
            } else if (element.boundingBox->IsObjectSpaceBBoxValid()) {
                finalBox.SetClipBox(element.boundingBox->ObjectSpaceBBox());
            } else {
                finalBox.SetClipBox(vislib::math::Cuboid<float>(-1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f));
            }

        } else {
            if (element.boundingBox->IsObjectSpaceBBoxValid()) {
                vislib::math::Cuboid<float> box(finalBox.BoundingBox());
                box.Union(element.boundingBox->ObjectSpaceBBox());
                finalBox.SetBoundingBox(box);
            } else if (element.boundingBox->IsObjectSpaceClipBoxValid()) {
                vislib::math::Cuboid<float> box(finalBox.BoundingBox());
                box.Union(element.boundingBox->ObjectSpaceClipBox());
                finalBox.SetBoundingBox(box);
            }
            if (element.boundingBox->IsObjectSpaceClipBoxValid()) {
                vislib::math::Cuboid<float> box(finalBox.ClipBox());
                box.Union(element.boundingBox->ObjectSpaceClipBox());
                finalBox.SetClipBox(box);
            } else if (element.boundingBox->IsObjectSpaceBBoxValid()) {
                vislib::math::Cuboid<float> box(finalBox.ClipBox());
                box.Union(element.boundingBox->ObjectSpaceBBox());
                finalBox.SetClipBox(box);
            }
        }
        frameCnt = vislib::math::Max(frameCnt, element.timeFramesCount);
    }

	cr.AccessBoundingBoxes().SetBoundingBox(finalBox.BoundingBox());
	cr.AccessBoundingBoxes().SetBoundingBox(finalBox.ClipBox());

    return true;
}

void OSPRayRenderer::getOpenGLDepthFromOSPPerspective(float* db) {

	//cr->GetCameraState().se

    const float fovy = cam.aperture_angle();
    const float aspect = cam.resolution_gate_aspect();
    const float zNear = cam.near_clipping_plane();
    const float zFar = cam.far_clipping_plane();

    const ospcommon::vec3f cameraUp(cam.up_vector().x(), cam.up_vector().y(), cam.up_vector().z());
    const ospcommon::vec3f cameraDir(cam.view_vector().x(), cam.view_vector().y(), cam.view_vector().z());

    // map OSPRay depth buffer from provided frame buffer
    const float* ospDepthBuffer = (const float*)ospMapFrameBuffer(this->framebuffer, OSP_FB_DEPTH);

    const size_t ospDepthBufferWidth = (size_t)this->imgSize.x;
    const size_t ospDepthBufferHeight = (size_t)this->imgSize.y;

    // transform from ray distance t to orthogonal Z depth
    ospcommon::vec3f dir_du = normalize(cross(cameraDir, cameraUp));
    ospcommon::vec3f dir_dv = normalize(cross(dir_du, cameraDir));

    const float imagePlaneSizeY = 2.f * tanf(fovy / 2.f * M_PI / 180.f);
    const float imagePlaneSizeX = imagePlaneSizeY * aspect;

    dir_du *= imagePlaneSizeX;
    dir_dv *= imagePlaneSizeY;

    const ospcommon::vec3f dir_00 = cameraDir - .5f * dir_du - .5f * dir_dv;

    const float A = -(zFar + zNear) / (zFar - zNear);
    const float B = -2. * zFar * zNear / (zFar - zNear);

    int j, i;
#pragma omp parallel for private(i)
    for (j = 0; j < ospDepthBufferHeight; j++)
        for (i = 0; i < ospDepthBufferWidth; i++) {
            const ospcommon::vec3f dir_ij = normalize(dir_00 + float(i) / float(ospDepthBufferWidth - 1) * dir_du +
                                                      float(j) / float(ospDepthBufferHeight - 1) * dir_dv);

            float tmp = ospDepthBuffer[j * ospDepthBufferWidth + i] * dot(cameraDir, dir_ij);
            float res = 0.5 * (-A * tmp + B) / tmp + 0.5;
            if (!std::isfinite(res)) res = 1.0f;
            db[j * ospDepthBufferWidth + i] = res;
        }

    // unmap OSPRay depth buffer
    ospUnmapFrameBuffer(ospDepthBuffer, this->framebuffer);
}
