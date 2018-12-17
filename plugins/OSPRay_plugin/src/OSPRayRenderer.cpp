/*
* OSPRayRenderer.cpp
* Copyright (C) 2009-2017 by MegaMol Team
* Alle Rechte vorbehalten.
*/

#include "stdafx.h"
#include "vislib/graphics/gl/IncludeAllGL.h"
#include "vislib/graphics/gl/ShaderSource.h"
#include "vislib/graphics/CameraParamsStore.h"
#include "vislib/math/Vector.h"
#include "vislib/sys/Log.h"
#include "OSPRayRenderer.h"
#include "mmcore/param/EnumParam.h"
#include "mmcore/param/BoolParam.h"

#include "mmcore/CoreInstance.h"

#include <functional>

#include "ospray/ospray.h"
#include "ospcommon/vec.h"

#include <stdint.h>
#include <sstream>

using namespace megamol::ospray;

/*
ospray::OSPRayRenderer::OSPRaySphereRenderer
*/
OSPRayRenderer::OSPRayRenderer(void) :
    AbstractOSPRayRenderer(),
    osprayShader(),

    getStructureSlot("getStructure", "Connects to an OSPRay structure")

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
            vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
                "Unable to compile ospray shader: Unknown error\n");
            return false;
        }
    }
    catch (vislib::graphics::gl::AbstractOpenGLShader::CompileException ce) {
        vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
            "Unable to compile ospray shader: (@%s): %s\n",
            vislib::graphics::gl::AbstractOpenGLShader::CompileException::CompileActionName(
            ce.FailedAction()), ce.GetMsgA());
        return false;
    }
    catch (vislib::Exception e) {
        vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
            "Unable to compile ospray shader: %s\n", e.GetMsgA());
        return false;
    }
    catch (...) {
        vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
            "Unable to compile ospray shader: Unknown exception\n");
        return false;
    }

    //this->initOSPRay(device);
    this->setupTextureScreen();
    //this->setupOSPRay(renderer, camera, world, "scivis");

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
bool OSPRayRenderer::Render(megamol::core::Call& call) {
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
            break;
        case MPI_RAYCAST: //< TODO: Probably only valid if device is a "mpi_distributed" device
            this->setupOSPRay(renderer, camera, world, "mpi_raycast");
            break;
        default:
            this->setupOSPRay(renderer, camera, world, "scivis");
        }
        renderer_has_changed = true;
        this->rd_type.ResetDirty();
    }

    core::view::CallRender3D *cr = dynamic_cast<core::view::CallRender3D*>(&call);
    if (cr == NULL) return false;



    CallOSPRayStructure *os = this->getStructureSlot.CallAs<CallOSPRayStructure>();
    if (os == NULL) return false;
    // read data
    os->setStructureMap(&structureMap);
    os->setTime(cr->Time());
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
    CallOSPRayLight *gl = this->getLightSlot.CallAs<CallOSPRayLight>();
    light_has_changed = false;
    if (gl != NULL) {
        gl->setLightMap(&lightMap);
        gl->fillLightMap();
        for (auto element : this->lightMap) {
            auto light = element.second;
            if (light.dataChanged) {
                light_has_changed = true;
            }
        }
    }


    // check data and camera hash
    if (camParams == NULL)
        camParams = new vislib::graphics::CameraParamsStore();

    if ((camParams->EyeDirection().PeekComponents()[0] != cr->GetCameraParameters()->EyeDirection().PeekComponents()[0]) ||
        (camParams->EyeDirection().PeekComponents()[1] != cr->GetCameraParameters()->EyeDirection().PeekComponents()[1]) ||
        (camParams->EyeDirection().PeekComponents()[2] != cr->GetCameraParameters()->EyeDirection().PeekComponents()[2])) {
        cam_has_changed = true;
    } else {
        cam_has_changed = false;
    }
    camParams->CopyFrom(cr->GetCameraParameters());


    //glDisable(GL_CULL_FACE);

    // new framebuffer at resize action
    //bool triggered = false;
    if (imgSize.x != cr->GetCameraParameters()->TileRect().Width() || imgSize.y != cr->GetCameraParameters()->TileRect().Height() || extraSamles.IsDirty()) {
        //triggered = true;
        // Breakpoint for Screenshooter debugging
        if (framebuffer != NULL) ospFreeFrameBuffer(framebuffer);
        //imgSize.x = cr->GetCameraParameters()->VirtualViewSize().GetWidth();
        //imgSize.y = cr->GetCameraParameters()->VirtualViewSize().GetHeight();
        imgSize.x = cr->GetCameraParameters()->TileRect().Width();
        imgSize.y = cr->GetCameraParameters()->TileRect().Height();
        framebuffer = newFrameBuffer(imgSize, OSP_FB_RGBA8, OSP_FB_COLOR | OSP_FB_DEPTH | OSP_FB_ACCUM);
        db.resize(imgSize.x * imgSize.y);
        ospCommit(framebuffer);
    }

    //// if user wants to switch renderer
    //if (this->rd_type.IsDirty()) {
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
    setupOSPRayCamera(camera, cr, this->scale);
    ospCommit(camera);

    osprayShader.Enable();
    // if nothing changes, the image is rendered multiple times
    if (data_has_changed ||
        material_has_changed ||
        light_has_changed ||
        cam_has_changed ||
        renderer_has_changed ||
        !(this->extraSamles.Param<core::param::BoolParam>()->Value()) ||
        time != cr->Time() ||
        this->InterfaceIsDirty()) {

        if (data_has_changed ||
            time != cr->Time() ||
            this->InterfaceIsDirty()) {
            if (!this->fillWorld()) return false;
            ospCommit(world);
        }
        if (material_has_changed && !data_has_changed) {
            this->changeMaterial();
        }
        this->InterfaceResetDirty();
        time = cr->Time();
        renderer_has_changed = false;

        /*
        if (this->maxDepthTexture) {
            ospRelease(this->maxDepthTexture);
        }
        this->maxDepthTexture = getOSPDepthTextureFromOpenGLPerspective(*cr);
        */
        RendererSettings(renderer);


        // Enable Lights
        if (gl != NULL) {
            auto eyeDir = cr->GetCameraParameters()->EyeDirection().PeekComponents();
            this->fillLightArray(eyeDir);
            lightsToRender = ospNewData(this->lightArray.size(), OSP_OBJECT, lightArray.data(), 0);
            ospCommit(lightsToRender);
            ospSetData(renderer, "lights", lightsToRender);
        }

        ospSetObject(renderer, "model", world);
        ospCommit(renderer);


        // setup framebuffer
        ospFrameBufferClear(framebuffer, OSP_FB_COLOR | OSP_FB_DEPTH | OSP_FB_ACCUM);
        ospRenderFrame(framebuffer, renderer, OSP_FB_COLOR | OSP_FB_DEPTH | OSP_FB_ACCUM);




        // get the texture from the framebuffer
        fb = (uint32_t*)ospMapFrameBuffer(framebuffer, OSP_FB_COLOR);
        if (this->useDB.Param<core::param::BoolParam>()->Value()) {
            getOpenGLDepthFromOSPPerspective(*cr, db.data());
        } else {
            db.clear();
        }

        // write a sequence of single pictures while the screenshooter is running
        // only for debugging
        //if (triggered) {
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
        this->renderTexture2D(osprayShader, fb, db.data(), imgSize.x, imgSize.y, *cr);

        // clear stuff
        ospUnmapFrameBuffer(fb, framebuffer);


        this->releaseOSPRayStuff();


    } else {
        ospRenderFrame(framebuffer, renderer, OSP_FB_COLOR | OSP_FB_DEPTH | OSP_FB_ACCUM);
        fb = (uint32_t*)ospMapFrameBuffer(framebuffer, OSP_FB_COLOR);

        this->renderTexture2D(osprayShader, fb, db.data(), imgSize.x, imgSize.y, *cr);
        ospUnmapFrameBuffer(fb, framebuffer);
    }

    osprayShader.Disable();

    return true;
}

/*
ospray::OSPRayRenderer::InterfaceIsDirty()
*/
bool OSPRayRenderer::InterfaceIsDirty() {
    if (
        this->AbstractIsDirty()
        ) {
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
bool OSPRayRenderer::GetExtents(megamol::core::Call& call) {

    megamol::core::view::CallRender3D *cr = dynamic_cast<megamol::core::view::CallRender3D*>(&call);
    if (cr == NULL) return false;
    CallOSPRayStructure *os = this->getStructureSlot.CallAs<CallOSPRayStructure>();
    if (os == NULL) return false;
    os->setTime(static_cast<int>(cr->Time()));
    os->setExtendMap(&(this->extendMap));
    if (!os->fillExtendMap()) return false;

    megamol::core::BoundingBoxes finalBox;
    unsigned int frameCnt = 0;
    for (auto pair : this->extendMap) {
        auto element = pair.second;

        if (frameCnt == 0) {
            if (element.boundingBox->IsObjectSpaceBBoxValid()) {
                finalBox.SetObjectSpaceBBox(element.boundingBox->ObjectSpaceBBox());
            } else if (element.boundingBox->IsObjectSpaceClipBoxValid()) {
                finalBox.SetObjectSpaceBBox(element.boundingBox->ObjectSpaceClipBox());
            } else {
                finalBox.SetObjectSpaceBBox(vislib::math::Cuboid<float>(-1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f));
            }
            if (element.boundingBox->IsObjectSpaceClipBoxValid()) {
                finalBox.SetObjectSpaceClipBox(element.boundingBox->ObjectSpaceClipBox());
            } else if (element.boundingBox->IsObjectSpaceBBoxValid()) {
                finalBox.SetObjectSpaceClipBox(element.boundingBox->ObjectSpaceBBox());
            } else {
                finalBox.SetObjectSpaceClipBox(vislib::math::Cuboid<float>(-1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f));
            }

        } else {
            if (element.boundingBox->IsObjectSpaceBBoxValid()) {
                vislib::math::Cuboid<float> box(finalBox.ObjectSpaceBBox());
                box.Union(element.boundingBox->ObjectSpaceBBox());
                finalBox.SetObjectSpaceBBox(box);
            } else if (element.boundingBox->IsObjectSpaceClipBoxValid()) {
                vislib::math::Cuboid<float> box(finalBox.ObjectSpaceBBox());
                box.Union(element.boundingBox->ObjectSpaceClipBox());
                finalBox.SetObjectSpaceBBox(box);
            }
            if (element.boundingBox->IsObjectSpaceClipBoxValid()) {
                vislib::math::Cuboid<float> box(finalBox.ObjectSpaceClipBox());
                box.Union(element.boundingBox->ObjectSpaceClipBox());
                finalBox.SetObjectSpaceClipBox(box);
            } else if (element.boundingBox->IsObjectSpaceBBoxValid()) {
                vislib::math::Cuboid<float> box(finalBox.ObjectSpaceClipBox());
                box.Union(element.boundingBox->ObjectSpaceBBox());
                finalBox.SetObjectSpaceClipBox(box);
            }
        }
        frameCnt = vislib::math::Max(frameCnt, element.timeFramesCount);

    }
    scale = 1.0f;
    if (frameCnt == 0) {
        frameCnt = 1;
        scale = 10.0f / finalBox.ObjectSpaceBBox().LongestEdge();
    } else {
        scale = 10.0f / finalBox.ObjectSpaceBBox().LongestEdge();
    }
    
    cr->SetTimeFramesCount(frameCnt);
    cr->AccessBoundingBoxes() = finalBox;
    cr->AccessBoundingBoxes().MakeScaledWorld(scale);

    return true;
}

OSPTexture2D OSPRayRenderer::getOSPDepthTextureFromOpenGLPerspective(megamol::core::Call& call) {

    megamol::core::view::CallRender3D *cr = dynamic_cast<megamol::core::view::CallRender3D*>(&call);
    if (cr == NULL) return NULL;

    const double fovy = cr->GetCameraParameters()->ApertureAngle();
    const double aspect = static_cast<float>(cr->GetCameraParameters()->VirtualViewSize().GetWidth()) /
        static_cast<float>(cr->GetCameraParameters()->VirtualViewSize().GetHeight());
        const double zNear = cr->GetCameraParameters()->NearClip();
    const double zFar = cr->GetCameraParameters()->FarClip();


    float up_x = cr->GetCameraParameters()->Up().GetX();
    float up_y = cr->GetCameraParameters()->Up().GetY();
    float up_z = cr->GetCameraParameters()->Up().GetZ();

    float* dir = cr->GetCameraParameters()->EyeDirection().PeekComponents();

    const ospcommon::vec3f  cameraUp(up_x, up_y, up_z );
    const ospcommon::vec3f cameraDir(dir[0], dir[1], dir[2]);


    // read OpenGL depth buffer
    auto fbo = cr->FrameBufferObject();
    if (fbo != NULL) {
        if (fbo->IsValid()) {
            if ((fbo->GetWidth() != imgSize.x) || (fbo->GetHeight() != imgSize.y)) {
                fbo->Release();
            }
        }
        if (!fbo->IsValid()) {
            fbo->Create(static_cast<const UINT>(imgSize.x), static_cast<const UINT>(imgSize.y), GL_RGBA, GL_RGBA, GL_UNSIGNED_BYTE, vislib::graphics::gl::FramebufferObject::ATTACHMENT_TEXTURE, GL_DEPTH_COMPONENT);
        }
        if (fbo->IsValid()) {
            fbo->Enable();
        }
        float *glDepthBuffer = new float[this->imgSize.x * this->imgSize.y];
        fbo->GetDepthTexture(glDepthBuffer, GL_DEPTH_COMPONENT, GL_FLOAT);
        //glReadPixels(0, 0, this->imgSize.x, this->imgSize.y, GL_DEPTH_COMPONENT, GL_FLOAT, (GLvoid *)glDepthBuffer);

    // get an OSPRay depth texture from the OpenGL depth buffer
    float *ospDepth = new float[this->imgSize.x * this->imgSize.y];

    // transform OpenGL depth to linear depth
    for (size_t i = 0; i<this->imgSize.x * this->imgSize.y; i++) {
        const double z_n = 2.0 * glDepthBuffer[i] - 1.0;
        ospDepth[i] = 2.0 * zNear * zFar / (zFar + zNear - z_n * (zFar - zNear));
    }

    // transform from orthogonal Z depth to ray distance t
    ospcommon::vec3f dir_du = ospcommon::normalize(ospcommon::cross(cameraDir, cameraUp));
    ospcommon::vec3f dir_dv = ospcommon::normalize(ospcommon::cross(dir_du, cameraDir));

    const float imagePlaneSizeY = 2.f * tanf(fovy / 2.f * M_PI / 180.f);
    const float imagePlaneSizeX = imagePlaneSizeY * aspect;

    dir_du *= imagePlaneSizeX;
    dir_dv *= imagePlaneSizeY;

    const ospcommon::vec3f dir_00 = cameraDir - .5f * dir_du - .5f * dir_dv;

    for (size_t j = 0; j < this->imgSize.y; j++) {
        for (size_t i = 0; i < this->imgSize.x; i++) {
            const ospcommon::vec3f dir_ij = ospcommon::normalize(dir_00 + float(i) / float(this->imgSize.x - 1) * dir_du + float(j) / float(this->imgSize.y - 1) * dir_dv);

            const float t = ospDepth[j*this->imgSize.x + i] / ospcommon::dot(cameraDir, dir_ij);
            ospDepth[j*this->imgSize.y + i] = t;
        }
    }

    // nearest texture filtering required for depth textures -- we don't want interpolation of depth values...
    OSPTexture2D depthTexture = ospNewTexture2D(this->imgSize, OSP_TEXTURE_R32F, ospDepth, OSP_TEXTURE_FILTER_NEAREST);

    // free allocated depth buffer
    delete[] glDepthBuffer;

    // return OSPRay depth texture
    return depthTexture;
    } else {
        return NULL;
    }
}

void OSPRayRenderer::getOpenGLDepthFromOSPPerspective(megamol::core::Call& call, float* db) {

    megamol::core::view::CallRender3D *cr = dynamic_cast<megamol::core::view::CallRender3D*>(&call);
    if (cr == NULL) return;

    const double fovy = cr->GetCameraParameters()->ApertureAngle();
    const double aspect = static_cast<float>(cr->GetCameraParameters()->VirtualViewSize().GetWidth()) /
        static_cast<float>(cr->GetCameraParameters()->VirtualViewSize().GetHeight());
    const double zNear = cr->GetCameraParameters()->NearClip() / this->scale;
    const double zFar = cr->GetCameraParameters()->FarClip()   / this->scale;

    float up_x = cr->GetCameraParameters()->Up().GetX();
    float up_y = cr->GetCameraParameters()->Up().GetY();
    float up_z = cr->GetCameraParameters()->Up().GetZ();

    float* dir = cr->GetCameraParameters()->EyeDirection().PeekComponents();

    const ospcommon::vec3f  cameraUp(up_x, up_y, up_z);
    const ospcommon::vec3f cameraDir(dir[0], dir[1], dir[2]);

    // map OSPRay depth buffer from provided frame buffer
    const float *ospDepthBuffer = (const float *)ospMapFrameBuffer(this->framebuffer, OSP_FB_DEPTH);


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

    const double A = -(zFar + zNear) / (zFar - zNear);
    const double B = -2. * zFar*zNear / (zFar - zNear);

    int j,i;
#pragma omp parallel for private(i)
    for (j = 0; j<ospDepthBufferHeight; j++)
        for (i = 0; i<ospDepthBufferWidth; i++) {
            const ospcommon::vec3f dir_ij = normalize(dir_00 + float(i) / float(ospDepthBufferWidth - 1) * dir_du + float(j) / float(ospDepthBufferHeight - 1) * dir_dv);

            float tmp = ospDepthBuffer[j*ospDepthBufferWidth + i] * dot(cameraDir, dir_ij);
            db[j*ospDepthBufferWidth + i] = 0.5*(-A*tmp + B) / tmp + 0.5;
        }

    // unmap OSPRay depth buffer
    ospUnmapFrameBuffer(ospDepthBuffer, this->framebuffer);
}
