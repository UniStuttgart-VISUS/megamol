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
    spheres = NULL;
    pln = NULL;
    vertexData = NULL;
    colorData = NULL;
    ModuleIsDirty = false;


    //tmp variable
    number = 0;


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

    this->initOSPRay(device);
    this->setupTextureScreen();
    this->setupOSPRay(renderer, camera, world, "scivis");

    return true;
}

/*
ospray::OSPRayRenderer::release
*/
void OSPRayRenderer::release() {
    if (camera != NULL) ospRelease(camera);
    if (world != NULL) ospRelease(world);
    if (renderer != NULL) ospRelease(renderer);
    if (spheres != NULL) ospRelease(spheres);
    if (pln != NULL) ospRelease(pln);
    releaseTextureScreen();
}

/*
ospray::OSPRayRenderer::Render
*/
bool OSPRayRenderer::Render(megamol::core::Call& call) {

    if (device != ospGetCurrentDevice()) {
        ospSetCurrentDevice(device);
    }
    core::view::CallRender3D *cr = dynamic_cast<core::view::CallRender3D*>(&call);
    if (cr == NULL)
        return false;



    CallOSPRayStructure *os = this->getStructureSlot.CallAs<CallOSPRayStructure>();
    // check if data has changed
    os->checkDatahash(&data_has_changed);

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
        framebuffer = newFrameBuffer(imgSize, OSP_FB_RGBA8, OSP_FB_COLOR | /*OSP_FB_DEPTH |*/ OSP_FB_ACCUM);
    }


    // if user wants to switch renderer
    if (this->rd_type.IsDirty()) {
        switch (this->rd_type.Param<core::param::EnumParam>()->Value()) {
        case SCIVIS:
            ospRelease(camera);
            ospRelease(world);
            ospRelease(renderer);
            this->setupOSPRay(renderer, camera, world, "scivis");
            break;
        case PATHTRACER:
            ospRelease(camera);
            ospRelease(world);
            ospRelease(renderer);
            this->setupOSPRay(renderer, camera, world, "pathtracer");
            break;
        }
        renderer_has_changed = true;
        this->rd_type.ResetDirty();
    }

    setupOSPRayCamera(camera, cr);
    ospCommit(camera);

    // Light setup
    CallOSPRayLight *gl = this->getLightSlot.CallAs<CallOSPRayLight>();
    if (gl != NULL) {
        gl->setLightMap(&lightMap);
        gl->setDirtyObj(&ModuleIsDirty);
        gl->fillLightMap();
    }

    osprayShader.Enable();
    // if nothing changes, the image is rendered multiple times
    if (data_has_changed || cam_has_changed || renderer_has_changed || !(this->extraSamles.Param<core::param::BoolParam>()->Value()) || time != cr->Time() || this->InterfaceIsDirty()) {
        time = cr->Time();
        renderer_has_changed = false;

        

        this->fillWorld();



        ospCommit(world);


        RendererSettings(renderer);


        // Enable Lights
        if (gl != NULL) {
            this->fillLightArray();
            lightsToRender = ospNewData(this->lightArray.size(), OSP_OBJECT, lightArray.data(), 0);
            ospSetData(renderer, "lights", lightsToRender);
        }

        ospCommit(renderer);


        // setup framebuffer
        ospFrameBufferClear(framebuffer, OSP_FB_COLOR | OSP_FB_ACCUM);
        ospRenderFrame(framebuffer, renderer, OSP_FB_COLOR | OSP_FB_ACCUM);


        // get the texture from the framebuffer
        fb = (uint32_t*)ospMapFrameBuffer(framebuffer, OSP_FB_COLOR);

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
        this->renderTexture2D(osprayShader, fb, imgSize.x, imgSize.y);

        // clear stuff
        ospUnmapFrameBuffer(fb, framebuffer);

        this->releaseOSPRayStuff();

        vd.clear();
        cd_rgba.clear();

    } else {
        ospRenderFrame(framebuffer, renderer, OSP_FB_COLOR | OSP_FB_ACCUM);
        fb = (uint32_t*)ospMapFrameBuffer(framebuffer, OSP_FB_COLOR);
        this->renderTexture2D(osprayShader, fb, imgSize.x, imgSize.y);
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
        this->AbstractIsDirty() ||
        this->ModuleIsDirty) {
        this->AbstractResetDirty();
        this->ModuleIsDirty = false;
        return true;
    } else {
        return false;
    }
}



/*
* ospray::OSPRaySphereRenderer::GetCapabilities
*/
bool OSPRayRenderer::GetCapabilities(megamol::core::Call& call) {
    megamol::core::view::CallRender3D *cr = dynamic_cast<megamol::core::view::CallRender3D*>(&call);
    if (cr == NULL) return false;

    cr->SetCapabilities(
        megamol::core::view::CallRender3D::CAP_RENDER
        | megamol::core::view::CallRender3D::CAP_LIGHTING
        | megamol::core::view::CallRender3D::CAP_ANIMATION
    );

    return true;
}


/*
* ospray::OSPRayRenderer::GetExtents
*/
bool OSPRayRenderer::GetExtents(megamol::core::Call& call) {
    megamol::core::view::CallRender3D *cr = dynamic_cast<megamol::core::view::CallRender3D*>(&call);
    if (cr == NULL) return false;
    CallOSPRayStructure *os = this->getStructureSlot.CallAs<CallOSPRayStructure>();
    if (os == NULL) return false;
    os->SetFrameID(static_cast<int>(cr->Time()));
    if (!(*os)(1)) return false;

    cr->SetTimeFramesCount(c2->FrameCount());
    cr->AccessBoundingBoxes().Clear();
    cr->AccessBoundingBoxes() = c2->AccessBoundingBoxes();
    cr->AccessBoundingBoxes().MakeScaledWorld(1.0f);

    return true;
}

