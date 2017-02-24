#include "stdafx.h"
#include "OSPRayStreamLineRenderer.h"
#include "mmcore/CoreInstance.h"
#include "vislib/graphics/gl/ShaderSource.h"
#include "vislib/graphics/CameraParamsStore.h"
#include "mmcore/param/EnumParam.h"
#include "mmcore/param/BoolParam.h"
#include "mmcore/param/Vector3fParam.h"
#include "mmcore/param/FloatParam.h"
#include "protein_calls/VTIDataCall.h"


using namespace megamol;

/*
ospray::OSPRayStreamLineRenderer
*/
ospray::OSPRayStreamLineRenderer::OSPRayStreamLineRenderer(void) :
    ospray::OSPRayRenderer::OSPRayRenderer(),
    osprayShader(),
    // caller Slots
    dataCallerSlot("getData", "Connects the stream line rendering with a data storage")
{
    imgSize.x = 0;
    imgSize.y = 0;
    time = 0;
    framebuffer = NULL;

    // set caller slot for different data calls
    this->dataCallerSlot.SetCompatibleCall<protein_calls::VTIDataCallDescription>();
    this->MakeSlotAvailable(&this->dataCallerSlot);
}


/*
ospray::OSPRayStreamLineRenderer::create
*/
bool ospray::OSPRayStreamLineRenderer::create() {
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
    } catch (vislib::graphics::gl::AbstractOpenGLShader::CompileException ce) {
        vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
            "Unable to compile ospray shader: (@%s): %s\n",
            vislib::graphics::gl::AbstractOpenGLShader::CompileException::CompileActionName(
                ce.FailedAction()), ce.GetMsgA());
        return false;
    } catch (vislib::Exception e) {
        vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
            "Unable to compile ospray shader: %s\n", e.GetMsgA());
        return false;
    } catch (...) {
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
ospray::OSPRayStreamLineRenderer::~OSPRayStreamLineRenderer
*/
ospray::OSPRayStreamLineRenderer::~OSPRayStreamLineRenderer(void) {
    this->Release();
}

/*
ospray::OSPRayStreamLineRenderer::release()
*/
void ospray::OSPRayStreamLineRenderer::release() {
    this->osprayShader.Release();
    this->releaseTextureScreen();
    return;
}


/*
ospray::OSPRayStreamLineRenderer::Render
*/
bool ospray::OSPRayStreamLineRenderer::Render(core::Call& call) {

    if (device != ospGetCurrentDevice()) {
        ospSetCurrentDevice(device);
    }
    // cast the call to Render3D
    core::view::CallRender3D *cr = dynamic_cast<core::view::CallRender3D*>(&call);
    if (cr == NULL)
        return false;

    // get pointer to VTIDataCall
    protein_calls::VTIDataCall* sld = this->dataCallerSlot.CallAs<protein_calls::VTIDataCall>();

    // set frame ID and call data
    if (sld) {
        sld->SetFrameID(static_cast<int>(cr->Time()));
        if (!(*sld)(protein_calls::VTIDataCall::CallForGetData)) {
            return false;
        }
    } else {
        return false;
    }

    // see if datahash or cameraParameters changed
    if (camParams == NULL)
        camParams = new vislib::graphics::CameraParamsStore();

    data_has_changed = (sld->DataHash() != this->m_datahash);
    this->m_datahash = sld->DataHash();

    if ((camParams->EyeDirection().PeekComponents()[0] != cr->GetCameraParameters()->EyeDirection().PeekComponents()[0]) ||
        (camParams->EyeDirection().PeekComponents()[1] != cr->GetCameraParameters()->EyeDirection().PeekComponents()[1]) ||
        (camParams->EyeDirection().PeekComponents()[2] != cr->GetCameraParameters()->EyeDirection().PeekComponents()[2])) {
        cam_has_changed = true;
    } else {
        cam_has_changed = false;
    }
    camParams->CopyFrom(cr->GetCameraParameters());

    // new framebuffer at resize action
    if (imgSize.x != cr->GetCameraParameters()->TileRect().Width() || imgSize.y != cr->GetCameraParameters()->TileRect().Height()) {
        if (framebuffer != NULL) ospFreeFrameBuffer(framebuffer);
        imgSize.x = cr->GetCameraParameters()->VirtualViewSize().GetWidth();
        imgSize.y = cr->GetCameraParameters()->VirtualViewSize().GetHeight();
        framebuffer = ospNewFrameBuffer(imgSize, OSP_FB_RGBA8, OSP_FB_COLOR | OSP_FB_ACCUM);
    }

    // if user wants to switch renderer
    if (this->rd_type.IsDirty()) {
        switch (this->rd_type.Param<core::param::EnumParam>()->Value()) {
        case SCIVIS:
            ospRelease(camera);
            ospRelease(world);
            ospRelease(renderer);
            ospRelease(lines);
            this->setupOSPRay(renderer, camera, world, "scivis");
            break;
        case PATHTRACER:
            ospRelease(camera);
            ospRelease(world);
            ospRelease(renderer);
            ospRelease(lines);
            this->setupOSPRay(renderer, camera, world, "pathtracer");
            break;
        }
        renderer_changed = true;
        this->rd_type.ResetDirty();
    }


    setupOSPRayCamera(camera, cr);
    ospCommit(camera);


    osprayShader.Enable();
    // if nothing changes, the image is rendered multiple times
    if (data_has_changed || cam_has_changed || !(this->extraSamles.Param<core::param::BoolParam>()->Value()) || time != cr->Time() || this->InterfaceIsDirty() || renderer_changed) {
        time = cr->Time();
        renderer_changed = false;


    }
}



/*
ospray::OSPRayStreamLineRenderer::GetCapabilities
*/
bool ospray::OSPRayStreamLineRenderer::GetCapabilities(core::Call& call) {
    core::view::CallRender3D *cr3d = dynamic_cast<core::view::CallRender3D *>(&call);
    if (cr3d == NULL) return false;

    cr3d->SetCapabilities(core::view::CallRender3D::CAP_RENDER |
        core::view::CallRender3D::CAP_LIGHTING |
        core::view::CallRender3D::CAP_ANIMATION);

    return true;
}



/*
ospray::OSPRayStreamLineRenderer::GetExtents
*/
bool ospray::OSPRayStreamLineRenderer::GetExtents(core::Call& call) {
    core::view::CallRender3D *cr = dynamic_cast<core::view::CallRender3D*>(&call);
    if (cr == NULL) return false;

    
    protein_calls::VTIDataCall *c2 = this->dataCallerSlot.CallAs<protein_calls::VTIDataCall>();
    c2->SetFrameID(static_cast<unsigned int>(cr->Time()), true);
    if ((c2 != NULL) && ((*c2)(1))) {
        cr->SetTimeFramesCount(c2->FrameCount());
        cr->AccessBoundingBoxes() = c2->AccessBoundingBoxes();

        this->scaling = cr->AccessBoundingBoxes().ObjectSpaceBBox().LongestEdge();
        if (this->scaling > 0.0000001) {
            this->scaling = 10.0f / this->scaling;
        } else {
            this->scaling = 1.0f;
        }
        cr->AccessBoundingBoxes().MakeScaledWorld(this->scaling);

    } else {
        cr->SetTimeFramesCount(1);
        cr->AccessBoundingBoxes().Clear();
    }

    return true;
}

/*
ospray::OSPRayStreamLineRenderer::InterfaceIsDirty
*/
bool ospray::OSPRayStreamLineRenderer::InterfaceIsDirty() {
	return true;
}