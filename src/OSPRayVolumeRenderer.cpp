#include "stdafx.h"
#include "OSPRayVolumeRenderer.h"
#include "vislib/graphics/gl/ShaderSource.h"
#include "vislib/graphics/gl/IncludeAllGL.h"
#include "mmcore/CoreInstance.h"
#include "mmcore/moldyn/VolumeDataCall.h"
#include "vislib/graphics/CameraParamsStore.h"
#include "mmcore/param/EnumParam.h"
#include "mmcore/param/BoolParam.h"



using namespace megamol;


/*
ospray::OSPRayVolumeRenderer::OSPRayVolumeRenderer
*/
ospray::OSPRayVolumeRenderer::OSPRayVolumeRenderer(void):
core::view::Renderer3DModule(),
osprayShader(),
volDataCallerSlot("getData", "Connects the volume rendering with data storage"),
secRenCallerSlot("secRen", "Connects the volume rendering with a secondary renderer"),
rd_type("Renderer::Type", "Select between SciVis and PathTracer"),
extraSamles("General::extraSamples", "Extra sampling when camera is not moved")
{

    imgSize.x = 0;
    imgSize.y = 0;
    time = 0;
    framebuffer = NULL;

    // set caller slot for different data calls
    this->volDataCallerSlot.SetCompatibleCall<core::moldyn::VolumeDataCallDescription>();
    this->MakeSlotAvailable(&this->volDataCallerSlot);

    // set renderer caller slot
    this->secRenCallerSlot.SetCompatibleCall<core::view::CallRender3DDescription>();
    this->MakeSlotAvailable(&this->secRenCallerSlot);

    core::param::EnumParam *rdt = new core::param::EnumParam(SCIVIS);
    rdt->SetTypePair(SCIVIS, "SciVis");
    rdt->SetTypePair(PATHTRACER, "PathTracer");

    this->rd_type << rdt;
    this->MakeSlotAvailable(&this->rd_type);

    this->extraSamles << new core::param::BoolParam(true);
    this->MakeSlotAvailable(&this->extraSamles);
}


/*
ospray::OSPRayVolumeRenderer::~OSPRayVolumeRenderer
*/
ospray::OSPRayVolumeRenderer::~OSPRayVolumeRenderer(void) {
    this->osprayShader.Release();
    this->Release();
}


/*
ospray::OSPRayVolumeRenderer::create
*/
bool ospray::OSPRayVolumeRenderer::create() {
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

    this->initOSPRay();
    this->setupTextureScreen();
    this->setupOSPRay(renderer, camera, world, volume, "shared_structured_volume", "scivis");

    return true;
}

/*
ospray::OSPRayVolumeRenderer::release()
*/
void ospray::OSPRayVolumeRenderer::release() {
    this->releaseTextureScreen();
    return;
}

/*
ospray::OSPRayVolumeRenderer::Render
*/
bool ospray::OSPRayVolumeRenderer::Render(core::Call& call) {

    // cast the call to Render3D
    core::view::CallRender3D *cr = dynamic_cast<core::view::CallRender3D*>(&call);
    if (cr == NULL)
        return false;

    // get pointer to VolumeDataCall
    core::moldyn::VolumeDataCall *vd = this->volDataCallerSlot.CallAs<core::moldyn::VolumeDataCall>();
    
    // set frame ID and call data
    if (vd) {
        vd->SetFrameID(static_cast<int>(cr->Time()));
        if (!(*vd)(core::moldyn::VolumeDataCall::CallForGetData)) {
            return false;
        }
    } else {
        return false;
    }

    if (camParams == NULL)
        camParams = new vislib::graphics::CameraParamsStore();

    data_has_changed = (vd->DataHash() != this->m_datahash);
    this->m_datahash = vd->DataHash();

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
        imgSize.x = cr->GetCameraParameters()->TileRect().Width();
        imgSize.y = cr->GetCameraParameters()->TileRect().Height();
        framebuffer = ospNewFrameBuffer(imgSize, OSP_FB_RGBA8, OSP_FB_COLOR | OSP_FB_ACCUM);
    }

    // if user wants to switch renderer
    if (this->rd_type.IsDirty()) {
        switch (this->rd_type.Param<core::param::EnumParam>()->Value()) {
        case SCIVIS:
            ospRelease(camera);
            ospRelease(world);
            ospRelease(renderer);
            ospRelease(volume);
            this->setupOSPRay(renderer, camera, world, volume, "shared_structured_volume", "scivis");
            break;
        case PATHTRACER:
            ospRelease(camera);
            ospRelease(world);
            ospRelease(renderer);
            ospRelease(volume);
            this->setupOSPRay(renderer, camera, world, volume, "shared_structured_volume", "pathtracer");
            break;
        }
        renderer_changed = true;
        this->rd_type.ResetDirty();
    }

    // setup camera
    ospSetf(camera, "aspect", cr->GetCameraParameters()->TileRect().AspectRatio());
    ospSet3fv(camera, "pos", cr->GetCameraParameters()->EyePosition().PeekCoordinates());
    ospSet3fv(camera, "dir", cr->GetCameraParameters()->EyeDirection().PeekComponents());
    ospSet3fv(camera, "up", cr->GetCameraParameters()->EyeUpVector().PeekComponents());
    ospCommit(camera);


    osprayShader.Enable();
    // if nothing changes, the image is rendered multiple times
    if (data_has_changed || cam_has_changed || !(this->extraSamles.Param<core::param::BoolParam>()->Value()) || time != cr->Time() || this->InterfaceIsDirty() || renderer_changed) {
        time = cr->Time();
        renderer_changed = false;


        int num_voxels = vd->VolumeDimension().GetDepth() * vd->VolumeDimension().GetHeight() * vd->VolumeDimension().GetWidth();
        voxels = ospNewData(num_voxels, OSP_FLOAT, vd->VoxelMap(), OSP_DATA_SHARED_BUFFER);
        ospCommit(voxels);
        ospSetString(volume, "voxelType", "float"); // loader only supports float
        //std::vector<int> dimensions = { vd->VolumeDimension().PeekDimension()[0], vd->VolumeDimension().PeekDimension()[1], vd->VolumeDimension().PeekDimension()[2] };
        ospSet3iv(volume, "dimensions", (const int*)vd->VolumeDimension().PeekDimension());

        ospSetData(volume, "voxelData", voxels);
        //ospSet3f(volume, "gridSpacing", vd->VolumeDimension().GetWidth(), vd->VolumeDimension().GetHeight(), vd->VolumeDimension().GetDepth());

        OSPTransferFunction tf = ospNewTransferFunction("piecewise_linear");
        std::vector<float> rgb = { 0.0f, 0.0f, 1.0f, 
                                   1.0f, 0.0f, 0.0f };
        std::vector<float> opa = { 0.3f, 0.5f };
        OSPData tf_rgb = ospNewData(2, OSP_FLOAT3, rgb.data());
        OSPData tf_opa = ospNewData(2, OSP_FLOAT, opa.data());
        ospSetData(tf, "colors", tf_rgb);
        ospSetData(tf, "opacities", tf_opa);


        ospSetObject(volume, "transferFunction", tf);

        // voxelRange is computed by OSPRay
        //ospSet2f(volume, "voxelRange", min_voxel, max_voxel);

        ospCommit(volume);
        ospCommit(world);


        // scivis renderer settings
        ospSet1f(renderer, "aoWeight", 1.0);
        ospSet1i(renderer, "aoSamples", 1);

        ospCommit(renderer);




        // setup framebuffer
        ospFrameBufferClear(framebuffer, OSP_FB_COLOR | OSP_FB_ACCUM);
        ospRenderFrame(framebuffer, renderer, OSP_FB_COLOR | OSP_FB_ACCUM);

        // get the texture from the framebuffer
        fb = (uint32_t*)ospMapFrameBuffer(framebuffer, OSP_FB_COLOR);

        this->renderTexture2D(osprayShader, fb, imgSize.x, imgSize.y);
        ospUnmapFrameBuffer(fb, framebuffer);
        ospRelease(voxels);
        


    } else {
        ospRenderFrame(framebuffer, renderer, OSP_FB_COLOR | OSP_FB_ACCUM);
        fb = (uint32_t*)ospMapFrameBuffer(framebuffer, OSP_FB_COLOR);
        this->renderTexture2D(osprayShader, fb, imgSize.x, imgSize.y);
        ospUnmapFrameBuffer(fb, framebuffer);

    }
    
    vd->Unlock();
    osprayShader.Disable();

    return true;
}






bool ospray::OSPRayVolumeRenderer::InterfaceIsDirty() {
    return true;
}



























/*
ospray::OSPRayVolumeRenderer::GetCapabilities
*/
bool ospray::OSPRayVolumeRenderer::GetCapabilities(core::Call& call) {
    core::view::CallRender3D *cr3d = dynamic_cast<core::view::CallRender3D *>(&call);
    if (cr3d == NULL) return false;

    cr3d->SetCapabilities(core::view::CallRender3D::CAP_RENDER |
        core::view::CallRender3D::CAP_LIGHTING |
        core::view::CallRender3D::CAP_ANIMATION);

    return true;
}

/*

*/
bool ospray::OSPRayVolumeRenderer::GetExtents(core::Call& call) {
    //TODO
    core::view::CallRender3D *cr3d = dynamic_cast<core::view::CallRender3D *>(&call);
    if (cr3d == NULL) return false;

    core::moldyn::VolumeDataCall *volume = this->volDataCallerSlot.CallAs<core::moldyn::VolumeDataCall>();

    float xoff, yoff, zoff;
    vislib::math::Cuboid<float> boundingBox;
    vislib::math::Point<float, 3> bbc;

    // try to call the volume data
    if (!(*volume)(core::moldyn::VolumeDataCall::CallForGetExtent)) return false;
    // get bounding box
    boundingBox = volume->BoundingBox();
    // get the pointer to CallRender3D for the secondary renderer 
    core::view::CallRender3D *cr3dSec = this->secRenCallerSlot.CallAs<core::view::CallRender3D>();
    if (cr3dSec) {
        (*cr3dSec)(1); // GetExtents
        core::BoundingBoxes &secRenBbox = cr3dSec->AccessBoundingBoxes();
        boundingBox.Union(secRenBbox.ObjectSpaceBBox());
    }

    this->unionBBox = boundingBox;
    bbc = boundingBox.CalcCenter();
    this->bboxCenter = bbc;
    xoff = -bbc.X();
    yoff = -bbc.Y();
    zoff = -bbc.Z();
    if (!vislib::math::IsEqual(this->unionBBox.LongestEdge(), 0.0f)) {
        this->scale = 2.0f / this->unionBBox.LongestEdge();
    } else {
        this->scale = 1.0f;
    }

    core::BoundingBoxes &bbox = cr3d->AccessBoundingBoxes();
    bbox.SetObjectSpaceBBox(this->unionBBox);
    bbox.MakeScaledWorld(this->scale);

    bbox.SetObjectSpaceClipBox(bbox.ObjectSpaceBBox());
    bbox.SetWorldSpaceClipBox(bbox.WorldSpaceBBox());

    return true;


}


