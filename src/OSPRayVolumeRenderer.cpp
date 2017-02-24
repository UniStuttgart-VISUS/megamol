#include "stdafx.h"
#include "OSPRayVolumeRenderer.h"
#include "vislib/graphics/gl/ShaderSource.h"
#include "vislib/graphics/gl/IncludeAllGL.h"
#include "mmcore/CoreInstance.h"
#include "mmcore/moldyn/VolumeDataCall.h"
#include "vislib/graphics/CameraParamsStore.h"
#include "mmcore/param/EnumParam.h"
#include "mmcore/param/BoolParam.h"
#include "mmcore/param/Vector3fParam.h"
#include "mmcore/param/FloatParam.h"
#include "mmcore/view/CallClipPlane.h"
#include "mmcore/view/CallGetTransferFunction.h"



using namespace megamol;


/*
ospray::OSPRayVolumeRenderer::OSPRayVolumeRenderer
*/
ospray::OSPRayVolumeRenderer::OSPRayVolumeRenderer(void) :
    ospray::OSPRayRenderer::OSPRayRenderer(),
    osprayShader(),
    // caller slots
    volDataCallerSlot("getData", "Connects the volume rendering with data storage"),
    secRenCallerSlot("secRen", "Connects the volume rendering with a secondary renderer"),
    TFSlot("transferfunction", "Connects to the transfer function module"),

    // API Variables
    showVolume("Volume::showVolume", "Displays the volume data"),
    showIsosurface("Volume::Isosurface::showIsosurface", "Displays the isosurface"),
    showSlice("Volume::Slice::showSlice", "Displays the slice"),

    sliceNormal("Volume::Slice::sliceNormal", "Direction of the slice normal"),
    sliceDist("Volume::Slice::sliceDist", "Distance of the slice in the direction of the normal vector"),

    clippingBoxActive("Volume::ClippingBox::Active", "Activates the clipping Box"),
    clippingBoxLower("Volume::ClippingBox::Left", "Left corner of the clipping Box"),
    clippingBoxUpper("Volume::ClippingBox::Right", "Right corner of the clipping Box")
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

    // set transferfunction caller slot
    this->TFSlot.SetCompatibleCall<core::view::CallGetTransferFunctionDescription>();
    this->MakeSlotAvailable(&this->TFSlot);

    // API Variables
    this->showVolume << new core::param::BoolParam(true);
    this->showIsosurface << new core::param::BoolParam(false);
    this->showSlice << new core::param::BoolParam(false);

    this->sliceNormal << new core::param::Vector3fParam({ 1.0f, 0.0f, 0.0f });
    this->sliceDist << new core::param::FloatParam(0.0f);

    this->clippingBoxActive << new core::param::BoolParam(false);
    this->clippingBoxLower << new::core::param::Vector3fParam({ -5.0f, -5.0f, -5.0f });
    this->clippingBoxUpper << new core::param::Vector3fParam({ 0.0f, 5.0f, 5.0f });

    this->MakeSlotAvailable(&this->showVolume);
    this->MakeSlotAvailable(&this->showIsosurface);
    this->MakeSlotAvailable(&this->showSlice);

    this->MakeSlotAvailable(&this->sliceNormal);
    this->MakeSlotAvailable(&this->sliceDist);

    this->MakeSlotAvailable(&this->clippingBoxActive);
    this->MakeSlotAvailable(&this->clippingBoxLower);
    this->MakeSlotAvailable(&this->clippingBoxUpper);
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

    this->initOSPRay(device);
    this->setupTextureScreen();
    this->setupOSPRay(renderer, camera, world, "scivis");

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

    if (device != ospGetCurrentDevice()) {
        ospSetCurrentDevice(device);
    }
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
        imgSize.x = cr->GetCameraParameters()->VirtualViewSize().GetWidth();
        imgSize.y = cr->GetCameraParameters()->VirtualViewSize().GetHeight();
        framebuffer = ospNewFrameBuffer(imgSize, OSP_FB_RGBA8, OSP_FB_COLOR | /*OSP_FB_DEPTH |*/ OSP_FB_ACCUM);
    }

    // if user wants to switch renderer
    if (this->rd_type.IsDirty()) {
        switch (this->rd_type.Param<core::param::EnumParam>()->Value()) {
        case SCIVIS:
            ospRelease(camera);
            ospRelease(world);
            ospRelease(renderer);
            ospRelease(volume);
            this->setupOSPRay(renderer, camera, world, "scivis");
            break;
        case PATHTRACER:
            ospRelease(camera);
            ospRelease(world);
            ospRelease(renderer);
            ospRelease(volume);
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



        int num_voxels = vd->VolumeDimension().GetDepth() * vd->VolumeDimension().GetHeight() * vd->VolumeDimension().GetWidth();
        // loader converts data to float
        voxels = ospNewData(num_voxels, OSP_FLOAT, vd->VoxelMap(), OSP_DATA_SHARED_BUFFER);
        ospCommit(voxels);


        volume = ospNewVolume("shared_structured_volume");


        ospSetString(volume, "voxelType", "float"); 
        // scaling properties of the volume
        ospSet3iv(volume, "dimensions", (const int*)vd->VolumeDimension().PeekDimension());
        unsigned int maxDim = vislib::math::Max<unsigned int>(vd->VolumeDimension().Depth(),
                              vislib::math::Max<unsigned int>(vd->VolumeDimension().Height(),
                                                              vd->VolumeDimension().Width()));
        float scale = 2*this->scaling; //scaling = 5
        ospSet3f(volume, "gridOrigin", -0.5f*scale, -0.5f*scale, -0.5f*scale);
        ospSet3f(volume, "gridSpacing", 1.0f*scale / (float)maxDim, 1.0f*scale / (float)maxDim, 1.0f*scale / (float)maxDim);
        
        // add data 
        ospSetData(volume, "voxelData", voxels);
        //ospSetRegion(volume, voxels, { 0, 0, 0 }, { (int)vd->VolumeDimension().GetWidth(), (int)vd->VolumeDimension().GetHeight(), (int)vd->VolumeDimension().GetDepth() });
        //ospSetRegion(volume, voxels, { 0, 0, 0 }, { 1, 1, 1 });
        //ospSet1i(volume, "gradientShadingEnabled", 1);


        // ClippingBox

        if (this->clippingBoxActive.Param<core::param::BoolParam>()->Value()) {
            ospSet3fv(volume, "volumeClippingBoxLower", this->clippingBoxLower.Param<core::param::Vector3fParam>()->Value().PeekComponents());
            ospSet3fv(volume, "volumeClippingBoxUpper", this->clippingBoxUpper.Param<core::param::Vector3fParam>()->Value().PeekComponents());
        } else {
            ospSetVec3f(volume, "volumeClippingBoxLower", {0.0f, 0.0f, 0.0f});
            ospSetVec3f(volume, "volumeClippingBoxUpper", {0.0f, 0.0f, 0.0f});
        }

        // get colors from transferfunction texture
        // TODO: wait for new transferfunction
        /*
        core::view::CallGetTransferFunction *cgtf = this->getTFSlot.CallAs<core::view::CallGetTransferFunction>();
        if (cgtf == NULL && (!(*cgtf)())) return false;
        float const* tf_tex = cgtf->GetTextureData();
        size_t tex_size = cgtf->TextureSize();
        */

        OSPTransferFunction tf = ospNewTransferFunction("piecewise_linear");
        std::vector<float> rgb = { 0.0f, 0.0f, 1.0f, 
                                   1.0f, 0.0f, 0.0f };
        std::vector<float> opa = { 0.01f, 0.05f };
        OSPData tf_rgb = ospNewData(2, OSP_FLOAT3, rgb.data());
        OSPData tf_opa = ospNewData(2, OSP_FLOAT, opa.data());
        ospSetData(tf, "colors", tf_rgb);
        ospSetData(tf, "opacities", tf_opa);

        ospCommit(tf);

        ospSetObject(volume, "transferFunction", tf);
        ospCommit(volume);

        // voxelRange is computed by OSPRay (if not set here)
        //ospSet2f(volume, "voxelRange", min_voxel, max_voxel);




        // isosurface
        if (this->showIsosurface.Param<core::param::BoolParam>()->Value()) {
            isosurface = ospNewGeometry("isosurfaces");
            std::vector<float> iv = { 0.5f };
            OSPData isovalues = ospNewData(2, OSP_FLOAT, iv.data());
            ospCommit(isovalues);
            ospSetData(isosurface, "isovalues", isovalues);
            ospSetObject(isosurface, "volume", volume);
            ospCommit(isosurface);

            ospAddGeometry(world, isosurface); // Show isosurface
        }

        // slices
        if (this->showSlice.Param<core::param::BoolParam>()->Value()) {
            slice = ospNewGeometry("slices");
            std::vector<float> pln(4);
            pln[0] = this->sliceNormal.Param<core::param::Vector3fParam>()->Value().X();
            pln[1] = this->sliceNormal.Param<core::param::Vector3fParam>()->Value().Y();
            pln[2] = this->sliceNormal.Param<core::param::Vector3fParam>()->Value().Z();
            pln[3] = this->sliceDist.Param<core::param::FloatParam>()->Value();
            OSPData planes = ospNewData(1, OSP_FLOAT4, pln.data());
            ospCommit(planes);
            ospSetData(slice, "planes", planes);
            ospSetObject(slice, "volume", volume);
            ospCommit(slice);

            ospAddGeometry(world, slice);  // Show slice

        }

        if (this->showVolume.Param<core::param::BoolParam>()->Value()) {
            ospAddVolume(world, volume);  // Show volume data
        }
        
        
        
        ospCommit(world);
        ospRemoveGeometry(world, slice);
        ospRemoveGeometry(world, isosurface);
        ospRemoveVolume(world, volume);

        RendererSettings(renderer);
        //OSPRayLights(renderer, call);
        ospCommit(renderer);


        // setup framebuffer
        ospFrameBufferClear(framebuffer, OSP_FB_COLOR | OSP_FB_ACCUM);
        ospRenderFrame(framebuffer, renderer, OSP_FB_COLOR | OSP_FB_ACCUM);


        // get the texture from the framebuffer
        fb = (uint32_t*)ospMapFrameBuffer(framebuffer, OSP_FB_COLOR);

        this->renderTexture2D(osprayShader, fb, imgSize.x, imgSize.y);
        ospUnmapFrameBuffer(fb, framebuffer);
        ospRelease(voxels);
        //ospRelease(volume);



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


/*
ospray::OSPRayVolumeRenderer::InterfaceIsDirty() {
*/
bool ospray::OSPRayVolumeRenderer::InterfaceIsDirty() {
    if (
        this->AbstractIsDirty() ||

        this->showVolume.IsDirty() || 
        this->showIsosurface.IsDirty() ||
        this->showSlice.IsDirty() ||
        this->sliceNormal.IsDirty() ||
        this->sliceDist.IsDirty() ||
        this->clippingBoxActive.IsDirty() ||
        this->clippingBoxLower.IsDirty() ||
        this->clippingBoxUpper.IsDirty() 
        )
    {
        this->AbstractResetDirty();
        this->showIsosurface.ResetDirty();
        this->showSlice.ResetDirty();
        this->sliceNormal.ResetDirty();
        this->sliceDist.ResetDirty();
        this->clippingBoxActive.ResetDirty();
        this->clippingBoxLower.ResetDirty();
        this->clippingBoxUpper.ResetDirty();
        return true;
    } else {
        return false;
    }
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
ospray::OSPRayVolumeRenderer::GetExtents
*/
bool ospray::OSPRayVolumeRenderer::GetExtents(core::Call& call) {
    core::view::CallRender3D *cr = dynamic_cast<core::view::CallRender3D*>(&call);
    if (cr == NULL) return false;

    core::moldyn::VolumeDataCall *c2 = this->volDataCallerSlot.CallAs<core::moldyn::VolumeDataCall>();
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


