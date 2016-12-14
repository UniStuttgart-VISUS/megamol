#include "stdafx.h"
#include "OSPRayVolumeRenderer.h"
#include "vislib/graphics/gl/ShaderSource.h"
#include "vislib/graphics/gl/IncludeAllGL.h"
#include "mmcore/CoreInstance.h"
#include "mmcore/moldyn/VolumeDataCall.h"


using namespace megamol;


/*
ospray::OSPRayVolumeRenderer::OSPRayVolumeRenderer
*/
ospray::OSPRayVolumeRenderer::OSPRayVolumeRenderer(void):
core::view::Renderer3DModule(),
osprayShader(),
volDataCallerSlot("getData", "Connects the volume rendering with data storage"),
secRenCallerSlot("secRen", "Connects the volume rendering with a secondary renderer")
{

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
    this->setupTextureScreen(vaScreen, vbo, tex);
    this->setupOSPRay(renderer, camera, world, volume, "shared", "scivis");

    return true;
}

/*
ospray::OSPRayVolumeRenderer::release()
*/
void ospray::OSPRayVolumeRenderer::release() {
    return;
}

/*
ospray::OSPRayVolumeRenderer::Render
*/
bool ospray::OSPRayVolumeRenderer::Render(core::Call& call) {
    core::view::CallRender3D *cr = dynamic_cast<core::view::CallRender3D*>(&call);
    if (cr == NULL)
        return false;

    //TODO

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


