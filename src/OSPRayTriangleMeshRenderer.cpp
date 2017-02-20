#include "stdafx.h"
#include "OSPRayTriangleMeshRenderer.h"
#include "mmcore/view/CallRender3D.h"
#include "vislib/graphics/gl/ShaderSource.h"
#include "mmcore/CoreInstance.h"
#include "mmstd_trisoup/CallTriMeshData.h"
#include "vislib/graphics/CameraParamsStore.h"
#include "mmcore/param/EnumParam.h"
#include "mmcore/param/BoolParam.h"


using namespace megamol;


/*
ospray::OSPRayTriangleMeshRenderer::OSPRayTriangleMeshRenderer
*/
ospray::OSPRayTriangleMeshRenderer::OSPRayTriangleMeshRenderer(void) :
    ospray::OSPRayRenderer::OSPRayRenderer(),
    osprayShader(),
    // caller slots
    meshDataCallerSlot("getData", "Connects the triangle mesh rendering with data storage")
{
    this->meshDataCallerSlot.SetCompatibleCall<trisoup::CallTriMeshDataDescription>();
    this->MakeSlotAvailable(&this->meshDataCallerSlot);

    imgSize.x = 0;
    imgSize.y = 0;
    time = 0;
    framebuffer = NULL;

    vertexdata = NULL;
    normaldata = NULL;
    coldata = NULL;
    texdata = NULL;
    indexdata = NULL;
}


/*
ospray::OSPRayTriangleMeshRenderer::~OSPRayTriangleMeshRenderer
*/
ospray::OSPRayTriangleMeshRenderer::~OSPRayTriangleMeshRenderer(void) {
    this->Release();
}

/*
ospray::OSPRayTriangleMeshRenderer::release()
*/
void ospray::OSPRayTriangleMeshRenderer::release() {
    this->osprayShader.Release();
    this->releaseTextureScreen();
    for (unsigned int i = 0; i < this->objectCount; i++) {
        ospRelease(trimesh[i]);
    }
    if (vertexdata != NULL) ospRelease(vertexdata);
    if (normaldata != NULL) ospRelease(normaldata);
    if (coldata != NULL) ospRelease(coldata);
    if (texdata != NULL) ospRelease(texdata);
    if (indexdata != NULL) ospRelease(indexdata);

    return;
}


/*
ospray::OSPRayTriangleMeshRenderer::create
*/
bool ospray::OSPRayTriangleMeshRenderer::create() {
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
    this->setupOSPRay(renderer, camera, world, "triangles", "scivis");

    return true;
}


/*
ospray::OSPRayTriangleMeshRenderer::Render
*/
bool ospray::OSPRayTriangleMeshRenderer::Render(core::Call& call) {

    if (device != ospGetCurrentDevice()) {
        ospSetCurrentDevice(device);
    }
    // cast the call to Render3D
    core::view::CallRender3D *cr = dynamic_cast<core::view::CallRender3D*>(&call);
    if (cr == NULL) return false;

    trisoup::CallTriMeshData *md = this->meshDataCallerSlot.CallAs<trisoup::CallTriMeshData>();
    if (md == NULL) return false;
    md->SetFrameID(static_cast<int>(cr->Time()));
    if (!(*md)(1)) return false;

    md->SetFrameID(static_cast<int>(cr->Time()));
    if (!(*md)(0)) return false;

    if (camParams == NULL)
        camParams = new vislib::graphics::CameraParamsStore();

    data_has_changed = (md->DataHash() != this->m_datahash);
    this->m_datahash = md->DataHash();

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
            this->setupOSPRay(renderer, camera, world, "triangles", "scivis");
            break;
        case PATHTRACER:
            ospRelease(camera);
            ospRelease(world);
            ospRelease(renderer);
            this->setupOSPRay(renderer, camera, world, "triangles", "pathtracer");
            break;
        }
        renderer_changed = true;
        this->rd_type.ResetDirty();
    }


    setupOSPRayCamera(camera, cr);
    ospCommit(camera);


    trimesh = new OSPGeometry [md->Count()];


    osprayShader.Enable();
    // if nothing changes, the image is rendered multiple times
    if (data_has_changed || cam_has_changed || !(this->extraSamles.Param<core::param::BoolParam>()->Value()) || time != cr->Time() || this->InterfaceIsDirty() || renderer_changed) {
        time = cr->Time();
        renderer_changed = false;

        this->objectCount = md->Count();

        for (unsigned int i = 0; i < this->objectCount; i++) {
            const trisoup::CallTriMeshData::Mesh& obj = md->Objects()[i];

            trimesh[i] = ospNewGeometry("triangles");

            // check vertex data type
            switch (obj.GetVertexDataType()) {
            case trisoup::CallTriMeshData::Mesh::DT_FLOAT:
                vertexdata = ospNewData(obj.GetVertexCount(), OSP_FLOAT3, obj.GetVertexPointerFloat());
                ospCommit(vertexdata);
                ospSetData(trimesh[i], "vertex", vertexdata);
                break;
                //case trisoup::CallTriMeshData::Mesh::DT_DOUBLE:
                    // ospray does not support double vectors
                    //OSPData vertexdata = ospNewData(obj.GetVertexCount, OSP_DOUBLE3, obj.GetVertexPointerFloat());
                //    break;
            }

            // check normal pointer
            if (obj.HasNormalPointer() != NULL) {
                switch (obj.GetNormalDataType()) {
                case trisoup::CallTriMeshData::Mesh::DT_FLOAT:
                    normaldata = ospNewData(obj.GetVertexCount(), OSP_FLOAT3, obj.GetNormalPointerFloat());
                    ospCommit(normaldata);
                    ospSetData(trimesh[i], "vertex.normal", normaldata);
                    break;
                    //case trisoup::CallTriMeshData::Mesh::DT_DOUBLE:
                    //    break;
                }
            }

            // check colorpointer and convert to rgba
            std::vector<float> cd;
            if (obj.HasColourPointer() != NULL) {
                switch (obj.GetColourDataType()) {
                case trisoup::CallTriMeshData::Mesh::DT_BYTE:
                    for (unsigned int i = 0; i < 3 * obj.GetVertexCount(); i++) {
                        cd.push_back((float)obj.GetColourPointerByte()[i] / 255.0f);
                        if ((i + 1) % 3 == 0) {
                            cd.push_back(1.0f);
                        }
                    }
                    coldata = ospNewData(obj.GetVertexCount(), OSP_FLOAT4, cd.data());
                    ospCommit(coldata);
                    ospSetData(trimesh[i], "vertex.color", coldata);
                    break;
                case trisoup::CallTriMeshData::Mesh::DT_FLOAT:
                    // TODO: not tested
                    for (unsigned int i = 0; i < 3 * obj.GetVertexCount(); i++) {
                        cd.push_back(obj.GetColourPointerFloat()[i]);
                        if ((i + 1) % 3 == 0) {
                            cd.push_back(1.0f);
                        }
                    }
                    coldata = ospNewData(obj.GetVertexCount(), OSP_FLOAT4, cd.data());
                    ospCommit(coldata);
                    ospSetData(trimesh[i], "vertex.color", coldata);
                    break;
                    //case trisoup::CallTriMeshData::Mesh::DT_DOUBLE:
                    //    break;
                }
            }


            // check texture array
            if (obj.HasTextureCoordinatePointer() != NULL) {
                switch (obj.GetTextureCoordinateDataType()) {
                case trisoup::CallTriMeshData::Mesh::DT_FLOAT:
                    texdata = ospNewData(obj.GetTriCount(), OSP_FLOAT2, obj.GetTextureCoordinatePointerFloat());
                    ospCommit(texdata);
                    ospSetData(trimesh[i], "vertex.texcoord", texdata);
                    break;
                    //case trisoup::CallTriMeshData::Mesh::DT_DOUBLE:
                    //    break;
                }
            }

            // check index pointer
            if (obj.HasTriIndexPointer() != NULL) {
                switch (obj.GetTriDataType()) {
                    //case trisoup::CallTriMeshData::Mesh::DT_BYTE:
                    //    break;
                    //case trisoup::CallTriMeshData::Mesh::DT_UINT16:
                    //    break;
                case trisoup::CallTriMeshData::Mesh::DT_UINT32:
                    indexdata = ospNewData(obj.GetTriCount(), OSP_UINT3, obj.GetTriIndexPointerUInt32());
                    //indexdata = ospNewData(obj.GetVertexCount(), OSP_UINT3, obj.GetTriIndexPointerUInt32());
                    ospCommit(indexdata);
                    ospSetData(trimesh[i], "index", indexdata);
                    break;
                }
            }

            // material properties
            if (obj.GetMaterial() != NULL) {
                const trisoup::CallTriMeshData::Material &mat = *obj.GetMaterial();

                material = ospNewMaterial(renderer, "OBJMaterial");
                ospSet3fv(material, "Kd", mat.GetKd());
                ospSet3fv(material, "Ks", mat.GetKs());
                ospSet1f(material, "Ns", mat.GetNs());
                ospSet1f(material, "d", mat.GetD());
                ospSet3fv(material, "Tf", mat.GetTf());
            } else {
                material = ospNewMaterial(renderer, "OBJMaterial");
            }
            ospCommit(material);
            ospSetMaterial(trimesh[i], material);

            ospCommit(trimesh[i]);
            ospAddGeometry(world, trimesh[i]);
            ospCommit(world);


            RendererSettings(renderer);
            //OSPRayLights(renderer, call);
            ospCommit(renderer);


            // setup framebuffer
            ospFrameBufferClear(framebuffer, OSP_FB_COLOR);
            ospRenderFrame(framebuffer, renderer, OSP_FB_COLOR);

            // get the texture from the framebuffer
            fb = (uint32_t*)ospMapFrameBuffer(framebuffer, OSP_FB_COLOR);

            this->renderTexture2D(osprayShader, fb, imgSize.x, imgSize.y);
            ospUnmapFrameBuffer(fb, framebuffer);


            cd.clear();


        } // end object loop

        for (unsigned int i = 0; i < this->objectCount; i++) {
            ospRemoveGeometry(world, trimesh[i]);
        }


    } else {
        ospRenderFrame(framebuffer, renderer, OSP_FB_COLOR | OSP_FB_ACCUM);
        fb = (uint32_t*)ospMapFrameBuffer(framebuffer, OSP_FB_COLOR);
        this->renderTexture2D(osprayShader, fb, imgSize.x, imgSize.y);
        ospUnmapFrameBuffer(fb, framebuffer);
    }

    md->Unlock();
    osprayShader.Disable();


    return true;
}


/*
ospray::OSPRayTriangleMeshRenderer::InterfaceIsDirty() {
*/
bool ospray::OSPRayTriangleMeshRenderer::InterfaceIsDirty() {
    if (
        this->AbstractIsDirty()
        )
    {
        this->AbstractResetDirty();

        return true;
    } else {
        return false;
    }
}


/*
bool ospray::OSPRayTriangleMeshRenderer::GetCapabilities
*/
bool ospray::OSPRayTriangleMeshRenderer::GetCapabilities(core::Call& call) {
    core::view::CallRender3D *cr3d = dynamic_cast<core::view::CallRender3D *>(&call);
    if (cr3d == NULL) return false;

    cr3d->SetCapabilities(core::view::CallRender3D::CAP_RENDER |
        core::view::CallRender3D::CAP_LIGHTING |
        core::view::CallRender3D::CAP_ANIMATION);

    return true;
}

/*
ospray::OSPRayTriangleMeshRenderer::GetExtents
*/
bool ospray::OSPRayTriangleMeshRenderer::GetExtents(core::Call& call) {
    core::view::CallRender3D *cr = dynamic_cast<core::view::CallRender3D*>(&call);
    if (cr == NULL) return false;
    trisoup::CallTriMeshData *ctmd = this->meshDataCallerSlot.CallAs<trisoup::CallTriMeshData>();
    if (ctmd == NULL) return false;
    ctmd->SetFrameID(static_cast<int>(cr->Time()));
    if (!(*ctmd)(1)) return false;

    cr->SetTimeFramesCount(ctmd->FrameCount());
    cr->AccessBoundingBoxes().Clear();
    cr->AccessBoundingBoxes() = ctmd->AccessBoundingBoxes();
    cr->AccessBoundingBoxes().MakeScaledWorld(1.0f);

    return true;
    
}