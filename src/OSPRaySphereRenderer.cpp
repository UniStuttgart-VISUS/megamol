/*
 * OSPRaySphereRenderer.cpp
 * Copyright (C) 2009-2017 by MegaMol Team
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "vislib/assert.h"
#include "vislib/graphics/gl/IncludeAllGL.h"
#include "vislib/graphics/gl/ShaderSource.h"
#include "vislib/graphics/CameraParamsStore.h"
#include "vislib/math/Vector.h"
#include "OSPRaySphereRenderer.h"
#include "mmcore/moldyn/MultiParticleDataCall.h"
#include "mmcore/param/FloatParam.h"
#include "mmcore/param/BoolParam.h"
#include "mmcore/param/IntParam.h"
#include "mmcore/param/Vector3fParam.h"
#include "mmcore/param/EnumParam.h"
#include "mmcore/param/FilePathParam.h"
#include "mmcore/CoreInstance.h"
#include "mmcore/view/CallGetTransferFunction.h"
#include "mmcore/view/CallClipPlane.h"

#include "ospray/ospray.h"

#include <stdint.h>

using namespace megamol;



VISLIB_FORCEINLINE float floatFromVoidArray(const megamol::core::moldyn::MultiParticleDataCall::Particles& p, size_t index) {
    //const float* parts = static_cast<const float*>(p.GetVertexData());
    //return parts[index * stride + offset];
    return static_cast<const float*>(p.GetVertexData())[index];
}

VISLIB_FORCEINLINE float unit8FromVoidArray(const megamol::core::moldyn::MultiParticleDataCall::Particles& p, size_t index) {
	return static_cast<const uint8_t*>(p.GetVertexData())[index];
}

typedef float(*floatFromArrayFunc)(const megamol::core::moldyn::MultiParticleDataCall::Particles& p, size_t index);
typedef float(*uint8FromArrayFunc)(const megamol::core::moldyn::MultiParticleDataCall::Particles& p, size_t index);

/*
ospray::OSPRaySphereRenderer::OSPRaySphereRenderer
*/
ospray::OSPRaySphereRenderer::OSPRaySphereRenderer(void) :
OSPRayRenderer(),
osprayShader(),

getDataSlot("getdata", "Connects to the data source"),
getTFSlot("gettransferfunction", "Connects to the transfer function module"),
getClipPlaneSlot("getclipplane", "Connects to a clipping plane module"),
// material parameteres
mat_Kd("Material::OBJMaterial::DiffuseColor", "Diffuse color"),
mat_Ks("Material::OBJMaterial::SpecularColor", "Specular color"),
mat_Ns("Material::OBJMaterial::Shininess", "Phong exponent"),
mat_d("Material::OBJMaterial::Opacity", "Opacity"),
mat_Tf("Material::OBJMaterial::TransparencyFilterColor", "Transparency filter color"),
mat_type("Material::Type", "Switches material types") {

    this->getDataSlot.SetCompatibleCall<core::moldyn::MultiParticleDataCallDescription>();
    this->MakeSlotAvailable(&this->getDataSlot);

    this->getTFSlot.SetCompatibleCall<core::view::CallGetTransferFunctionDescription>();
    this->MakeSlotAvailable(&this->getTFSlot);

    this->getClipPlaneSlot.SetCompatibleCall<core::view::CallClipPlaneDescription>();
    this->MakeSlotAvailable(&this->getClipPlaneSlot);

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


    // Material
    core::param::EnumParam *mt = new core::param::EnumParam(OBJMATERIAL);
    mt->SetTypePair(OBJMATERIAL, "OBJMaterial");
    mt->SetTypePair(GLASS, "Glass (only PathTracer)");
    mt->SetTypePair(MATTE, "Matte (only PathTracer)");
    mt->SetTypePair(METAL, "Metal (only PathTracer)");
    mt->SetTypePair(METALLICPAINT, "MetallicPaint (only PathTracer)");
    mt->SetTypePair(PLASTIC, "Plastic (only PathTracer)");
    mt->SetTypePair(THINGLASS, "ThinGlass (only PathTracer)");
    mt->SetTypePair(VELVET, "Velvet (only PathTracer)");

    this->mat_Kd << new core::param::Vector3fParam(vislib::math::Vector<float, 3>(0.8f, 0.8f, 0.8f));
    this->mat_Ks << new core::param::Vector3fParam(vislib::math::Vector<float, 3>(0.0f, 0.0f, 0.0f));
    this->mat_Ns << new core::param::FloatParam(10.0f);
    this->mat_d << new core::param::FloatParam(1.0f);
    this->mat_Tf << new core::param::Vector3fParam(vislib::math::Vector<float, 3>(0.0f, 0.0f, 0.0f));
    this->mat_type << mt;
    this->MakeSlotAvailable(&this->mat_Kd);
    this->MakeSlotAvailable(&this->mat_Ks);
    this->MakeSlotAvailable(&this->mat_Ns);
    this->MakeSlotAvailable(&this->mat_d);
    this->MakeSlotAvailable(&this->mat_Tf);
    this->MakeSlotAvailable(&this->mat_type);

}



/*
ospray::OSPRaySphereRenderer::~OSPRaySphereRenderer
*/
ospray::OSPRaySphereRenderer::~OSPRaySphereRenderer(void) {
    this->osprayShader.Release();
    this->Release();
}


/*
ospray::OSPRaySphereRenderer::create
*/
bool ospray::OSPRaySphereRenderer::create() {
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
    this->setupOSPRay(renderer, camera, world, spheres, "spheres", "scivis");

    return true;
}

/*
ospray::OSPRaySphereRenderer::release
*/
void ospray::OSPRaySphereRenderer::release() {
    if (camera != NULL) ospRelease(camera);
    if (world != NULL) ospRelease(world);
    if (renderer != NULL) ospRelease(renderer);
    if (spheres != NULL) ospRelease(spheres);
    if (pln != NULL) ospRelease(pln);
    releaseTextureScreen();
}

/*
ospray::OSPRaySphereRenderer::Render
*/
bool ospray::OSPRaySphereRenderer::Render(core::Call& call) {

    core::view::CallRender3D *cr = dynamic_cast<core::view::CallRender3D*>(&call);
    if (cr == NULL)
        return false;

    float scaling = 1.0f;
    core::moldyn::MultiParticleDataCall *c2 = this->getData(static_cast<unsigned int>(cr->Time()), scaling);
    if (c2 == NULL)
        return false;


    // check data and camera hash
    if (camParams == NULL)
        camParams = new vislib::graphics::CameraParamsStore();
   
    data_has_changed = (c2->DataHash() != this->m_datahash);
    this->m_datahash = c2->DataHash();
    
    if ((camParams->EyeDirection().PeekComponents()[0] != cr->GetCameraParameters()->EyeDirection().PeekComponents()[0]) ||
        (camParams->EyeDirection().PeekComponents()[1] != cr->GetCameraParameters()->EyeDirection().PeekComponents()[1]) ||
        (camParams->EyeDirection().PeekComponents()[2] != cr->GetCameraParameters()->EyeDirection().PeekComponents()[2])) {
        cam_has_changed = true;
    } else {
        cam_has_changed = false;
    }
        camParams->CopyFrom(cr->GetCameraParameters());
    

    glDisable(GL_CULL_FACE);

    // new framebuffer at resize action
    if (imgSize.x != cr->GetCameraParameters()->TileRect().Width() || imgSize.y != cr->GetCameraParameters()->TileRect().Height()) {
        // Breakpoint for Screenshooter debugging
        if (framebuffer != NULL) ospFreeFrameBuffer(framebuffer);
        imgSize.x = cr->GetCameraParameters()->VirtualViewSize().GetWidth();
        imgSize.y = cr->GetCameraParameters()->VirtualViewSize().GetHeight();
        //framebuffer = ospNewFrameBuffer(imgSize, OSP_FB_RGBA8, OSP_FB_COLOR | /*OSP_FB_DEPTH |*/ OSP_FB_ACCUM);
        framebuffer = ospNewFrameBuffer(imgSize, OSP_FB_RGBA8, OSP_FB_COLOR);

    }


    // if user wants to switch renderer
    if (this->rd_type.IsDirty()) {
        switch (this->rd_type.Param<core::param::EnumParam>()->Value()) {
        case SCIVIS:
            ospRelease(camera);
            ospRelease(world);
            ospRelease(renderer);
            ospRelease(spheres);
            this->setupOSPRay(renderer, camera, world, spheres, "spheres", "scivis");
            break;
        case PATHTRACER:
            ospRelease(camera);
            ospRelease(world);
            ospRelease(renderer);
            ospRelease(spheres);
            this->setupOSPRay(renderer, camera, world, spheres, "spheres", "pathtracer");
            break;
        }
        renderer_changed = true;
        this->rd_type.ResetDirty();
    }

    // calculate image parts for e.g. screenshooter
    std::vector<float> imgStart(2, 0);
    std::vector<float> imgEnd(2, 0);
    imgStart[0] = cr->GetCameraParameters()->TileRect().GetLeft() / (float)cr->GetCameraParameters()->VirtualViewSize().GetWidth();
    imgStart[1] = cr->GetCameraParameters()->TileRect().GetBottom() / (float)cr->GetCameraParameters()->VirtualViewSize().GetHeight();

    imgEnd[0] = cr->GetCameraParameters()->TileRect().GetRight() / (float)cr->GetCameraParameters()->VirtualViewSize().GetWidth();
    imgEnd[1] = cr->GetCameraParameters()->TileRect().GetTop() / (float)cr->GetCameraParameters()->VirtualViewSize().GetHeight();


    // setup camera
    ospSet2fv(camera, "image_start", imgStart.data());
    ospSet2fv(camera, "image_end", imgEnd.data());
    ospSetf(camera, "aspect", cr->GetCameraParameters()->TileRect().AspectRatio());
    ospSet3fv(camera, "pos", cr->GetCameraParameters()->EyePosition().PeekCoordinates());
    ospSet3fv(camera, "dir", cr->GetCameraParameters()->EyeDirection().PeekComponents());
    ospSet3fv(camera, "up", cr->GetCameraParameters()->EyeUpVector().PeekComponents());
    ospSet1f(camera, "fovy", cr->GetCameraParameters()->ApertureAngle());
    ospCommit(camera);




    osprayShader.Enable();
    // if nothing changes, the image is rendered multiple times
    if (data_has_changed || cam_has_changed || !(this->extraSamles.Param<core::param::BoolParam>()->Value()) || time != cr->Time() || this->InterfaceIsDirty() || renderer_changed) {
        time = cr->Time();
        renderer_changed = false;

        for (unsigned int i = 0; i < c2->GetParticleListCount(); i++) {
            core::moldyn::MultiParticleDataCall::Particles &parts = c2->AccessParticles(i);
            // Vertex data type check
            if (parts.GetVertexDataType() == core::moldyn::MultiParticleDataCall::Particles::VERTDATA_FLOAT_XYZ) {
                vertexLength = 3;
                vertexType = OSP_FLOAT3;
            } else if (parts.GetVertexDataType() == core::moldyn::MultiParticleDataCall::Particles::VERTDATA_FLOAT_XYZR) {
                vertexLength = 3;
                vertexType = OSP_FLOAT3;
            }
            // Color data type check
            if (parts.GetColourDataType() == core::moldyn::MultiParticleDataCall::Particles::COLDATA_FLOAT_RGBA) {
                colorLength = 4;
                colorType = OSP_FLOAT4;
            } else if (parts.GetColourDataType() == core::moldyn::MultiParticleDataCall::Particles::COLDATA_FLOAT_I) {
                colorLength = 1;
                colorType = OSP_FLOAT4;
            } else if (parts.GetColourDataType() == core::moldyn::MultiParticleDataCall::Particles::COLDATA_FLOAT_RGB) {
                colorLength = 3;
                colorType = OSP_FLOAT3;
            } else if (parts.GetColourDataType() == core::moldyn::MultiParticleDataCall::Particles::COLDATA_UINT8_RGBA) {
                colorLength = 4;
                colorType = OSP_UINT4; // TODO
            } else if (parts.GetColourDataType() == core::moldyn::MultiParticleDataCall::Particles::COLDATA_UINT8_RGB) {
                colorLength = 3;
                colorType = OSP_UINT3; // TODO
            }

            std::vector<float> cd_rgba;
            cd_rgba.reserve(parts.GetCount() * 4);
            std::vector<float> vd;
            vd.reserve(parts.GetCount() * vertexLength);
            if (parts.GetColourDataType() == core::moldyn::MultiParticleDataCall::Particles::COLDATA_UINT8_RGB) {
                std::vector<unsigned char> vd_bytes;
                vd_bytes.reserve(parts.GetCount() * (parts.GetVertexDataStride() - colorLength));
                for (size_t loop = 1; loop < (parts.GetCount() + 1); loop++) {
                    for (size_t i = parts.GetVertexDataStride(); i > colorLength ; i--) {
                        vd_bytes.push_back(static_cast<const unsigned char*>(parts.GetVertexData())[loop * parts.GetVertexDataStride() - i]);
                    }
                    ASSERT(vd_bytes.size() % 12 == 0);
                    cd_rgba.push_back((float)static_cast<const uint8_t*>(parts.GetVertexData())[loop * parts.GetVertexDataStride() - 3] / 255.0f);
                    cd_rgba.push_back((float)static_cast<const uint8_t*>(parts.GetVertexData())[loop * parts.GetVertexDataStride() - 2] / 255.0f);
                    cd_rgba.push_back((float)static_cast<const uint8_t*>(parts.GetVertexData())[loop * parts.GetVertexDataStride() - 1] / 255.0f);
                    cd_rgba.push_back(1.0f);

                }

                //floatFromArrayFunc ffaf;
                //ffaf = floatFromVoidArray;
                //for (size_t loop = 0; loop < (parts.GetCount() * parts.GetVertexDataStride() / sizeof(float)); loop++) {
                //    if (!(loop % (vertexLength + colorLength) >= vertexLength)) {
                //        vd.push_back(ffaf(parts, loop));
                //    }
                //}

                auto blub = reinterpret_cast<float*>(vd_bytes.data());
                vd.assign(blub, blub + parts.GetCount()*vertexLength);
            }


            if (parts.GetColourDataType() == core::moldyn::MultiParticleDataCall::Particles::COLDATA_FLOAT_I) {
                floatFromArrayFunc ffaf;
                ffaf = floatFromVoidArray;
                std::vector<float> cd;

                for (size_t loop = 0; loop < (parts.GetCount() * parts.GetVertexDataStride() / sizeof(float)); loop++) {
                    if (loop % (vertexLength + colorLength) >= vertexLength) {
                        cd.push_back(ffaf(parts, loop));
                    } else {
                        vd.push_back(ffaf(parts, loop));
                    }
                }

                // Color transfer call and calculation
                core::view::CallGetTransferFunction *cgtf = this->getTFSlot.CallAs<core::view::CallGetTransferFunction>();
                if (cgtf != NULL && ((*cgtf)())) {
                    float const* tf_tex = cgtf->GetTextureData();
                    tex_size = cgtf->TextureSize();
                    this->colorTransferGray(cd, tf_tex, tex_size, cd_rgba);
                } else {
                    this->colorTransferGray(cd, NULL, 0, cd_rgba);
                }
                colorLength = 4;
            }

            // test for spheres with inidividual radii
            /*
         
            std::vector<float> vd_test = { 1.0f, 0.0f, 0.0f, 1.0f,
                0.0f, 10.0f, 0.0f, 10.0f };
            std::vector<float> cd_rgba_test = { 0.5f, 0.5f, 0.5f, 1.0f,
                1.0f, 0.0f, 0.0f, 1.0f };

            vertexData = ospNewData(2, OSP_FLOAT4, vd_test.data());
            colorData = ospNewData(2, colorType, cd_rgba_test.data());

            for (int i = 0; i < (vd_test.size() / 4); i++) {
                OSPGeometry sphere = ospNewGeometry("spheres");
                std::vector<float> tmp = { vd_test[4 * i + 0], vd_test[4 * i + 1], vd_test[4 * i + 2], vd_test[4 * i + 3] };
                OSPData ospvd = ospNewData(1, OSP_FLOAT4, tmp.data());
                ospCommit(ospvd);
                ospSetData(sphere, "spheres", ospvd);
                ospSet1f(sphere, "radius", vd_test[4 * i + 3]);
                ospCommit(sphere);
                ospAddGeometry(world, sphere);
                ospCommit(world);
            }
            */

   //         vertexData = ospNewData(parts.GetCount(), vertexType, vd.data());
			//if (parts.GetColourDataType() == core::moldyn::MultiParticleDataCall::Particles::COLDATA_FLOAT_I) {
			//	colorData = ospNewData(parts.GetCount(), colorType, cd_rgba.data());
			//}
			//else {
			//	colorData = ospNewData(parts.GetCount(), OSP_FLOAT3, cd.data());
			//}
   //         ospCommit(vertexData);
   //         ospCommit(colorData);
			vertexData = ospNewData(parts.GetCount(), vertexType, vd.data());
			colorData = ospNewData(parts.GetCount(), OSP_FLOAT4, cd_rgba.data());
			ospCommit(vertexData);
			ospCommit(colorData);

			//ospSet1i(spheres, "bytes_per_sphere", parts.GetVertexDataStride());
            ospSet1i(spheres, "bytes_per_sphere", vertexLength*sizeof(float));

			//ospSet1i(spheres, "color_stride", parts.GetVertexDataStride());
			//ospSet1i(spheres, "color_offset", 12);
			//ospSet1i(spheres, "color_stride", 3);
			ospSetData(spheres, "spheres", vertexData);
            ospSetData(spheres, "color", colorData);
            ospSet1f(spheres, "radius", parts.GetGlobalRadius());

            // clipPlane setup
            std::vector<float> clipDat(4);
            std::vector<float> clipCol(4);
            this->getClipData(clipDat.data(), clipCol.data());

            if (!std::all_of(clipDat.begin(), clipDat.end() - 1, [](float i) { return i == 0; })) {
                pln = ospNewPlane("clipPlane");
                ospSet1f(pln, "dist", clipDat[3]);
                ospSet3fv(pln, "normal", clipDat.data());
                ospSet4fv(pln, "color", clipCol.data());
                ospCommit(pln);
                ospSetObject(spheres, "clipPlane", pln);
            } else {
                ospSetObject(spheres, "clipPlane", NULL);
            }

            // custom material settings
            OSPMaterial material;
            switch (this->mat_type.Param<core::param::EnumParam>()->Value()) {
            case OBJMATERIAL:
                material = ospNewMaterial(renderer, "OBJMaterial");
                ospSet3fv(material, "Kd", this->mat_Kd.Param<core::param::Vector3fParam>()->Value().PeekComponents());
                ospSet3fv(material, "Ks", this->mat_Ks.Param<core::param::Vector3fParam>()->Value().PeekComponents());
                ospSet1f(material, "Ns", this->mat_Ns.Param<core::param::FloatParam>()->Value());
                ospSet1f(material, "d", this->mat_d.Param<core::param::FloatParam>()->Value());
                ospSet3fv(material, "Tf", this->mat_Tf.Param<core::param::Vector3fParam>()->Value().PeekComponents());
                break;
            case GLASS:
                material = ospNewMaterial(renderer, "Glass");
                break;
            case MATTE:
                material = ospNewMaterial(renderer, "Matte");
                break;
            case METAL:
                material = ospNewMaterial(renderer, "Metal");
                break;
            case METALLICPAINT:
                material = ospNewMaterial(renderer, "MetallicPaint");
                break;
            case PLASTIC:
                material = ospNewMaterial(renderer, "Plastic");
                break;
            case THINGLASS:
                material = ospNewMaterial(renderer, "ThinGlass");
                break;
            case VELVET:
                material = ospNewMaterial(renderer, "Velvet");
                break;
            }

            ospCommit(material);
            ospSetMaterial(spheres, material);
            ospCommit(spheres);
            ospCommit(world);



            RendererSettings(renderer);

            OSPRayLights(renderer, call);
            ospCommit(renderer);

            
            // setup framebuffer
            ospFrameBufferClear(framebuffer, OSP_FB_COLOR);
            ospRenderFrame(framebuffer, renderer, OSP_FB_COLOR);
            //ospFrameBufferClear(framebuffer, OSP_FB_COLOR | OSP_FB_ACCUM);
            //ospRenderFrame(framebuffer, renderer, OSP_FB_COLOR | OSP_FB_ACCUM);

            // get the texture from the framebuffer
            fb = (uint32_t*)ospMapFrameBuffer(framebuffer, OSP_FB_COLOR);

            //writePPM("ospframe.ppm", imgSize, fb);

            this->renderTexture2D(osprayShader, fb, imgSize.x, imgSize.y);

            // clear stuff
            ospUnmapFrameBuffer(fb, framebuffer);
            ospRelease(vertexData);
            ospRelease(colorData);
            ospRelease(material);

            vd.clear();
            cd_rgba.clear();

        }
    } else {
            ospRenderFrame(framebuffer, renderer, OSP_FB_COLOR);
            //ospRenderFrame(framebuffer, renderer, OSP_FB_COLOR | OSP_FB_ACCUM);
            fb = (uint32_t*)ospMapFrameBuffer(framebuffer, OSP_FB_COLOR);
            this->renderTexture2D(osprayShader, fb, imgSize.x, imgSize.y);
            ospUnmapFrameBuffer(fb, framebuffer);
    }

    c2->Unlock();
    osprayShader.Disable();

    return true;
}

/*
ospray::OSPRaySphereRenderer::InterfaceIsDirty()
*/
bool ospray::OSPRaySphereRenderer::InterfaceIsDirty() {
    if (
        this->AbstractIsDirty() ||
        this->mat_Kd.IsDirty() ||
        this->mat_Ks.IsDirty() ||
        this->mat_Ns.IsDirty() ||
        this->mat_d.IsDirty() ||
        this->mat_Tf.IsDirty())
    {
        this->AbstractResetDirty();
        this->mat_Kd.ResetDirty();
        this->mat_Ks.ResetDirty();
        this->mat_Ns.ResetDirty();
        this->mat_d.ResetDirty();
        this->mat_Tf.ResetDirty();
        return true;
    } else {
        return false;
    }
}

/*
* ospray::OSPRaySphereRenderer::getData
*/
core::moldyn::MultiParticleDataCall *ospray::OSPRaySphereRenderer::getData(unsigned int t, float& outScaling) {
    core::moldyn::MultiParticleDataCall *c2 = this->getDataSlot.CallAs<core::moldyn::MultiParticleDataCall>();
    outScaling = 1.0f;
    if (c2 != NULL) {
        c2->SetFrameID(t, true); // isTimeForced flag set to true
        if (!(*c2)(1)) return NULL;

        // calculate scaling
        outScaling = c2->AccessBoundingBoxes().ObjectSpaceBBox().LongestEdge();
        if (outScaling > 0.0000001) {
            outScaling = 10.0f / outScaling;
        } else {
            outScaling = 1.0f;
        }

        c2->SetFrameID(t, true); // isTimeForced flag set to true
        if (!(*c2)(0)) return NULL;

        return c2;
    } else {
        return NULL;
    }
}


/*
* ospray::OSPRaySphereRenderer::getClipData
*/
void ospray::OSPRaySphereRenderer::getClipData(float *clipDat, float *clipCol) {
    core::view::CallClipPlane *ccp = this->getClipPlaneSlot.CallAs<core::view::CallClipPlane>();
    if ((ccp != NULL) && (*ccp)()) {
        clipDat[0] = ccp->GetPlane().Normal().X();
        clipDat[1] = ccp->GetPlane().Normal().Y();
        clipDat[2] = ccp->GetPlane().Normal().Z();
        vislib::math::Vector<float, 3> grr(ccp->GetPlane().Point().PeekCoordinates());
        clipDat[3] = grr.Dot(ccp->GetPlane().Normal());
        clipCol[0] = static_cast<float>(ccp->GetColour()[0]) / 255.0f;
        clipCol[1] = static_cast<float>(ccp->GetColour()[1]) / 255.0f;
        clipCol[2] = static_cast<float>(ccp->GetColour()[2]) / 255.0f;
        clipCol[3] = static_cast<float>(ccp->GetColour()[3]) / 255.0f;

    } else {
        clipDat[0] = clipDat[1] = clipDat[2] = clipDat[3] = 0.0f;
        clipCol[0] = clipCol[1] = clipCol[2] = 0.75f;
        clipCol[3] = 1.0f;
    }
}


/*
* ospray::OSPRaySphereRenderer::GetCapabilities
*/
bool ospray::OSPRaySphereRenderer::GetCapabilities(core::Call& call) {
    core::view::CallRender3D *cr = dynamic_cast<core::view::CallRender3D*>(&call);
    if (cr == NULL) return false;

    cr->SetCapabilities(
        core::view::CallRender3D::CAP_RENDER
        | core::view::CallRender3D::CAP_LIGHTING
        | core::view::CallRender3D::CAP_ANIMATION
    );

    return true;
}


/*
* moldyn::AbstractSimpleSphereRenderer::GetExtents
*/
bool ospray::OSPRaySphereRenderer::GetExtents(core::Call& call) {
    //core::view::CallRender3D *cr = dynamic_cast<core::view::CallRender3D*>(&call);
    //if (cr == NULL) return false;

    //core::moldyn::MultiParticleDataCall *c2 = this->getDataSlot.CallAs<core::moldyn::MultiParticleDataCall>();
    //c2->SetFrameID(static_cast<unsigned int>(cr->Time()), true); // isTimeForced flag set to true
    //if ((c2 != NULL) && ((*c2)(1))) {
    //    cr->SetTimeFramesCount(c2->FrameCount());
    //    cr->AccessBoundingBoxes() = c2->AccessBoundingBoxes();

    //    float scaling = cr->AccessBoundingBoxes().ObjectSpaceBBox().LongestEdge();
    //    if (scaling > 0.0000001) {
    //        scaling = 10.0f / scaling;
    //    } else {
    //        scaling = 1.0f;
    //    }
    //    cr->AccessBoundingBoxes().MakeScaledWorld(scaling);

    //} else {
    //    cr->SetTimeFramesCount(1);
    //    cr->AccessBoundingBoxes().Clear();
    //}

	core::view::CallRender3D *cr = dynamic_cast<core::view::CallRender3D*>(&call);
	if (cr == NULL) return false;
	core::moldyn::MultiParticleDataCall *c2 = this->getDataSlot.CallAs<core::moldyn::MultiParticleDataCall>();
	if (c2 == NULL) return false;
	c2->SetFrameID(static_cast<int>(cr->Time()));
	if (!(*c2)(1)) return false;

	cr->SetTimeFramesCount(c2->FrameCount());
	cr->AccessBoundingBoxes().Clear();
	cr->AccessBoundingBoxes() = c2->AccessBoundingBoxes();
	cr->AccessBoundingBoxes().MakeScaledWorld(1.0f);

    return true;
}

