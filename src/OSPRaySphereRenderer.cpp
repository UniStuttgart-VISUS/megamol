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
#include "vislib/sys/Log.h"
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
#include <functional>

#include "ospray/ospray.h"

#include <stdint.h>
#include <sstream>

using namespace megamol;



VISLIB_FORCEINLINE float floatFromVoidArray(const megamol::core::moldyn::MultiParticleDataCall::Particles& p, size_t index) {
    //const float* parts = static_cast<const float*>(p.GetVertexData());
    //return parts[index * stride + offset];
    return static_cast<const float*>(p.GetVertexData())[index];
}

VISLIB_FORCEINLINE unsigned char byteFromVoidArray(const megamol::core::moldyn::MultiParticleDataCall::Particles& p, size_t index) {
	return static_cast<const unsigned char*>(p.GetVertexData())[index];
}

typedef float(*floatFromArrayFunc)(const megamol::core::moldyn::MultiParticleDataCall::Particles& p, size_t index);
typedef unsigned char(*byteFromArrayFunc)(const megamol::core::moldyn::MultiParticleDataCall::Particles& p, size_t index);

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
mat_type("Material::Type", "Switches material types"),
particleList("General::ParticleList", "Switches between particle lists")
{
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

    //tmp variable
    number = 0;

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

    this->particleList << new core::param::IntParam(0);
    this->MakeSlotAvailable(&this->particleList);
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

    this->initOSPRay(device);
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

    if (device != ospGetCurrentDevice()) {
        ospSetCurrentDevice(device);
    }
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

    setupOSPRayCamera(camera, cr);
    ospCommit(camera);

    osprayShader.Enable();
    // if nothing changes, the image is rendered multiple times
    if (data_has_changed || cam_has_changed || !(this->extraSamles.Param<core::param::BoolParam>()->Value()) || time != cr->Time() || this->InterfaceIsDirty() || renderer_changed) {
        time = cr->Time();
        renderer_changed = false;

        if (this->particleList.Param<core::param::IntParam>()->Value() > (c2->GetParticleListCount() - 1)) {
            this->particleList.Param<core::param::IntParam>()->SetValue(0);
        }

        core::moldyn::MultiParticleDataCall::Particles &parts = c2->AccessParticles(this->particleList.Param<core::param::IntParam>()->Value());

        // Vertex data type check
        if (parts.GetVertexDataType() == core::moldyn::MultiParticleDataCall::Particles::VERTDATA_FLOAT_XYZ) {
            vertexLength = 3;
            vertexType = OSP_FLOAT3;
            ospSet1i(spheres, "bytes_per_sphere", vertexLength * sizeof(float));
            ospSet1f(spheres, "radius", parts.GetGlobalRadius());
        } else if (parts.GetVertexDataType() == core::moldyn::MultiParticleDataCall::Particles::VERTDATA_FLOAT_XYZR) {
            vertexLength = 4;
            vertexType = OSP_FLOAT4;
            ospSet1i(spheres, "bytes_per_sphere", vertexLength * sizeof(float));
            ospSet1i(spheres, "offset_radius", (vertexLength - 1) * sizeof(float));
        }
        // reserve space for vertex data object
        vd.reserve(parts.GetCount() * vertexLength);

        // Color data type check
        if (parts.GetColourDataType() == core::moldyn::MultiParticleDataCall::Particles::COLDATA_FLOAT_RGBA) {
            colorLength = 4;
            convertedColorType = OSP_FLOAT4;
            cd_rgba.reserve(parts.GetCount() * colorLength);

            floatFromArrayFunc ffaf;
            ffaf = floatFromVoidArray;

            for (size_t loop = 0; loop < (parts.GetCount() * parts.GetVertexDataStride() / sizeof(float)); loop++) {
                if (loop % (vertexLength + colorLength) >= vertexLength) {
                    cd_rgba.push_back(ffaf(parts, loop));
                } else {
                    vd.push_back(ffaf(parts, loop));
                }
            }
        } else if (parts.GetColourDataType() == core::moldyn::MultiParticleDataCall::Particles::COLDATA_FLOAT_I) {
            // this colorType will be transformed to:
            convertedColorType = OSP_FLOAT4;
            colorLength = 4;
            cd_rgba.reserve(parts.GetCount() * colorLength);

            floatFromArrayFunc ffaf;
            ffaf = floatFromVoidArray;
            std::vector<float> cd;

            for (size_t loop = 0; loop < (parts.GetCount() * parts.GetVertexDataStride() / sizeof(float)); loop++) {
                if (loop % (vertexLength + 1) >= vertexLength) {
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


        } else if (parts.GetColourDataType() == core::moldyn::MultiParticleDataCall::Particles::COLDATA_FLOAT_RGB) {
            colorLength = 3;
            convertedColorType = OSP_FLOAT3;
            cd_rgba.reserve(parts.GetCount() * colorLength);

            floatFromArrayFunc ffaf;
            ffaf = floatFromVoidArray;

            for (size_t loop = 0; loop < (parts.GetCount() * parts.GetVertexDataStride() / sizeof(float)); loop++) {
                if (loop % (vertexLength + colorLength) >= vertexLength) {
                    cd_rgba.push_back(ffaf(parts, loop));
                } else {
                    vd.push_back(ffaf(parts, loop));
                }
            }
        } else if (parts.GetColourDataType() == core::moldyn::MultiParticleDataCall::Particles::COLDATA_UINT8_RGBA) {
            colorLength = 4;
            convertedColorType = OSP_FLOAT4;
            cd_rgba.reserve(parts.GetCount() * colorLength);

            float alpha = 1.0f;
            byteFromArrayFunc bfaf;
            bfaf = byteFromVoidArray;

            vd.resize(parts.GetCount() * vertexLength);
            auto data = static_cast<const float*>(parts.GetVertexData());

            for (size_t i = 0; i < parts.GetCount(); i++) {
                std::copy_n(data + (i * parts.GetVertexDataStride() / sizeof(float)),
                    vertexLength,
                    vd.begin() + (i * vertexLength));
                for (size_t j = 0; j < colorLength; j++) {
                    cd_rgba.push_back((float)bfaf(parts, i * parts.GetVertexDataStride() + vertexLength * sizeof(float) + j) / 255.0f);
                }
            }
        } else if (parts.GetColourDataType() == core::moldyn::MultiParticleDataCall::Particles::COLDATA_UINT8_RGB) {
            vislib::sys::Log::DefaultLog.WriteError("File format deprecated. Convert your data.");
        }



        if (parts.GetColourDataType() != core::moldyn::MultiParticleDataCall::Particles::COLDATA_NONE) {
            vertexData = ospNewData(parts.GetCount(), vertexType, vd.data());
            ospCommit(vertexData);
            ospSetData(spheres, "spheres", vertexData);

            colorData = ospNewData(parts.GetCount(), convertedColorType, cd_rgba.data());
            ospCommit(colorData);
            ospSetData(spheres, "color", colorData);

            //OSPData data = ospNewData(parts.GetCount(), OSP_FLOAT4, parts.GetVertexData());
            //ospCommit(data);
            //ospSetData(spheres, "spheres", data);
            //ospSet1i(spheres, "bytes_per_sphere", 4*sizeof(float));
            //ospSet1i(spheres, "color_offset", 3*sizeof(float));
            //ospSet1i(spheres, "color_stride", 1*sizeof(float));
        } else {

            vertexData = ospNewData(parts.GetCount(), vertexType, parts.GetVertexData());
            ospCommit(vertexData);
            ospSetData(spheres, "spheres", vertexData);
        }



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

        // Light callback
        LightDelegate delegate_addLight = std::bind(&ospray::OSPRaySphereRenderer::addLight, this, std::placeholders::_1, std::placeholders::_2);
        CallOSPRayLight *gl = this->getLightSlot.CallAs<CallOSPRayLight>();
        if (gl != NULL) {
            gl->SetID(0);
            gl->SetDelegate(delegate_addLight);
            if (!(*gl)(0)) {
                vislib::sys::Log::DefaultLog.WriteError("Error in getLight callback");
            }
            lightArray = ospNewData(this->lightsToAdd.size(), OSP_OBJECT, lightsToAdd.data(), 0);
            ospSetData(renderer, "lights", lightArray);
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
        ospRelease(vertexData);
        ospRelease(colorData);
        ospRelease(material);

        vd.clear();
        cd_rgba.clear();

    } else {
            ospRenderFrame(framebuffer, renderer, OSP_FB_COLOR | OSP_FB_ACCUM);
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
        this->mat_Tf.IsDirty() ||
        this->particleList.IsDirty())
    {
        this->AbstractResetDirty();
        this->mat_Kd.ResetDirty();
        this->mat_Ks.ResetDirty();
        this->mat_Ns.ResetDirty();
        this->mat_d.ResetDirty();
        this->mat_Tf.ResetDirty();
        this->particleList.ResetDirty();
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


