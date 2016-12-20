/*
 * OSPRaySphereRenderer.cpp
 * Copyright (C) 2009-2015 by MegaMol Team
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


using namespace megamol;



VISLIB_FORCEINLINE float floatFromVoidArray(const megamol::core::moldyn::MultiParticleDataCall::Particles& p, size_t index) {
    //const float* parts = static_cast<const float*>(p.GetVertexData());
    //return parts[index * stride + offset];
    return static_cast<const float*>(p.GetVertexData())[index];
}

typedef float(*floatFromArrayFunc)(const megamol::core::moldyn::MultiParticleDataCall::Particles& p, size_t index);

/*
ospray::OSPRaySphereRenderer::OSPRaySphereRenderer
*/
ospray::OSPRaySphereRenderer::OSPRaySphereRenderer(void) : core::moldyn::AbstractSimpleSphereRenderer(),
osprayShader(),
extraSamles("General::extraSamples", "Extra sampling when camera is not moved"),
// General light parameters
lightColor("Light::General::LightColor", "Sets the color of the Light"),
shadows("Light::General::Shadows", "Enables/Disables computation of hard shadows"),
lightIntensity("Light::General::LightIntensity", "Intensity of the Light"),
lightType("Light::Type::LightType", "Type of the light"),
// Distant light parameters
dl_direction("Light::DistantLight::LightDirection", "Direction of the Light"),
dl_angularDiameter("Light::DistantLight::AngularDiameter", "If greater than zero results in soft shadows"),
dl_eye_direction("Light::DistantLight::EyeDirection", "Sets the light direction as view direction"),
// point light parameters
pl_position("Light::PointLight::Position", ""),
pl_radius("Light::PointLight::Radius", ""),
// spot light parameters
sl_position("Light::SpotLight::Position", ""),
sl_direction("Light::SpotLight::Direction", ""),
sl_openingAngle("Light::SpotLight::openingAngle", ""),
sl_penumbraAngle("Light::SpotLight::penumbraAngle", ""),
sl_radius("Light::SpotLight::Radius", ""),
// quad light parameters
ql_position("Light::QuadLight::Position", ""),
ql_edgeOne("Light::QuadLight::Edge1", ""),
ql_edgeTwo("Light::QuadLight::Edge2", ""),
// hdri light parameteres
hdri_up("Light::HDRILight::up", ""),
hdri_direction("Light::HDRILight::Direction", ""),
hdri_evnfile("Light::HDRILight::EvironmentFile", ""),
// general renderer parameters
rd_epsilon("Renderer::Epsilon", "Ray epsilon to avoid self-intersections"),
rd_spp("Renderer::SamplesPerPixel", "Samples per pixel"),
rd_maxRecursion("Renderer::maxRecursion", "Maximum ray recursion depth"),
rd_type("Renderer::Type", "Select between SciVis and PathTracer"),
// scivis renderer parameters
AOweight("Renderer::SciVis::AOweight", "Amount of ambient occlusion added in shading"),
AOsamples("Renderer::SciVis::AOsamples", "Number of rays per sample to compute ambient occlusion"),
AOdistance("Renderer::SciVis::AOdistance", "Maximum distance to consider for ambient occlusion"),
// pathtracer renderer parameters
rd_ptBackground("Renderer::PathTracer::BackgroundTexture", "Texture image used as background, replacing visible lights in infinity"),
// material parameteres
mat_Kd("Material::OBJMaterial::DiffuseColor", "Diffuse color"),
mat_Ks("Material::OBJMaterial::SpecularColor", "Specular color"),
mat_Ns("Material::OBJMaterial::Shininess", "Phong exponent"),
mat_d("Material::OBJMaterial::Opacity", "Opacity"),
mat_Tf("Material::OBJMaterial::TransparencyFilterColor", "Transparency filter color"),
mat_type("Material::Type", "Switches material types")
{
    imgSize.x = 0;
    imgSize.y = 0;
    time = 0;
    framebuffer = NULL;
    light = NULL;
    core::param::EnumParam *lt = new core::param::EnumParam(NONE);
    lt->SetTypePair(NONE, "None");
    lt->SetTypePair(DISTANTLIGHT, "DistantLight");
    lt->SetTypePair(POINTLIGHT, "PointLight");
    lt->SetTypePair(SPOTLIGHT, "SpotLight");
    lt->SetTypePair(QUADLIGHT, "QuadLight");
    lt->SetTypePair(AMBIENTLIGHT, "AmbientLight");
    lt->SetTypePair(HDRILIGHT, "HDRILight");

    core::param::EnumParam *rdt = new core::param::EnumParam(SCIVIS);
    rdt->SetTypePair(SCIVIS, "SciVis");
    rdt->SetTypePair(PATHTRACER, "PathTracer");

    core::param::EnumParam *mt = new core::param::EnumParam(OBJMATERIAL);
    mt->SetTypePair(OBJMATERIAL, "OBJMaterial");
    mt->SetTypePair(GLASS, "Glass (only PathTracer)");
    mt->SetTypePair(MATTE, "Matte (only PathTracer)");
    mt->SetTypePair(METAL, "Metal (only PathTracer)");
    mt->SetTypePair(METALLICPAINT, "MetallicPaint (only PathTracer)");
    mt->SetTypePair(PLASTIC, "Plastic (only PathTracer)");
    mt->SetTypePair(THINGLASS, "ThinGlass (only PathTracer)");
    mt->SetTypePair(VELVET, "Velvet (only PathTracer)");

    // Ambient parameters
    this->AOweight << new core::param::FloatParam(1.0f);
    this->AOsamples << new core::param::IntParam(1);
    this->AOdistance << new core::param::FloatParam(1e20f);
    this->extraSamles << new core::param::BoolParam(true);
    this->MakeSlotAvailable(&this->AOweight);
    this->MakeSlotAvailable(&this->AOsamples);
    this->MakeSlotAvailable(&this->AOdistance);
    this->MakeSlotAvailable(&this->extraSamles);

    // general light
    this->shadows << new core::param::BoolParam(0);
    this->lightColor << new core::param::Vector3fParam(vislib::math::Vector<float,3>( 1.0f, 1.0f, 1.0f ));
    this->lightType << lt;
    this->lightIntensity << new core::param::FloatParam(1.0f);
    this->MakeSlotAvailable(&this->lightIntensity);
    this->MakeSlotAvailable(&this->lightColor);
    this->MakeSlotAvailable(&this->shadows);
    this->MakeSlotAvailable(&this->lightType);

    // distant light
    this->dl_angularDiameter << new core::param::FloatParam(0.0f);
    this->dl_direction << new core::param::Vector3fParam(vislib::math::Vector<float, 3>(0.0f, -1.0f, 0.0f));
    this->dl_eye_direction << new core::param::BoolParam(0);
    this->MakeSlotAvailable(&this->dl_direction);
    this->MakeSlotAvailable(&this->dl_angularDiameter);
    this->MakeSlotAvailable(&this->dl_eye_direction);

    // point light
    this->pl_position << new core::param::Vector3fParam(vislib::math::Vector<float, 3>(0.0f, 0.0f, 0.0f));
    this->pl_radius << new core::param::FloatParam(0.0f);
    this->MakeSlotAvailable(&this->pl_position);
    this->MakeSlotAvailable(&this->pl_radius);

    // spot light
    this->sl_position << new core::param::Vector3fParam(vislib::math::Vector<float, 3>(0.0f, 0.0f, 0.0f));
    this->sl_direction << new core::param::Vector3fParam(vislib::math::Vector<float, 3>(0.0f, 1.0f, 0.0f));
    this->sl_openingAngle << new core::param::FloatParam(0.0f);
    this->sl_penumbraAngle << new core::param::FloatParam(0.0f);
    this->sl_radius << new core::param::FloatParam(0.0f);
    this->MakeSlotAvailable(&this->sl_position);
    this->MakeSlotAvailable(&this->sl_direction);
    this->MakeSlotAvailable(&this->sl_openingAngle);
    this->MakeSlotAvailable(&this->sl_penumbraAngle);
    this->MakeSlotAvailable(&this->sl_radius);

    // quad light
    this->ql_position << new core::param::Vector3fParam(vislib::math::Vector<float, 3>(1.0f, 0.0f, 0.0f));
    this->ql_edgeOne << new core::param::Vector3fParam(vislib::math::Vector<float, 3>(0.0f, 1.0f, 0.0f));
    this->ql_edgeTwo << new core::param::Vector3fParam(vislib::math::Vector<float, 3>(0.0f, 0.0f, 1.0f));
    this->MakeSlotAvailable(&this->ql_position);
    this->MakeSlotAvailable(&this->ql_edgeOne);
    this->MakeSlotAvailable(&this->ql_edgeTwo);

    // HDRI light
    this->hdri_up << new core::param::Vector3fParam(vislib::math::Vector<float, 3>(0.0f, 1.0f, 0.0f));
    this->hdri_direction << new core::param::Vector3fParam(vislib::math::Vector<float, 3>(0.0f, 0.0f, 1.0f));
    this->hdri_evnfile << new core::param::FilePathParam("");
    this->MakeSlotAvailable(&this->hdri_up);
    this->MakeSlotAvailable(&this->hdri_direction);
    this->MakeSlotAvailable(&this->hdri_evnfile);

    // General Renderer
    this->rd_epsilon << new core::param::FloatParam(1e-4f);
    this->rd_spp << new core::param::IntParam(1);
    this->rd_maxRecursion << new core::param::IntParam(10);
    this->rd_type << rdt;
    this->MakeSlotAvailable(&this->rd_epsilon);
    this->MakeSlotAvailable(&this->rd_spp);
    this->MakeSlotAvailable(&this->rd_maxRecursion);
    this->MakeSlotAvailable(&this->rd_type);

    //PathTracer
    this->rd_ptBackground << new core::param::FilePathParam("");
    this->MakeSlotAvailable(&this->rd_ptBackground);

    // Material
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
    ospRelease(camera);
    ospRelease(world);
    ospRelease(renderer);
    ospRelease(spheres);
    ospRelease(light);
    ospRelease(lightArray);
    ospRelease(pln);
    releaseTextureScreen();
    core::moldyn::AbstractSimpleSphereRenderer::release();
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
        if (framebuffer != NULL) ospFreeFrameBuffer(framebuffer);
        imgSize.x = cr->GetCameraParameters()->TileRect().Width();
        imgSize.y = cr->GetCameraParameters()->TileRect().Height();
        framebuffer = ospNewFrameBuffer(imgSize, OSP_FB_RGBA8, OSP_FB_COLOR | /*OSP_FB_DEPTH |*/ OSP_FB_ACCUM);
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

        for (unsigned int i = 0; i < c2->GetParticleListCount(); i++) {
            core::moldyn::MultiParticleDataCall::Particles &parts = c2->AccessParticles(i);
            // Vertex data type check
            if (parts.GetVertexDataType() == core::moldyn::MultiParticleDataCall::Particles::VERTDATA_FLOAT_XYZ) {
                vertexLength = 3;
                vertexType = OSP_FLOAT3;
            } else if (parts.GetVertexDataType() == core::moldyn::MultiParticleDataCall::Particles::VERTDATA_FLOAT_XYZR) {
                vertexLength = 4;
                vertexType = OSP_FLOAT4;
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
            }


            floatFromArrayFunc ffaf;
            ffaf = floatFromVoidArray;
            std::vector<float> vd;
            std::vector<float> cd;
            std::vector<float> cd_rgba;
            for (size_t loop = 0; loop < (parts.GetCount() * parts.GetVertexDataStride()/sizeof(float)); loop++) {
                if (loop % (vertexLength+colorLength) >= vertexLength) {
                    cd.push_back(ffaf(parts, loop));
                } else {
                    vd.push_back(ffaf(parts, loop));
                }
            }

            if (parts.GetColourDataType() == core::moldyn::MultiParticleDataCall::Particles::COLDATA_FLOAT_I) {
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

            vertexData = ospNewData(parts.GetCount(), vertexType, vd.data());
            colorData = ospNewData(parts.GetCount(), colorType, cd_rgba.data());
            ospCommit(vertexData);
            ospCommit(colorData);

            ospSet1i(spheres, "bytes_per_sphere", (vertexLength * sizeof(float)));
            ospSet1i(spheres, "color_stride", (colorLength * sizeof(float)));
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

            // general renderer settings
            ospSet1f(renderer, "epsilon", this->rd_epsilon.Param<core::param::FloatParam>()->Value());
            ospSet1i(renderer, "spp", this->rd_spp.Param<core::param::IntParam>()->Value());
            ospSet1i(renderer, "maxDepth", this->rd_maxRecursion.Param<core::param::IntParam>()->Value());

            switch (this->rd_type.Param<core::param::EnumParam>()->Value()) {
            case SCIVIS:
                // scivis renderer settings
                ospSet1f(renderer, "aoWeight", this->AOweight.Param<core::param::FloatParam>()->Value());
                ospSet1i(renderer, "aoSamples", this->AOsamples.Param<core::param::IntParam>()->Value());
                ospSet1i(renderer, "shadowsEnabled", this->shadows.Param<core::param::BoolParam>()->Value());
                ospSet1f(renderer, "aoOcclusionDistance", this->AOdistance.Param<core::param::FloatParam>()->Value());
                GLfloat bgcolor[4];
                glGetFloatv(GL_COLOR_CLEAR_VALUE, bgcolor);
                ospSet3fv(renderer, "bgColor", bgcolor);
                /* Not implemented
                ospSet1i(renderer, "oneSidedLighting", 0);
                ospSet1i(renderer, "backgroundEnabled", 0);
                */
                break;
            case PATHTRACER:
                if (this->rd_ptBackground.Param<core::param::FilePathParam>()->Value() != vislib::TString("")) {
                    OSPTexture2D bkgnd_tex = this->TextureFromFile(this->rd_ptBackground.Param<core::param::FilePathParam>()->Value());
                    ospSetObject(renderer, "backplate", bkgnd_tex);
                }
                break;
            }

            //ospCommit(renderer);

            // create custom ospray light
            switch (this->lightType.Param<core::param::EnumParam>()->Value()) {
            case NONE:
                light = NULL;
                break;
            case DISTANTLIGHT:
                light = ospNewLight(renderer, "distant");
                if (this->dl_eye_direction.Param<core::param::BoolParam>()->Value() == true) {
                    //GLfloat lightdir[4];
                    //glGetLightfv(GL_LIGHT0, GL_POSITION, lightdir);
                    //ospSetVec3f(light, "direction", { lightdir[0], lightdir[1], lightdir[2] });
                    ospSet3fv(light, "direction", cr->GetCameraParameters()->EyeDirection().PeekComponents());
                } else {
                    ospSet3fv(light, "direction", this->dl_direction.Param<core::param::Vector3fParam>()->Value().PeekComponents());
                }
                ospSet1f(light, "angularDiameter", this->dl_angularDiameter.Param<core::param::FloatParam>()->Value());
                break;
            case POINTLIGHT:
                light = ospNewLight(renderer, "point");
                ospSet3fv(light, "position", this->pl_position.Param<core::param::Vector3fParam>()->Value().PeekComponents());
                ospSet1f(light, "radius", this->pl_radius.Param<core::param::FloatParam>()->Value());
                break;
            case SPOTLIGHT:
                light = ospNewLight(renderer, "spot");
                ospSet3fv(light, "position", this->sl_position.Param<core::param::Vector3fParam>()->Value().PeekComponents());
                ospSet3fv(light, "direction", this->sl_direction.Param<core::param::Vector3fParam>()->Value().PeekComponents());
                ospSet1f(light, "openingAngle", this->sl_openingAngle.Param<core::param::FloatParam>()->Value());
                ospSet1f(light, "penumbraAngle", this->sl_penumbraAngle.Param<core::param::FloatParam>()->Value());
                ospSet1f(light, "radius", this->sl_radius.Param<core::param::FloatParam>()->Value());
                break;
            case QUADLIGHT:
                light = ospNewLight(renderer, "quad");
                ospSet3fv(light, "position", this->ql_position.Param<core::param::Vector3fParam>()->Value().PeekComponents());
                ospSet3fv(light, "edge1", this->ql_edgeOne.Param<core::param::Vector3fParam>()->Value().PeekComponents());
                ospSet3fv(light, "edge2", this->ql_edgeTwo.Param<core::param::Vector3fParam>()->Value().PeekComponents());
                break;
            case HDRILIGHT:
                light = ospNewLight(renderer, "hdri");
                ospSet3fv(light, "up", this->hdri_up.Param<core::param::Vector3fParam>()->Value().PeekComponents());
                ospSet3fv(light, "dir", this->hdri_direction.Param<core::param::Vector3fParam>()->Value().PeekComponents());
                if (this->hdri_evnfile.Param<core::param::FilePathParam>()->Value() != vislib::TString("")) {
                    OSPTexture2D hdri_tex = this->TextureFromFile(this->hdri_evnfile.Param<core::param::FilePathParam>()->Value());
                    ospSetObject(renderer, "backplate", hdri_tex);
                }
                break;
            case AMBIENTLIGHT:
                light = ospNewLight(renderer, "ambient");
                break;
            }

            if (light == NULL) {
                ospSetData(renderer, "lights", NULL);
            } else {
                ospSet1f(light, "intensity", this->lightIntensity.Param<core::param::FloatParam>()->Value());
                vislib::math::Vector<float, 3> lc = this->lightColor.Param<core::param::Vector3fParam>()->Value();
                ospSetVec3f(light, "color", { lc.X(), lc.Y(), lc.Z() });
                ospCommit(light);
                lightArray = ospNewData(1, OSP_OBJECT, &light, 0);
                ospSetData(renderer, "lights", lightArray);
            }
            ospCommit(renderer);

            
            // setup framebuffer
            ospFrameBufferClear(framebuffer, OSP_FB_COLOR | OSP_FB_ACCUM);
            ospRenderFrame(framebuffer, renderer, OSP_FB_COLOR | OSP_FB_ACCUM);

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
            cd.clear();

        }
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
    if (this->AOsamples.IsDirty() ||
        this->AOweight.IsDirty() ||
        this->AOdistance.IsDirty() ||
        this->extraSamles.IsDirty() ||
        this->lightIntensity.IsDirty() ||
        this->lightType.IsDirty() ||
        this->lightColor.IsDirty() ||
        this->shadows.IsDirty() ||
        this->dl_angularDiameter.IsDirty() ||
        this->dl_direction.IsDirty() ||
        this->dl_eye_direction.IsDirty() ||
        this->pl_position.IsDirty() ||
        this->pl_radius.IsDirty() ||
        this->sl_position.IsDirty() ||
        this->sl_direction.IsDirty() ||
        this->sl_openingAngle.IsDirty() ||
        this->sl_penumbraAngle.IsDirty() ||
        this->sl_radius.IsDirty() ||
        this->ql_position.IsDirty() ||
        this->ql_edgeOne.IsDirty() ||
        this->ql_edgeTwo.IsDirty() ||
        this->hdri_up.IsDirty() ||
        this->hdri_direction.IsDirty() ||
        this->hdri_evnfile.IsDirty() ||
        this->rd_epsilon.IsDirty() ||
        this->rd_spp.IsDirty() ||
        this->rd_maxRecursion.IsDirty() ||
        this->rd_ptBackground.IsDirty() ||
        this->mat_Kd.IsDirty() ||
        this->mat_Ks.IsDirty() ||
        this->mat_Ns.IsDirty() ||
        this->mat_d.IsDirty() ||
        this->mat_Tf.IsDirty())
    {
        this->AOsamples.ResetDirty();
        this->AOweight.ResetDirty();
        this->AOdistance.ResetDirty();
        this->extraSamles.ResetDirty();
        this->lightIntensity.ResetDirty();
        this->lightType.ResetDirty();
        this->lightColor.ResetDirty();
        this->shadows.ResetDirty();
        this->dl_angularDiameter.ResetDirty();
        this->dl_direction.ResetDirty();
        this->dl_eye_direction.ResetDirty();
        this->pl_position.ResetDirty();
        this->pl_radius.ResetDirty();
        this->sl_position.ResetDirty();
        this->sl_direction.ResetDirty();
        this->sl_openingAngle.ResetDirty();
        this->sl_penumbraAngle.ResetDirty();
        this->sl_radius.ResetDirty();
        this->ql_position.ResetDirty();
        this->ql_edgeOne.ResetDirty();
        this->ql_edgeTwo.ResetDirty();
        this->hdri_up.ResetDirty();
        this->hdri_direction.ResetDirty();
        this->hdri_evnfile.ResetDirty();
        this->rd_epsilon.ResetDirty();
        this->rd_spp.ResetDirty();
        this->rd_maxRecursion.ResetDirty();
        this->rd_ptBackground.ResetDirty();
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
