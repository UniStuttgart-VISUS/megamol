/*
 * GrimRenderer.cpp
 *
 * Copyright (C) 2009 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#include "GrimRenderer.h"

#include "geometry_calls/MultiParticleDataCall.h"
#include "mmstd/light/DistantLight.h"

#include <glm/ext.hpp>

#include "OpenGL_Context.h"

#include "mmcore_gl/utility/ShaderSourceFactory.h"


using namespace megamol::core;
using namespace megamol::moldyn_gl::rendering;


// #define SPEAK_CELL_USAGE 1
#define SPEAK_VRAM_CACHE_USAGE 1
//#define VRAM_UPLOAD_QUOTA 0
#define VRAM_UPLOAD_QUOTA 25
//#define VRAM_UPLOAD_QUOTA 100

//#define SUPSAMP_LOOP 1
//#define SUPSAMP_LOOPCNT 1
//#define SUPSAMP_LOOPCNT 2
//#define SUPSAMP_LOOPCNT 4
//#define SUPSAMP_LOOPCNT 16
//#define SUPSAMP_LOOPCNT 64


/****************************************************************************/
// CellInfo

GrimRenderer::CellInfo::CellInfo(void) {

    glGenOcclusionQueriesNV(1, &this->oQuery);
}


GrimRenderer::CellInfo::~CellInfo(void) {

    glDeleteOcclusionQueriesNV(1, &this->oQuery);
    this->cache.clear();
}

/****************************************************************************/
// GrimRenderer

GrimRenderer::GrimRenderer(void)
        : mmstd_gl::Renderer3DModuleGL()
        , sphereShader()
        , vanillaSphereShader()
        , initDepthShader()
        , initDepthMapShader()
        , depthMipShader()
        , pointShader()
        , initDepthPointShader()
        , vertCntShader()
        , vertCntShade2r()
        , fbo()
        , getDataSlot("getdata", "Connects to the data source")
        , getTFSlot("gettransferfunction", "Connects to the transfer function module")
        , getLightsSlot("lights", "Lights are retrieved over this slot.")
        , useCellCullSlot("useCellCull", "Flag to activate per cell culling")
        , useVertCullSlot("useVertCull", "Flag to activate per vertex culling")
        , speakCellPercSlot("speakCellPerc", "Flag to activate output of percentage of culled cells")
        , speakVertCountSlot("speakVertCount", "Flag to activate output of number of vertices")
        , deferredShadingSlot("deferredShading", "De-/Activates deferred shading with normal generation")
        , greyTF(0)
        , cellDists()
        , cellInfos(0)
        , cacheSize(0)
        , cacheSizeUsed(0)
        , deferredSphereShader()
        , deferredVanillaSphereShader()
        , deferredPointShader()
        , deferredShader()
        , inhash(0) {

    this->getDataSlot.SetCompatibleCall<moldyn::ParticleGridDataCallDescription>();
    this->MakeSlotAvailable(&this->getDataSlot);

    this->getTFSlot.SetCompatibleCall<mmstd_gl::CallGetTransferFunctionGLDescription>();
    this->MakeSlotAvailable(&this->getTFSlot);

    this->getLightsSlot.SetCompatibleCall<core::view::light::CallLightDescription>();
    this->MakeSlotAvailable(&this->getLightsSlot);

    this->useCellCullSlot << new param::BoolParam(true);
    this->MakeSlotAvailable(&this->useCellCullSlot);

    this->useVertCullSlot.SetParameter(new param::BoolParam(true));
    this->MakeSlotAvailable(&this->useVertCullSlot);

    this->speakCellPercSlot << new param::BoolParam(false);
    this->MakeSlotAvailable(&this->speakCellPercSlot);

    this->speakVertCountSlot << new param::BoolParam(false);
    this->MakeSlotAvailable(&this->speakVertCountSlot);

    this->deferredShadingSlot << new param::BoolParam(true);
    this->MakeSlotAvailable(&this->deferredShadingSlot);
    this->deferredShadingSlot.ForceSetDirty();

    this->cacheSize = 6 * 1024 * 1024 * 1024; // TODO: Any way to get this better?
    //this->cacheSize = 256 * 1024 * 1024; // TODO: Any way to get this better?
    //this->cacheSize = 256 * 1024; // TODO: Any way to get this better?
    //this->cacheSize = 1; // TODO: Any way to get this better?
}


GrimRenderer::~GrimRenderer(void) {

    this->Release();
}


bool GrimRenderer::create(void) {

    ASSERT(IsAvailable());

    auto const& ogl_ctx = frontend_resources.get<frontend_resources::OpenGL_Context>();
    if (!ogl_ctx.isExtAvailable("GL_NV_occlusion_query") || !ogl_ctx.isExtAvailable("GL_ARB_multitexture") ||
        !ogl_ctx.isExtAvailable("GL_ARB_vertex_buffer_object") ||
        !ogl_ctx.areExtAvailable(vislib_gl::graphics::gl::GLSLShader::RequiredExtensions()) ||
        !ogl_ctx.areExtAvailable(vislib_gl::graphics::gl::FramebufferObject::RequiredExtensions()))
        return false;

    vislib_gl::graphics::gl::ShaderSource vert, geom, frag;
    auto ssf = std::make_shared<core_gl::utility::ShaderSourceFactory>(instance()->Configuration().ShaderDirectories());

    const char* shaderName = "sphere";
    try {

        shaderName = "sphereShader";
        if (!ssf->MakeShaderSource("mipdepth::theOtherSphereVertex", vert)) {
            return false;
        }
        if (!ssf->MakeShaderSource("mipdepth::simplesphere::fragment", frag)) {
            return false;
        }
        //printf("\nVertex Shader:\n%s\n\nFragment Shader:\n%s\n",
        //    vert.WholeCode().PeekBuffer(),
        //    frag.WholeCode().PeekBuffer());
        if (!this->sphereShader.Create(vert.Code(), vert.Count(), frag.Code(), frag.Count())) {
            megamol::core::utility::log::Log::DefaultLog.WriteError(
                "Unable to compile %s: Unknown error\n", shaderName);
            return false;
        }

        shaderName = "vanillaSphereShader";
        if (!ssf->MakeShaderSource("mipdepth::simplesphere::vertex", vert)) {
            return false;
        }
        if (!ssf->MakeShaderSource("mipdepth::simplesphere::fragment", frag)) {
            return false;
        }
        if (!this->vanillaSphereShader.Create(vert.Code(), vert.Count(), frag.Code(), frag.Count())) {
            megamol::core::utility::log::Log::DefaultLog.WriteError(
                "Unable to compile %s: Unknown error\n", shaderName);
            return false;
        }

        shaderName = "initDepthShader";
        if (!ssf->MakeShaderSource("mipdepth::init::vertex", vert)) {
            return false;
        }
        if (!ssf->MakeShaderSource("mipdepth::init::fragment", frag)) {
            return false;
        }
        if (!this->initDepthShader.Create(vert.Code(), vert.Count(), frag.Code(), frag.Count())) {
            megamol::core::utility::log::Log::DefaultLog.WriteError(
                "Unable to compile %s: Unknown error\n", shaderName);
            return false;
        }

        shaderName = "initDepthMapShader";
        if (!ssf->MakeShaderSource("mipdepth::depthmap::initvert", vert)) {
            return false;
        }
        if (!ssf->MakeShaderSource("mipdepth::depthmap::initfrag", frag)) {
            return false;
        }
        if (!this->initDepthMapShader.Create(vert.Code(), vert.Count(), frag.Code(), frag.Count())) {
            megamol::core::utility::log::Log::DefaultLog.WriteError(
                "Unable to compile %s: Unknown error\n", shaderName);
            return false;
        }

        shaderName = "depthMipShader";
        if (!ssf->MakeShaderSource("mipdepth::depthmap::initvert", vert)) {
            return false;
        }
        if (!ssf->MakeShaderSource("mipdepth::depthmap::mipfrag", frag)) {
            return false;
        }
        if (!this->depthMipShader.Create(vert.Code(), vert.Count(), frag.Code(), frag.Count())) {
            megamol::core::utility::log::Log::DefaultLog.WriteError(
                "Unable to compile %s: Unknown error\n", shaderName);
            return false;
        }

        shaderName = "pointShader";
        if (!ssf->MakeShaderSource("mipdepth::point::vert", vert)) {
            return false;
        }
        if (!ssf->MakeShaderSource("mipdepth::point::frag", frag)) {
            return false;
        }
        if (!this->pointShader.Create(vert.Code(), vert.Count(), frag.Code(), frag.Count())) {
            megamol::core::utility::log::Log::DefaultLog.WriteError(
                "Unable to compile %s: Unknown error\n", shaderName);
            return false;
        }

        shaderName = "initDepthPointShader";
        if (!ssf->MakeShaderSource("mipdepth::point::simplevert", vert)) {
            return false;
        }
        if (!ssf->MakeShaderSource("mipdepth::point::simplefrag", frag)) {
            return false;
        }
        if (!this->initDepthPointShader.Create(vert.Code(), vert.Count(), frag.Code(), frag.Count())) {
            megamol::core::utility::log::Log::DefaultLog.WriteError(
                "Unable to compile %s: Unknown error\n", shaderName);
            return false;
        }

        shaderName = "vertCntShader";
        if (!ssf->MakeShaderSource("mipdepth::point::simplevert", vert)) {
            return false;
        }
        if (!ssf->MakeShaderSource("mipdepth::point::simplefrag", frag)) {
            return false;
        }
        if (!this->vertCntShader.Create(vert.Code(), vert.Count(), frag.Code(), frag.Count())) {
            megamol::core::utility::log::Log::DefaultLog.WriteError(
                "Unable to compile %s: Unknown error\n", shaderName);
            return false;
        }

        shaderName = "vertCntShade2r";
        if (!ssf->MakeShaderSource("mipdepth::point2::lesssimplevert", vert)) {
            return false;
        }
        if (!ssf->MakeShaderSource("mipdepth::point::simplefrag", frag)) {
            return false;
        }
        if (!this->vertCntShade2r.Create(vert.Code(), vert.Count(), frag.Code(), frag.Count())) {
            megamol::core::utility::log::Log::DefaultLog.WriteError(
                "Unable to compile %s: Unknown error\n", shaderName);
            return false;
        }


        shaderName = "deferredSphereShader";
        if (!ssf->MakeShaderSource("mipdepth::deferred::otherSphereVertex", vert)) {
            return false;
        }
        if (!ssf->MakeShaderSource("mipdepth::deferred::spherefragment", frag)) {
            return false;
        }
        if (!this->deferredSphereShader.Create(vert.Code(), vert.Count(), frag.Code(), frag.Count())) {
            megamol::core::utility::log::Log::DefaultLog.WriteError(
                "Unable to compile %s: Unknown error\n", shaderName);
            return false;
        }

        shaderName = "deferredVanillaSphereShader";
        if (!ssf->MakeShaderSource("mipdepth::deferred::spherevertex", vert)) {
            return false;
        }
        if (!ssf->MakeShaderSource("mipdepth::deferred::spherefragment", frag)) {
            return false;
        }
        if (!this->deferredVanillaSphereShader.Create(vert.Code(), vert.Count(), frag.Code(), frag.Count())) {
            megamol::core::utility::log::Log::DefaultLog.WriteError(
                "Unable to compile %s: Unknown error\n", shaderName);
            return false;
        }

        shaderName = "deferredPointShader";
        if (!ssf->MakeShaderSource("mipdepth6::pointvertex", vert)) {
            return false;
        }
        if (!ssf->MakeShaderSource("mipdepth::deferred::pointfragment", frag)) {
            return false;
        }
        if (!this->deferredPointShader.Create(vert.Code(), vert.Count(), frag.Code(), frag.Count())) {
            megamol::core::utility::log::Log::DefaultLog.WriteError(
                "Unable to compile %s: Unknown error\n", shaderName);
            return false;
        }

        shaderName = "deferredShader";
        if (!ssf->MakeShaderSource("mipdepth6::deferredShader::vert", vert)) {
            return false;
        }
        if (!ssf->MakeShaderSource("mipdepth6::deferredShader::frag", frag)) {
            return false;
        }
        if (!this->deferredShader.Create(vert.Code(), vert.Count(), frag.Code(), frag.Count())) {
            megamol::core::utility::log::Log::DefaultLog.WriteError(
                "Unable to compile %s: Unknown error\n", shaderName);
            return false;
        }


    } catch (vislib_gl::graphics::gl::AbstractOpenGLShader::CompileException ce) {
        megamol::core::utility::log::Log::DefaultLog.WriteError("Unable to compile %s shader (@%s): %s\n", shaderName,
            vislib_gl::graphics::gl::AbstractOpenGLShader::CompileException::CompileActionName(ce.FailedAction()),
            ce.GetMsgA());
        return false;
    } catch (vislib::Exception e) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "Unable to compile %s shader: %s\n", shaderName, e.GetMsgA());
        return false;
    } catch (...) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "Unable to compile %s shader: Unknown exception\n", shaderName);
        return false;
    }

    // Fallback transfer function texture
    glGenTextures(1, &this->greyTF);
    unsigned char tex[6] = {0, 0, 0, 255, 255, 255};
    glBindTexture(GL_TEXTURE_1D, this->greyTF);
    glTexImage1D(GL_TEXTURE_1D, 0, GL_RGBA, 2, 0, GL_RGB, GL_UNSIGNED_BYTE, tex);
    glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_WRAP_S, GL_CLAMP);
    glBindTexture(GL_TEXTURE_1D, 0);

    this->fbo.Create(1, 1); // year, right.
    this->depthmap[0].Create(1, 1);
    this->depthmap[1].Create(1, 1);

    return true;
}


bool GrimRenderer::GetExtents(mmstd_gl::CallRender3DGL& call) {

    auto cr = &call;
    if (cr == NULL)
        return false;

    moldyn::ParticleGridDataCall* pgdc = this->getDataSlot.CallAs<moldyn::ParticleGridDataCall>();
    if (pgdc == NULL)
        return false;
    if (!(*pgdc)(1))
        return false;

    cr->SetTimeFramesCount(pgdc->FrameCount());
    cr->AccessBoundingBoxes() = pgdc->AccessBoundingBoxes();

    return true;
}


void GrimRenderer::release(void) {

    this->sphereShader.Release();
    this->initDepthMapShader.Release();
    this->initDepthShader.Release();
    this->depthMipShader.Release();
    this->pointShader.Release();
    this->fbo.Release();
    this->depthmap[0].Release();
    this->depthmap[1].Release();
    glDeleteTextures(1, &this->greyTF);
    this->cellDists.clear();
    this->cellInfos.clear();
    this->deferredSphereShader.Release();
    this->deferredVanillaSphereShader.Release();
    this->deferredPointShader.Release();
    this->deferredShader.Release();
}


void GrimRenderer::set_cam_uniforms(vislib_gl::graphics::gl::GLSLShader& shader, glm::mat4 view_matrix_inv,
    glm::mat4 view_matrix_inv_transp, glm::mat4 mvp_matrix, glm::mat4 mvp_matrix_transp, glm::mat4 mvp_matrix_inv,
    glm::vec4 camPos, glm::vec4 curlightDir) {

    glUniformMatrix4fv(shader.ParameterLocation("mv_inv"), 1, GL_FALSE, glm::value_ptr(view_matrix_inv));
    glUniformMatrix4fv(shader.ParameterLocation("mv_inv_transp"), 1, GL_FALSE, glm::value_ptr(view_matrix_inv_transp));
    glUniformMatrix4fv(shader.ParameterLocation("mvp"), 1, GL_FALSE, glm::value_ptr(mvp_matrix));
    glUniformMatrix4fv(shader.ParameterLocation("mvp_transp"), 1, GL_FALSE, glm::value_ptr(mvp_matrix_transp));
    glUniformMatrix4fv(shader.ParameterLocation("mvp_inv"), 1, GL_FALSE, glm::value_ptr(mvp_matrix_inv));
    glUniform4fv(shader.ParameterLocation("light_dir"), 1, glm::value_ptr(curlightDir));
    glUniform4fv(shader.ParameterLocation("cam_pos"), 1, glm::value_ptr(camPos));
}


bool GrimRenderer::Render(mmstd_gl::CallRender3DGL& call) {

    auto cr = &call;
    if (cr == NULL)
        return false;

    moldyn::ParticleGridDataCall* pgdc = this->getDataSlot.CallAs<moldyn::ParticleGridDataCall>();
    if (pgdc == NULL)
        return false;

    static unsigned int tod = 0;
    unsigned int todi = vislib::sys::GetTicksOfDay();
    bool speak = false;
    if ((todi < tod) || (todi > tod + 1000)) {
        speak = true;
        tod = todi;
    }

    bool useCellCull = this->useCellCullSlot.Param<param::BoolParam>()->Value();
    bool useVertCull = this->useVertCullSlot.Param<param::BoolParam>()->Value();
    bool speakCellPerc = speak /*&& useCellCull*/ && this->speakCellPercSlot.Param<param::BoolParam>()->Value();
    bool speakVertCount = /*speak && */ this->speakVertCountSlot.Param<param::BoolParam>()->Value();
    bool deferredShading = this->deferredShadingSlot.Param<param::BoolParam>()->Value();
    vislib_gl::graphics::gl::GLSLShader* daSphereShader =
        useVertCull ? &this->sphereShader : &this->vanillaSphereShader;
    vislib_gl::graphics::gl::GLSLShader* daPointShader = &this->pointShader;
    if (deferredShading) {
        daSphereShader = useVertCull ? &this->deferredSphereShader : &this->deferredVanillaSphereShader;
        daPointShader = &this->deferredPointShader;
    }
    unsigned int cial = glGetAttribLocationARB(*daSphereShader, "colIdx");
    unsigned int cial2 = glGetAttribLocationARB(*daPointShader, "colIdx");

    // ask for extend to calculate the data scaling
    pgdc->SetFrameID(static_cast<unsigned int>(cr->Time()));
    if (!(*pgdc)(1))
        return false;

    const float scaling = 1.0f;

    // fetch real data
    pgdc->SetFrameID(static_cast<unsigned int>(cr->Time()));
    if (!(*pgdc)(0))
        return false;
    if (this->inhash != pgdc->DataHash()) {
        this->inhash = pgdc->DataHash();
        // invalidate ALL VBOs
        SIZE_T cnt = this->cellInfos.size();
        for (SIZE_T i = 0; i < cnt; i++) {
            SIZE_T cnt2 = this->cellInfos[i].cache.size();
            for (SIZE_T j = 0; j < cnt2; j++) {
                glDeleteBuffersARB(2, this->cellInfos[i].cache[j].data);
                this->cellInfos[i].cache[j].data[0] = 0;
                this->cellInfos[i].cache[j].data[1] = 0;
            }
        }
        this->cacheSizeUsed = 0;
    }

    unsigned int cellcnt = pgdc->CellsCount();
    unsigned int typecnt = pgdc->TypesCount();

    // Camera
    core::view::Camera cam = call.GetCamera();
    auto cam_pose = cam.get<core::view::Camera::Pose>();
    auto cam_intrinsics = cam.get<core::view::Camera::PerspectiveParameters>();
    auto fbo = call.GetFramebuffer();

    glm::mat4 view_matrix = cam.getViewMatrix();
    glm::mat4 proj_matrix = cam.getProjectionMatrix();
    glm::mat4 view_matrix_inv = glm::inverse(view_matrix);
    glm::mat4 view_matrix_inv_transp = glm::transpose(view_matrix_inv);
    glm::mat4 mvp_matrix = proj_matrix * view_matrix;
    glm::mat4 mvp_matrix_transp = glm::transpose(mvp_matrix);
    glm::mat4 mvp_matrix_inv = glm::inverse(mvp_matrix);
    glm::vec4 camPos = glm::vec4(cam_pose.position, 1.0f);
    glm::vec4 camView = glm::vec4(cam_pose.direction, 1.0f);
    glm::vec4 camUp = glm::vec4(cam_pose.up, 1.0f);
    glm::vec4 camRight = glm::vec4(glm::cross(cam_pose.direction, cam_pose.up), 1.0f);
    float half_aperture_angle = cam_intrinsics.fovy / 2.0f;

    // Lights
    glm::vec4 curlightDir = {0.0f, 0.0f, 0.0f, 1.0f};

    auto call_light = getLightsSlot.CallAs<core::view::light::CallLight>();
    if (call_light != nullptr) {
        if (!(*call_light)(0)) {
            return false;
        }

        auto lights = call_light->getData();
        auto distant_lights = lights.get<core::view::light::DistantLightType>();

        if (distant_lights.size() > 1) {
            megamol::core::utility::log::Log::DefaultLog.WriteWarn(
                "[GrimRenderer] Only one single 'Distant Light' source is supported by this renderer");
        } else if (distant_lights.empty()) {
            megamol::core::utility::log::Log::DefaultLog.WriteWarn("[GrimRenderer] No 'Distant Light' found");
        }

        for (auto const& light : distant_lights) {
            auto use_eyedir = light.eye_direction;
            if (use_eyedir) {
                curlightDir = camView;
            } else {
                auto lightDir = light.direction;
                if (lightDir.size() == 3) {
                    curlightDir[0] = lightDir[0];
                    curlightDir[1] = lightDir[1];
                    curlightDir[2] = lightDir[2];
                }
                if (lightDir.size() == 4) {
                    curlightDir[3] = lightDir[3];
                }
                /// View Space Lighting. Comment line to change to Object Space Lighting.
                // this->curlightDir = this->curMVtransp * this->curlightDir;
            }
            /// TODO Implement missing distant light parameters:
            // light.second.dl_angularDiameter;
            // light.second.lightColor;
            // light.second.lightIntensity;
        }
    }

    // update fbo size, if required ///////////////////////////////////////////
    if ((this->fbo.GetWidth() != fbo->getWidth()) || (this->fbo.GetHeight() != fbo->getHeight()) ||
        this->deferredShadingSlot.IsDirty()) {
        this->deferredShadingSlot.ResetDirty();

        glPushDebugGroup(GL_DEBUG_SOURCE_APPLICATION, 1, -1, "grim-fbo-resize");
        this->fbo.Release();
        this->fbo.Create(fbo->getWidth(), fbo->getHeight(), GL_RGBA8, GL_RGBA, GL_UNSIGNED_BYTE, // colour buffer
            vislib_gl::graphics::gl::FramebufferObject::ATTACHMENT_TEXTURE,
            GL_DEPTH_COMPONENT24); // depth buffer

        unsigned int dmw = vislib::math::NextPowerOfTwo(fbo->getWidth());
        unsigned int dmh = vislib::math::NextPowerOfTwo(fbo->getHeight());
        dmh += dmh / 2;
        if ((this->depthmap[0].GetWidth() != dmw) || (this->depthmap[0].GetHeight() != dmh)) {
            for (int i = 0; i < 2; i++) {
                this->depthmap[i].Release();
                this->depthmap[i].Create(dmw, dmh, GL_LUMINANCE32F_ARB, GL_LUMINANCE, GL_FLOAT,
                    vislib_gl::graphics::gl::FramebufferObject::ATTACHMENT_DISABLED);
            }
        }

        this->dsFBO.Release();
        if (deferredShading) {
            // attachments:
            //  colour (RGBA-byte)
            //  normal (RGBA-float16; xyz + confidence)
            //  depth (24 bit)
            //  stencil (none)
            vislib_gl::graphics::gl::FramebufferObject::ColourAttachParams cap[3];
            cap[0].format = GL_RGBA;
            cap[0].internalFormat = GL_RGBA8;
            cap[0].type = GL_UNSIGNED_BYTE;
            cap[1].format = GL_RGBA;
            cap[1].internalFormat = GL_RGBA16F;
            cap[1].type = GL_HALF_FLOAT;
            cap[2].format = GL_RGBA;
            cap[2].internalFormat = GL_RGBA32F;
            cap[2].type = GL_FLOAT;
            vislib_gl::graphics::gl::FramebufferObject::DepthAttachParams dap;
            dap.state = vislib_gl::graphics::gl::FramebufferObject::ATTACHMENT_TEXTURE;
            dap.format = GL_DEPTH_COMPONENT24;
            vislib_gl::graphics::gl::FramebufferObject::StencilAttachParams sap;
            sap.state = vislib_gl::graphics::gl::FramebufferObject::ATTACHMENT_DISABLED;
            sap.format = GL_STENCIL_INDEX;

            try {
                if (!this->dsFBO.Create(fbo->getWidth(), fbo->getHeight(), 3, cap, dap, sap)) {
                    throw vislib::Exception("dsFBO.Create failed\n", __FILE__, __LINE__);
                }
            } catch (vislib::Exception ex) {
                megamol::core::utility::log::Log::DefaultLog.WriteError("Failed to created dsFBO: %s", ex.GetMsgA());
            }
        }
        glPopDebugGroup();
    }

    if (this->cellDists.size() != cellcnt) {
        this->cellDists.resize(cellcnt);
        this->cellInfos.resize(cellcnt);
        for (unsigned int i = 0; i < cellcnt; i++) {
            this->cellDists[i].First() = i;
            this->cellInfos[i].wasvisible = true; // TODO: refine with Reina-Approach (wtf?)
            this->cellInfos[i].maxrad = 0.0f;
            this->cellInfos[i].cache.clear();
            this->cellInfos[i].cache.resize(typecnt);
            for (unsigned int j = 0; j < typecnt; j++) {
                this->cellInfos[i].maxrad = glm::max(
                    this->cellInfos[i].maxrad, pgdc->Cells()[i].AccessParticleLists()[j].GetMaxRadius() * scaling);
            }
        }
        this->cacheSizeUsed = 0;
    }

    // depth-sort of cells ////////////////////////////////////////////////////

    float viewDist = 0.5f * fbo->getHeight() / tanf(half_aperture_angle);

    std::vector<vislib::Pair<unsigned int, float>>& dists = this->cellDists;
    std::vector<CellInfo>& infos = this->cellInfos;
    // -> The usage of these references is required in order to get performance !!! WTF !!!

    for (unsigned int i = 0; i < cellcnt; i++) {
        unsigned int idx = dists[i].First();
        const moldyn::ParticleGridDataCall::GridCell& cell = pgdc->Cells()[idx];
        CellInfo& info = infos[idx];
        const vislib::math::Cuboid<float>& bbox = cell.GetBoundingBox();

        glm::vec3 cellPos((bbox.Left() + bbox.Right()) * 0.5f * scaling, (bbox.Bottom() + bbox.Top()) * 0.5f * scaling,
            (bbox.Back() + bbox.Front()) * 0.5f * scaling);

        glm::vec3 cellDistV = cellPos - glm::vec3(camPos);
        float cellDist = glm::dot(glm::vec3(camView), cellDistV);

        dists[i].Second() = cellDist;

        // calculate view size of the max sphere
        float sphereImgRad = info.maxrad * viewDist / cellDist;
        info.dots = (sphereImgRad < 0.75f);

        info.isvisible = true;
        // Testing against the viewing frustum would be nice, but I don't care
    }
    std::sort(dists.begin(), dists.end(), GrimRenderer::depthSort);

    // init depth points //////////////////////////////////////////////////////
#ifdef _WIN32
#pragma region Depthbuffer initialization
#endif // _WIN32

    glLineWidth(5.0f);
    glDisable(GL_BLEND);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_VERTEX_PROGRAM_POINT_SIZE);

    // upload to gpu-cache
    int vramUploadQuota = VRAM_UPLOAD_QUOTA; // upload no more then X VBO per frame

    glPushDebugGroup(GL_DEBUG_SOURCE_APPLICATION, 2, -1, "grim-init-depth-points");
    // z-buffer-filling
#if defined(DEBUG) || defined(_DEBUG)
    UINT oldlevel = vislib::Trace::GetInstance().GetLevel();
    vislib::Trace::GetInstance().SetLevel(vislib::Trace::LEVEL_NONE);
#endif

    this->fbo.Enable();
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

#ifdef SPEAK_CELL_USAGE
    printf("[initd1");
#endif
    this->initDepthPointShader.Enable();
    set_cam_uniforms(this->initDepthPointShader, view_matrix_inv, view_matrix_inv_transp, mvp_matrix, mvp_matrix_transp,
        mvp_matrix_inv, camPos, curlightDir);
    glPointSize(1.0f);
    for (int i = cellcnt - 1; i >= 0; i--) { // front to back
        unsigned int idx = dists[i].First();
        const moldyn::ParticleGridDataCall::GridCell* cell = &pgdc->Cells()[idx];
        CellInfo& info = infos[idx];
        if (!info.wasvisible)
            continue;
        // only draw cells which were visible last frame
        if (!info.dots)
            continue;

#ifdef SPEAK_CELL_USAGE
        printf("-%d", i);
#endif
        for (unsigned int j = 0; j < typecnt; j++) {
            const moldyn::ParticleGridDataCall::Particles& parts = cell->AccessParticleLists()[j];
            const moldyn::ParticleGridDataCall::ParticleType& ptype = pgdc->Types()[j];
            CellInfo::CacheItem& ci = info.cache[j];
            unsigned int vbpp = 1, cbpp = 1;
            switch (ptype.GetVertexDataType()) {
            case geocalls::MultiParticleDataCall::Particles::VERTDATA_FLOAT_XYZ:
                vbpp = 3 * sizeof(float);
                break;
            case geocalls::MultiParticleDataCall::Particles::VERTDATA_FLOAT_XYZR:
                vbpp = 4 * sizeof(float);
                break;
            case geocalls::MultiParticleDataCall::Particles::VERTDATA_SHORT_XYZ:
                vbpp = 3 * sizeof(short);
                break;
            default:
                continue;
            }
            switch (ptype.GetColourDataType()) {
            case geocalls::MultiParticleDataCall::Particles::COLDATA_UINT8_RGB:
                cbpp = 3;
                break;
            case geocalls::MultiParticleDataCall::Particles::COLDATA_UINT8_RGBA:
                cbpp = 4;
                break;
            case geocalls::MultiParticleDataCall::Particles::COLDATA_FLOAT_RGB:
                cbpp = 3 * sizeof(float);
                break;
            case geocalls::MultiParticleDataCall::Particles::COLDATA_FLOAT_RGBA:
                cbpp = 4 * sizeof(float);
                break;
            case geocalls::MultiParticleDataCall::Particles::COLDATA_FLOAT_I:
                cbpp = sizeof(float);
                break;
            default:
                break;
            }

            if ((ci.data[0] == 0) && (vramUploadQuota > 0) && (parts.GetCount() > 0) &&
                (((vbpp + cbpp) * parts.GetCount()) < (this->cacheSize - this->cacheSizeUsed))) {
                // upload
                glGetError();
                glGenBuffersARB(2, ci.data);
                if (glGetError() != GL_NO_ERROR) {
                    megamol::core::utility::log::Log::DefaultLog.WriteError("glGenBuffersARB failed");
                    throw vislib::Exception("glGenBuffersARB failed", __FILE__, __LINE__);
                }
                vramUploadQuota--;
                glBindBufferARB(GL_ARRAY_BUFFER, ci.data[0]);
                glGetError();
                if (parts.GetVertexDataStride() == 0) {
                    GLenum err;
                    glBufferDataARB(GL_ARRAY_BUFFER, vbpp * parts.GetCount(), parts.GetVertexData(), GL_STATIC_DRAW);
                    if ((err = glGetError()) != GL_NO_ERROR) {
                        megamol::core::utility::log::Log::DefaultLog.WriteError("glBufferDataARB failed: %u", err);
                        throw vislib::Exception("glBufferDataARB failed", __FILE__, __LINE__);
                    }
                } else {
                    megamol::core::utility::log::Log::DefaultLog.WriteError(
                        "Currently only data without stride is supported for caching");
                    throw vislib::Exception(
                        "Currently only data without stride is supported for caching", __FILE__, __LINE__);
                }
                this->cacheSizeUsed += vbpp * parts.GetCount();
                glBindBufferARB(GL_ARRAY_BUFFER, ci.data[1]);
                if (parts.GetColourDataStride() == 0) {
                    GLenum err;
                    glBufferDataARB(GL_ARRAY_BUFFER, cbpp * parts.GetCount(), parts.GetColourData(), GL_STATIC_DRAW);
                    if ((err = glGetError()) != GL_NO_ERROR) {
                        megamol::core::utility::log::Log::DefaultLog.WriteError("glBufferDataARB failed: %u", err);
                        throw vislib::Exception("glBufferDataARB failed", __FILE__, __LINE__);
                    }
                } else {
                    megamol::core::utility::log::Log::DefaultLog.WriteError(
                        "Currently only data without stride is supported for caching");
                    throw vislib::Exception(
                        "Currently only data without stride is supported for caching", __FILE__, __LINE__);
                }
                this->cacheSizeUsed += cbpp * parts.GetCount();
#ifdef SPEAK_VRAM_CACHE_USAGE
                printf("VRAM-Cache: Add[%d; %u] %u/%u\n", i, j, this->cacheSizeUsed, this->cacheSize);
#endif // SPEAK_VRAM_CACHE_USAGE
            }

            // radius and position
            switch (ptype.GetVertexDataType()) {
            case geocalls::MultiParticleDataCall::Particles::VERTDATA_NONE:
                continue;
            case geocalls::MultiParticleDataCall::Particles::VERTDATA_FLOAT_XYZ:
                glEnableClientState(GL_VERTEX_ARRAY);
                if (ci.data[0] != 0) {
                    glBindBufferARB(GL_ARRAY_BUFFER, ci.data[0]);
                    glVertexPointer(3, GL_FLOAT, 0, NULL);
                } else {
                    glVertexPointer(3, GL_FLOAT, parts.GetVertexDataStride(), parts.GetVertexData());
                }
                break;
            case geocalls::MultiParticleDataCall::Particles::VERTDATA_FLOAT_XYZR:
                glEnableClientState(GL_VERTEX_ARRAY);
                if (ci.data[0] != 0) {
                    glBindBufferARB(GL_ARRAY_BUFFER, ci.data[0]);
                    glVertexPointer(3, GL_FLOAT, 16, NULL);
                } else {
                    glVertexPointer(3, GL_FLOAT, glm::max(16U, parts.GetVertexDataStride()), parts.GetVertexData());
                }
                break;
            case geocalls::MultiParticleDataCall::Particles::VERTDATA_SHORT_XYZ: {
                megamol::core::utility::log::Log::DefaultLog.WriteError(
                    "GrimRenderer: vertices with short coords are deprecated!");
            } break;

            default:
                continue;
            }
            glDrawArrays(GL_POINTS, 0, static_cast<GLsizei>(parts.GetCount()));
            glBindBufferARB(GL_ARRAY_BUFFER, 0);
            glDisableClientState(GL_VERTEX_ARRAY);
        }
    }
#ifdef SPEAK_CELL_USAGE
    printf("]\n");
#endif
    this->initDepthPointShader.Disable();
    glPopDebugGroup();

    // init depth disks ///////////////////////////////////////////////////////
    glPushDebugGroup(GL_DEBUG_SOURCE_APPLICATION, 3, -1, "grim-init-depth-disks");

    float viewportStuff[4] = {0.0f, 0.0f, fbo->getWidth(), fbo->getHeight()};
    float defaultPointSize = glm::max(viewportStuff[2], viewportStuff[3]);
    if (viewportStuff[2] < 1.0f)
        viewportStuff[2] = 1.0f;
    if (viewportStuff[3] < 1.0f)
        viewportStuff[3] = 1.0f;
    viewportStuff[2] = 2.0f / viewportStuff[2];
    viewportStuff[3] = 2.0f / viewportStuff[3];

    glPointSize(defaultPointSize);

    this->initDepthShader.Enable();
    set_cam_uniforms(this->initDepthShader, view_matrix_inv, view_matrix_inv_transp, mvp_matrix, mvp_matrix_transp,
        mvp_matrix_inv, camPos, curlightDir);

    glUniform4fv(this->initDepthShader.ParameterLocation("viewAttr"), 1, viewportStuff);
    glUniform3fv(this->initDepthShader.ParameterLocation("camIn"), 1, glm::value_ptr(camView));
    glUniform3fv(this->initDepthShader.ParameterLocation("camRight"), 1, glm::value_ptr(camRight));
    glUniform3fv(this->initDepthShader.ParameterLocation("camUp"), 1, glm::value_ptr(camUp));

    // no clipping plane for now
    glColor4ub(192, 192, 192, 255);
    glDisableClientState(GL_COLOR_ARRAY);

#ifdef SPEAK_CELL_USAGE
    printf("[initd2");
#endif
    for (int i = cellcnt - 1; i >= 0; i--) { // front to back
        unsigned int idx = dists[i].First();
        const moldyn::ParticleGridDataCall::GridCell* cell = &pgdc->Cells()[idx];
        CellInfo& info = infos[idx];
        if (!info.wasvisible)
            continue;
        // only draw cells which were visible last frame
        if (info.dots)
            continue;

        //glColor4ub(192, 192, 192, 255);
        float a = static_cast<float>(i) / static_cast<float>(cellcnt - 1);
        ASSERT((a >= 0.0) && (a <= 1.0f));
        glColor3f(1.0f - a, 0.0f, a);
        if (info.dots) {
            glColor3ub(255, 0, 0);
        }

#ifdef SPEAK_CELL_USAGE
        printf("-%d", i);
#endif
        for (unsigned int j = 0; j < typecnt; j++) {
            const moldyn::ParticleGridDataCall::Particles& parts = cell->AccessParticleLists()[j];
            const moldyn::ParticleGridDataCall::ParticleType& ptype = pgdc->Types()[j];
            CellInfo::CacheItem& ci = info.cache[j];
            unsigned int vbpp = 1, cbpp = 1;
            switch (ptype.GetVertexDataType()) {
            case geocalls::MultiParticleDataCall::Particles::VERTDATA_FLOAT_XYZ:
                vbpp = 3 * sizeof(float);
                break;
            case geocalls::MultiParticleDataCall::Particles::VERTDATA_FLOAT_XYZR:
                vbpp = 4 * sizeof(float);
                break;
            case geocalls::MultiParticleDataCall::Particles::VERTDATA_SHORT_XYZ:
                vbpp = 3 * sizeof(short);
                break;
            default:
                continue;
            }
            switch (ptype.GetColourDataType()) {
            case geocalls::MultiParticleDataCall::Particles::COLDATA_UINT8_RGB:
                cbpp = 3;
                break;
            case geocalls::MultiParticleDataCall::Particles::COLDATA_UINT8_RGBA:
                cbpp = 4;
                break;
            case geocalls::MultiParticleDataCall::Particles::COLDATA_FLOAT_RGB:
                cbpp = 3 * sizeof(float);
                break;
            case geocalls::MultiParticleDataCall::Particles::COLDATA_FLOAT_RGBA:
                cbpp = 4 * sizeof(float);
                break;
            case geocalls::MultiParticleDataCall::Particles::COLDATA_FLOAT_I:
                cbpp = sizeof(float);
                break;
            default:
                break;
            }

            if ((ci.data[0] == 0) && (vramUploadQuota > 0) && (parts.GetCount() > 0) &&
                (((vbpp + cbpp) * parts.GetCount()) < (this->cacheSize - this->cacheSizeUsed))) {
                // upload
                glGetError();
                glGenBuffersARB(2, ci.data);
                if (glGetError() != GL_NO_ERROR) {
                    megamol::core::utility::log::Log::DefaultLog.WriteError("glGenBuffersARB failed");
                    throw vislib::Exception("glGenBuffersARB failed", __FILE__, __LINE__);
                }
                vramUploadQuota--;
                glBindBufferARB(GL_ARRAY_BUFFER, ci.data[0]);
                glGetError();
                if (parts.GetVertexDataStride() == 0) {
                    GLenum err;
                    glBufferDataARB(GL_ARRAY_BUFFER, vbpp * parts.GetCount(), parts.GetVertexData(), GL_STATIC_DRAW);
                    if ((err = glGetError()) != GL_NO_ERROR) {
                        megamol::core::utility::log::Log::DefaultLog.WriteError("glBufferDataARB failed: %u", err);
                        throw vislib::Exception("glBufferDataARB failed", __FILE__, __LINE__);
                    }
                } else {
                    megamol::core::utility::log::Log::DefaultLog.WriteError(
                        "Currently only data without stride is supported for caching");
                    throw vislib::Exception(
                        "Currently only data without stride is supported for caching", __FILE__, __LINE__);
                }
                this->cacheSizeUsed += vbpp * parts.GetCount();
                glBindBufferARB(GL_ARRAY_BUFFER, ci.data[1]);
                if (parts.GetColourDataStride() == 0) {
                    GLenum err;
                    glBufferDataARB(GL_ARRAY_BUFFER, cbpp * parts.GetCount(), parts.GetColourData(), GL_STATIC_DRAW);
                    if ((err = glGetError()) != GL_NO_ERROR) {
                        megamol::core::utility::log::Log::DefaultLog.WriteError("glBufferDataARB failed: %u", err);
                        throw vislib::Exception("glBufferDataARB failed", __FILE__, __LINE__);
                    }
                } else {
                    megamol::core::utility::log::Log::DefaultLog.WriteError(
                        "Currently only data without stride is supported for caching");
                    throw vislib::Exception(
                        "Currently only data without stride is supported for caching", __FILE__, __LINE__);
                }
                this->cacheSizeUsed += cbpp * parts.GetCount();
#ifdef SPEAK_VRAM_CACHE_USAGE
                printf("VRAM-Cache: Add[%d; %u] %u/%u\n", i, j, this->cacheSizeUsed, this->cacheSize);
#endif // SPEAK_VRAM_CACHE_USAGE
            }

            // radius and position
            switch (ptype.GetVertexDataType()) {
            case geocalls::MultiParticleDataCall::Particles::VERTDATA_NONE:
                continue;
            case geocalls::MultiParticleDataCall::Particles::VERTDATA_FLOAT_XYZ:
                glEnableClientState(GL_VERTEX_ARRAY);
                glUniform4f(
                    this->initDepthShader.ParameterLocation("inConsts1"), ptype.GetGlobalRadius(), 0.0f, 0.0f, 0.0f);
                if (ci.data[0] != 0) {
                    glBindBufferARB(GL_ARRAY_BUFFER, ci.data[0]);
                    glVertexPointer(3, GL_FLOAT, 0, NULL);
                } else {
                    glVertexPointer(3, GL_FLOAT, parts.GetVertexDataStride(), parts.GetVertexData());
                }
                break;
            case geocalls::MultiParticleDataCall::Particles::VERTDATA_FLOAT_XYZR:
                glEnableClientState(GL_VERTEX_ARRAY);
                glUniform4f(this->initDepthShader.ParameterLocation("inConsts1"), -1.0f, 0.0f, 0.0f, 0.0f);
                if (ci.data[0] != 0) {
                    glBindBufferARB(GL_ARRAY_BUFFER, ci.data[0]);
                    glVertexPointer(4, GL_FLOAT, 0, NULL);
                } else {
                    glVertexPointer(4, GL_FLOAT, parts.GetVertexDataStride(), parts.GetVertexData());
                }
                break;
            case geocalls::MultiParticleDataCall::Particles::VERTDATA_SHORT_XYZ: {
                megamol::core::utility::log::Log::DefaultLog.WriteError(
                    "GrimRenderer: vertices with short coords are deprecated!");
            } break;

            default:
                continue;
            }
            glDrawArrays(GL_POINTS, 0, static_cast<GLsizei>(parts.GetCount()));
            glBindBufferARB(GL_ARRAY_BUFFER, 0);
            glDisableClientState(GL_VERTEX_ARRAY);
        }

        //glBegin(GL_LINES);
        //glVertex3f(cell->GetBoundingBox().Left(), cell->GetBoundingBox().Bottom(), cell->GetBoundingBox().Back());
        //glVertex3f(cell->GetBoundingBox().Right(), cell->GetBoundingBox().Top(), cell->GetBoundingBox().Front());
        //glVertex3f(cell->GetBoundingBox().Right(), cell->GetBoundingBox().Bottom(), cell->GetBoundingBox().Back());
        //glVertex3f(cell->GetBoundingBox().Left(), cell->GetBoundingBox().Top(), cell->GetBoundingBox().Front());
        //glVertex3f(cell->GetBoundingBox().Left(), cell->GetBoundingBox().Top(), cell->GetBoundingBox().Back());
        //glVertex3f(cell->GetBoundingBox().Right(), cell->GetBoundingBox().Bottom(), cell->GetBoundingBox().Front());
        //glVertex3f(cell->GetBoundingBox().Right(), cell->GetBoundingBox().Top(), cell->GetBoundingBox().Back());
        //glVertex3f(cell->GetBoundingBox().Left(), cell->GetBoundingBox().Bottom(), cell->GetBoundingBox().Front());
        //glEnd();
    }
#ifdef SPEAK_CELL_USAGE
    printf("]\n");
#endif

    this->initDepthShader.Disable();
    glPopDebugGroup();

#ifdef _WIN32
#pragma endregion Depthbuffer initialization
#endif // _WIN32

    // issue queries //////////////////////////////////////////////////////
#ifdef _WIN32
#pragma region issue occlusion queries for all cells to find hidden ones
#endif // _WIN32

    if (useCellCull) {
        glPushDebugGroup(GL_DEBUG_SOURCE_APPLICATION, 4, -1, "grim-issue-queries");

        // occlusion queries ftw
        glColorMask(GL_FALSE, GL_FALSE, GL_FALSE, GL_FALSE);
        glDepthMask(GL_FALSE);
        glDisable(GL_CULL_FACE);

        // this shader is so simple it should also work for the boxes.
        this->initDepthPointShader.Enable();
        set_cam_uniforms(this->initDepthPointShader, view_matrix_inv, view_matrix_inv_transp, mvp_matrix,
            mvp_matrix_transp, mvp_matrix_inv, camPos, curlightDir);

        // also disable texturing and any fancy shading features
        for (int i = cellcnt - 1; i >= 0; i--) { // front to back
            const moldyn::ParticleGridDataCall::GridCell& cell = pgdc->Cells()[i];
            CellInfo& info = infos[i];
            const vislib::math::Cuboid<float>& bbox = cell.GetBoundingBox();
            if (!info.isvisible)
                continue; // frustum culling

            glBeginOcclusionQueryNV(info.oQuery);

            // render bounding box for cell idx
            glBegin(GL_QUADS);

            glVertex3f(bbox.Left(), bbox.Bottom(), bbox.Back());
            glVertex3f(bbox.Right(), bbox.Bottom(), bbox.Back());
            glVertex3f(bbox.Right(), bbox.Top(), bbox.Back());
            glVertex3f(bbox.Left(), bbox.Top(), bbox.Back());

            glVertex3f(bbox.Left(), bbox.Bottom(), bbox.Front());
            glVertex3f(bbox.Left(), bbox.Top(), bbox.Front());
            glVertex3f(bbox.Right(), bbox.Top(), bbox.Front());
            glVertex3f(bbox.Right(), bbox.Bottom(), bbox.Front());

            glVertex3f(bbox.Left(), bbox.Bottom(), bbox.Back());
            glVertex3f(bbox.Right(), bbox.Bottom(), bbox.Back());
            glVertex3f(bbox.Right(), bbox.Bottom(), bbox.Front());
            glVertex3f(bbox.Left(), bbox.Bottom(), bbox.Front());

            glVertex3f(bbox.Left(), bbox.Top(), bbox.Front());
            glVertex3f(bbox.Left(), bbox.Top(), bbox.Back());
            glVertex3f(bbox.Right(), bbox.Top(), bbox.Back());
            glVertex3f(bbox.Right(), bbox.Top(), bbox.Front());

            glVertex3f(bbox.Left(), bbox.Bottom(), bbox.Back());
            glVertex3f(bbox.Left(), bbox.Top(), bbox.Back());
            glVertex3f(bbox.Left(), bbox.Top(), bbox.Front());
            glVertex3f(bbox.Left(), bbox.Bottom(), bbox.Front());

            glVertex3f(bbox.Right(), bbox.Bottom(), bbox.Front());
            glVertex3f(bbox.Right(), bbox.Bottom(), bbox.Back());
            glVertex3f(bbox.Right(), bbox.Top(), bbox.Back());
            glVertex3f(bbox.Right(), bbox.Top(), bbox.Front());

            glEnd();

            glEndOcclusionQueryNV();
        }

        glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
        glDepthMask(GL_TRUE);
        // reenable other state
        glPopDebugGroup();
    }
#ifdef _WIN32
#pragma endregion issue occlusion queries
#endif // _WIN32

    this->fbo.Disable();
    // END Depth buffer initialized

    glEnable(GL_CULL_FACE);

    // depth mipmap ///////////////////////////////////////////////////////////
#ifdef _WIN32
#pragma region depth buffer mipmaping
#endif // _WIN32

    int maxLevel = 0;
    if (useVertCull) {
        glPushDebugGroup(GL_DEBUG_SOURCE_APPLICATION, 5, -1, "grim-depth-mipmap");
        // create depth mipmap
        this->depthmap[0].Enable();

        //glClearColor(0.5f, 0.5f, 0.5f, 0.5f);
        glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        glEnable(GL_TEXTURE_2D);
        glDisable(GL_LIGHTING);
        glDisable(GL_DEPTH_TEST);
        glActiveTextureARB(GL_TEXTURE0_ARB);
        this->fbo.BindDepthTexture();

        glMatrixMode(GL_PROJECTION);
        glPushMatrix();
        glLoadIdentity();
        glMatrixMode(GL_MODELVIEW);
        glPushMatrix();
        glLoadIdentity();

        this->initDepthMapShader.Enable();
        set_cam_uniforms(this->initDepthMapShader, view_matrix_inv, view_matrix_inv_transp, mvp_matrix,
            mvp_matrix_transp, mvp_matrix_inv, camPos, curlightDir);
        this->initDepthMapShader.SetParameter("datex", 0);

        glBegin(GL_QUADS);
        float xf = float(this->fbo.GetWidth()) / float(this->depthmap[0].GetWidth());
        float yf = float(this->fbo.GetHeight()) / float(this->depthmap[0].GetHeight());
        glVertex2f(-1.0f, -1.0f);
        glVertex2f(-1.0f + 2.0f * xf, -1.0f);
        glVertex2f(-1.0f + 2.0f * xf, -1.0f + 2.0f * yf);
        glVertex2f(-1.0f, -1.0f + 2.0f * yf);
        glEnd();

        this->initDepthMapShader.Disable();

        int lw = this->depthmap[0].GetWidth() / 2;
        int ly = this->depthmap[0].GetHeight() * 2 / 3;
        int lh = ly / 2;
        int ls = vislib::math::Min(lh, lw);

        this->depthMipShader.Enable();
        set_cam_uniforms(this->depthMipShader, view_matrix_inv, view_matrix_inv_transp, mvp_matrix, mvp_matrix_transp,
            mvp_matrix_inv, camPos, curlightDir);
        this->depthMipShader.SetParameter("datex", 0);
        this->depthMipShader.SetParameter("src", 0, 0);
        this->depthMipShader.SetParameter("dst", 0, ly);

        maxLevel = 1; // we created one! hui!
        glBegin(GL_QUADS);
        glVertex2f(-1.0f + 2.0f * 0.0f, -1.0f + 2.0f * float(ly) / float(this->depthmap[0].GetHeight()));
        glVertex2f(-1.0f + 2.0f * float(this->fbo.GetWidth() / 2) / float(this->depthmap[0].GetWidth()),
            -1.0f + 2.0f * float(ly) / float(this->depthmap[0].GetHeight()));
        glVertex2f(-1.0f + 2.0f * float(this->fbo.GetWidth() / 2) / float(this->depthmap[0].GetWidth()),
            -1.0f + 2.0f * float(ly + this->fbo.GetHeight() / 2) / float(this->depthmap[0].GetHeight()));
        glVertex2f(-1.0f + 2.0f * 0.0f,
            -1.0f + 2.0f * float(ly + this->fbo.GetHeight() / 2) / float(this->depthmap[0].GetHeight()));
        glEnd();

        this->depthmap[0].Disable();

        int lx = lw;
        while (ls > 1) {
            this->depthmap[maxLevel % 2].Enable();
            this->depthmap[1 - (maxLevel % 2)].BindColourTexture();

            this->depthMipShader.SetParameter("src", lx - lw, ly);
            this->depthMipShader.SetParameter("dst", lx, ly);

            lw /= 2;
            lh /= 2;
            ls /= 2;

            float x1, x2, y1, y2;

            x1 = float(lx) / float(this->depthmap[0].GetWidth());
            x2 = float(lx + lw) / float(this->depthmap[0].GetWidth());
            y1 = float(ly) / float(this->depthmap[0].GetHeight());
            y2 = float(ly + lh) / float(this->depthmap[0].GetHeight());

            glBegin(GL_QUADS);
            glVertex2f(-1.0f + 2.0f * x1, -1.0f + 2.0f * y1);
            glVertex2f(-1.0f + 2.0f * x2, -1.0f + 2.0f * y1);
            glVertex2f(-1.0f + 2.0f * x2, -1.0f + 2.0f * y2);
            glVertex2f(-1.0f + 2.0f * x1, -1.0f + 2.0f * y2);
            glEnd();

            this->depthmap[maxLevel % 2].Disable();
            glBindTexture(GL_TEXTURE_2D, 0);

            lx += lw;
            maxLevel++;
        }

        this->depthMipShader.Disable();

        this->depthmap[0].Enable();
        this->initDepthMapShader.Enable();
        set_cam_uniforms(this->initDepthMapShader, view_matrix_inv, view_matrix_inv_transp, mvp_matrix,
            mvp_matrix_transp, mvp_matrix_inv, camPos, curlightDir);
        this->initDepthMapShader.SetParameter("datex", 0);
        this->depthmap[1].BindColourTexture();

        lw = this->depthmap[0].GetWidth() / 2;
        ly = this->depthmap[0].GetHeight() * 2 / 3;
        lh = ly / 2;
        ls = vislib::math::Min(lh, lw);
        lx = lw;
        while (ls > 1) {

            lw /= 2;
            lh /= 2;
            ls /= 2;

            float x1, x2, y1, y2;

            x1 = float(lx) / float(this->depthmap[0].GetWidth());
            x2 = float(lx + lw) / float(this->depthmap[0].GetWidth());
            y1 = float(ly) / float(this->depthmap[0].GetHeight());
            y2 = float(ly + lh) / float(this->depthmap[0].GetHeight());

            glBegin(GL_QUADS);
            glVertex2f(-1.0f + 2.0f * x1, -1.0f + 2.0f * y1);
            glVertex2f(-1.0f + 2.0f * x2, -1.0f + 2.0f * y1);
            glVertex2f(-1.0f + 2.0f * x2, -1.0f + 2.0f * y2);
            glVertex2f(-1.0f + 2.0f * x1, -1.0f + 2.0f * y2);
            glEnd();

            lx += lw;

            // and skip one
            lw /= 2;
            lh /= 2;
            ls /= 2;
            lx += lw;
        }

        this->initDepthMapShader.Disable();
        this->depthmap[0].Disable();

        glMatrixMode(GL_PROJECTION);
        glPopMatrix();
        glMatrixMode(GL_MODELVIEW);
        glPopMatrix();

        glBindTexture(GL_TEXTURE_2D, 0);
        glPopDebugGroup();
        // END generation of depth-max mipmap
    }
#ifdef _WIN32
#pragma endregion depth buffer mipmaping
#endif // _WIN32

#if defined(DEBUG) || defined(_DEBUG)
    vislib::Trace::GetInstance().SetLevel(oldlevel);
#endif

    // image output ///////////////////////////////////////////////////////////
    unsigned int visCnt = 0;
    unsigned int visPart = 0;

    if (speakVertCount) {
        //
        // outputs the number of vertices surviving the vertex culling stage
        // usually not done when just drawing pictures
        //
        // THIS WILL NOT GENERATE ANY VISIBLE IMAGE OUTPUT !!!
        //
#ifdef _WIN32
#pragma region speakVertCount
#endif // _WIN32

        glPushDebugGroup(GL_DEBUG_SOURCE_APPLICATION, 5, -1, "grim-count-visible-points");
        GLuint allQuery;
        glGenOcclusionQueriesNV(1, &allQuery);
        glBeginOcclusionQueryNV(allQuery);

        glDisable(GL_DEPTH_TEST);

        glPointSize(1.0f);
        if (useVertCull) {
            this->vertCntShade2r.Enable();
            set_cam_uniforms(this->vertCntShade2r, view_matrix_inv, view_matrix_inv_transp, mvp_matrix,
                mvp_matrix_transp, mvp_matrix_inv, camPos, curlightDir);

            glUniform4fv(this->vertCntShade2r.ParameterLocation("viewAttr"), 1, viewportStuff);
            glUniform3fv(this->vertCntShade2r.ParameterLocation("camIn"), 1, glm::value_ptr(camView));
            glUniform3fv(this->vertCntShade2r.ParameterLocation("camRight"), 1, glm::value_ptr(camRight));
            glUniform3fv(this->vertCntShade2r.ParameterLocation("camUp"), 1, glm::value_ptr(camUp));
            this->vertCntShade2r.SetParameter(
                "depthTexParams", this->depthmap[0].GetWidth(), this->depthmap[0].GetHeight() * 2 / 3, maxLevel);

            glEnable(GL_TEXTURE_2D);
            glActiveTextureARB(GL_TEXTURE2_ARB);
            this->depthmap[0].BindColourTexture();
            this->vertCntShade2r.SetParameter("depthTex", 2);
            glActiveTextureARB(GL_TEXTURE0_ARB);

            glColor3ub(128, 128, 128);
            glDisableClientState(GL_COLOR_ARRAY);
        } else {
            this->vertCntShader.Enable();
            set_cam_uniforms(this->vertCntShader, view_matrix_inv, view_matrix_inv_transp, mvp_matrix,
                mvp_matrix_transp, mvp_matrix_inv, camPos, curlightDir);
        }

#ifdef SPEAK_CELL_USAGE
        printf("[vertCnt");
#endif
        for (int i = 0; i < static_cast<int>(cellcnt); i++) { // front to back
            const moldyn::ParticleGridDataCall::GridCell& cell = pgdc->Cells()[i];
            CellInfo& info = infos[i];
            unsigned int pixelCount;
            if (!info.isvisible)
                continue; // frustum culling

            if (useCellCull) {
                glGetOcclusionQueryuivNV(info.oQuery, GL_PIXEL_COUNT_NV, &pixelCount);
                info.isvisible = (pixelCount > 0);
                //printf("PixelCount of cell %u is %u\n", idx, pixelCount);
                if (!info.isvisible)
                    continue; // occlusion culling
            } else {
                info.isvisible = true;
            }
            visCnt++;

#ifdef SPEAK_CELL_USAGE
            printf("-%d", i);
#endif
            for (unsigned int j = 0; j < typecnt; j++) {
                const moldyn::ParticleGridDataCall::Particles& parts = cell.AccessParticleLists()[j];
                const moldyn::ParticleGridDataCall::ParticleType& ptype = pgdc->Types()[j];
                CellInfo::CacheItem& ci = info.cache[j];
                float minC = 0.0f, maxC = 0.0f;
                unsigned int colTabSize = 0;
                visPart += parts.GetCount();

                // radius and position
                switch (ptype.GetVertexDataType()) {
                case geocalls::MultiParticleDataCall::Particles::VERTDATA_NONE:
                    continue;
                case geocalls::MultiParticleDataCall::Particles::VERTDATA_FLOAT_XYZ:
                    glEnableClientState(GL_VERTEX_ARRAY);
                    if (useVertCull) {
                        glUniform4f(this->vertCntShade2r.ParameterLocation("inConsts1"), ptype.GetGlobalRadius(), minC,
                            maxC, float(colTabSize));
                    }
                    if (ci.data[0] != 0) {
                        glBindBufferARB(GL_ARRAY_BUFFER, ci.data[0]);
                        glVertexPointer(3, GL_FLOAT, 0, NULL);
                    } else {
                        glVertexPointer(3, GL_FLOAT, parts.GetVertexDataStride(), parts.GetVertexData());
                    }
                    break;
                case geocalls::MultiParticleDataCall::Particles::VERTDATA_FLOAT_XYZR:
                    glEnableClientState(GL_VERTEX_ARRAY);
                    if (useVertCull) {
                        glUniform4f(
                            this->vertCntShade2r.ParameterLocation("inConsts1"), -1.0f, minC, maxC, float(colTabSize));
                    }
                    if (ci.data[0] != 0) {
                        glBindBufferARB(GL_ARRAY_BUFFER, ci.data[0]);
                        glVertexPointer(4, GL_FLOAT, 0, NULL);
                    } else {
                        glVertexPointer(4, GL_FLOAT, parts.GetVertexDataStride(), parts.GetVertexData());
                    }
                    break;
                case geocalls::MultiParticleDataCall::Particles::VERTDATA_SHORT_XYZ: {
                    megamol::core::utility::log::Log::DefaultLog.WriteError(
                        "GrimRenderer: vertices with short coords are deprecated!");
                } break;

                default:
                    continue;
                }

                glDrawArrays(GL_POINTS, 0, static_cast<GLsizei>(parts.GetCount()));
                glBindBufferARB(GL_ARRAY_BUFFER, 0);
                //glDisableClientState(GL_COLOR_ARRAY);
                glDisableClientState(GL_VERTEX_ARRAY);
            }
        }
        (useVertCull ? this->vertCntShade2r : this->vertCntShader).Disable();
#ifdef SPEAK_CELL_USAGE
        printf("]\n");
#endif

        unsigned int totalSchnitzels = 0;
        glEndOcclusionQueryNV();
        glFlush();
        glGetOcclusionQueryuivNV(allQuery, GL_PIXEL_COUNT_NV, &totalSchnitzels);
        glDeleteOcclusionQueriesNV(1, &allQuery);

        if (speak && speakVertCount) {
            unsigned int totalSpheres = 0;
            for (int i = 0; i < static_cast<int>(cellcnt); i++) {
                const moldyn::ParticleGridDataCall::GridCell& cell = pgdc->Cells()[i];
                for (unsigned int j = 0; j < typecnt; j++) {
                    const moldyn::ParticleGridDataCall::Particles& parts = cell.AccessParticleLists()[j];
                    totalSpheres += parts.GetCount();
                }
            }
            printf("VERTEX COUNT: %u (%f%%)\n", static_cast<unsigned int>(totalSchnitzels),
                static_cast<float>(totalSchnitzels) / static_cast<float>(totalSpheres) * 100.0f);
        }
        glPopDebugGroup();
#ifdef _WIN32
#pragma endregion speakVertCount
#endif // _WIN32

    } else {
        //
        // GENERATE VISIBLE IMAGE OUTPUT
        //
        if (deferredShading) {
#if defined(DEBUG) || defined(_DEBUG)
            UINT oldlevel = vislib::Trace::GetInstance().GetLevel();
            vislib::Trace::GetInstance().SetLevel(vislib::Trace::LEVEL_NONE);
#endif
            this->dsFBO.EnableMultiple(3, GL_COLOR_ATTACHMENT0_EXT, GL_COLOR_ATTACHMENT1_EXT, GL_COLOR_ATTACHMENT2_EXT);
            glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
            glClearDepth(1.0f);
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); // not sure about this one
#if defined(DEBUG) || defined(_DEBUG)
            vislib::Trace::GetInstance().SetLevel(oldlevel);
#endif
        } else {

            // REACTIVATE TARGET FBO
            cr->GetFramebuffer()->bind();
        }

        glEnable(GL_DEPTH_TEST);
        glPointSize(1.0f);
        glDisableClientState(GL_COLOR_ARRAY);

        // draw points ///////////////////////////////////////////////////////
#ifdef SPEAK_CELL_USAGE
        printf("[drawd");
#endif
        glPushDebugGroup(GL_DEBUG_SOURCE_APPLICATION, 6, -1, "grim-draw-dots");
        // draw visible data (dots)
        daPointShader->Enable();
        set_cam_uniforms(*daPointShader, view_matrix_inv, view_matrix_inv_transp, mvp_matrix, mvp_matrix_transp,
            mvp_matrix_inv, camPos, curlightDir);
        for (int i = cellcnt - 1; i >= 0; i--) { // front to back
            const moldyn::ParticleGridDataCall::GridCell& cell = pgdc->Cells()[i];
            CellInfo& info = infos[i];
            unsigned int pixelCount;
            if (!info.isvisible)
                continue; // frustum culling
            if (!info.dots)
                continue;

            if (useCellCull) {
                glGetOcclusionQueryuivNV(info.oQuery, GL_PIXEL_COUNT_NV, &pixelCount);
                info.isvisible = (pixelCount > 0);
                //printf("PixelCount of cell %u is %u\n", idx, pixelCount);
                if (!info.isvisible)
                    continue; // occlusion culling
            } else {
                info.isvisible = true;
            }
            visCnt++;

#ifdef SPEAK_CELL_USAGE
            printf("-%d", i);
#endif
            for (unsigned int j = 0; j < typecnt; j++) {
                const moldyn::ParticleGridDataCall::Particles& parts = cell.AccessParticleLists()[j];
                const moldyn::ParticleGridDataCall::ParticleType& ptype = pgdc->Types()[j];
                CellInfo::CacheItem& ci = info.cache[j];
                float minC = 0.0f, maxC = 0.0f;
                unsigned int colTabSize = 0;
                visPart += parts.GetCount();

                // colour
                switch (ptype.GetColourDataType()) {
                case geocalls::MultiParticleDataCall::Particles::COLDATA_NONE:
                    glColor3ubv(ptype.GetGlobalColour());
                    break;
                case geocalls::MultiParticleDataCall::Particles::COLDATA_UINT8_RGB:
                    glEnableClientState(GL_COLOR_ARRAY);
                    if (ci.data[0] != 0) {
                        glBindBufferARB(GL_ARRAY_BUFFER, ci.data[1]);
                        glColorPointer(3, GL_UNSIGNED_BYTE, 0, NULL);
                    } else {
                        glColorPointer(3, GL_UNSIGNED_BYTE, parts.GetColourDataStride(), parts.GetColourData());
                    }
                    break;
                case geocalls::MultiParticleDataCall::Particles::COLDATA_UINT8_RGBA:
                    glEnableClientState(GL_COLOR_ARRAY);
                    if (ci.data[0] != 0) {
                        glBindBufferARB(GL_ARRAY_BUFFER, ci.data[1]);
                        glColorPointer(4, GL_UNSIGNED_BYTE, 0, NULL);
                    } else {
                        glColorPointer(4, GL_UNSIGNED_BYTE, parts.GetColourDataStride(), parts.GetColourData());
                    }
                    break;
                case geocalls::MultiParticleDataCall::Particles::COLDATA_FLOAT_RGB:
                    glEnableClientState(GL_COLOR_ARRAY);
                    if (ci.data[0] != 0) {
                        glBindBufferARB(GL_ARRAY_BUFFER, ci.data[1]);
                        glColorPointer(3, GL_FLOAT, 0, NULL);
                    } else {
                        glColorPointer(3, GL_FLOAT, parts.GetColourDataStride(), parts.GetColourData());
                    }
                    break;
                case geocalls::MultiParticleDataCall::Particles::COLDATA_FLOAT_RGBA:
                    glEnableClientState(GL_COLOR_ARRAY);
                    if (ci.data[0] != 0) {
                        glBindBufferARB(GL_ARRAY_BUFFER, ci.data[1]);
                        glColorPointer(4, GL_FLOAT, 0, NULL);
                    } else {
                        glColorPointer(4, GL_FLOAT, parts.GetColourDataStride(), parts.GetColourData());
                    }
                    break;
                case geocalls::MultiParticleDataCall::Particles::COLDATA_FLOAT_I: {
                    glEnableVertexAttribArrayARB(cial2);
                    if (ci.data[0] != 0) {
                        glBindBufferARB(GL_ARRAY_BUFFER, ci.data[1]);
                        glVertexAttribPointerARB(cial2, 1, GL_FLOAT, GL_FALSE, 0, NULL);
                    } else {
                        glVertexAttribPointerARB(
                            cial2, 1, GL_FLOAT, GL_FALSE, parts.GetColourDataStride(), parts.GetColourData());
                    }

                    // Bind transfer function texture
                    glEnable(GL_TEXTURE_1D);
                    mmstd_gl::CallGetTransferFunctionGL* cgtf =
                        this->getTFSlot.CallAs<mmstd_gl::CallGetTransferFunctionGL>();
                    if ((cgtf != NULL) && ((*cgtf)())) {
                        glBindTexture(GL_TEXTURE_1D, cgtf->OpenGLTexture());
                        colTabSize = cgtf->TextureSize();
                    } else {
                        glBindTexture(GL_TEXTURE_1D, this->greyTF);
                        colTabSize = 2;
                    }

                    glUniform1i(daPointShader->ParameterLocation("colTab"), 0);
                    minC = ptype.GetMinColourIndexValue();
                    maxC = ptype.GetMaxColourIndexValue();
                    glColor3ub(127, 127, 127);
                } break;
                default:
                    glColor3ub(127, 127, 127);
                    break;
                }

                // radius and position
                switch (ptype.GetVertexDataType()) {
                case geocalls::MultiParticleDataCall::Particles::VERTDATA_NONE:
                    continue;
                case geocalls::MultiParticleDataCall::Particles::VERTDATA_FLOAT_XYZ:
                    glEnableClientState(GL_VERTEX_ARRAY);
                    glUniform4f(daPointShader->ParameterLocation("inConsts1"), ptype.GetGlobalRadius(), minC, maxC,
                        float(colTabSize));
                    if (ci.data[0] != 0) {
                        glBindBufferARB(GL_ARRAY_BUFFER, ci.data[0]);
                        glVertexPointer(3, GL_FLOAT, 0, NULL);
                    } else {
                        glVertexPointer(3, GL_FLOAT, parts.GetVertexDataStride(), parts.GetVertexData());
                    }
                    break;
                case geocalls::MultiParticleDataCall::Particles::VERTDATA_FLOAT_XYZR:
                    glEnableClientState(GL_VERTEX_ARRAY);
                    glUniform4f(daPointShader->ParameterLocation("inConsts1"), -1.0f, minC, maxC, float(colTabSize));
                    if (ci.data[0] != 0) {
                        glBindBufferARB(GL_ARRAY_BUFFER, ci.data[0]);
                        glVertexPointer(3, GL_FLOAT, 16, NULL);
                    } else {
                        glVertexPointer(
                            3, GL_FLOAT, vislib::math::Max(16U, parts.GetVertexDataStride()), parts.GetVertexData());
                    }
                    break;
                case geocalls::MultiParticleDataCall::Particles::VERTDATA_SHORT_XYZ: {
                    megamol::core::utility::log::Log::DefaultLog.WriteError(
                        "GrimRenderer: vertices with short coords are deprecated!");
                } break;

                default:
                    continue;
                }
                glDrawArrays(GL_POINTS, 0, static_cast<GLsizei>(parts.GetCount()));
                glBindBufferARB(GL_ARRAY_BUFFER, 0);
                glDisableClientState(GL_COLOR_ARRAY);
                glDisableClientState(GL_VERTEX_ARRAY);
                glDisableVertexAttribArrayARB(cial2);
                glDisable(GL_TEXTURE_1D);
            }
        }
#ifdef SPEAK_CELL_USAGE
        printf("]\n");
#endif
        daPointShader->Disable();
        glPopDebugGroup();

        // draw spheres ///////////////////////////////////////////////////////
#ifdef SPEAK_CELL_USAGE
        printf("[draws");
#endif
        // draw visible data (spheres)
        glPushDebugGroup(GL_DEBUG_SOURCE_APPLICATION, 7, -1, "grim-draw-spheres");
        daSphereShader->Enable();
        set_cam_uniforms(*daSphereShader, view_matrix_inv, view_matrix_inv_transp, mvp_matrix, mvp_matrix_transp,
            mvp_matrix_inv, camPos, curlightDir);
#ifdef SUPSAMP_LOOP
        for (int supsamppass = 0; supsamppass < SUPSAMP_LOOPCNT; supsamppass++) {
#endif // SUPSAMP_LOOP

            glUniform4fv(daSphereShader->ParameterLocation("viewAttr"), 1, viewportStuff);
            glUniform3fv(daSphereShader->ParameterLocation("camIn"), 1, glm::value_ptr(camView));
            glUniform3fv(daSphereShader->ParameterLocation("camRight"), 1, glm::value_ptr(camRight));
            glUniform3fv(daSphereShader->ParameterLocation("camUp"), 1, glm::value_ptr(camUp));
            glUniform1i(daSphereShader->ParameterLocation("use_shading"), static_cast<int>(!deferredShading));

            if (useVertCull) {
                daSphereShader->SetParameter(
                    "depthTexParams", this->depthmap[0].GetWidth(), this->depthmap[0].GetHeight() * 2 / 3, maxLevel);
                glEnable(GL_TEXTURE_2D);
                glActiveTextureARB(GL_TEXTURE2_ARB);
                this->depthmap[0].BindColourTexture();
                daSphereShader->SetParameter("depthTex", 2);
                glActiveTextureARB(GL_TEXTURE0_ARB);
            } else {
                daSphereShader->SetParameter("clipDat", 0.0f, 0.0f, 0.0f, 0.0f);
                daSphereShader->SetParameter("clipCol", 0.0f, 0.0f, 0.0f);
            }
            glPointSize(defaultPointSize);

            for (int i = cellcnt - 1; i >= 0; i--) { // front to back
                unsigned int idx = dists[i].First();
                const moldyn::ParticleGridDataCall::GridCell& cell = pgdc->Cells()[idx];
                CellInfo& info = infos[idx];

                unsigned int pixelCount;
                if (!info.isvisible)
                    continue; // frustum culling
                if (info.dots)
                    continue;

                if (useCellCull) {
                    glGetOcclusionQueryuivNV(info.oQuery, GL_PIXEL_COUNT_NV, &pixelCount);
                    info.isvisible = (pixelCount > 0);
                    //printf("PixelCount of cell %u is %u\n", idx, pixelCount);
                    if (!info.isvisible)
                        continue; // occlusion culling
                } else {
                    info.isvisible = true;
                }
                visCnt++;

#ifdef SPEAK_CELL_USAGE
                printf("-%d", i);
#endif
                for (unsigned int j = 0; j < typecnt; j++) {
                    const moldyn::ParticleGridDataCall::Particles& parts = cell.AccessParticleLists()[j];
                    const moldyn::ParticleGridDataCall::ParticleType& ptype = pgdc->Types()[j];
                    CellInfo::CacheItem& ci = info.cache[j];
                    float minC = 0.0f, maxC = 0.0f;
                    unsigned int colTabSize = 0;
                    visPart += parts.GetCount();

                    // colour
                    switch (ptype.GetColourDataType()) {
                    case geocalls::MultiParticleDataCall::Particles::COLDATA_NONE:
                        glColor3ubv(ptype.GetGlobalColour());
                        break;
                    case geocalls::MultiParticleDataCall::Particles::COLDATA_UINT8_RGB:
                        glEnableClientState(GL_COLOR_ARRAY);
                        if (ci.data[0] != 0) {
                            glBindBufferARB(GL_ARRAY_BUFFER, ci.data[1]);
                            glColorPointer(3, GL_UNSIGNED_BYTE, 0, NULL);
                        } else {
                            glColorPointer(3, GL_UNSIGNED_BYTE, parts.GetColourDataStride(), parts.GetColourData());
                        }
                        break;
                    case geocalls::MultiParticleDataCall::Particles::COLDATA_UINT8_RGBA:
                        glEnableClientState(GL_COLOR_ARRAY);
                        if (ci.data[0] != 0) {
                            glBindBufferARB(GL_ARRAY_BUFFER, ci.data[1]);
                            glColorPointer(4, GL_UNSIGNED_BYTE, 0, NULL);
                        } else {
                            glColorPointer(4, GL_UNSIGNED_BYTE, parts.GetColourDataStride(), parts.GetColourData());
                        }
                        break;
                    case geocalls::MultiParticleDataCall::Particles::COLDATA_FLOAT_RGB:
                        glEnableClientState(GL_COLOR_ARRAY);
                        if (ci.data[0] != 0) {
                            glBindBufferARB(GL_ARRAY_BUFFER, ci.data[1]);
                            glColorPointer(3, GL_FLOAT, 0, NULL);
                        } else {
                            glColorPointer(3, GL_FLOAT, parts.GetColourDataStride(), parts.GetColourData());
                        }
                        break;
                    case geocalls::MultiParticleDataCall::Particles::COLDATA_FLOAT_RGBA:
                        glEnableClientState(GL_COLOR_ARRAY);
                        if (ci.data[0] != 0) {
                            glBindBufferARB(GL_ARRAY_BUFFER, ci.data[1]);
                            glColorPointer(4, GL_FLOAT, 0, NULL);
                        } else {
                            glColorPointer(4, GL_FLOAT, parts.GetColourDataStride(), parts.GetColourData());
                        }
                        break;
                    case geocalls::MultiParticleDataCall::Particles::COLDATA_FLOAT_I: {
                        glEnableVertexAttribArrayARB(cial);
                        if (ci.data[0] != 0) {
                            glBindBufferARB(GL_ARRAY_BUFFER, ci.data[1]);
                            glVertexAttribPointerARB(cial, 1, GL_FLOAT, GL_FALSE, 0, NULL);
                        } else {
                            glVertexAttribPointerARB(
                                cial, 1, GL_FLOAT, GL_FALSE, parts.GetColourDataStride(), parts.GetColourData());
                        }

                        // Bind transfer function texture
                        glEnable(GL_TEXTURE_1D);
                        mmstd_gl::CallGetTransferFunctionGL* cgtf =
                            this->getTFSlot.CallAs<mmstd_gl::CallGetTransferFunctionGL>();
                        if ((cgtf != NULL) && ((*cgtf)())) {
                            glBindTexture(GL_TEXTURE_1D, cgtf->OpenGLTexture());
                            colTabSize = cgtf->TextureSize();
                        } else {
                            glBindTexture(GL_TEXTURE_1D, this->greyTF);
                            colTabSize = 2;
                        }

                        glUniform1i(daSphereShader->ParameterLocation("colTab"), 0);
                        minC = ptype.GetMinColourIndexValue();
                        maxC = ptype.GetMaxColourIndexValue();
                        glColor3ub(127, 127, 127);
                    } break;
                    default:
                        glColor3ub(127, 127, 127);
                        break;
                    }

                    // radius and position
                    switch (ptype.GetVertexDataType()) {
                    case geocalls::MultiParticleDataCall::Particles::VERTDATA_NONE:
                        continue;
                    case geocalls::MultiParticleDataCall::Particles::VERTDATA_FLOAT_XYZ:
                        glEnableClientState(GL_VERTEX_ARRAY);
                        glUniform4f(daSphereShader->ParameterLocation("inConsts1"), ptype.GetGlobalRadius(), minC, maxC,
                            float(colTabSize));
                        if (ci.data[0] != 0) {
                            glBindBufferARB(GL_ARRAY_BUFFER, ci.data[0]);
                            glVertexPointer(3, GL_FLOAT, 0, NULL);
                        } else {
                            glVertexPointer(3, GL_FLOAT, parts.GetVertexDataStride(), parts.GetVertexData());
                        }
                        break;
                    case geocalls::MultiParticleDataCall::Particles::VERTDATA_FLOAT_XYZR:
                        glEnableClientState(GL_VERTEX_ARRAY);
                        glUniform4f(
                            daSphereShader->ParameterLocation("inConsts1"), -1.0f, minC, maxC, float(colTabSize));
                        if (ci.data[0] != 0) {
                            glBindBufferARB(GL_ARRAY_BUFFER, ci.data[0]);
                            glVertexPointer(4, GL_FLOAT, 0, NULL);
                        } else {
                            glVertexPointer(4, GL_FLOAT, parts.GetVertexDataStride(), parts.GetVertexData());
                        }
                        break;
                    case geocalls::MultiParticleDataCall::Particles::VERTDATA_SHORT_XYZ: {
                        megamol::core::utility::log::Log::DefaultLog.WriteError(
                            "GrimRenderer: vertices with short coords are deprecated!");
                    } break;

                    default:
                        continue;
                    }

                    glDrawArrays(GL_POINTS, 0, static_cast<GLsizei>(parts.GetCount()));
                    glBindBufferARB(GL_ARRAY_BUFFER, 0);
                    glDisableClientState(GL_COLOR_ARRAY);
                    glDisableClientState(GL_VERTEX_ARRAY);
                    glDisableVertexAttribArrayARB(cial);
                    glDisable(GL_TEXTURE_1D);
                }
            }
#ifdef SPEAK_CELL_USAGE
            printf("]\n");
#endif

#ifdef SUPSAMP_LOOP
        }
#endif // SUPSAMP_LOOP

        if (deferredShading) {
            this->dsFBO.Disable();
        }

        daSphereShader->Disable();
        glDisable(GL_VERTEX_PROGRAM_POINT_SIZE);
        glDisable(GL_TEXTURE_2D);

        glPopDebugGroup();
    }

    if (speakCellPerc) {
        printf("CELLS VISIBLE: %f%%\n", float(visCnt * 100) / float(cellcnt));
        printf("PARTICLES IN VISIBLE CELLS: %u\n", visPart);
    }

    // remove unused cache item ///////////////////////////////////////////////
    if ((this->cacheSizeUsed * 5 / 4) > this->cacheSize) {
        for (int i = cellcnt - 1; i >= 0; i--) { // front to back
            unsigned int idx = dists[i].First();
            const moldyn::ParticleGridDataCall::GridCell& cell = pgdc->Cells()[idx];
            CellInfo& info = infos[idx];

            if (info.wasvisible)
                continue; // this one is still required

            for (unsigned int j = 0; j < typecnt; j++) {
                const moldyn::ParticleGridDataCall::Particles& parts = cell.AccessParticleLists()[j];
                const moldyn::ParticleGridDataCall::ParticleType& ptype = pgdc->Types()[j];
                CellInfo::CacheItem& ci = info.cache[j];

                if ((ci.data[0] == 0) || (parts.GetCount() == 0))
                    continue; // not cached or no data

                unsigned int vbpp = 1, cbpp = 1;
                switch (ptype.GetVertexDataType()) {
                case geocalls::MultiParticleDataCall::Particles::VERTDATA_FLOAT_XYZ:
                    vbpp = 3 * sizeof(float);
                    break;
                case geocalls::MultiParticleDataCall::Particles::VERTDATA_FLOAT_XYZR:
                    vbpp = 4 * sizeof(float);
                    break;
                case geocalls::MultiParticleDataCall::Particles::VERTDATA_SHORT_XYZ:
                    vbpp = 3 * sizeof(short);
                    break;
                default:
                    continue;
                }
                switch (ptype.GetColourDataType()) {
                case geocalls::MultiParticleDataCall::Particles::COLDATA_UINT8_RGB:
                    cbpp = 3;
                    break;
                case geocalls::MultiParticleDataCall::Particles::COLDATA_UINT8_RGBA:
                    cbpp = 4;
                    break;
                case geocalls::MultiParticleDataCall::Particles::COLDATA_FLOAT_RGB:
                    cbpp = 3 * sizeof(float);
                    break;
                case geocalls::MultiParticleDataCall::Particles::COLDATA_FLOAT_RGBA:
                    cbpp = 4 * sizeof(float);
                    break;
                case geocalls::MultiParticleDataCall::Particles::COLDATA_FLOAT_I:
                    cbpp = sizeof(float);
                    break;
                default:
                    break;
                }

                glDeleteBuffersARB(2, ci.data);
                ci.data[0] = ci.data[1] = 0;

                this->cacheSizeUsed -= (vbpp + cbpp) * parts.GetCount();
#ifdef SPEAK_VRAM_CACHE_USAGE
                printf("VRAM-Cache: Del[%d; %u] %u/%u\n", i, j, this->cacheSizeUsed, this->cacheSize);
#endif // SPEAK_VRAM_CACHE_USAGE
            }
        }
    }

    // deferred shading ///////////////////////////////////////////////////////
    if (deferredShading) {
        glPushDebugGroup(GL_DEBUG_SOURCE_APPLICATION, 8, -1, "grim-deferred-shading");
        cr->GetFramebuffer()->bind();

        glEnable(GL_TEXTURE_2D);
        glDisable(GL_LIGHTING);
        glDisable(GL_DEPTH_TEST);
        glDisable(GL_CULL_FACE);

        this->deferredShader.Enable();
        // useless, everything is identity here
        //set_cam_uniforms(this->deferredShader, view_matrix_inv, view_matrix_inv_transp, mvp_matrix,
        //    mvp_matrix_transp, mvp_matrix_inv, camPos, curlightDir);

        glActiveTextureARB(GL_TEXTURE0_ARB);
        this->dsFBO.BindColourTexture(0);
        glActiveTextureARB(GL_TEXTURE1_ARB);
        this->dsFBO.BindColourTexture(1);
        glActiveTextureARB(GL_TEXTURE2_ARB);
        this->dsFBO.BindColourTexture(2);

        this->deferredShader.SetParameter("colour", 0);
        this->deferredShader.SetParameter("normal", 1);
        this->deferredShader.SetParameter("pos", 2);


        vislib::math::Vector<float, 3> ray(glm::value_ptr(camView));
        vislib::math::Vector<float, 3> up(glm::value_ptr(camUp));
        vislib::math::Vector<float, 3> right(glm::value_ptr(camRight));

        //vislib::math::Vector<float, 4> lightDir;
        //vislib::math::ShallowVector<float, 3> lp(lightDir.PeekComponents());
        //lp = right;
        //lp *= -0.5f;
        //lp -= ray;
        //lp += up;
        //lightDir[3] = 0.0f;
        //this->deferredShader.SetParameterArray3("lightDir", 1, lightDir.PeekComponents());
        this->deferredShader.SetParameterArray4("lightDir", 1, glm::value_ptr(curlightDir));

        up *= sinf(half_aperture_angle);
        right *= sinf(half_aperture_angle) * static_cast<float>(fbo->getWidth()) / static_cast<float>(fbo->getHeight());

        this->deferredShader.SetParameterArray3("ray", 1, glm::value_ptr(camView));

        glMatrixMode(GL_PROJECTION);
        glPushMatrix();
        glLoadIdentity();
        glMatrixMode(GL_MODELVIEW);
        glPushMatrix();
        glLoadIdentity();
        glColor3ub(255, 255, 255);
        glBegin(GL_QUADS);
        glNormal3fv((ray - right - up).PeekComponents());
        glTexCoord2f(0.0f, 0.0f);
        glVertex2i(-1, -1);
        glNormal3fv((ray + right - up).PeekComponents());
        glTexCoord2f(1.0f, 0.0f);
        glVertex2i(1, -1);
        glNormal3fv((ray + right + up).PeekComponents());
        glTexCoord2f(1.0f, 1.0f);
        glVertex2i(1, 1);
        glNormal3fv((ray - right + up).PeekComponents());
        glTexCoord2f(0.0f, 1.0f);
        glVertex2i(-1, 1);
        glEnd();

        glMatrixMode(GL_PROJECTION);
        glPopMatrix();
        glMatrixMode(GL_MODELVIEW);
        glPopMatrix();

        glActiveTextureARB(GL_TEXTURE0_ARB);
        glBindTexture(GL_TEXTURE_2D, 0);
        glActiveTextureARB(GL_TEXTURE1_ARB);
        glBindTexture(GL_TEXTURE_2D, 0);
        glActiveTextureARB(GL_TEXTURE2_ARB);
        glBindTexture(GL_TEXTURE_2D, 0);
        glActiveTextureARB(GL_TEXTURE0_ARB);

        this->deferredShader.Disable();
        glPopDebugGroup();
    }

    //// DEBUG OUTPUT OF FBO --------------------------------------------------
    //cr->EnableOutputBuffer();
    //glEnable(GL_TEXTURE_2D);
    //glDisable(GL_LIGHTING);
    //glDisable(GL_DEPTH_TEST);

    ////this->fbo.BindDepthTexture();
    ////this->fbo.BindColourTexture();
    ////this->depthmap[0].BindColourTexture();

    ////this->dsFBO.BindColourTexture(0);
    ////this->dsFBO.BindColourTexture(1);
    //this->dsFBO.BindColourTexture(2);
    ////this->dsFBO.BindDepthTexture();

    //glMatrixMode(GL_PROJECTION);
    //glPushMatrix();
    //glLoadIdentity();
    //glMatrixMode(GL_MODELVIEW);
    //glPushMatrix();
    //glLoadIdentity();
    //glColor3ub(255, 255, 255);
    //glBegin(GL_QUADS);
    //glTexCoord2f(0.0f, 0.0f);
    //glVertex2i(-1, -1);
    //glTexCoord2f(1.0f, 0.0f);
    //glVertex2i(1, -1);
    //glTexCoord2f(1.0f, 1.0f);
    //glVertex2i(1, 1);
    //glTexCoord2f(0.0f, 1.0f);
    //glVertex2i(-1, 1);
    //glEnd();
    //glMatrixMode(GL_PROJECTION);
    //glPopMatrix();
    //glMatrixMode(GL_MODELVIEW);
    //glPopMatrix();
    //glBindTexture(GL_TEXTURE_2D, 0);

    // done!
    pgdc->Unlock();

    for (int i = cellcnt - 1; i >= 0; i--) {
        CellInfo& info = infos[i];
        info.wasvisible = info.isvisible;
    }

    // Reset default OpenGL state ---------------------------------------------
    glDisable(GL_VERTEX_PROGRAM_POINT_SIZE);
    glDisable(GL_CLIP_DISTANCE0);
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LESS);
    glDisable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glDisable(GL_POINT_SPRITE);
    glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
    glDepthMask(GL_TRUE);
    glDisable(GL_CULL_FACE);
    glDisable(GL_TEXTURE_2D);
    glDisable(GL_TEXTURE_1D);
    glDisable(GL_LIGHTING);
    glPointSize(1.0f);
    glLineWidth(1.0f);

    return true;
}


bool GrimRenderer::depthSort(
    const vislib::Pair<unsigned int, float>& lhs, const vislib::Pair<unsigned int, float>& rhs) {

    return (rhs.Second() < lhs.Second());

    //float d = rhs.Second() - lhs.Second();
    //if (d > vislib::math::FLOAT_EPSILON) return ;
    //if (d < -vislib::math::FLOAT_EPSILON) return -1;
    //return 0;
}
