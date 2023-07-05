/**
 * MegaMol
 * Copyright (c) 2021, MegaMol Dev Team
 * All rights reserved.
 */

#include "ModernTrisoupRenderer.h"

#include "compositing_gl/CompositingCalls.h"
#include "geometry_calls_gl/CallTriMeshDataGL.h"
#include "mmcore/param/BoolParam.h"
#include "mmcore/param/ColorParam.h"
#include "mmcore/param/EnumParam.h"
#include "mmcore/param/FloatParam.h"
#include "mmcore_gl/utility/ShaderFactory.h"
#include "mmstd/light/DistantLight.h"
#include "mmstd/light/PointLight.h"

using namespace megamol;
using namespace megamol::trisoup_gl;

ModernTrisoupRenderer::ModernTrisoupRenderer()
        : getDataSlot_("getData", "Connects the renderer to a data provider to retrieve data")
        , getLightsSlot_("getLights", "")
        , getFramebufferSlot_("getFramebuffer", "Connects the renderer to an external framebuffer object")
        , lightingParam_("lighting::enableLighting", "Flag to enable the lighting of the mesh")
        , frontStyleParam_("frontstyle", "Rendering style for the front surface")
        , backStyleParam_("backstyle", "Rendering style for the back surface")
        , windingRuleParam_("windingrule", "Triangle edge winding rule")
        , colorParam_("color", "The color of the mesh that is used if no color is provided in the data")
        , ambientColorParam_("lighting::ambientColor", "Ambient color of the used lights")
        , diffuseColorParam_("lighting::diffuseColor", "Diffuse color of the used lights")
        , specularColorParam_("lighting::specularColor", "Specular color of the used lights")
        , ambientFactorParam_("lighting::ambientFactor", "Factor for the ambient lighting of Blinn-Phong")
        , diffuseFactorParam_("lighting::diffuseFactor", "Factor for the diffuse lighting of Blinn-Phong")
        , specularFactorParam_("lighting::specularFactor", "Factor for the specular lighting of Blinn-Phong")
        , specularExponentParam_("lighting::specularExponent", "Exponent for the specular lighting of Blinn-Phong")
        , useLambertParam_(
              "lighting::lambertShading", "If turned on, the local lighting uses lambert instead of Blinn-Phong.")
        , meshShader_(nullptr)
        , positionBuffer_(nullptr)
        , colorBuffer_(nullptr)
        , normalBuffer_(nullptr)
        , texcoordBuffer_(nullptr)
        , indexBuffer_(nullptr)
        , pointLightBuffer_(nullptr)
        , directionalLightBuffer_(nullptr)
        , vertexArray_(0) {

    getDataSlot_.SetCompatibleCall<megamol::geocalls_gl::CallTriMeshDataGLDescription>();
    getDataSlot_.SetNecessity(megamol::core::AbstractCallSlotPresentation::Necessity::SLOT_REQUIRED);
    this->MakeSlotAvailable(&getDataSlot_);

    getLightsSlot_.SetCompatibleCall<megamol::core::view::light::CallLightDescription>();
    getLightsSlot_.SetNecessity(megamol::core::AbstractCallSlotPresentation::Necessity::SLOT_REQUIRED);
    this->MakeSlotAvailable(&getLightsSlot_);

    getFramebufferSlot_.SetCompatibleCall<megamol::compositing_gl::CallFramebufferGLDescription>();
    this->MakeSlotAvailable(&getFramebufferSlot_);

    core::param::EnumParam* ep = new core::param::EnumParam(static_cast<int>(RenderingMode::FILLED));
    ep->SetTypePair(static_cast<int>(RenderingMode::FILLED), "Filled");
    ep->SetTypePair(static_cast<int>(RenderingMode::WIREFRAME), "Wireframe");
    ep->SetTypePair(static_cast<int>(RenderingMode::POINTS), "Points");
    ep->SetTypePair(static_cast<int>(RenderingMode::NONE), "None");
    frontStyleParam_ << ep;
    this->MakeSlotAvailable(&frontStyleParam_);

    ep = new core::param::EnumParam(static_cast<int>(RenderingMode::NONE));
    ep->SetTypePair(static_cast<int>(RenderingMode::FILLED), "Filled");
    ep->SetTypePair(static_cast<int>(RenderingMode::WIREFRAME), "Wireframe");
    ep->SetTypePair(static_cast<int>(RenderingMode::POINTS), "Points");
    ep->SetTypePair(static_cast<int>(RenderingMode::NONE), "None");
    backStyleParam_ << ep;
    this->MakeSlotAvailable(&backStyleParam_);

    ep = new core::param::EnumParam(static_cast<int>(WindingRule::COUNTER_CLOCK_WISE));
    ep->SetTypePair(static_cast<int>(WindingRule::COUNTER_CLOCK_WISE), "Counter-Clock Wise");
    ep->SetTypePair(static_cast<int>(WindingRule::CLOCK_WISE), "Clock Wise");
    windingRuleParam_ << ep;
    this->MakeSlotAvailable(&windingRuleParam_);

    colorParam_.SetParameter(new core::param::ColorParam("#ffffff"));
    this->MakeSlotAvailable(&colorParam_);

    ambientColorParam_.SetParameter(new core::param::ColorParam("#ffffff"));
    this->MakeSlotAvailable(&ambientColorParam_);

    diffuseColorParam_.SetParameter(new core::param::ColorParam("#ffffff"));
    this->MakeSlotAvailable(&diffuseColorParam_);

    specularColorParam_.SetParameter(new core::param::ColorParam("#ffffff"));
    this->MakeSlotAvailable(&specularColorParam_);

    ambientFactorParam_.SetParameter(new core::param::FloatParam(0.2f, 0.0f, 1.0f));
    this->MakeSlotAvailable(&ambientFactorParam_);

    diffuseFactorParam_.SetParameter(new core::param::FloatParam(0.798f, 0.0f, 1.0f));
    this->MakeSlotAvailable(&diffuseFactorParam_);

    specularFactorParam_.SetParameter(new core::param::FloatParam(0.02f, 0.0f, 1.0f));
    this->MakeSlotAvailable(&specularFactorParam_);

    specularExponentParam_.SetParameter(new core::param::FloatParam(120.0f, 1.0f, 1000.0f));
    this->MakeSlotAvailable(&specularExponentParam_);

    useLambertParam_.SetParameter(new core::param::BoolParam(false));
    this->MakeSlotAvailable(&useLambertParam_);

    lightingParam_.SetParameter(new core::param::BoolParam(true));
    this->MakeSlotAvailable(&lightingParam_);
}

ModernTrisoupRenderer::~ModernTrisoupRenderer() {
    this->Release();
}

bool ModernTrisoupRenderer::create() {

    try {
        auto const shdr_options = core::utility::make_path_shader_options(
            frontend_resources.get<megamol::frontend_resources::RuntimeConfig>());
        meshShader_ = core::utility::make_shared_glowl_shader("mesh", shdr_options,
            std::filesystem::path("trisoup_gl/trisoup.vert.glsl"),
            std::filesystem::path("trisoup_gl/trisoup.frag.glsl"));
    } catch (glowl::GLSLProgramException const& ex) {
        megamol::core::utility::log::Log::DefaultLog.WriteError("[ModernTrisoupRenderer] %s", ex.what());
    } catch (std::exception const& ex) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "[ModernTrisoupRenderer] Unable to compile shader: Unknown exception: %s", ex.what());
    } catch (...) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "[ModernTrisoupRenderer] Unable to compile shader: Unknown exception.");
    }

    positionBuffer_ = std::make_unique<glowl::BufferObject>(GL_ARRAY_BUFFER, nullptr, 0, GL_DYNAMIC_DRAW);
    colorBuffer_ = std::make_unique<glowl::BufferObject>(GL_ARRAY_BUFFER, nullptr, 0, GL_DYNAMIC_DRAW);
    normalBuffer_ = std::make_unique<glowl::BufferObject>(GL_ARRAY_BUFFER, nullptr, 0, GL_DYNAMIC_DRAW);
    texcoordBuffer_ = std::make_unique<glowl::BufferObject>(GL_ARRAY_BUFFER, nullptr, 0, GL_DYNAMIC_DRAW);
    indexBuffer_ = std::make_unique<glowl::BufferObject>(GL_ELEMENT_ARRAY_BUFFER, nullptr, 0, GL_DYNAMIC_DRAW);
    pointLightBuffer_ = std::make_unique<glowl::BufferObject>(GL_SHADER_STORAGE_BUFFER, nullptr, 0, GL_DYNAMIC_DRAW);
    directionalLightBuffer_ =
        std::make_unique<glowl::BufferObject>(GL_SHADER_STORAGE_BUFFER, nullptr, 0, GL_DYNAMIC_DRAW);

    glGenVertexArrays(1, &vertexArray_);
    glBindVertexArray(vertexArray_);

    positionBuffer_->bind();
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, nullptr);

    colorBuffer_->bind();
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, nullptr);

    normalBuffer_->bind();
    glEnableVertexAttribArray(2);
    glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 0, nullptr);

    texcoordBuffer_->bind();
    glEnableVertexAttribArray(3);
    glVertexAttribPointer(3, 2, GL_FLOAT, GL_FALSE, 0, nullptr);

    indexBuffer_->bind();

    glBindVertexArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glDisableVertexAttribArray(0);
    glDisableVertexAttribArray(1);
    glDisableVertexAttribArray(2);
    glDisableVertexAttribArray(3);

    return true;
}

void ModernTrisoupRenderer::release() {
    if (vertexArray_ != 0) {
        glDeleteVertexArrays(1, &vertexArray_);
        vertexArray_ = 0;
    }
}

bool ModernTrisoupRenderer::GetExtents(mmstd_gl::CallRender3DGL& call) {

    auto ctmd = this->getDataSlot_.CallAs<geocalls_gl::CallTriMeshDataGL>();
    if (ctmd == nullptr) {
        return false;
    }
    ctmd->SetFrameID(static_cast<int>(call.Time()));
    if (!(*ctmd)(1)) {
        return false;
    }

    call.SetTimeFramesCount(ctmd->FrameCount());
    call.AccessBoundingBoxes().Clear();
    call.AccessBoundingBoxes() = ctmd->AccessBoundingBoxes();

    return true;
}

bool ModernTrisoupRenderer::Render(mmstd_gl::CallRender3DGL& call) {
    auto call_fbo = call.GetFramebuffer();
    auto cam = call.GetCamera();

    auto view_mat = cam.getViewMatrix();
    auto proj_mat = cam.getProjectionMatrix();
    auto mvp_mat = proj_mat * view_mat;

    bool has_external_fbo = false;
    auto cfbo = getFramebufferSlot_.CallAs<compositing_gl::CallFramebufferGL>();
    auto fbo = call.GetFramebuffer();
    if (cfbo != nullptr) {
        cfbo->operator()(compositing_gl::CallFramebufferGL::CallGetMetaData);
        cfbo->operator()(compositing_gl::CallFramebufferGL::CallGetData);
        if (cfbo->getData() != nullptr) {
            fbo = cfbo->getData();
            has_external_fbo = true;
        }
    }

    fbo->bind();

    auto lc = this->getLightsSlot_.CallAs<core::view::light::CallLight>();
    this->updateLights(lc, cam.getPose().direction);

    megamol::geocalls_gl::CallTriMeshDataGL* ctmd =
        this->getDataSlot_.CallAs<megamol::geocalls_gl::CallTriMeshDataGL>();
    if (ctmd == nullptr) {
        return false;
    }

    ctmd->SetFrameID(static_cast<int>(call.Time()));
    if (!(*ctmd)(1)) {
        return false;
    }

    ctmd->SetFrameID(static_cast<int>(call.Time()));
    if (!(*ctmd)(0)) {
        return false;
    }

    GLint frontStyle = GL_NONE;
    GLint backStyle = GL_NONE;
    GLint culling = GL_NONE;
    GLint frontFace = GL_NONE;

    switch (static_cast<RenderingMode>(this->frontStyleParam_.Param<core::param::EnumParam>()->Value())) {
    case RenderingMode::FILLED:
        frontStyle = GL_FILL;
        break;
    case RenderingMode::WIREFRAME:
        frontStyle = GL_LINE;
        break;
    case RenderingMode::POINTS:
        frontStyle = GL_POINT;
        break;
    case RenderingMode::NONE:
        frontStyle = GL_FILL;
        culling = GL_FRONT;
        break;
    default:
        break;
    }

    switch (static_cast<RenderingMode>(this->backStyleParam_.Param<core::param::EnumParam>()->Value())) {
    case RenderingMode::FILLED:
        backStyle = GL_FILL;
        break;
    case RenderingMode::WIREFRAME:
        backStyle = GL_LINE;
        break;
    case RenderingMode::POINTS:
        backStyle = GL_POINT;
        break;
    case RenderingMode::NONE:
        backStyle = GL_FILL;
        culling = (culling == 0) ? GL_BACK : GL_FRONT_AND_BACK;
        break;
    default:
        break;
    }

    glPolygonMode(GL_FRONT, frontStyle);
    glPolygonMode(GL_BACK, backStyle);
    if (culling == GL_NONE) {
        glDisable(GL_CULL_FACE);
    } else {
        glEnable(GL_CULL_FACE);
        glCullFace(culling);
    }

    glGetIntegerv(GL_FRONT_FACE, &frontFace);
    if (this->windingRuleParam_.Param<core::param::EnumParam>()->Value() ==
        static_cast<int>(WindingRule::COUNTER_CLOCK_WISE)) {
        glFrontFace(GL_CCW);
    } else {
        glFrontFace(GL_CW);
    }

    std::vector<float> positions;
    std::vector<float> normals;
    std::vector<float> colors;
    std::vector<float> texCoords;
    std::vector<uint32_t> vertIds;

    for (uint32_t i = 0; i < ctmd->Count(); ++i) {
        const auto& obj = ctmd->Objects()[i];
        auto vertCount = obj.GetVertexCount();
        auto triCount = obj.GetTriCount();
        positions.resize(vertCount * 3);
        normals.resize(vertCount * 3);
        colors.resize(vertCount * 3);
        texCoords.resize(vertCount * 2);
        vertIds.resize(triCount * 3);

        bool has_normals = false;
        bool has_colors = false;
        bool has_texCoords = false;
        bool has_vertIds = false;

        switch (obj.GetVertexDataType()) {
        case geocalls_gl::CallTriMeshDataGL::Mesh::DT_FLOAT:
            std::memcpy(positions.data(), obj.GetVertexPointerFloat(), sizeof(float) * positions.size());
            break;
        case geocalls_gl::CallTriMeshDataGL::Mesh::DT_DOUBLE:
            std::transform(obj.GetVertexPointerDouble(), obj.GetVertexPointerDouble() + positions.size(),
                positions.data(), [](auto a) { return static_cast<float>(a); });
            break;
        default:
            continue;
        }

        if (obj.HasNormalPointer()) {
            has_normals = true;
            switch (obj.GetNormalDataType()) {
            case geocalls_gl::CallTriMeshDataGL::Mesh::DT_FLOAT:
                std::memcpy(normals.data(), obj.GetNormalPointerFloat(), sizeof(float) * normals.size());
                break;
            case geocalls_gl::CallTriMeshDataGL::Mesh::DT_DOUBLE:
                std::transform(obj.GetNormalPointerDouble(), obj.GetNormalPointerDouble() + normals.size(),
                    normals.data(), [](auto a) { return static_cast<float>(a); });
                break;
            default:
                continue;
            }
        }

        if (obj.HasColourPointer()) {
            has_colors = true;
            switch (obj.GetColourDataType()) {
            case geocalls_gl::CallTriMeshDataGL::Mesh::DT_BYTE:
                std::transform(obj.GetColourPointerByte(), obj.GetColourPointerByte() + colors.size(), colors.data(),
                    [](auto a) { return static_cast<float>(a) / 255.0f; });
                break;
            case geocalls_gl::CallTriMeshDataGL::Mesh::DT_FLOAT:
                std::memcpy(colors.data(), obj.GetColourPointerFloat(), sizeof(float) * colors.size());
                break;
            case geocalls_gl::CallTriMeshDataGL::Mesh::DT_DOUBLE:
                std::transform(obj.GetColourPointerDouble(), obj.GetColourPointerDouble() + colors.size(),
                    colors.data(), [](auto a) { return static_cast<float>(a); });
                break;
            default:
                continue;
            }
        }

        if (obj.HasTextureCoordinatePointer()) {
            has_texCoords = true;
            switch (obj.GetTextureCoordinateDataType()) {
            case geocalls_gl::CallTriMeshDataGL::Mesh::DT_FLOAT:
                std::memcpy(texCoords.data(), obj.GetTextureCoordinatePointerFloat(), sizeof(float) * texCoords.size());
                break;
            case geocalls_gl::CallTriMeshDataGL::Mesh::DT_DOUBLE:
                std::transform(obj.GetTextureCoordinatePointerDouble(),
                    obj.GetTextureCoordinatePointerDouble() + texCoords.size(), texCoords.data(),
                    [](auto a) { return static_cast<float>(a); });
                break;
            default:
                continue;
            }
        }

        if (obj.HasTriIndexPointer()) {
            has_vertIds = true;
            switch (obj.GetTriDataType()) {
            case geocalls_gl::CallTriMeshDataGL::Mesh::DT_BYTE:
                std::transform(obj.GetTriIndexPointerByte(), obj.GetTriIndexPointerByte() + vertIds.size(),
                    vertIds.data(), [](auto a) { return static_cast<uint32_t>(a); });
                break;
            case geocalls_gl::CallTriMeshDataGL::Mesh::DT_UINT16:
                std::transform(obj.GetTriIndexPointerUInt16(), obj.GetTriIndexPointerUInt16() + vertIds.size(),
                    vertIds.data(), [](auto a) { return static_cast<uint32_t>(a); });
                break;
            case geocalls_gl::CallTriMeshDataGL::Mesh::DT_UINT32:
                std::memcpy(vertIds.data(), obj.GetTriIndexPointerUInt32(), sizeof(uint32_t) * vertIds.size());
                break;
            }
        }

        positionBuffer_->rebuffer(positions);
        normalBuffer_->rebuffer(normals);
        colorBuffer_->rebuffer(colors);
        texcoordBuffer_->rebuffer(texCoords);
        indexBuffer_->rebuffer(vertIds);

        glEnable(GL_DEPTH_TEST);
        glBindVertexArray(vertexArray_);
        meshShader_->use();

        pointLightBuffer_->bind(1);
        directionalLightBuffer_->bind(2);

        meshShader_->setUniform("mvp", mvp_mat);
        meshShader_->setUniform("point_light_cnt", static_cast<int>(pointLights_.size()));
        meshShader_->setUniform("distant_light_cnt", static_cast<int>(directionalLights_.size()));
        meshShader_->setUniform("cam_pos", cam.getPose().position);

        meshShader_->setUniform(
            "ambientColor", glm::make_vec4(ambientColorParam_.Param<core::param::ColorParam>()->Value().data()));
        meshShader_->setUniform(
            "diffuseColor", glm::make_vec4(diffuseColorParam_.Param<core::param::ColorParam>()->Value().data()));
        meshShader_->setUniform(
            "specularColor", glm::make_vec4(specularColorParam_.Param<core::param::ColorParam>()->Value().data()));
        meshShader_->setUniform("k_amb", ambientFactorParam_.Param<core::param::FloatParam>()->Value());
        meshShader_->setUniform("k_diff", diffuseFactorParam_.Param<core::param::FloatParam>()->Value());
        meshShader_->setUniform("k_spec", specularFactorParam_.Param<core::param::FloatParam>()->Value());
        meshShader_->setUniform("k_exp", specularExponentParam_.Param<core::param::FloatParam>()->Value());
        meshShader_->setUniform(
            "meshColor", glm::make_vec3(colorParam_.Param<core::param::ColorParam>()->Value().data()));

        meshShader_->setUniform("use_lambert", useLambertParam_.Param<core::param::BoolParam>()->Value());
        bool performLighting = lightingParam_.Param<core::param::BoolParam>()->Value() && !has_external_fbo;
        meshShader_->setUniform("enable_lighting", performLighting);

        meshShader_->setUniform("has_colors", has_colors);

        if (has_vertIds) {
            glDrawElements(GL_TRIANGLES, triCount * 3, GL_UNSIGNED_INT, nullptr);
        } else {
            glDrawArrays(GL_TRIANGLES, 0, vertCount);
        }

        glUseProgram(0);
        glBindVertexArray(0);
        glDisable(GL_DEPTH_TEST);

        glFrontFace(frontFace);
        glPolygonMode(GL_FRONT, GL_FILL);
        glPolygonMode(GL_BACK, GL_FILL);
        glDisable(GL_CULL_FACE);
    }

    // reset to old fbo
    call_fbo->bind();

    return true;
}

void ModernTrisoupRenderer::updateLights(core::view::light::CallLight* lightCall, glm::vec3 camDir) {
    if (lightCall == nullptr || !(*lightCall)(core::view::light::CallLight::CallGetData)) {
        pointLights_.clear();
        directionalLights_.clear();
        core::utility::log::Log::DefaultLog.WriteWarn(
            "[ModernTrisoupRenderer]: There are no proper lights connected, no shading is happening");
    } else {
        auto& lights = lightCall->getData();

        pointLights_.clear();
        directionalLights_.clear();

        auto point_lights = lights.get<core::view::light::PointLightType>();
        auto distant_lights = lights.get<core::view::light::DistantLightType>();

        for (const auto& pl : point_lights) {
            pointLights_.push_back({pl.position[0], pl.position[1], pl.position[2], pl.intensity});
        }

        for (const auto& dl : distant_lights) {
            if (dl.eye_direction) {
                auto cd = glm::normalize(camDir); // paranoia
                directionalLights_.push_back({cd.x, cd.y, cd.z, dl.intensity});
            } else {
                directionalLights_.push_back({dl.direction[0], dl.direction[1], dl.direction[2], dl.intensity});
            }
        }
    }

    pointLightBuffer_->rebuffer(pointLights_);
    directionalLightBuffer_->rebuffer(directionalLights_);

    if (pointLights_.empty() && directionalLights_.empty()) {
        core::utility::log::Log::DefaultLog.WriteWarn("[ModernTrisoupRenderer]: There are no directional or "
                                                      "positional lights connected. Lighting not available.");
    }
}
