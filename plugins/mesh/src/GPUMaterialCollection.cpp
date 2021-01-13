/*
 * GPUMaterialCollection.pp
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VISUS).
 * All rights reserved.
 */

#include "stdafx.h"
#include "mesh/GPUMaterialCollection.h"

#include "mmcore/CoreInstance.h"
#include "vislib/graphics/gl/ShaderSource.h"


namespace megamol {
namespace mesh {

void GPUMaterialCollection::addMaterial(megamol::core::CoreInstance* mm_core_inst, std::string const& identifier,
    std::string const& shader_btf_name, std::vector<std::shared_ptr<glowl::Texture>> const& textures) {

    vislib::graphics::gl::ShaderSource vert_shader_src;
    vislib::graphics::gl::ShaderSource frag_shader_src;
    vislib::graphics::gl::ShaderSource geom_shader_src;
    vislib::graphics::gl::ShaderSource tessCtrl_shader_src;
    vislib::graphics::gl::ShaderSource tessEval_shader_src;
    // TODO get rid of vislib StringA...
    vislib::StringA shader_base_name(shader_btf_name.c_str());

    auto vertShaderName = shader_base_name + "::vertex";
    auto fragShaderName = shader_base_name + "::fragment";
    auto geoShaderName = shader_base_name + "::geometry";
    auto tessCtrlShaderName = shader_base_name + "::tessellation_control";
    auto tessEvalShaderName = shader_base_name + "::tessellation_evaluation";

    mm_core_inst->ShaderSourceFactory().MakeShaderSource(vertShaderName.PeekBuffer(), vert_shader_src);
    mm_core_inst->ShaderSourceFactory().MakeShaderSource(fragShaderName.PeekBuffer(), frag_shader_src);
    auto geom_shdr_success =
        mm_core_inst->ShaderSourceFactory().MakeShaderSource(geoShaderName.PeekBuffer(), geom_shader_src);
    auto tessCtrl_shdr_success =
        mm_core_inst->ShaderSourceFactory().MakeShaderSource(tessCtrlShaderName.PeekBuffer(), tessCtrl_shader_src);
    auto tessEval_shdr_success =
        mm_core_inst->ShaderSourceFactory().MakeShaderSource(tessEvalShaderName.PeekBuffer(), tessEval_shader_src);

    std::string vertex_src(vert_shader_src.WholeCode(), (vert_shader_src.WholeCode()).Length());
    std::string tessellationControl_src(tessCtrl_shader_src.WholeCode(), (tessCtrl_shader_src.WholeCode()).Length());
    std::string tessellationEvaluation_src(tessEval_shader_src.WholeCode(), (tessEval_shader_src.WholeCode()).Length());
    std::string geometry_src(geom_shader_src.WholeCode(), (geom_shader_src.WholeCode()).Length());
    std::string fragment_src(frag_shader_src.WholeCode(), (frag_shader_src.WholeCode()).Length());
    std::string compute_src;

    std::vector<std::pair<glowl::GLSLProgram::ShaderType, std::string>> shader_srcs;

    if (!vertex_src.empty()) {
        shader_srcs.push_back({glowl::GLSLProgram::ShaderType::Vertex, vertex_src});
    }
    if (!fragment_src.empty()) {
        shader_srcs.push_back({glowl::GLSLProgram::ShaderType::Fragment, fragment_src});
    }
    if (!geometry_src.empty()) {
        shader_srcs.push_back({glowl::GLSLProgram::ShaderType::Geometry, geometry_src});
    }
    if (!tessellationControl_src.empty()) {
        shader_srcs.push_back({glowl::GLSLProgram::ShaderType::TessControl, tessellationControl_src});
    }
    if (!tessellationEvaluation_src.empty()) {
        shader_srcs.push_back({glowl::GLSLProgram::ShaderType::TessEvaluation, tessellationEvaluation_src});
    }
    if (!compute_src.empty()) {
        shader_srcs.push_back({glowl::GLSLProgram::ShaderType::Compute, compute_src});
    }

    std::shared_ptr<Shader> shader(nullptr);
    try {
        shader = std::make_shared<Shader>(shader_srcs);
    } catch (glowl::GLSLProgramException const& exc) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "Error during shader program creation of\"%s\": %s. [%s, %s, line %d]\n", shader_btf_name.c_str(),
            exc.what(), __FILE__, __FUNCTION__, __LINE__);
    }

    addMaterial(identifier, shader, textures);
}

void GPUMaterialCollection::addMaterial(std::string const& identifier, std::shared_ptr<Shader> const& shader,
    std::vector<std::shared_ptr<glowl::Texture>> const& textures) {
    auto new_mtl = m_materials.insert({identifier,Material()});

    if (new_mtl.second == true) {
        new_mtl.first->second.shader_program = shader;
        new_mtl.first->second.textures = textures;
    } else {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "Could not add material, identifier already exists. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
    }
}

void GPUMaterialCollection::addMaterial(std::string const& identifier, Material const& material) {
    m_materials.insert({identifier,material});
}

void GPUMaterialCollection::updateMaterialTexture(
    std::string const& identifier,
    size_t tex_idx, std::shared_ptr<glowl::Texture> const& texture) {

    auto query = m_materials.find(identifier);

    if (query != m_materials.end()) {
        query->second.textures[tex_idx] = texture;
    } else {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "Could not update material texture, identifier not found. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
    }
}

void GPUMaterialCollection::deleteMaterial(std::string const& identifier) {

    auto erased_cnt = m_materials.erase(identifier);

    if (erased_cnt == 0) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "Could not delete material, identifier not found. [%s, %s, line %d]\n", __FILE__, __FUNCTION__,
            __LINE__);
    }
}

void GPUMaterialCollection::clear() { m_materials.clear(); }

GPUMaterialCollection::Material const& GPUMaterialCollection::getMaterial(std::string const& identifier) {
    auto query = m_materials.find(identifier);

    if (query != m_materials.end()) {
        return query->second;
    } else {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "Could not get material, identifier not found. [%s, %s, line %d]\n", __FILE__, __FUNCTION__,
            __LINE__);
    }

    return Material();
}

} // namespace mesh
} // namespace megamol
