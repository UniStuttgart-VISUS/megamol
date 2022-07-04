/*
 * GPUMaterialCollection.pp
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VISUS).
 * All rights reserved.
 */

#include "mesh_gl/GPUMaterialCollection.h"

#include "mmcore/CoreInstance.h"
#include "mmcore_gl/utility/ShaderFactory.h"
#include "mmcore_gl/utility/ShaderSourceFactory.h"
#include "vislib_gl/graphics/gl/ShaderSource.h"


namespace megamol {
namespace mesh_gl {

void GPUMaterialCollection::addMaterial(megamol::core::CoreInstance* mm_core_inst, std::string const& identifier,
    std::string const& shader_btf_name, std::vector<std::shared_ptr<glowl::Texture>> const& textures) {

    vislib_gl::graphics::gl::ShaderSource vert_shader_src;
    vislib_gl::graphics::gl::ShaderSource frag_shader_src;
    vislib_gl::graphics::gl::ShaderSource geom_shader_src;
    vislib_gl::graphics::gl::ShaderSource tessCtrl_shader_src;
    vislib_gl::graphics::gl::ShaderSource tessEval_shader_src;
    // TODO get rid of vislib StringA...
    vislib::StringA shader_base_name(shader_btf_name.c_str());

    auto vertShaderName = shader_base_name + "::vertex";
    auto fragShaderName = shader_base_name + "::fragment";
    auto geoShaderName = shader_base_name + "::geometry";
    auto tessCtrlShaderName = shader_base_name + "::tessellation_control";
    auto tessEvalShaderName = shader_base_name + "::tessellation_evaluation";

    auto ssf =
        std::make_shared<core_gl::utility::ShaderSourceFactory>(mm_core_inst->Configuration().ShaderDirectories());
    ssf->MakeShaderSource(vertShaderName.PeekBuffer(), vert_shader_src);
    ssf->MakeShaderSource(fragShaderName.PeekBuffer(), frag_shader_src);
    auto geom_shdr_success = ssf->MakeShaderSource(geoShaderName.PeekBuffer(), geom_shader_src);
    auto tessCtrl_shdr_success = ssf->MakeShaderSource(tessCtrlShaderName.PeekBuffer(), tessCtrl_shader_src);
    auto tessEval_shdr_success = ssf->MakeShaderSource(tessEvalShaderName.PeekBuffer(), tessEval_shader_src);

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

void GPUMaterialCollection::addMaterial(megamol::core::CoreInstance* mm_core_inst, std::string const& identifier,
    std::vector<std::filesystem::path> const& shader_filepaths,
    std::vector<std::shared_ptr<glowl::Texture>> const& textures) {

    msf::LineTranslator translator;
    auto const shader_options = ::msf::ShaderFactoryOptionsOpenGL(mm_core_inst->GetShaderPaths());

    try {

        glowl::GLSLProgram::ShaderSourceList shader_src_list;

        for (auto const& shader_filepath : shader_filepaths) {
            shader_src_list.emplace_back(
                core::utility::make_glowl_shader_source(shader_filepath, shader_options, translator));
            if (static_cast<unsigned int>(shader_src_list.back().first) == 0) {
                throw std::exception("Invalid shader type");
            }
        }

        auto program = std::make_shared<glowl::GLSLProgram>(shader_src_list);
        program->setDebugLabel(identifier);

        addMaterial(identifier, program, textures);

    } catch (glowl::GLSLProgramException const& ex) {
        throw glowl::GLSLProgramException("GPUMaterialCollection - Error building shader program \"" + identifier +
                                          "\":\n" + translator.translateErrorLog(ex.what()));
    }
    catch (std::exception const& ex) {
        throw glowl::GLSLProgramException("GPUMaterialCollection - Error adding material \"" + identifier +
                                          "\":\n" + translator.translateErrorLog(ex.what()));
    }
}

void GPUMaterialCollection::addMaterial(std::string const& identifier, std::shared_ptr<Shader> const& shader,
    std::vector<std::shared_ptr<glowl::Texture>> const& textures) {
    auto new_mtl = m_materials.insert({identifier, Material()});

    if (new_mtl.second == true) {
        new_mtl.first->second.shader_program = shader;
        new_mtl.first->second.textures = textures;
    } else {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "Could not add material, identifier already exists. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
    }
}

void GPUMaterialCollection::addMaterial(std::string const& identifier, Material const& material) {
    m_materials.insert({identifier, material});
}

void GPUMaterialCollection::updateMaterialTexture(
    std::string const& identifier, size_t tex_idx, std::shared_ptr<glowl::Texture> const& texture) {

    auto query = m_materials.find(identifier);

    if (query != m_materials.end()) {
        query->second.textures[tex_idx] = texture;
    } else {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "Could not update material texture, identifier not found. [%s, %s, line %d]\n", __FILE__, __FUNCTION__,
            __LINE__);
    }
}

void GPUMaterialCollection::deleteMaterial(std::string const& identifier) {

    auto erased_cnt = m_materials.erase(identifier);

    if (erased_cnt == 0) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "Could not delete material, identifier not found. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
    }
}

void GPUMaterialCollection::clear() {
    m_materials.clear();
}

GPUMaterialCollection::Material const& GPUMaterialCollection::getMaterial(std::string const& identifier) {
    auto query = m_materials.find(identifier);

    if (query != m_materials.end()) {
        return query->second;
    } else {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "Could not get material, identifier not found. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
    }

    return Material();
}

} // namespace mesh_gl
} // namespace megamol
