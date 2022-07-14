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
    std::vector<std::filesystem::path> const& shader_filepaths,
    std::vector<std::shared_ptr<glowl::Texture>> const& textures) {

    msf::LineTranslator translator;
    auto const shader_options = ::msf::ShaderFactoryOptionsOpenGL(mm_core_inst->GetShaderPaths());

    glowl::GLSLProgram::ShaderSourceList shader_src_list;

    for (auto const& shader_filepath : shader_filepaths) {
        try {
            shader_src_list.emplace_back(
                core::utility::make_glowl_shader_source(shader_filepath, shader_options, translator));
            if (static_cast<unsigned int>(shader_src_list.back().first) == 0) {
                throw std::runtime_error("Invalid shader type");
            }
        } catch (std::runtime_error const& ex) {
            throw std::runtime_error("GPUMaterialCollection - Error processing shader \"" + shader_filepath.string() +
                                     "\":\n" + translator.translateErrorLog(ex.what()));
        }
    }

    try {
        auto program = std::make_shared<glowl::GLSLProgram>(shader_src_list);
        program->setDebugLabel(identifier);
        addMaterial(identifier, program, textures);
    } catch (glowl::GLSLProgramException const& ex) {
        throw std::runtime_error("GPUMaterialCollection - Error creating GLSLprogram \"" + identifier + "\":\n" +
                                 translator.translateErrorLog(ex.what()));
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
