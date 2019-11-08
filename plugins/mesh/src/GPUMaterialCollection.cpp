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

void GPUMaterialCollecton::addMaterial(
    megamol::core::CoreInstance* mm_core_inst,
    std::string  const& shader_btf_name,
    std::vector<std::shared_ptr<glowl::Texture>> const& textures) 
{
    std::shared_ptr<Shader> shader = std::make_shared<Shader>();

    vislib::graphics::gl::ShaderSource vert_shader_src;
    vislib::graphics::gl::ShaderSource frag_shader_src;
    vislib::graphics::gl::ShaderSource geom_shader_src;
    // TODO get rid of vislib StringA...
    vislib::StringA shader_base_name(shader_btf_name.c_str());

    auto vertShaderName = shader_base_name + "::vertex";
    auto fragShaderName = shader_base_name + "::fragment";
    auto geoShaderName = shader_base_name + "::geometry";

    mm_core_inst->ShaderSourceFactory().MakeShaderSource(vertShaderName.PeekBuffer(), vert_shader_src);
    mm_core_inst->ShaderSourceFactory().MakeShaderSource(fragShaderName.PeekBuffer(), frag_shader_src);
    auto geom_shdr_success =
        mm_core_inst->ShaderSourceFactory().MakeShaderSource(geoShaderName.PeekBuffer(), geom_shader_src);

    std::string vertex_src( vert_shader_src.WholeCode() , (vert_shader_src.WholeCode()).Length());
    std::string tessellationControl_src;
    std::string tessellationEvaluation_src;
    std::string geometry_src(geom_shader_src.WholeCode(), (geom_shader_src.WholeCode()).Length());
    std::string fragment_src(frag_shader_src.WholeCode(), (frag_shader_src.WholeCode()).Length());
    std::string compute_src;

    bool prgm_error = false;

    if (!vertex_src.empty())
        prgm_error |= !shader->compileShaderFromString(&vertex_src, Shader::VertexShader);
    if (!fragment_src.empty())
        prgm_error |= !shader->compileShaderFromString(&fragment_src, Shader::FragmentShader);
    if (!geometry_src.empty())
        prgm_error |= !shader->compileShaderFromString(&geometry_src, Shader::GeometryShader);
    if (!tessellationControl_src.empty())
        prgm_error |= !shader->compileShaderFromString(&tessellationControl_src, Shader::TessellationControl);
    if (!tessellationEvaluation_src.empty())
        prgm_error |= !shader->compileShaderFromString(&tessellationEvaluation_src, Shader::TessellationEvaluation);
    if (!compute_src.empty())
        prgm_error |= !shader->compileShaderFromString(&compute_src, Shader::ComputeShader);

    prgm_error |= !shader->link();

    if (prgm_error) {
        std::cout << "Error during shader program creation of \"" << shader->getDebugLabel() << "\""
                  << std::endl;
        std::cout << shader->getLog();
    }

    addMaterial(shader, textures);
}

void GPUMaterialCollecton::addMaterial(
    std::shared_ptr<Shader> const& shader, 
    std::vector<std::shared_ptr<glowl::Texture>> const& textures) 
{
    m_materials.push_back(Material());
    m_materials.back().shader_program = shader;
    m_materials.back().textures = textures;
}

void GPUMaterialCollecton::updateMaterialTexture(
    size_t mtl_idx, size_t tex_idx, std::shared_ptr<glowl::Texture> const& texture) {
    m_materials[mtl_idx].textures[tex_idx] = texture;
}

void GPUMaterialCollecton::clearMaterials() { m_materials.clear(); }

} // namespace mesh
} // namespace megamol
