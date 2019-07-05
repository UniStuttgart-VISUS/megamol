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
    megamol::core::CoreInstance* mm_core_inst, std::string shader_btf_name, std::vector<GLuint> texture_names) {
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

    try {

        if (geom_shdr_success) {
            if (!shader->Compile(vert_shader_src.Code(), vert_shader_src.Count(), geom_shader_src.Code(),
                    geom_shader_src.Count(), frag_shader_src.Code(), frag_shader_src.Count())) {
                vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
                    "Unable to compile %s: Unknown error\n", shader_base_name.PeekBuffer());
                // return false;
            }
            if (!shader->Link()) {
                vislib::sys::Log::DefaultLog.WriteMsg(
                    vislib::sys::Log::LEVEL_ERROR, "Unable to link %s: Unknown error\n", shader_base_name.PeekBuffer());
                // return false;
            }
        } else {
            shader->Create(
                vert_shader_src.Code(), vert_shader_src.Count(), frag_shader_src.Code(), frag_shader_src.Count());
        }
    } catch (vislib::graphics::gl::AbstractOpenGLShader::CompileException ce) {
        vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR, "Unable to compile %s (@%s):\n%s\n",
            shader_base_name.PeekBuffer(),
            vislib::graphics::gl::AbstractOpenGLShader::CompileException::CompileActionName(ce.FailedAction()),
            ce.GetMsgA());
        // return false;
    } catch (vislib::Exception e) {
        vislib::sys::Log::DefaultLog.WriteMsg(
            vislib::sys::Log::LEVEL_ERROR, "Unable to compile %s:\n%s\n", shader_base_name.PeekBuffer(), e.GetMsgA());
        // return false;
    } catch (...) {
        vislib::sys::Log::DefaultLog.WriteMsg(
            vislib::sys::Log::LEVEL_ERROR, "Unable to compile %s: Unknown exception\n", shader_base_name.PeekBuffer());
        // return false;
    }

    addMaterial(shader, texture_names);
}

void GPUMaterialCollecton::addMaterial(std::shared_ptr<Shader> const& shader, std::vector<GLuint> texture_names) {
    m_materials.push_back(Material());
    m_materials.back().shader_program = shader;
    m_materials.back().textures_names = texture_names;
}

void GPUMaterialCollecton::updateMaterialTexture(size_t mtl_idx, size_t tex_idx, GLuint texture_name) {
    m_materials[mtl_idx].textures_names[tex_idx] = texture_name;
}

void GPUMaterialCollecton::clearMaterials() { m_materials.clear(); }

} // namespace mesh
} // namespace megamol
