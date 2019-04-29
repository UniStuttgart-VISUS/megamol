/*
* GPUMaterialCollection.h
*
* Copyright (C) 2019 by Universitaet Stuttgart (VISUS).
* All rights reserved.
*/

#include "stdafx.h"
#include "ng_mesh/GPUMaterialCollection.h"

#include "mmcore/CoreInstance.h"
#include "vislib/graphics/gl/ShaderSource.h"

void megamol::ngmesh::GPUMaterialCollecton::addMaterial(megamol::core::CoreInstance* mm_core_inst, std::string shader_btf_name)
{
	std::shared_ptr<GLSLShader> shader = std::make_shared<GLSLShader>();

	vislib::graphics::gl::ShaderSource vert_shader_src;
	vislib::graphics::gl::ShaderSource frag_shader_src;
	// TODO get rid of vislib StringA...
	vislib::StringA shader_base_name(shader_btf_name.c_str());

	mm_core_inst->ShaderSourceFactory().MakeShaderSource(shader_base_name + "::vertex", vert_shader_src);
	mm_core_inst->ShaderSourceFactory().MakeShaderSource(shader_base_name + "::fragment", frag_shader_src);

	shader->Create(vert_shader_src.Code(), vert_shader_src.Count(), frag_shader_src.Code(), frag_shader_src.Count());

	addMaterial(shader);
}

void megamol::ngmesh::GPUMaterialCollecton::addMaterial(std::shared_ptr<GLSLShader> const& shader)
{
	m_materials.push_back(Material());
	m_materials.back().shader_program = shader;
}

void megamol::ngmesh::GPUMaterialCollecton::clearMaterials()
{
	m_materials.clear();
}
