#include "DebugGPUMaterialDataSource.h"
/*
* DebugGPUMaterialDataSource.cpp
*
* Copyright (C) 2019 by Universitaet Stuttgart (VISUS).
* All rights reserved.
*/

#include "mmcore/CoreInstance.h"
#include "vislib/graphics/gl/ShaderSource.h"

#include "mesh/CallGPUMaterialData.h"

megamol::mesh::DebugGPUMaterialDataSource::DebugGPUMaterialDataSource()
{
}

megamol::mesh::DebugGPUMaterialDataSource::~DebugGPUMaterialDataSource()
{
}

bool megamol::mesh::DebugGPUMaterialDataSource::create()
{
	load();

	return true;
}

bool megamol::mesh::DebugGPUMaterialDataSource::getDataCallback(core::Call & caller)
{
	CallGPUMaterialData* matl_call = dynamic_cast<CallGPUMaterialData*>(&caller);
	if (matl_call == NULL)
		return false;

	matl_call->setMaterialStorage(m_gpu_materials);

	return true;
}

bool megamol::mesh::DebugGPUMaterialDataSource::load()
{
	// Create shader program
	auto shader = std::make_shared<Shader>();

	vislib::graphics::gl::ShaderSource vert_shader_src;
	vislib::graphics::gl::ShaderSource frag_shader_src;
	// TODO get rid of vislib StringA...
	vislib::StringA shader_base_name("MeshDebug");
	instance()->ShaderSourceFactory().MakeShaderSource(shader_base_name + "::vertex", vert_shader_src);
	instance()->ShaderSourceFactory().MakeShaderSource(shader_base_name + "::fragment", frag_shader_src);
	shader->Create(vert_shader_src.Code(), vert_shader_src.Count(), frag_shader_src.Code(), frag_shader_src.Count());

	m_gpu_materials->addMaterial(shader,{});
	//m_gpu_materials->addMaterial("meshDebug");

	return true;
}
