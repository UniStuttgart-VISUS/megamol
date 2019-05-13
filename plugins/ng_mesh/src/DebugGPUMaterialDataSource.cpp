#include "DebugGPUMaterialDataSource.h"
/*
* DebugGPUMaterialDataSource.cpp
*
* Copyright (C) 2019 by Universitaet Stuttgart (VISUS).
* All rights reserved.
*/

#include "mmcore/CoreInstance.h"
#include "vislib/graphics/gl/ShaderSource.h"

#include "ng_mesh/GPUMaterialDataCall.h"

megamol::ngmesh::DebugGPUMaterialDataSource::DebugGPUMaterialDataSource()
{
}

megamol::ngmesh::DebugGPUMaterialDataSource::~DebugGPUMaterialDataSource()
{
}

bool megamol::ngmesh::DebugGPUMaterialDataSource::create()
{
	load();

	return true;
}

bool megamol::ngmesh::DebugGPUMaterialDataSource::getDataCallback(core::Call & caller)
{
	GPUMaterialDataCall* matl_call = dynamic_cast<GPUMaterialDataCall*>(&caller);
	if (matl_call == NULL)
		return false;

	matl_call->setMaterialStorage(m_gpu_materials);

	return true;
}

bool megamol::ngmesh::DebugGPUMaterialDataSource::load()
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

	m_gpu_materials->addMaterial(shader);
	//m_gpu_materials->addMaterial("NGMeshDebug");

	return true;
}
