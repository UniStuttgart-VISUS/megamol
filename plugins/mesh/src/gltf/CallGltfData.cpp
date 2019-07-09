#include "stdafx.h"

#include "mesh/CallGltfData.h"

#ifndef TINYGLTF_IMPLEMENTATION
#define TINYGLTF_IMPLEMENTATION
#endif // !TINYGLTF_IMPLEMENTATION
#ifndef STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#endif // !STB_IMAGE_IMPLEMENTATION
#ifndef STB_IMAGE_WRITE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#endif // !STB_IMAGE_WRITE_IMPLEMENTATION
#include "tiny_gltf.h"

void megamol::mesh::CallGlTFData::setGlTFModel(std::shared_ptr<tinygltf::Model> const & gltf_model)
{
	m_gltf_model = gltf_model;
}

std::shared_ptr<tinygltf::Model> megamol::mesh::CallGlTFData::getGlTFModel()
{
	return m_gltf_model;
}

void megamol::mesh::CallGlTFData::setUpdateFlag()
{
	m_update_flag = true;
}

bool megamol::mesh::CallGlTFData::getUpdateFlag()
{
	return m_update_flag;
}

void megamol::mesh::CallGlTFData::clearUpdateFlag()
{
	m_update_flag = false;
}
