#include "..\include\ng_mesh\SimpleGPUMtlDataSource.h"

#include "mmcore/param/FilePathParam.h"

#include "ng_mesh/GPUMaterialDataCall.h"

megamol::ngmesh::SimpleGPUMtlDataSource::SimpleGPUMtlDataSource()
	: m_btf_filename_slot("BTF filename", "The name of the btf file to load")
{
	this->m_btf_filename_slot << new core::param::FilePathParam("");
	this->MakeSlotAvailable(&this->m_btf_filename_slot);
}

megamol::ngmesh::SimpleGPUMtlDataSource::~SimpleGPUMtlDataSource()
{
}

bool megamol::ngmesh::SimpleGPUMtlDataSource::create()
{
	return false;
}

bool megamol::ngmesh::SimpleGPUMtlDataSource::getDataCallback(core::Call & caller)
{
	GPUMaterialDataCall* mtl_call = dynamic_cast<GPUMaterialDataCall*>(&caller);
	if (mtl_call == NULL)
		return false;

	// clear update?

	if (this->m_btf_filename_slot.IsDirty())
	{
		m_btf_filename_slot.ResetDirty();

		auto vislib_filename = m_btf_filename_slot.Param<core::param::FilePathParam>()->Value();
		std::string filename(vislib_filename.PeekBuffer());

		m_gpu_materials->clearMaterials();

		m_gpu_materials->addMaterial(filename);
	}

	mtl_call->setMaterialStorage(m_gpu_materials);

	// set update?

	return true;
}

bool megamol::ngmesh::SimpleGPUMtlDataSource::load()
{
	return false;
}
