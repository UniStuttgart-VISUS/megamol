#include "stdafx.h"

#include "glTFMeshesDataSource.h"

megamol::ngmesh::GlTFMeshesDataSource::GlTFMeshesDataSource()
	: m_glTF_callerSlot("getGlTFFile","Connects the data source with a loaded glTF file")
{
	this->m_glTF_callerSlot.SetCompatibleCall<GlTFDataCallDescription>();
	this->MakeSlotAvailable(&this->m_glTF_callerSlot);
}

megamol::ngmesh::GlTFMeshesDataSource::~GlTFMeshesDataSource()
{
}

bool megamol::ngmesh::GlTFMeshesDataSource::getDataCallback(core::Call & caller)
{
	BatchedMeshesDataCall* mesh_call = dynamic_cast<BatchedMeshesDataCall*>(&caller);
	if (mesh_call == NULL)
		return false;

	GlTFDataCall* gltf_call = this->m_glTF_callerSlot.CallAs<GlTFDataCall>();
	if (gltf_call == NULL)
		return false;

	if (!(*gltf_call)(0))
		return false;

	if (gltf_call->getUpdateFlag())
	{
		BatchedMeshesDataAccessor retval;

		// set update_all_flag?
	}

	return true;
}
