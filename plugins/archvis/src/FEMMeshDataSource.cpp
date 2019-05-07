#include "FEMMeshDataSource.h"

#include "FEMDataCall.h"
#include "ng_mesh/GPUMeshDataCall.h"

megamol::archvis::FEMMeshDataSource::FEMMeshDataSource()
	: m_fem_callerSlot("getFEMFile", "Connects the data source with loaded FEM data")
{
	this->m_fem_callerSlot.SetCompatibleCall<FEMDataCallDescription>();
	this->MakeSlotAvailable(&this->m_fem_callerSlot);
}

megamol::archvis::FEMMeshDataSource::~FEMMeshDataSource()
{
}

bool megamol::archvis::FEMMeshDataSource::create()
{
	return false;
}

bool megamol::archvis::FEMMeshDataSource::getDataCallback(core::Call & caller)
{
	ngmesh::GPUMeshDataCall* mc = dynamic_cast<ngmesh::GPUMeshDataCall*>(&caller);
	if (mc == NULL)
		return false;

	FEMDataCall* fem_call = this->m_fem_callerSlot.CallAs<FEMDataCall>();
	if (fem_call == NULL)
		return false;

	if (!(*fem_call)(0))
		return false;




	return true;
}
