#include "FEMGPUMeshDataSource.h"

#include "FEMDataCall.h"

megamol::archvis::FEMGPUMeshDataSource::FEMGPUMeshDataSource()
	: m_fem_callerSlot("getFEMFile", "Connects the data source with loaded FEM data")
{
	this->m_fem_callerSlot.SetCompatibleCall<FEMDataCallDescription>();
	this->MakeSlotAvailable(&this->m_fem_callerSlot);
}

megamol::archvis::FEMGPUMeshDataSource::~FEMGPUMeshDataSource()
{
}

bool megamol::archvis::FEMGPUMeshDataSource::create()
{
	return false;
}

bool megamol::archvis::FEMGPUMeshDataSource::getDataCallback(core::Call & caller)
{
	return false;
}
