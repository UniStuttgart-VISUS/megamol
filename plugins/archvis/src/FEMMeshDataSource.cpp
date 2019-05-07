#include "FEMMeshDataSource.h"

#include "FEMDataCall.h"

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
	return false;
}
