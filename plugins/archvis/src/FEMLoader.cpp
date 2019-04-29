#include "stdafx.h"

#include "FEMTxtLoader.h"

#include "mmcore/param/FilePathParam.h"

megamol::archvis::FEMLoader::FEMLoader()
	: core::Module(),
	m_femNodes_filename_slot("FEM node list filename","The name of the txt file containing the FEM nodes"),
	m_femElements_filename_slot("FEM element list filename","The name of the txt file containing the FEM elemets"),
	m_getData_slot("getData","The slot for publishing the loaded data")
{
	this->m_getData_slot.SetCallback(FEMDataCall::ClassName(), "GetData", &FEMLoader::getDataCallback);
	this->MakeSlotAvailable(&this->m_getData_slot);

	this->m_femNodes_filename_slot << new core::param::FilePathParam("");
	this->MakeSlotAvailable(&this->m_femNodes_filename_slot);

	this->m_femElements_filename_slot << new core::param::FilePathParam("");
	this->MakeSlotAvailable(&this->m_femElements_filename_slot);
}

megamol::archvis::FEMLoader::~FEMLoader()
{
}

bool megamol::archvis::FEMLoader::create(void)
{
	return false;
}

bool megamol::archvis::FEMLoader::getDataCallback(core::Call & caller)
{
	return false;
}

void megamol::archvis::FEMLoader::release()
{
}
