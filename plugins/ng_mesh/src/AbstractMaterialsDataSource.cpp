/*
* AbstractMaterialsDataSource.cpp
*
* Copyright (C) 2019 by Universitaet Stuttgart (VISUS).
* All rights reserved.
*/

#include "stdafx.h"

#include "ng_mesh/AbstractMaterialsDataSource.h"
#include "ng_mesh/MaterialsDataCall.h"

megamol::ngmesh::AbstractMaterialsDataSource::AbstractMaterialsDataSource()
	: core::Module(), m_getData_slot("getData", "The slot publishing the loaded data")
{
	this->m_getData_slot.SetCallback(MaterialsDataCall::ClassName(), "GetData", &AbstractMaterialsDataSource::getDataCallback);
	this->MakeSlotAvailable(&this->m_getData_slot);
}

megamol::ngmesh::AbstractMaterialsDataSource::~AbstractMaterialsDataSource()
{
	this->Release();
}

bool megamol::ngmesh::AbstractMaterialsDataSource::create(void)
{
	// intentionally empty ?
	return true;
}

void megamol::ngmesh::AbstractMaterialsDataSource::release()
{
	// intentionally empty ?
}
