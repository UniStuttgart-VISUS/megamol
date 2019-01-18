/*
* DebugMaterialsDataSource.cpp
*
* Copyright (C) 2019 by Universitaet Stuttgart (VISUS).
* All rights reserved.
*/

#include "stdafx.h"

#include "DebugMaterialsDataSource.h"

#include "ng_mesh/MaterialsDataCall.h"

megamol::ngmesh::DebugMaterialsDataSource::DebugMaterialsDataSource()
{
}

megamol::ngmesh::DebugMaterialsDataSource::~DebugMaterialsDataSource()
{
}

bool megamol::ngmesh::DebugMaterialsDataSource::getDataCallback(core::Call & caller)
{
	MaterialsDataCall* matl_call = dynamic_cast<MaterialsDataCall*>(&caller);
	if (matl_call == NULL)
		return false;

	load();
	
	matl_call->setMaterialsData(m_material_storage);

	return true;
}

bool megamol::ngmesh::DebugMaterialsDataSource::load()
{
	return false;
}
