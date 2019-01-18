/*
* DebugBatchedMeshesDataSource.cpp
*
* Copyright(C) 2019 by Universitaet Stuttgart(VISUS).
* All rights reserved.
*/

#include "stdafx.h"

#include "..\include\ng_mesh\MeshDataStorage.h"

using namespace megamol::ngmesh;

BatchedMeshesDataAccessor megamol::ngmesh::MeshDataStorage::generateDataAccessor()
{
	return BatchedMeshesDataAccessor();
}
