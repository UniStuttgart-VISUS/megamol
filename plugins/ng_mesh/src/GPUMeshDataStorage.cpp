#include "../include/ng_mesh/GPUMeshDataStorage.h"

std::vector<megamol::ngmesh::GPUMeshDataStorage::BatchedMeshes> const & megamol::ngmesh::GPUMeshDataStorage::getMeshes()
{
	return m_batched_meshes;
}

std::vector<megamol::ngmesh::GPUMeshDataStorage::SubMeshData> const & megamol::ngmesh::GPUMeshDataStorage::getSubMeshData()
{
	return m_sub_mesh_data;
}
