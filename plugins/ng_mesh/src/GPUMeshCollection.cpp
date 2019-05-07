#include "../include/ng_mesh/GPUMeshCollection.h"

std::vector<megamol::ngmesh::GPUMeshCollection::BatchedMeshes> const & megamol::ngmesh::GPUMeshCollection::getMeshes()
{
	return m_batched_meshes;
}

std::vector<megamol::ngmesh::GPUMeshCollection::SubMeshData> const & megamol::ngmesh::GPUMeshCollection::getSubMeshData()
{
	return m_sub_mesh_data;
}
