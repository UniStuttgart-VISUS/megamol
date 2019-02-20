#include "stdafx.h"

#include "glTFMeshesDataSource.h"
#include "tiny_gltf.h"
#include "ng_mesh/GPUMeshDataCall.h"

megamol::ngmesh::GlTFMeshesDataSource::GlTFMeshesDataSource()
	: m_glTF_callerSlot("getGlTFFile","Connects the data source with a loaded glTF file")
{
	this->m_glTF_callerSlot.SetCompatibleCall<GlTFDataCallDescription>();
	this->MakeSlotAvailable(&this->m_glTF_callerSlot);
}

megamol::ngmesh::GlTFMeshesDataSource::~GlTFMeshesDataSource()
{
}

bool megamol::ngmesh::GlTFMeshesDataSource::create()
{
	m_gpu_meshes = std::make_shared<GPUMeshDataStorage>();

	m_bbox = { -1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f };

	return true;
}

bool megamol::ngmesh::GlTFMeshesDataSource::getDataCallback(core::Call & caller)
{
	GPUMeshDataCall* mc = dynamic_cast<GPUMeshDataCall*>(&caller);
	if (mc == NULL)
		return false;

	GlTFDataCall* gltf_call = this->m_glTF_callerSlot.CallAs<GlTFDataCall>();
	if (gltf_call == NULL)
		return false;

	if (!(*gltf_call)(0))
		return false;

	if (gltf_call->getUpdateFlag())
	{
		auto model = gltf_call->getGlTFModel();

		for (size_t mesh_idx = 0; mesh_idx < model->meshes.size(); mesh_idx++)
		{
			std::vector<VertexLayout::Attribute> attribs;
			std::vector<std::pair< std::vector<unsigned char>::iterator, std::vector<unsigned char>::iterator>> vb_iterators;
			std::pair< std::vector<unsigned char>::iterator, std::vector<unsigned char>::iterator> ib_iterators;

			//TODO for now support a single primitive per mesh
			auto& indices_accessor = model->accessors[model->meshes[mesh_idx].primitives.back().indices];
			auto& indices_bufferView = model->bufferViews[indices_accessor.bufferView];
			auto& indices_buffer = model->buffers[indices_bufferView.buffer];

			ib_iterators = {
				indices_buffer.data.begin() + indices_bufferView.byteOffset + indices_accessor.byteOffset,
				indices_buffer.data.begin() + indices_bufferView.byteOffset + indices_accessor.byteOffset
				+ (indices_accessor.count * indices_accessor.ByteStride(indices_bufferView))
			};

			auto& vertex_attributes = model->meshes[mesh_idx].primitives.back().attributes;
			for (auto attrib : vertex_attributes)
			{
				auto& vertexAttrib_accessor = model->accessors[attrib.second];
				auto& vertexAttrib_bufferView = model->bufferViews[vertexAttrib_accessor.bufferView];
				auto& vertexAttrib_buffer = model->buffers[vertexAttrib_bufferView.buffer];

				attribs.push_back(VertexLayout::Attribute(
					vertexAttrib_accessor.type,
					vertexAttrib_accessor.componentType,
					vertexAttrib_accessor.normalized,
					vertexAttrib_accessor.byteOffset)
				);

				//TODO vb_iterators
				vb_iterators.push_back(
					{
						vertexAttrib_buffer.data.begin() + vertexAttrib_bufferView.byteOffset + vertexAttrib_accessor.byteOffset,
						vertexAttrib_buffer.data.begin() + vertexAttrib_bufferView.byteOffset + vertexAttrib_accessor.byteOffset
							+ (vertexAttrib_accessor.count * vertexAttrib_accessor.ByteStride(vertexAttrib_bufferView))
					}
				);
			}

			VertexLayout vertex_descriptor(0, attribs);
			m_gpu_meshes->addMesh(vertex_descriptor, vb_iterators, ib_iterators, indices_accessor.componentType, GL_STATIC_DRAW, GL_TRIANGLES);
			
		}

		// set update_all_flag?
	}

	mc->setGPUMeshes(m_gpu_meshes.get());

	return true;
}
