/*
* AbstractNGMeshDataSource.h
*
* Copyright (C) 2017 by Universitaet Stuttgart (VISUS).
* All rights reserved.
*/

#ifndef ABSTRACT_NG_MESH_DATASOURCE_H_INCLUDED
#define ABSTRACT_NG_MESH_DATASOURCE_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "mmcore/CalleeSlot.h"

#include "ng_mesh/CallNGMeshRenderBatches.h"

namespace megamol {
namespace ngmesh {

	class NG_MESH_API AbstractNGMeshDataSource : public core::Module
	{
	public:
		AbstractNGMeshDataSource();

		virtual ~AbstractNGMeshDataSource();

	protected:

		/**
		*
		*/
		class RenderBatchesData
		{
		private:
			struct MeshData
			{
				std::list<std::vector<uint8_t>>	vertex_data;
				std::vector<uint8_t>			index_data;

				VertexLayout	vertex_descriptor;
				GLenum			index_type;
				GLenum			usage;
				GLenum			primitive_type;
			};

			struct RenderBatch
			{
				std::string							program_name;
				MeshData							mesh_data;
				std::vector<DrawElementsCommand>	draw_commands;
				std::vector<uint8_t>				perObject_shader_params;
				std::vector<uint8_t>				perMaterial_shader_params;
				uint32_t							update_flags;
			};

			std::list<RenderBatch> m_render_batches;

		public:
			template<typename VertexBufferContainer,
				typename IndexBufferContainer,
				typename ObjectShaderParamsContainer,
				typename MaterialShaderParamsContainer>
			void addRenderTask(
				std::string program_name,
				VertexBufferContainer vertex_buffer_data,
				IndexBufferContainer index_buffer_data,
				VertexLayout vertex_descriptor,
				ObjectShaderParamsContainer obj_shdr_params,
				MaterialShaderParamsContainer mtl_shdr_params)
			{
				//TODO check if batch with same program and vertex layout already exits
			}

			RenderBatchesDataAccessor getAccessor()
			{

			}


		};

		/**
		* Implementation of 'Create'.
		*
		* @return 'true' on success, 'false' otherwise.
		*/
		virtual bool create(void);

		/**
		* Gets the data from the source.
		*
		* @param caller The calling call.
		*
		* @return 'true' on success, 'false' on failure.
		*/
		virtual bool getDataCallback(core::Call& caller) = 0;

		/**
		* Gets the data from the source.
		*
		* @param caller The calling call.
		*
		* @return 'true' on success, 'false' on failure.
		*/
		virtual bool getExtentCallback(core::Call& caller);

		/**
		* Implementation of 'Release'.
		*/
		virtual void release();

		/** The data storage for the render batches */
		CallNGMeshRenderBatches::RenderBatchesData m_render_batches;

		RenderBatchesData m_new_batches;

		/** The bounding box */
		vislib::math::Cuboid<float> m_bbox;

	private:

		/** The slot for requesting data */
		megamol::core::CalleeSlot m_getData_slot;
	};

}
}

#endif

