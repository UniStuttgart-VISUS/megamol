/*
* CallNGMeshRenderBatches.h
*
* Copyright (C) 2017 by Universitaet Stuttgart (VISUS).
* All rights reserved.
*/

#ifndef CALL_NG_MESH_RENDER_BATCHES_H_INCLUDED
#define CALL_NG_MESH_RENDER_BATCHES_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include <array>
#include <string>
#include <vector>

#include "ng_mesh.h"
#include "mmcore/AbstractGetData3DCall.h"
#include "mmcore/factories/CallAutoDescription.h"
#include "vislib/assert.h"
#include "vislib/Array.h"
#include "vislib/String.h"
#include "vislib/macro_utils.h"

#include "VertexLayout.h"

namespace megamol {
namespace ngmesh {

	class NG_MESH_API CallNGMeshRenderBatches : public megamol::core::AbstractGetData3DCall
	{
		public:

			class NG_MESH_API RenderBatchesData
			{
			public:

				static uint32_t shaderUpdateBit()			{ return 0x1; }
				static uint32_t meshUpdateBit()				{ return 0x2; }
				static uint32_t drawCommandsUpdateBit()		{ return 0x4; }
				static uint32_t meshParamsUpdateBit()		{ return 0x8; }
				static uint32_t materialsParamsUpdateBit()	{ return 0x10; }

				struct ShaderPrgmData
				{
					char*	raw_string;
					size_t	char_cnt;
				};

				struct MeshData
				{
					struct VertexData
					{
						uint8_t*	raw_data;
						size_t		byte_size;
					};

					struct IndexData
					{
						uint8_t*	raw_data;
						size_t		byte_size;
						GLenum		index_type;
					};

					VertexData		vertex_data;
					IndexData		index_data;
					VertexLayout	vertex_descriptor;
				};

				struct DrawCommandData
				{
					struct DrawElementsCommand
					{
						GLuint vertex_cnt;
						GLuint instance_cnt;
						GLuint first_idx;
						GLuint base_vertex;
						GLuint base_instance;
					};

					DrawElementsCommand*	data;
					GLsizei					draw_cnt;
				};

				struct MeshShaderParams
				{
					uint8_t*	raw_data;
					size_t		byte_size;
				};

				struct MaterialParameters
				{
					//TODO texture paths?
				};

				struct MaterialShaderParams
				{
					MaterialParameters*	data;
					size_t				elements_cnt;
				};

			private:

				uint8_t*	raw_buffer;
				size_t		used_byte_size;
				size_t		allocated_byte_size;
				size_t		batch_cnt;

				ShaderPrgmData*			shader_prgms;
				MeshData*				meshes;
				DrawCommandData*		draw_commands;
				MeshShaderParams*		mesh_shader_params;
				MaterialShaderParams*	mtl_shader_params;
				uint32_t*				update_flags;

			public:
				RenderBatchesData()
					: raw_buffer(nullptr),
					used_byte_size(0),
					allocated_byte_size(0),
					batch_cnt(0),
					shader_prgms(nullptr),
					meshes(nullptr),
					draw_commands(nullptr),
					mesh_shader_params(nullptr),
					mtl_shader_params(nullptr),
					update_flags(nullptr) {}
				~RenderBatchesData()
				{
					if (raw_buffer != nullptr)
						delete raw_buffer;
				}

				RenderBatchesData& operator=(const RenderBatchesData& rhs)
				{
					used_byte_size = rhs.used_byte_size;
					allocated_byte_size = rhs.allocated_byte_size;
					batch_cnt = rhs.batch_cnt;

					delete raw_buffer;
					raw_buffer = new uint8_t[allocated_byte_size];
					std::memcpy(raw_buffer, rhs.raw_buffer, used_byte_size);

					// set individual pointers to new values
					size_t base_offset = 0;
					size_t offset = 0;
					for (int i = 0; i < batch_cnt; ++i)
					{
						shader_prgms = reinterpret_cast<ShaderPrgmData*>(raw_buffer);
						offset += sizeof(ShaderPrgmData) + shader_prgms[i].char_cnt;

						meshes = reinterpret_cast<MeshData*>(raw_buffer + base_offset + offset);
						offset += sizeof(MeshData) + meshes[i].vertex_data.byte_size + meshes[i].index_data.byte_size;

						draw_commands = reinterpret_cast<DrawCommandData*>(raw_buffer + base_offset + offset);
						offset += sizeof(DrawCommandData) + sizeof(DrawCommandData::DrawElementsCommand) * draw_commands[i].draw_cnt;

						mesh_shader_params = reinterpret_cast<MeshShaderParams*>(raw_buffer + base_offset + offset);
						offset += sizeof(MeshShaderParams) + mesh_shader_params[i].byte_size;

						mtl_shader_params = reinterpret_cast<MaterialShaderParams*>(raw_buffer + base_offset + offset);
						offset += sizeof(MaterialShaderParams);

						update_flags = reinterpret_cast<uint32_t*>(raw_buffer + base_offset + offset);
						offset += sizeof(uint32_t);

						base_offset += offset;
						offset = 0;
					}

					return *this;
				}

				RenderBatchesData(const RenderBatchesData& cpy) = delete;
				RenderBatchesData(RenderBatchesData&& other) = delete;
				RenderBatchesData& operator=(RenderBatchesData&& rhs) = delete;
				

				void reallocate(size_t byte_size)
				{
					if (byte_size <= allocated_byte_size)
					{
						vislib::sys::Log::DefaultLog.WriteError("Reallocation size for RenderBatches not feasible");
						return;
					}

					uint8_t* new_raw_buffer = new uint8_t[byte_size];
					std::memcpy(new_raw_buffer, raw_buffer, used_byte_size);
					allocated_byte_size = byte_size;

					// set individual pointers to new values
					size_t base_offset = 0;
					size_t offset = 0;
					for (int i = 0; i < batch_cnt; ++i)
					{
						shader_prgms = reinterpret_cast<ShaderPrgmData*>(new_raw_buffer);
						offset += sizeof(ShaderPrgmData) + shader_prgms[i].char_cnt;

						meshes = reinterpret_cast<MeshData*>(new_raw_buffer + base_offset + offset);
						offset += sizeof(MeshData) + meshes[i].vertex_data.byte_size + meshes[i].index_data.byte_size;

						draw_commands = reinterpret_cast<DrawCommandData*>(new_raw_buffer + base_offset + offset);
						offset += sizeof(DrawCommandData) + sizeof(DrawCommandData::DrawElementsCommand) * draw_commands[i].draw_cnt;

						mesh_shader_params = reinterpret_cast<MeshShaderParams*>(new_raw_buffer + base_offset + offset);
						offset += sizeof(MeshShaderParams) + mesh_shader_params[i].byte_size;

						mtl_shader_params = reinterpret_cast<MaterialShaderParams*>(new_raw_buffer + base_offset + offset);
						offset += sizeof(MaterialShaderParams);

						update_flags = reinterpret_cast<uint32_t*>(new_raw_buffer + base_offset + offset);
						offset += sizeof(uint32_t);

						base_offset += offset;
						offset = 0;
					}

					delete raw_buffer;
					raw_buffer = new_raw_buffer;
				}

				void addBatch(ShaderPrgmData shader_prgm,
					MeshData mesh_data,
					DrawCommandData draw_commands,
					MeshShaderParams mesh_shader_params,
					MaterialShaderParams)
				{
					//TODO calculate byte size of batch

					//TODO check if allocated size fits old data + new data

					//TODO add new render batch
				}

			};
		
		public:
			CallNGMeshRenderBatches();
			~CallNGMeshRenderBatches();
		
		
			/**
			* Answer the name of the objects of this description.
			*
			* @return The name of the objects of this description.
			*/
			static const char *ClassName(void) {
				return "CallNGMeshRenderBatches";
			}
		
			/**
			* Gets a human readable description of the module.
			*
			* @return A human readable description of the module.
			*/
			static const char *Description(void) {
				return "Call transporting mesh data organized in batches for rendering.";
			}
		
			/**
			* Answer the number of functions used for this call.
			*
			* @return The number of functions used for this call.
			*/
			static unsigned int FunctionCount(void) {
				return AbstractGetData3DCall::FunctionCount();
			}
		
			/**
			* Answer the name of the function used for this call.
			*
			* @param idx The index of the function to return it's name.
			*
			* @return The name of the requested function.
			*/
			static const char * FunctionName(unsigned int idx) {
				return AbstractGetData3DCall::FunctionName(idx);
			}

			void setRenderBatches(RenderBatchesData const* render_batches)
			{
				m_render_batches = render_batches;
			}

			RenderBatchesData const* getRenderBatches()
			{
				return m_render_batches;
			}
		
		private:
		
			RenderBatchesData const* m_render_batches;
		
	};

	/** Description class typedef */
	typedef megamol::core::factories::CallAutoDescription<CallNGMeshRenderBatches> CallNGMeshRenderBatchesDescription;

}
}

#endif