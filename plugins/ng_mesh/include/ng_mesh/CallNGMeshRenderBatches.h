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
#include "vislib/graphics/gl/IncludeAllGL.h"
#include "vislib/assert.h"
#include "vislib/Array.h"
#include "vislib/String.h"
#include "vislib/macro_utils.h"


namespace megamol {
namespace ngmesh {

	class NG_MESH_API CallNGMeshRenderBatches : public megamol::core::AbstractGetData3DCall
	{
		public:

			class NG_MESH_API RenderBatchesData
			{
			public:

				enum UpdateBits {
					SHADER_BIT			= 0x1,
					MESH_BIT			= 0x2,
					DRAWCOMMANDS_BIT	= 0x4,
					MESHPARAMS_BIT		= 0x8,
					MATERIAL_BIT		= 0x10
				};

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

					struct VertexLayoutData
					{
						struct Attribute
						{
							GLint		size;
							GLenum		type;
							GLboolean	normalized;
							GLsizei		offset;
						};

						GLsizei		stride;
						GLuint		attribute_cnt;
						Attribute*	attributes;
					};

					VertexData			vertex_data;
					IndexData			index_data;
					VertexLayoutData	vertex_descriptor;
				};

				struct DrawCommandData
				{
					struct DrawElementsCommand
					{
						GLuint cnt;
						GLuint instance_cnt;
						GLuint first_idx;
						GLuint base_vertex;
						GLuint base_instance;
					};

					DrawElementsCommand*	data;
					GLsizei					draw_cnt;
				};

				struct ObjectShaderParams
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
				
				/*
				* Meta data of the render batches. Stores information on byte sizes and memory locations of actual data.
				*/
				struct Head
				{
					uint8_t*	raw_buffer;
					size_t		used_batch_cnt;
					size_t		available_batch_cnt;

					ShaderPrgmData*			shader_prgms;
					MeshData*				meshes;
					DrawCommandData*		draw_commands;
					ObjectShaderParams*		obj_shader_params;
					MaterialShaderParams*	mtl_shader_params;
					uint32_t*				update_flags;
				};				

				/*
				* Raw data storage of actual render batch data.
				*/
				struct Data
				{
					uint8_t*	raw_buffer;
					size_t		used_byte_size;
					size_t		allocated_byte_size;
				};

				Head m_head;
				Data m_data;

			public:
				RenderBatchesData()
				{
					m_head.raw_buffer = nullptr;
					m_head.used_batch_cnt = 0;
					m_head.available_batch_cnt = 0;

					m_data.raw_buffer = nullptr;
					m_data.used_byte_size = 0;
					m_data.allocated_byte_size = 0;
				}
				~RenderBatchesData()
				{
					if (m_head.raw_buffer != nullptr)
						delete m_head.raw_buffer;

					if (m_data.raw_buffer != nullptr)
						delete m_data.raw_buffer;
				}

				RenderBatchesData& operator=(const RenderBatchesData& rhs)
				{
					delete m_head.raw_buffer;
					delete m_data.raw_buffer;

					// compute new size of head buffer
					size_t byte_size = rhs.m_head.used_batch_cnt * (
						sizeof(ShaderPrgmData)
						+ sizeof(MeshData)
						+ sizeof(DrawCommandData)
						+ sizeof(ObjectShaderParams)
						+ sizeof(MaterialShaderParams)
						+ sizeof(uint32_t));

					m_head.raw_buffer = new uint8_t[byte_size];
					m_head.used_batch_cnt = rhs.m_head.used_batch_cnt;
					m_head.available_batch_cnt = rhs.m_head.used_batch_cnt;

					m_head.shader_prgms = (ShaderPrgmData*)m_head.raw_buffer;
					m_head.meshes = (MeshData*)(m_head.shader_prgms + m_head.used_batch_cnt);
					m_head.draw_commands = (DrawCommandData*)(m_head.meshes + m_head.used_batch_cnt);
					m_head.obj_shader_params = (ObjectShaderParams*)(m_head.draw_commands + m_head.used_batch_cnt);
					m_head.mtl_shader_params = (MaterialShaderParams*)(m_head.obj_shader_params + m_head.used_batch_cnt);
					m_head.update_flags = (uint32_t*)(m_head.mtl_shader_params + m_head.used_batch_cnt);

					std::memcpy(m_head.shader_prgms, rhs.m_head.shader_prgms, m_head.used_batch_cnt * sizeof(ShaderPrgmData));
					std::memcpy(m_head.meshes, rhs.m_head.meshes, m_head.used_batch_cnt * sizeof(MeshData));
					std::memcpy(m_head.draw_commands, rhs.m_head.draw_commands, m_head.used_batch_cnt * sizeof(DrawCommandData));
					std::memcpy(m_head.obj_shader_params, rhs.m_head.obj_shader_params, m_head.used_batch_cnt * sizeof(ObjectShaderParams));
					std::memcpy(m_head.mtl_shader_params, rhs.m_head.mtl_shader_params, m_head.used_batch_cnt * sizeof(MaterialShaderParams));
					std::memcpy(m_head.update_flags, rhs.m_head.update_flags, m_head.used_batch_cnt * sizeof(uint32_t));

					
					m_data.raw_buffer = new uint8_t[rhs.m_data.used_byte_size];
					m_data.used_byte_size = rhs.m_data.used_byte_size;
					m_data.allocated_byte_size = rhs.m_data.allocated_byte_size;
					std::memcpy(m_data.raw_buffer, rhs.m_data.raw_buffer, rhs.m_data.used_byte_size);

					// set head pointers to new buffer
					size_t base_offset = 0;
					size_t offset = 0;
					for (int i = 0; i < m_head.used_batch_cnt; ++i)
					{
						m_head.shader_prgms[i].raw_string = reinterpret_cast<char*>(m_data.raw_buffer + base_offset + offset);
						offset += sizeof(char) + m_head.shader_prgms[i].char_cnt;

						m_head.meshes[i].vertex_data.raw_data = reinterpret_cast<uint8_t*>(m_data.raw_buffer + base_offset + offset);
						offset += m_head.meshes[i].vertex_data.byte_size;
						m_head.meshes[i].index_data.raw_data = reinterpret_cast<uint8_t*>(m_data.raw_buffer + base_offset + offset);
						offset += m_head.meshes[i].index_data.byte_size;
						m_head.meshes[i].vertex_descriptor.attributes = reinterpret_cast<MeshData::VertexLayoutData::Attribute*>(m_data.raw_buffer + base_offset + offset);
						offset += sizeof(MeshData::VertexLayoutData::Attribute) * m_head.meshes[i].vertex_descriptor.attribute_cnt;

						m_head.draw_commands[i].data = reinterpret_cast<DrawCommandData::DrawElementsCommand*>(m_data.raw_buffer + base_offset + offset);
						offset += sizeof(DrawCommandData::DrawElementsCommand) * m_head.draw_commands[i].draw_cnt;

						m_head.obj_shader_params[i].raw_data = reinterpret_cast<uint8_t*>(m_data.raw_buffer + base_offset + offset);
						offset += m_head.obj_shader_params[i].byte_size;

						m_head.mtl_shader_params[i].data = reinterpret_cast<MaterialParameters*>(m_data.raw_buffer + base_offset + offset);
						offset += sizeof(MaterialParameters) * m_head.mtl_shader_params[i].elements_cnt;

						base_offset += offset;
						offset = 0;
					}

					return *this;
				}

				RenderBatchesData(const RenderBatchesData& cpy) = delete;
				RenderBatchesData(RenderBatchesData&& other) = delete;
				RenderBatchesData& operator=(RenderBatchesData&& rhs) = delete;
				
				void reallocateHeadBuffer(size_t new_batch_cnt)
				{
					if (new_batch_cnt <= m_head.used_batch_cnt)
					{
						vislib::sys::Log::DefaultLog.WriteError("Reallocation size for RenderBatches not feasible");
						return;
					}
	
					Head new_head;

					// compute new size of head buffer
					size_t byte_size = new_batch_cnt * (
						sizeof(ShaderPrgmData)
						+ sizeof(MeshData)
						+ sizeof(DrawCommandData)
						+ sizeof(ObjectShaderParams)
						+ sizeof(MaterialShaderParams)
						+ sizeof(uint32_t));

					new_head.raw_buffer = new uint8_t[byte_size];
					new_head.used_batch_cnt = m_head.used_batch_cnt;
					new_head.available_batch_cnt = new_batch_cnt;

					new_head.shader_prgms = (ShaderPrgmData*) new_head.raw_buffer;
					new_head.meshes = (MeshData*)(new_head.shader_prgms + new_batch_cnt);
					new_head.draw_commands = (DrawCommandData*)(new_head.meshes + new_batch_cnt);
					new_head.obj_shader_params = (ObjectShaderParams*)(new_head.draw_commands + new_batch_cnt);
					new_head.mtl_shader_params = (MaterialShaderParams*)(new_head.obj_shader_params + new_batch_cnt);
					new_head.update_flags = (uint32_t*)(new_head.mtl_shader_params + new_batch_cnt);

					std::memcpy(new_head.shader_prgms, m_head.shader_prgms, m_head.used_batch_cnt * sizeof(ShaderPrgmData));
					std::memcpy(new_head.meshes, m_head.meshes, m_head.used_batch_cnt * sizeof(MeshData));
					std::memcpy(new_head.draw_commands, m_head.draw_commands, m_head.used_batch_cnt * sizeof(DrawCommandData));
					std::memcpy(new_head.obj_shader_params, m_head.obj_shader_params, m_head.used_batch_cnt * sizeof(ObjectShaderParams));
					std::memcpy(new_head.mtl_shader_params, m_head.mtl_shader_params, m_head.used_batch_cnt * sizeof(MaterialShaderParams));
					std::memcpy(new_head.update_flags, m_head.update_flags, m_head.used_batch_cnt * sizeof(uint32_t));

					delete m_head.raw_buffer;
					m_head = new_head;
				}

				void reallocateDataBuffer(size_t new_byte_size)
				{
					if (new_byte_size <= m_data.allocated_byte_size)
					{
						vislib::sys::Log::DefaultLog.WriteError("Reallocation size for RenderBatches not feasible");
						return;
					}

					Data new_data;
					new_data.raw_buffer = new uint8_t[new_byte_size];
					new_data.used_byte_size = m_data.used_byte_size;
					new_data.allocated_byte_size = new_byte_size;
					std::memcpy(new_data.raw_buffer, m_data.raw_buffer, m_data.used_byte_size);

					// set head pointers to new buffer
					size_t base_offset = 0;
					size_t offset = 0;
					for (int i = 0; i < m_head.used_batch_cnt; ++i)
					{
						m_head.shader_prgms[i].raw_string = reinterpret_cast<char*>(new_data.raw_buffer + base_offset + offset);
						offset += sizeof(char) + m_head.shader_prgms[i].char_cnt;

						m_head.meshes[i].vertex_data.raw_data = reinterpret_cast<uint8_t*>(new_data.raw_buffer + base_offset + offset);
						offset += m_head.meshes[i].vertex_data.byte_size;
						m_head.meshes[i].index_data.raw_data = reinterpret_cast<uint8_t*>(new_data.raw_buffer + base_offset + offset);
						offset += m_head.meshes[i].index_data.byte_size;
						m_head.meshes[i].vertex_descriptor.attributes = reinterpret_cast<MeshData::VertexLayoutData::Attribute*>(new_data.raw_buffer + base_offset + offset);
						offset += sizeof(MeshData::VertexLayoutData::Attribute) * m_head.meshes[i].vertex_descriptor.attribute_cnt;

						m_head.draw_commands[i].data = reinterpret_cast<DrawCommandData::DrawElementsCommand*>(new_data.raw_buffer + base_offset + offset);
						offset += sizeof(DrawCommandData::DrawElementsCommand) * m_head.draw_commands[i].draw_cnt;

						m_head.obj_shader_params[i].raw_data = reinterpret_cast<uint8_t*>(new_data.raw_buffer + base_offset + offset);
						offset += m_head.obj_shader_params[i].byte_size;

						m_head.mtl_shader_params[i].data = reinterpret_cast<MaterialParameters*>(new_data.raw_buffer + base_offset + offset);
						offset += sizeof(MaterialParameters) * m_head.mtl_shader_params[i].elements_cnt;

						base_offset += offset;
						offset = 0;
					}

					delete m_data.raw_buffer;
					m_data = new_data;
				}

				void addBatch(
					ShaderPrgmData			shader_prgm,
					MeshData				mesh_data,
					DrawCommandData			draw_commands,
					ObjectShaderParams		obj_shader_params,
					MaterialShaderParams	mtl_shader_params)
				{
					if (m_head.used_batch_cnt == m_head.available_batch_cnt)
						reallocateHeadBuffer(m_head.available_batch_cnt + 5);

					// calculate byte size of batch data
					size_t byte_size = shader_prgm.char_cnt
						+ mesh_data.index_data.byte_size + mesh_data.vertex_data.byte_size
						+ draw_commands.draw_cnt * sizeof(DrawCommandData::DrawElementsCommand)
						+ obj_shader_params.byte_size
						+ mtl_shader_params.elements_cnt * sizeof(MaterialParameters);

					// check if allocated size fits old data + new data
					if ((m_data.used_byte_size + byte_size) > m_data.allocated_byte_size)
						reallocateDataBuffer(m_data.used_byte_size + 2 * byte_size);


					// add new render batch
					++m_head.used_batch_cnt;
					assert(m_head.used_batch_cnt <= m_head.available_batch_cnt);

					size_t idx = m_head.used_batch_cnt -1;
					size_t offset = m_data.used_byte_size;
					m_head.shader_prgms[idx].raw_string = reinterpret_cast<char*>(m_data.raw_buffer + offset);
					m_head.shader_prgms[idx].char_cnt = shader_prgm.char_cnt;
					offset += m_head.shader_prgms[idx].char_cnt;
					std::memcpy(m_head.shader_prgms[idx].raw_string, shader_prgm.raw_string, shader_prgm.char_cnt);

					m_head.meshes[idx].vertex_data.raw_data = reinterpret_cast<uint8_t*>(m_data.raw_buffer + offset);
					m_head.meshes[idx].vertex_data.byte_size = mesh_data.vertex_data.byte_size;
					offset += m_head.meshes[idx].vertex_data.byte_size;
					std::memcpy(m_head.meshes[idx].vertex_data.raw_data, mesh_data.vertex_data.raw_data, mesh_data.vertex_data.byte_size);
					m_head.meshes[idx].index_data.raw_data = reinterpret_cast<uint8_t*>(m_data.raw_buffer + offset);
					m_head.meshes[idx].index_data.byte_size = mesh_data.index_data.byte_size;
					m_head.meshes[idx].index_data.index_type = mesh_data.index_data.index_type;
					offset += m_head.meshes[idx].index_data.byte_size;
					std::memcpy(m_head.meshes[idx].index_data.raw_data, mesh_data.index_data.raw_data, mesh_data.index_data.byte_size);
					m_head.meshes[idx].vertex_descriptor.stride = mesh_data.vertex_descriptor.stride;
					m_head.meshes[idx].vertex_descriptor.attribute_cnt = mesh_data.vertex_descriptor.attribute_cnt;
					m_head.meshes[idx].vertex_descriptor.attributes = reinterpret_cast<MeshData::VertexLayoutData::Attribute*>(m_data.raw_buffer + offset);
					offset += m_head.meshes[idx].vertex_descriptor.attribute_cnt * sizeof(MeshData::VertexLayoutData::Attribute);
					std::memcpy(m_head.meshes[idx].vertex_descriptor.attributes, mesh_data.vertex_descriptor.attributes, mesh_data.vertex_descriptor.attribute_cnt * sizeof(MeshData::VertexLayoutData::Attribute));

					m_head.draw_commands[idx].data = reinterpret_cast<DrawCommandData::DrawElementsCommand*>(m_data.raw_buffer + offset);
					m_head.draw_commands[idx].draw_cnt = draw_commands.draw_cnt;
					offset += sizeof(DrawCommandData::DrawElementsCommand) * m_head.draw_commands[idx].draw_cnt;
					std::memcpy(m_head.draw_commands[idx].data, draw_commands.data, draw_commands.draw_cnt * sizeof(DrawCommandData::DrawElementsCommand));

					m_head.obj_shader_params[idx].raw_data = reinterpret_cast<uint8_t*>(m_data.raw_buffer + offset); 
					m_head.obj_shader_params[idx].byte_size = obj_shader_params.byte_size;
					offset += m_head.obj_shader_params[idx].byte_size;
					std::memcpy(m_head.obj_shader_params[idx].raw_data, obj_shader_params.raw_data, obj_shader_params.byte_size);

					m_head.mtl_shader_params[idx].data = reinterpret_cast<MaterialParameters*>(m_data.raw_buffer + offset);
					m_head.mtl_shader_params[idx].elements_cnt = mtl_shader_params.elements_cnt;
					offset += sizeof(MaterialParameters) * m_head.mtl_shader_params[idx].elements_cnt;
					std::memcpy(m_head.mtl_shader_params[idx].data, mtl_shader_params.data, mtl_shader_params.elements_cnt * sizeof(MaterialParameters));
				}

				void resetUpdateFlags(size_t batch_idx) { m_head.update_flags[batch_idx] = 0; }

				size_t getBatchCount() const { return m_head.used_batch_cnt; }

				ShaderPrgmData const&		getShaderProgramData(size_t batch_idx) const { return m_head.shader_prgms[batch_idx]; }
				MeshData const&				getMeshData(size_t batch_idx) const { return m_head.meshes[batch_idx]; }
				DrawCommandData	const&		getDrawCommandData(size_t batch_idx) const { return m_head.draw_commands[batch_idx]; }
				ObjectShaderParams const&	getObjectShaderParams(size_t batch_idx) const { return m_head.obj_shader_params[batch_idx]; }
				MaterialShaderParams const&	getMaterialShaderParams(size_t batch_idx) const { return m_head.mtl_shader_params[batch_idx]; }
				uint32_t					getUpdateFlags(size_t batch_idx) const { return m_head.update_flags[batch_idx]; }
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

			/**
			 *
			 */
			void setRenderBatches(RenderBatchesData const* render_batches)
			{
				m_render_batches = render_batches;
			}

			/**
			 *
			 */
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