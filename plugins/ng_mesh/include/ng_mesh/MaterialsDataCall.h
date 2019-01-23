/*
* MaterialsDataCall.h
*
* Copyright (C) 2019 by Universitaet Stuttgart (VISUS).
* All rights reserved.
*/

#ifndef MATERIALS_DATA_CALL_H_INCLUDED
#define MATERIALS_DATA_CALL_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "ng_mesh.h"
#include "NGMeshStructs.h"
#include "MaterialDataStorage.h"
#include "mmcore/AbstractGetDataCall.h"

namespace megamol {
	namespace ngmesh {

		// C-Style Accessors for Material Data
		//	struct MaterialDataAccessor
		//	{
		//		char*          btf_filename;
		//		size_t         char_cnt;
		//	
		//		size_t         texture_data_buffer_base_index;
		//		TextureLayout* texture_layouts;
		//		size_t         texture_cnt;
		//	};
		//	
		//	class NG_MESH_API MaterialsDataAccessor
		//	{
		//	public:
		//		MaterialsDataAccessor();
		//		~MaterialsDataAccessor();
		//	
		//	private:
		//		BufferAccessor*       texture_data_buffer_accessors;
		//		size_t                buffer_accessor_cnt;
		//	
		//		MaterialDataAccessor* material_data_accessors;
		//	};

		class NG_MESH_API MaterialsDataCall : public megamol::core::AbstractGetDataCall
		{
		public:
			inline MaterialsDataCall() : AbstractGetDataCall(), m_materials(nullptr) {}
			~MaterialsDataCall() = default;

			/**
			* Answer the name of the objects of this description.
			*
			* @return The name of the objects of this description.
			*/
			static const char *ClassName(void) {
				return "MaterialsDataCall";
			}

			/**
			* Gets a human readable description of the module.
			*
			* @return A human readable description of the module.
			*/
			static const char *Description(void) {
				return "Call that gives access to material data.";
			}

			/**
			* Answer the number of functions used for this call.
			*
			* @return The number of functions used for this call.
			*/
			static unsigned int FunctionCount(void) {
				return AbstractGetDataCall::FunctionCount();
			}

			/**
			* Answer the name of the function used for this call.
			*
			* @param idx The index of the function to return it's name.
			*
			* @return The name of the requested function.
			*/
			static const char * FunctionName(unsigned int idx) {
				return AbstractGetDataCall::FunctionName(idx);
			}

			void setMaterialsData(std::shared_ptr<MaterialsDataStorage> const& materials_data) {
				m_materials = materials_data;
			}

			std::shared_ptr<MaterialsDataStorage> getMaterialsData(){
				return m_materials;
			}

			uint32_t getUpdateFlags() {
				return m_update_flags;
			}

			void resetUpdateFlags() {
				m_update_flags = 0;
			}

		private:
			std::shared_ptr<MaterialsDataStorage> m_materials;
			uint32_t                              m_update_flags;
		};

		/** Description class typedef */
		typedef megamol::core::factories::CallAutoDescription<MaterialsDataCall> MaterialsDataCallDescription;
	}
}

#endif // !MATERIALS_DATA_CALL_H_INCLUDED
