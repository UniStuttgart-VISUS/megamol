/*
* CallGlTFData.h
*
* Copyright (C) 2019 by Universitaet Stuttgart (VISUS).
* All rights reserved.
*/

#ifndef CALL_GLTF_DATA_H_INCLUDED
#define CALL_GLTF_DATA_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "mesh.h"
#include "mmcore/AbstractGetDataCall.h"

namespace tinygltf {
	class Model;
}

namespace megamol {
	namespace mesh {
		class MESH_API CallGlTFData : public megamol::core::AbstractGetDataCall
		{
		public:
			inline CallGlTFData() 
				: AbstractGetDataCall(),
				m_gltf_model(nullptr),
				m_update_flag(false) {}
			~CallGlTFData() = default;

			/**
			* Answer the name of the objects of this description.
			*
			* @return The name of the objects of this description.
			*/
			static const char *ClassName(void) {
				return "CallGlTFData";
			}

			/**
			* Gets a human readable description of the module.
			*
			* @return A human readable description of the module.
			*/
			static const char *Description(void) {
				return "Call that gives access to a loaded gltf model.";
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

			void setGlTFModel(std::shared_ptr<tinygltf::Model> const& gltf_model);
			
			std::shared_ptr<tinygltf::Model> getGlTFModel();

			void setUpdateFlag();

			bool getUpdateFlag();

			void clearUpdateFlag();

		private:
			std::shared_ptr<tinygltf::Model> m_gltf_model;
			bool                             m_update_flag;
		};

		/** Description class typedef */
		typedef megamol::core::factories::CallAutoDescription<CallGlTFData> CallGlTFDataDescription;
	}
}


#endif // !CALL_GLTF_DATA_H_INCLUDED
