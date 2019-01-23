/*
* glTFDataCall.h
*
* Copyright (C) 2019 by Universitaet Stuttgart (VISUS).
* All rights reserved.
*/

#ifndef GLTF_DATA_CALL_H_INCLUDED
#define GLTF_DATA_CALL_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "ng_mesh.h"
#include "mmcore/AbstractGetDataCall.h"

namespace tinygltf {
	class Model;
}

namespace megamol {
	namespace ngmesh {
		class NG_MESH_API GlTFDataCall : public megamol::core::AbstractGetDataCall
		{
		public:
			inline GlTFDataCall() 
				: AbstractGetDataCall(),
				m_gltf_model(nullptr),
				m_update_flag(false) {}
			~GlTFDataCall() = default;

			/**
			* Answer the name of the objects of this description.
			*
			* @return The name of the objects of this description.
			*/
			static const char *ClassName(void) {
				return "GlTFDataCall";
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
		typedef megamol::core::factories::CallAutoDescription<GlTFDataCall> GlTFDataCallDescription;
	}
}


#endif // !GLTF_DATA_CALL_H_INCLUDED
