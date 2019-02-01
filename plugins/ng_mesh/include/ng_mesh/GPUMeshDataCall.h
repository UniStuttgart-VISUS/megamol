/*
* GPUMeshDataCall.h
*
* Copyright (C) 2019 by Universitaet Stuttgart (VISUS).
* All rights reserved.
*/

#ifndef GPU_MESH_DATA_CALL_H_INCLUDED
#define GPU_MESH_DATA_CALL_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "ng_mesh.h"
#include "mmcore/AbstractGetData3DCall.h"
#include "GPUMeshDataStorage.h"

namespace megamol {
	namespace ngmesh {

		class GPUMeshDataCall : public megamol::core::AbstractGetData3DCall
		{
		public:
			GPUMeshDataCall() : AbstractGetData3DCall(), m_gpu_meshes(nullptr) {}
			~GPUMeshDataCall() = default;

			/**
			* Answer the name of the objects of this description.
			*
			* @return The name of the objects of this description.
			*/
			static const char *ClassName(void) {
				return "GPUMeshDataCall";
			}

			/**
			* Gets a human readable description of the module.
			*
			* @return A human readable description of the module.
			*/
			static const char *Description(void) {
				return "Call that gives access to meshes stored in batches on the GPU for rendering.";
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

			void setGPUMeshes(GPUMeshDataStorage* gpu_meshes) {
				m_gpu_meshes = gpu_meshes;
			}

			GPUMeshDataStorage* getGPUMeshes() {
				return m_gpu_meshes;
			}

		private:
			GPUMeshDataStorage* m_gpu_meshes;
		};

		/** Description class typedef */
		typedef megamol::core::factories::CallAutoDescription<GPUMeshDataCall> GPUMeshDataCallDescription;

	}
}


#endif // !GPU_MESH_DATA_CALL_H_INCLUDED
