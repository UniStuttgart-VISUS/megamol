/*
* GPURenderTasksDataCall.h
*
* Copyright (C) 2019 by Universitaet Stuttgart (VISUS).
* All rights reserved.
*/

#ifndef CALL_GPU_RENDER_TASK_DATA_H_INCLUDED
#define CALL_GPU_RENDER_TASK_DATA_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "mmcore/AbstractGetData3DCall.h"
#include "mesh.h"
#include "GPURenderTaskCollection.h"

namespace megamol
{
	namespace mesh
	{
		class MESH_API CallGPURenderTaskData : public megamol::core::AbstractGetData3DCall
		{
		public:
			inline CallGPURenderTaskData() : AbstractGetData3DCall(), m_gpu_render_tasks(nullptr) {}
			~CallGPURenderTaskData() {};

			/**
			* Answer the name of the objects of this description.
			*
			* @return The name of the objects of this description.
			*/
			static const char *ClassName(void) {
				return "CallGPURenderTaskData";
			}

			/**
			* Gets a human readable description of the module.
			*
			* @return A human readable description of the module.
			*/
			static const char *Description(void) {
				return "Call that gives access to render tasks.";
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

			void setRenderTaskData(std::shared_ptr<GPURenderTaskCollection> render_tasks) {
				m_gpu_render_tasks = render_tasks;
			}

			std::shared_ptr<GPURenderTaskCollection> getRenderTaskData() {
				return m_gpu_render_tasks;
			}

		private:
			std::shared_ptr<GPURenderTaskCollection> m_gpu_render_tasks;
		};

		/** Description class typedef */
		typedef megamol::core::factories::CallAutoDescription<CallGPURenderTaskData> GPURenderTasksDataCallDescription;
	}
}

#endif // !CALL_GPU_RENDER_TASK_DATA_H_INCLUDED
