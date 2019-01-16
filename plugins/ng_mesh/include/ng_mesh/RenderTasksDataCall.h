/*
* RenderTasksDataCall.h
*
* Copyright (C) 2019 by Universitaet Stuttgart (VISUS).
* All rights reserved.
*/

#ifndef RENDER_TASKS_DATA_CALL_H_INCLUDED
#define RENDER_TASKS_DATA_CALL_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "mmcore/AbstractGetDataCall.h"
#include "ng_mesh.h"
#include "RenderTaskDataStorage.h"

namespace megamol
{
	namespace ngmesh
	{
		class NG_MESH_API RenderTasksDataCall : public megamol::core::AbstractGetDataCall
		{
		public:
			inline RenderTasksDataCall() : AbstractGetDataCall(), m_render_tasks(nullptr) {}
			~RenderTasksDataCall() = default;

			/**
			* Answer the name of the objects of this description.
			*
			* @return The name of the objects of this description.
			*/
			static const char *ClassName(void) {
				return "RenderTasksDataCall";
			}

			/**
			* Gets a human readable description of the module.
			*
			* @return A human readable description of the module.
			*/
			static const char *Description(void) {
				return "Call that gives access to render task data.";
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

		private:
			std::shared_ptr<RenderTaskDataStorage> m_render_tasks;
		};
	}
}

#endif // !RENDER_TASKS_DATA_CALL_H_INCLUDED
