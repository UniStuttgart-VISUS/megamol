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

	class AbstractNGMeshDataSource
	{
	public:
		AbstractNGMeshDataSource();
		~AbstractNGMeshDataSource();

	protected:

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
		virtual bool getDataCallback(core::Call& caller);

		/**
		* Gets the data from the source.
		*
		* @param caller The calling call.
		*
		* @return 'true' on success, 'false' on failure.
		*/
		virtual bool getExtentCallback(core::Call& caller);


		CallNGMeshRenderBatches::RenderBatchesData m_render_batches;

	private:

		/** The slot for requesting data */
		megamol::core::CalleeSlot getDataSlot;


	};

}
}

#endif

