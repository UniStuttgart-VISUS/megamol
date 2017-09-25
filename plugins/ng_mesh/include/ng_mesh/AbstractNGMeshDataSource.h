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

	class AbstractNGMeshDataSource : public core::Module
	{
	public:
		AbstractNGMeshDataSource();

		virtual ~AbstractNGMeshDataSource();

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

		/** The bounding box */
		vislib::math::Cuboid<float> m_bbox;

	private:

		/** The slot for requesting data */
		megamol::core::CalleeSlot m_getData_slot;
	};

}
}

#endif

