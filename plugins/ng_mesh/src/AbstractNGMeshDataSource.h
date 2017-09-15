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

namespace megamol {
namespace ngmesh {

	class AbstractNGMeshDataSource
	{
	public:
		AbstractNGMeshDataSource();
		~AbstractNGMeshDataSource();

	private:

		/** The slot for requesting data */
		CalleeSlot getDataSlot;


	};

}
}

#endif

