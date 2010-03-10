///////////////////////////////////////////////////////////////////////////////
// contourbuildupKernel.cl
//
// Copyright (C) 2010 by Michael Krone <michael.krone@vis.uni-stuttgart.de>
//                         Visualization Research Center (VISUS),
//                         Universitaet Stuttgart, Germany
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

__kernel void sphereCut( unsigned int atomCount, unsigned int neighborCount, 
	__global float4* atomPos, __global float4* neighborPos, 
	__global float4* circles, float probeRad) {
	// get IDs
    int atomId = get_global_id(0);
    int neighborId = get_global_id(1);	
	// compute array index for neighbor 
    int arrayIdx = atomId * neighborCount + neighborId;

}
