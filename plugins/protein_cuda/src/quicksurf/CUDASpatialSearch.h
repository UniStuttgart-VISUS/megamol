/***************************************************************************
 *cr
 *cr            (C) Copyright 1995-2019 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ***************************************************************************/
/***************************************************************************
 * RCS INFORMATION:
 *
 *      $RCSfile: CUDASpatialSearch.h,v $
 *      $Author: johns $        $Locker:  $             $State: Exp $
 *      $Revision: 1.3 $      $Date: 2020/02/26 19:26:47 $
 *
 ***************************************************************************/
/**
 * \file CUDASpatialSearch.h
 * \brief CUDA kernels that build data structures for spatial sorting, 
 *        hashing, and searching, used by QuickSurf, MDFF, etc.
 */

#define GRID_CELL_EMPTY 0xffffffff

int vmd_cuda_build_density_atom_grid(int natoms,
                                     const float4 * xyzr_d,
                                     const float4 * color_d,
                                     float4 * sorted_xyzr_d,
                                     float4 * sorted_color_d,
                                     unsigned int *atomIndex_d,
                                     unsigned int *sorted_atomIndex_d,
                                     unsigned int *atomHash_d,
                                     uint2 * cellStartEnd_d,
                                     int3 volsz,
                                     float invgridspacing);


int vmd_cuda_build_density_atom_grid(int natoms,
                                     const float4 * xyzr_d,
                                     float4 *& sorted_xyzr_d,
                                     uint2 *& cellStartEnd_d,
                                     int3 volsz,
                                     float invgridspacing);


