/*
 * Filter.cpp
 *
 * Copyright (C) 2010 by University of Stuttgart (VISUS).
 * All rights reserved.
 */

#include "stdafx.h"
#include <omp.h>

#include "CoreInstance.h"
#include "CalleeSlot.h"
#include "CallerSlot.h"
#include "Filter.h"
#include "MolecularDataCall.h"

//#if (defined(WITH_CUDA) && (WITH_CUDA))

//#include "FilterCuda.cuh"

/*#include <cutil_inline.h>
#include <cutil_gl_inline.h>
#include <cuda_gl_interop.h>*/

//#endif // (defined(WITH_CUDA) && (WITH_CUDA))

using namespace megamol;
using namespace megamol::core;
using namespace megamol::protein;


/*
 * Filter::Filter
 */
Filter::Filter(void) : core::Module(), 
    molDataCallerSlot("getData", "Connects the filter with molecule data storage"), 
    dataOutSlot("dataout", "The slot providing the filtered data"),
    calltimeOld(-1.0), atmCnt(0) {

    // Enable caller slot
    this->molDataCallerSlot.SetCompatibleCall<MolecularDataCallDescription>();
    this->MakeSlotAvailable(&this->molDataCallerSlot);

    // Enable dataout slot
    this->dataOutSlot.SetCallback(MolecularDataCall::ClassName(), 
        MolecularDataCall::FunctionName(MolecularDataCall::CallForGetData), 
        &Filter::getData);
    this->dataOutSlot.SetCallback(MolecularDataCall::ClassName(), 
        MolecularDataCall::FunctionName(MolecularDataCall::CallForGetExtent), 
        &Filter::getExtent);
    this->MakeSlotAvailable(&this->dataOutSlot);

}


/*
 * Filter::~Filter
 */
Filter::~Filter(void)  {
    
    this->Release();
}


/*
 * Filter::create
 */
bool Filter::create(void) {

    return true;
}


/*
 * Filter::release
 */
void Filter::release(void) {
    //cudppDestroyPlan(this->sortHandle);
    //cudppDestroyPlan(this->sortConfig);
}


//#if (defined(WITH_CUDA) && (WITH_CUDA))

/*
 * Filter::initCuda
 */
void Filter::initCuda() {


    /*uint3 gridSize;
    uint numCells;
    float3 worldOrigin;
    float3 cellSize;
    uint numBodies;
    float probeRadius;*/

    // Init params 
        
    // Use CUDA device with highest Gflops/s
    //cudaGLSetGLDevice(cutGetMaxGflopsDeviceId());

    // set parameters
    /*this->params.gridSize = gridSize;
    this->params.numCells = numGridCells;
    this->params.numBodies = this->atmCnt;
    this->params.worldOrigin.x = mol->AccessBoundingBoxes().ObjectSpaceBBox().GetLeftBottomBack().GetX();
    this->params.worldOrigin.y = mol->AccessBoundingBoxes().ObjectSpaceBBox().GetLeftBottomBack().GetY();
    this->params.worldOrigin.z = mol->AccessBoundingBoxes().ObjectSpaceBBox().GetLeftBottomBack().GetZ();

    this->params.cellSize.x = worldsize.x / gridSize.x;
    this->params.cellSize.y = worldsize.y / gridSize.y;
    this->params.cellSize.z = worldsize.z / gridSize.z;

    this->params.probeRadius = 3.0;
    this->params.maxNumNeighbors = 50;
    
    this->sortConfig.algorithm = CUDPP_SORT_RADIX;
    this->sortConfig.datatype = CUDPP_UINT;
    this->sortConfig.op = CUDPP_ADD;
    this->sortConfig.options = CUDPP_OPTION_KEY_VALUE_PAIRS;
    
    cudppPlan(&this->sortHandle, this->sortConfig, this->atmCnt - this->solventAtmCnt, 1, 0);*/

}

//#endif // (defined(WITH_CUDA) && (WITH_CUDA))


/*
 * Filter::getData 
 */
bool Filter::getData(megamol::core::Call& call) {
    
    using vislib::sys::Log;
    
    int cnt;
    
    // Get a pointer to the outgoing data call
    MolecularDataCall *molOut = this->molDataCallerSlot.CallAs<MolecularDataCall>();
    if(molOut == NULL) return false;
    
    // Get a pointer to the incoming data call
    MolecularDataCall *molIn = dynamic_cast<MolecularDataCall*>(&call);
    if(molIn == NULL) return false;
    
    // Calltime hasn't changed
    if(molIn->Calltime() == this->calltimeOld) {

        // Set the frame ID, calltime and call data
        molOut->SetFrameID(static_cast<int>(molIn->Calltime()));
        molOut->SetCalltime(molIn->Calltime());
        if(!(*molOut)(MolecularDataCall::CallForGetData)) return false;   

    }
    // Interpolate and filter if calltime has changed
    else {

        // Get the second frame first
        molOut->SetCalltime(molIn->Calltime());
        if(((static_cast<int>(molIn->Calltime()) + 1) < int(molOut->FrameCount()))) 
            molOut->SetFrameID(static_cast<int>(molIn->Calltime()) + 1);
        else
            molOut->SetFrameID(static_cast<int>(molIn->Calltime()));

        if(!(*molOut)(MolecularDataCall::CallForGetData)) return false;
        
        // Check wheter the data source has changed
        if(molOut->AtomCount() != this->atmCnt) {
            updateParams(molOut);
        }
        
        // Get positions of the second frame
        float *pos1 = new float[molOut->AtomCount() * 3];
        memcpy(pos1, molOut->AtomPositions(), molOut->AtomCount()*3*sizeof(float));
        
        // Unlock the second frame and get the first frame
        molOut->Unlock();
        molOut->SetFrameID(static_cast<int>(molIn->Calltime()));
        
        if (!(*molOut)(MolecularDataCall::CallForGetData)) {
            delete[] pos1;
            return false;
        }
        
        float *pos0 = new float[molOut->AtomCount() * 3];
        memcpy(pos0, molOut->AtomPositions(), molOut->AtomCount() * 3 * sizeof( float));
    
        // Interpolate atom positions between frames
        float *posInter = new float [molOut->AtomCount() * 3];
        
        float inter = molIn->Calltime() - static_cast<float>(static_cast<int>(molIn->Calltime()));
        float threshold = vislib::math::Min(molOut->AccessBoundingBoxes().ObjectSpaceBBox().Width(),
            vislib::math::Min(molOut->AccessBoundingBoxes().ObjectSpaceBBox().Height(),
            molOut->AccessBoundingBoxes().ObjectSpaceBBox().Depth())) * 0.75f;
#pragma omp parallel for
        for(cnt = 0; cnt < int(molOut->AtomCount()); ++cnt ) {
            if( std::sqrt( std::pow( pos0[3*cnt+0] - pos1[3*cnt+0], 2) +
                    std::pow( pos0[3*cnt+1] - pos1[3*cnt+1], 2) +
                    std::pow( pos0[3*cnt+2] - pos1[3*cnt+2], 2) ) < threshold ) {
                posInter[3*cnt+0] = (1.0f - inter) * pos0[3*cnt+0] + inter * pos1[3*cnt+0];
                posInter[3*cnt+1] = (1.0f - inter) * pos0[3*cnt+1] + inter * pos1[3*cnt+1];
                posInter[3*cnt+2] = (1.0f - inter) * pos0[3*cnt+2] + inter * pos1[3*cnt+2];
            } else if( inter < 0.5f ) {
                posInter[3*cnt+0] = pos0[3*cnt+0];
                posInter[3*cnt+1] = pos0[3*cnt+1];
                posInter[3*cnt+2] = pos0[3*cnt+2];
            } else {
                posInter[3*cnt+0] = pos1[3*cnt+0];
                posInter[3*cnt+1] = pos1[3*cnt+1];
                posInter[3*cnt+2] = pos1[3*cnt+2];
            }
        }
        
        // Filter 
        
        filterSolventAtoms(molOut, posInter, 50.0f);        
        
        this->calltimeOld = molIn->Calltime();
        
        delete[] pos0;
        delete[] pos1;
        delete[] posInter;       
    }
    
    // Write filter information
    molIn->SetFilter(this->atomVisibility.PeekElements());

    // Transfer data from outgoing to incoming data call
    *molIn = *molOut;

    // Set unlocker object for incoming data call
    molIn->SetUnlocker(new Filter::Unlocker(*molOut));

    return true;
}


/*
 * Filter::getExtent
 */
bool Filter::getExtent(core::Call& call) {

    // Get a pointer to the incoming data call.
    MolecularDataCall *molIn = dynamic_cast<MolecularDataCall*>(&call);
    if(molIn == NULL) return false;
    
    // Get a pointer to the outgoing data call.
    MolecularDataCall *molOut = this->molDataCallerSlot.CallAs<MolecularDataCall>();
    if(molOut == NULL) return false;
    
    // Get extend.
    if(!(*molOut)(MolecularDataCall::CallForGetExtent)) return false;
    
    // TODO: Verify this.
    // Set extend.
    molIn->AccessBoundingBoxes().Clear();
    molIn->SetExtent(molOut->FrameCount(), molOut->AccessBoundingBoxes());

    return true;
}


/*
 * Filter::updateParams
 */
void Filter::updateParams(const MolecularDataCall *mol) {
    
    this->atmCnt = mol->AtomCount();
    this->atomVisibility.SetCount(this->atmCnt);
    
    flagSolventAtoms(mol);

//#if (defined(WITH_CUDA) && (WITH_CUDA))
    
    // set grid dimensions
    /*uint3 gridSize;
    gridSize.x = gridSize.y = gridSize.z = 16;
    unsigned int numGridCells = gridSize.x * gridSize.y * gridSize.z;
    
    float3 worldsize;
    worldsize.x = mol->AccessBoundingBoxes().ObjectSpaceBBox().Width();
    worldsize.y = mol->AccessBoundingBoxes().ObjectSpaceBBox().Height();
    worldsize.z = mol->AccessBoundingBoxes().ObjectSpaceBBox().Depth();
    
    initCuda();*/
    
    // copy arrays etc
    
//#endif // (defined(WITH_CUDA) && (WITH_CUDA))
}


/*
 * Filter::initVisibility
 */
void Filter::initVisibility(bool visibility) {

    for(unsigned int i = 0; i < this->atmCnt; i++) {
        this->atomVisibility[i] = visibility;
    }
}


/*
 * Filter::flagSolventAtoms
 */
void Filter::flagSolventAtoms(const MolecularDataCall *mol) {
    
    unsigned int prot=0, solv=0;
    unsigned int ch, at, res, m;
    
    //this->solventAtmCnt=0;
    this->isSolventAtom.SetCount(mol->AtomCount());


    // Loop through all chains
    for(ch = 0; ch < mol->ChainCount(); ch++) {
        
        // Chain contains solvent atoms
        if(mol->Chains()[ch].Type() == MolecularDataCall::Chain::SOLVENT) {
            
            // Loop through all molecules of this chain
            for(m = mol->Chains()[ch].FirstMoleculeIndex(); 
                m < mol->Chains()[ch].MoleculeCount() + mol->Chains()[ch].FirstMoleculeIndex(); 
                m++) {
                // Loop through all residues of this molecule
                for(res = mol->Molecules()[m].FirstResidueIndex(); 
                    res < mol->Molecules()[m].ResidueCount() + mol->Molecules()[m].FirstResidueIndex(); 
                    res++) {
                    // Loop through all atoms of this residue
                    for(at = mol->Residues()[res]->FirstAtomIndex(); 
                        at < mol->Residues()[res]->AtomCount() + mol->Residues()[res]->FirstAtomIndex();
                        at++) {
                        
                        this->isSolventAtom[at] = true;
                         
                    }
                }
            }
        }
        // Chain doesn't contain solvent atoms
        else { 
            // Loop through all molecules of this chain
            for(m = mol->Chains()[ch].FirstMoleculeIndex(); 
                m < mol->Chains()[ch].MoleculeCount() + mol->Chains()[ch].FirstMoleculeIndex(); 
                m++) {
                // Loop through all residues of this molecule
                for(res = mol->Molecules()[m].FirstResidueIndex(); 
                    res < mol->Molecules()[m].ResidueCount() + mol->Molecules()[m].FirstResidueIndex(); 
                    res++) {
                    // Loop through all atoms of this residue
                    for(at = mol->Residues()[res]->FirstAtomIndex(); 
                        at < mol->Residues()[res]->AtomCount() + mol->Residues()[res]->FirstAtomIndex();
                        at++) {
                            
                        this->isSolventAtom[at] = false;
                    }
                }
            }
        }
    }
}


/*
 * Filter::filerSolventAtoms
 */
void Filter::filterSolventAtoms(MolecularDataCall *mol, float *atomPos, float rad) {

    
//#if (defined(WITH_CUDA) && (WITH_CUDA))

/*    initCuda(mol);
    
    setParameters(&this->params);
    
    unsigned int atomCntProt = this->atmCnt - this->solventAtmCnt;
    
    // Calculate grid hash for non solvent atoms
    float *atomPosProteinD;
    unsigned int *gridAtomHashProteinD;
    unsigned int *gridAtomIndexProteinD;
    
    allocateArray((void **)&atomPosProteinD, sizeof(float)*atomCntProt*3);
    cudaMemcpy(atomPosProteinD, this->atomPosProt.PeekElements(), sizeof(float)*atomCntProt*3, cudaMemcpyHostToDevice);
    
    allocateArray((void **)&gridAtomHashProteinD, sizeof(unsigned int)*atomCntProt);
    allocateArray((void **)&gridAtomIndexProteinD, sizeof(unsigned int)*atomCntProt);
    
    calcHash(gridAtomHashProteinD, gridAtomIndexProteinD, atomPosProteinD, atomCntProt);    

    // Sort atoms based on hash 
    unsigned int gridSortBits = 18;

    cudppSort(this->sortHandle, gridAtomHashProteinD, gridAtomIndexProteinD, gridSortBits, atomCntProt);
    
    // Reorder position array into sorted order and find start and end of each cell 
    unsigned int *cellStartD;
    unsigned int *cellEndD;
    float *atomPosProteinSortedD;
    
    unsigned int gridDim = 16; // TODO: every axis?

    allocateArray((void **)&cellStartD, sizeof(unsigned int) * gridDim * gridDim * gridDim);
    allocateArray((void **)&cellEndD, sizeof(unsigned int) * gridDim * gridDim * gridDim);
    allocateArray((void **)&atomPosProteinSortedD, sizeof(float)*atomCntProt*3);
    
    reorderDataAndFindCellStart(cellStartD, cellEndD, atomPosProteinSortedD, gridAtomHashProteinD,
                                gridAtomIndexProteinD, atomPosProteinD, atomCntProt,
                                gridDim*gridDim*gridDim);
                                
    // Calculate visibility of solvent atoms 
    bool *atomVisibilityD;
    bool *isSolventAtomD;
    float *atomPosD;
    
    allocateArray((void **)&atomVisibilityD, sizeof(bool)*this->atmCnt);
    allocateArray((void **)&atomPosD, sizeof(float)*this->atmCnt*3);
    allocateArray((void **)&isSolventAtomD, sizeof(bool)*this->atmCnt);

    cudaMemcpy(isSolventAtomD, this->isSolventAtom.PeekElements(), sizeof(bool)*this->atmCnt, cudaMemcpyHostToDevice);

    setParameters(&this->params);
    // Calculate visibility
    calcSolventVisibility(cellStartD,
                          cellEndD,
                          atomPosD,
                          atomPosProteinSortedD,
                          isSolventAtomD,
                          this->atmCnt,
                          rad,
                          atomVisibilityD); // output
                               
    // Copy visibility information from device to host
    /*bool *atomVisibilityH = new bool[this->atmCnt];
    cudaMemcpy(atomVisibilityH, atomVisibilityD, sizeof(bool)*this->atmCnt, cudaMemcpyDeviceToHost);
    
    // Clean up
    delete[] atomVisibilityH;
    freeArray(atomPosProteinD);
    freeArray(gridAtomHashProteinD);
    freeArray(gridAtomIndexProteinD);
    freeArray(atomPosD);
    freeArray(atomPosProteinSortedD);
    freeArray(cellStartD);
    freeArray(cellEndD);
    freeArray(atomVisibilityD);
    freeArray(isSolventAtomD);*/
    

//#else // CPU

    unsigned int at, b;
    for(at = 0; at < this->atmCnt; at++) {
        
        if(this->isSolventAtom[at]) {
            
            // Check whether there are non-solvent atoms within the range
            this->atomVisibility[at] = 0;            
            for(b = 0; b < this->atmCnt; b++) {
                if(!this->isSolventAtom[b]) {
                    if(sqrt(pow(atomPos[3*at+0]-atomPos[3*b+0], 2)
                              + pow(atomPos[3*at+1]-atomPos[3*b+1], 2)
                              + pow(atomPos[3*at+2]-atomPos[3*b+2], 2)) <= rad) {
                        
                        this->atomVisibility[at] = 1;
                        break;
                    }
                }
            }           
        }
        else { 
            
            // Non-solvent atoms are visible
            this->atomVisibility[at] = 1; 

        }

    }

//#endif // (defined(WITH_CUDA) && (WITH_CUDA))

}
