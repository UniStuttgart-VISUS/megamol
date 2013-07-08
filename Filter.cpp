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
#include "param/EnumParam.h"
#include "param/FloatParam.h"
#include "param/BoolParam.h"
#include "param/IntParam.h"
#include "MolecularDataCall.h"

#include "glh/glh_extensions.h"
#include <GL/glu.h>

#include "Filter.h"

#if (defined(WITH_CUDA) && (WITH_CUDA))

#include "cuda_helper.h"
#include <cuda_gl_interop.h>

#include "filter_cuda.cuh"
#include "particles_kernel.cuh"
#include "particleSystem.cuh"

#include <vector_types.h>

#endif // (defined(WITH_CUDA) && (WITH_CUDA))

using namespace megamol;
using namespace megamol::core;
using namespace megamol::protein;


/*
 * Filter::Filter
 */
Filter::Filter(void) : core::Module(), 
    molDataCallerSlot("getData", "Connects the filter with molecule data storage"), 
    dataOutSlot("dataout", "The slot providing the filtered data"),
    hierarchyParam("hierarchy", "Bottom-up/Top-down parsing of the hierarchy"),
    solvRadiusParam("solvRadius", "Range of solvent atom neighbourhood filtering"),
    interpolParam("posInterpolation", "Enable positional interpolation between frames"),
    gridSizeParam("cudaGridSize", "Gridsize"),
    filterParam("filter", "Choose the filter to be used"),
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
    
    // Set hierarchical visibility information
    param::EnumParam *h = new param::EnumParam(TOPDOWN);
    h->SetTypePair(TOPDOWN, "TopDown");
    h->SetTypePair(BOTTOMUP, "BottomUp");
    this->hierarchyParam << h;
    this->MakeSlotAvailable(&this->hierarchyParam);
    
    // Set range of solvent atom neighbourhood
    this->solvRadiusParam.SetParameter(new param::FloatParam(75.0f, 0.0f));
    this->MakeSlotAvailable(&this->solvRadiusParam);
    
    // En-/disable positional interpolation
    this->interpolParam.SetParameter(new param::BoolParam(true));
    this->MakeSlotAvailable(&this->interpolParam);
    
    param::EnumParam *gs = new param::EnumParam(2);
    gs->SetTypePair(2, "2");
    gs->SetTypePair(4, "4");
    gs->SetTypePair(8, "8");
    gs->SetTypePair(16, "16");
    gs->SetTypePair(32, "32");
    gs->SetTypePair(64, "64");
    this->gridSizeParam << gs;
    
    // Set filter param
    param::EnumParam *f = new param::EnumParam(NONE);
    f->SetTypePair(NONE, "None");
    f->SetTypePair(SOLVENT, "Solvent");
    //f->SetTypePair(SOLVENTALT, "SolventAlt");
    this->filterParam << f;
    this->MakeSlotAvailable(&this->filterParam);
    
    this->atomVisibility = NULL;
    this->atmPosProt     = NULL;
    this->isSolventAtom  = NULL;
    
#if (defined(WITH_CUDA) && (WITH_CUDA))

    this->MakeSlotAvailable(&this->gridSizeParam);
    
    this->atomPosD = NULL;
    this->atomPosProtD = NULL;
    this->atomPosProtSortedD = NULL;
    
    this->gridAtomHashD = NULL;
    this->gridAtomIndexD = NULL;
    
    this->isSolventAtomD = NULL;
    
    this->cellStartD = NULL;
    this->cellEndD = NULL;
    
    this->atomVisibilityD = NULL;
    
    this->neighbourCellPosD = NULL;

#endif // (defined(WITH_CUDA) && (WITH_CUDA))

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
    
    if(this->atomVisibility != NULL) delete[] this->atomVisibility;
    if(this->atmPosProt != NULL) delete[] this->atmPosProt;
    if(this->isSolventAtom != NULL) delete[] this->isSolventAtom;
    
#if (defined(WITH_CUDA) && (WITH_CUDA))

    if(this->atomPosD != NULL) cutilSafeCall(cudaFree(this->atomPosD));
    if(this->atomPosProtD != NULL) cutilSafeCall(cudaFree(this->atomPosProtD));
    if(this->atomPosProtSortedD != NULL) cutilSafeCall(cudaFree(this->atomPosProtSortedD));
    
    if(this->gridAtomHashD != NULL) cutilSafeCall(cudaFree(this->gridAtomHashD));
    if(this->gridAtomIndexD != NULL) cutilSafeCall(cudaFree(this->gridAtomIndexD));
    
    if(this->isSolventAtomD != NULL) cutilSafeCall(cudaFree(this->isSolventAtomD));

    if(this->cellStartD != NULL) cutilSafeCall(cudaFree(this->cellStartD));
    if(this->cellEndD != NULL) cutilSafeCall(cudaFree(this->cellEndD));;    
    
    if(this->atomVisibilityD != NULL) cutilSafeCall(cudaFree(this->atomVisibilityD));
    
    if(this->neighbourCellPosD != NULL) cutilSafeCall(cudaFree(this->neighbourCellPosD));
    
    //cudppDestroyPlan(this->sortHandle);

#endif // (defined(WITH_CUDA) && (WITH_CUDA))

}


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
        molOut->SetFrameID(molIn->FrameID());
        molOut->SetCalltime(molIn->Calltime());
        if(!(*molOut)(MolecularDataCall::CallForGetData)) return false;   

    }
    // Interpolate and filter if calltime has changed
    else {

        // Get the second frame first
        molOut->SetCalltime(molIn->Calltime());
        
        if((molIn->FrameID()+1 < molOut->FrameCount()) &&
            (this->interpolParam.Param<param::BoolParam>()->Value())) { 
            molOut->SetFrameID(molIn->FrameID()+1);
        }
        else {
            molOut->SetFrameID(molIn->FrameID());
        }

        if(!(*molOut)(MolecularDataCall::CallForGetData)) return false;
        
        // Update parameters
        this->updateParams(molOut);
        
        // Get positions of the second frame
        float *pos1 = new float[molOut->AtomCount() * 3];
        memcpy(pos1, molOut->AtomPositions(), molOut->AtomCount()*3*sizeof(float));
        
        // Unlock the second frame and get the first frame
        molOut->Unlock();
        molOut->SetFrameID(molIn->FrameID());
        
        if (!(*molOut)(MolecularDataCall::CallForGetData)) {
            delete[] pos1;
            return false;
        }
        
        float *pos0 = new float[molOut->AtomCount() * 3];
        memcpy(pos0, molOut->AtomPositions(), molOut->AtomCount()*3*sizeof(float));
    
        // Interpolate atom positions between frames
        float *posInter = new float [molOut->AtomCount()*3];
        
        float inter = molIn->Calltime() - static_cast<float>(molIn->FrameID());
        float threshold = vislib::math::Min(molOut->AccessBoundingBoxes().ObjectSpaceBBox().Width(),
                          vislib::math::Min(molOut->AccessBoundingBoxes().ObjectSpaceBBox().Height(),
                          molOut->AccessBoundingBoxes().ObjectSpaceBBox().Depth()))*0.75f;
            
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
        switch(this->filterParam.Param<param::EnumParam>()->Value()) {
            case NONE       : this->initAtomVisibility(1); break;
            case SOLVENT    : this->filterSolventAtoms(posInter); break;
            case SOLVENTALT : this->filterSolventAtomsAlt(posInter); break;
        }
           
        this->calltimeOld = molIn->Calltime();
        
        delete[] pos0;
        delete[] pos1;
        delete[] posInter;       
    }
    
    // Write filter information
    molIn->SetFilter(this->atomVisibility);
    this->setHierarchicalVisibility(molIn);

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
void Filter::updateParams(MolecularDataCall *mol) {
    
    using namespace vislib::sys;
    
    if((mol->AtomCount() != this->atmCnt) 
        || this->solvRadiusParam.IsDirty()
        || this->gridSizeParam.IsDirty()) {
            
        this->solvRadiusParam.ResetDirty();
        this->gridSizeParam.ResetDirty();
    
        this->atmCnt = mol->AtomCount();
    
        if(this->atomVisibility != NULL) delete[] this->atomVisibility;
        this->atomVisibility = new int[this->atmCnt];
    
        this->flagSolventAtoms(mol);
    
        if(this->atmPosProt != NULL) delete[] this->atmPosProt;
        this->atmPosProt = new float[(this->atmCnt - this->solvAtmCnt)*3];
    
#if (defined(WITH_CUDA) && (WITH_CUDA))
        
        // Set parameters
        
        this->params.gridSize.x = this->gridSizeParam.Param<param::EnumParam>()->Value();
        this->params.gridSize.y = this->gridSizeParam.Param<param::EnumParam>()->Value();
        this->params.gridSize.z = this->gridSizeParam.Param<param::EnumParam>()->Value();
        
        this->params.worldOrigin.x = mol->AccessBoundingBoxes().ObjectSpaceBBox().GetLeftBottomBack().GetX();
        this->params.worldOrigin.y = mol->AccessBoundingBoxes().ObjectSpaceBBox().GetLeftBottomBack().GetY();
        this->params.worldOrigin.z = mol->AccessBoundingBoxes().ObjectSpaceBBox().GetLeftBottomBack().GetZ();
        
        this->params.cellSize.x = mol->AccessBoundingBoxes().ObjectSpaceBBox().Width()  / this->params.gridSize.x;
        this->params.cellSize.y = mol->AccessBoundingBoxes().ObjectSpaceBBox().Height() / this->params.gridSize.y;
        this->params.cellSize.z = mol->AccessBoundingBoxes().ObjectSpaceBBox().Depth()  / this->params.gridSize.z;
        
        this->params.numCells   = this->params.gridSize.x * this->params.gridSize.y * this->params.gridSize.z;
        
        this->params.solvRange  = this->solvRadiusParam.Param<param::FloatParam>()->Value();
        this->params.solvRangeSq = this->params.solvRange * this->params.solvRange;
        
        this->params.atmCnt     = this->atmCnt;
        this->params.atmCntProt = this->atmCnt - this->solvAtmCnt;
        
        this->params.discRange.x = ceil(this->params.solvRange / this->params.cellSize.x);
        this->params.discRange.y = ceil(this->params.solvRange / this->params.cellSize.y);
        this->params.discRange.z = ceil(this->params.solvRange / this->params.cellSize.z);
        
        this->params.discRangeWide.x = (this->params.discRange.x*2 + 1);
        this->params.discRangeWide.y = (this->params.discRange.y*2 + 1);
        this->params.discRangeWide.z = (this->params.discRange.z*2 + 1);
        
        this->params.numNeighbours = (this->params.discRange.x*2 + 1)
                                    *(this->params.discRange.y*2 + 1)
                                    *(this->params.discRange.z*2 + 1);
        
        //Log::DefaultLog.WriteMsg(Log::LEVEL_INFO, "numNeighbours: %u",this->params.numNeighbours );   
        
        float cellBodyDiagonal = sqrt(this->params.cellSize.x * this->params.cellSize.x +
                                this->params.cellSize.y * this->params.cellSize.y +
                                this->params.cellSize.z * this->params.cellSize.z);
        
        this->params.innerCellRange = (int)(this->params.solvRange / cellBodyDiagonal); 
        
        setFilterParams(&this->params);
        
        // Create CUDPP radix sort        
        //CUDPPConfiguration sortConfig;
        //sortConfig.algorithm = CUDPP_SORT_RADIX;
        //sortConfig.datatype  = CUDPP_UINT;
        //sortConfig.op        = CUDPP_ADD;
        //sortConfig.options   = CUDPP_OPTION_KEY_VALUE_PAIRS;
        //if(cudppPlan(&this->sortHandle, sortConfig, this->atmCnt, 1, 0) != CUDPP_SUCCESS) {
        //    Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, "Error creating CUDPPPlan");
        //    exit(-1);
        //}

        // Clean up 
        if(this->atomPosD != NULL) cutilSafeCall(cudaFree(this->atomPosD));
        if(this->atomPosProtD != NULL) cutilSafeCall(cudaFree(this->atomPosProtD));
        if(this->atomPosProtSortedD != NULL) cutilSafeCall(cudaFree(this->atomPosProtSortedD));
        
        if(this->gridAtomHashD != NULL) cutilSafeCall(cudaFree(this->gridAtomHashD));
        if(this->gridAtomIndexD != NULL) cutilSafeCall(cudaFree(this->gridAtomIndexD));
        
        if(this->isSolventAtomD != NULL) cutilSafeCall(cudaFree(this->isSolventAtomD));
    
        if(this->cellStartD != NULL) cutilSafeCall(cudaFree(this->cellStartD));
        if(this->cellEndD != NULL) cutilSafeCall(cudaFree(this->cellEndD)); 
        
        if(this->atomVisibilityD != NULL) cutilSafeCall(cudaFree(this->atomVisibilityD));
        if(this->neighbourCellPosD != NULL) cutilSafeCall(cudaFree(this->neighbourCellPosD));

        // Allocate device memory 
        
        cutilSafeCall(cudaMalloc((void **)&this->atomPosD, sizeof(float)*this->atmCnt*3));
        cutilSafeCall(cudaMalloc((void **)&this->atomPosProtD, sizeof(float)*(this->atmCnt - this->solvAtmCnt)*3));   
        cutilSafeCall(cudaMalloc((void **)&this->atomPosProtSortedD, sizeof(float)*(this->atmCnt - this->solvAtmCnt)*3)); 
        
        cutilSafeCall(cudaMalloc((void **)&this->gridAtomHashD, sizeof(unsigned int)*(this->atmCnt - this->solvAtmCnt)));
        cutilSafeCall(cudaMalloc((void **)&this->gridAtomIndexD, sizeof(unsigned int)*(this->atmCnt - this->solvAtmCnt)));
        
        cutilSafeCall(cudaMalloc((void **)&this->isSolventAtomD, sizeof(bool)*this->atmCnt));
        
        cutilSafeCall(cudaMalloc((void **)&this->cellStartD, sizeof(unsigned int)*this->params.numCells));
        cutilSafeCall(cudaMalloc((void **)&this->cellEndD, sizeof(unsigned int)*this->params.numCells));
        
    
        cutilSafeCall(cudaMalloc((void **)&this->atomVisibilityD, sizeof(int)*this->atmCnt));
        
        
        // Calculate positions of neighbour cells in respect to the hash-value
        int *neighbourCellPos = new int[sizeof(int)*3*this->params.numNeighbours];
        for(unsigned int i = 0; i < this->params.numNeighbours; i++) {
            neighbourCellPos[i*3+0] = (i % this->params.discRangeWide.x) - this->params.discRange.x;
            neighbourCellPos[i*3+1] = ((i / this->params.discRangeWide.x) % this->params.discRangeWide.y) - this->params.discRange.y;
            neighbourCellPos[i*3+2] = ((i / this->params.discRangeWide.x) / this->params.discRangeWide.y) - this->params.discRange.z;
        }
               
        cutilSafeCall(cudaMalloc((void **)&this->neighbourCellPosD, sizeof(int)*3*this->params.numNeighbours));
        
        cutilSafeCall(cudaMemcpy(this->neighbourCellPosD, neighbourCellPos, 
            sizeof(int)*3*this->params.numNeighbours, cudaMemcpyHostToDevice));
        
        delete[] neighbourCellPos; 
        
        // Copy data to device memory
        
        cutilSafeCall(cudaMemcpy(this->isSolventAtomD, this->isSolventAtom, 
            sizeof(bool)*this->atmCnt, cudaMemcpyHostToDevice));
    
#endif // (defined(WITH_CUDA) && (WITH_CUDA))
    }
}


/*
 * Filter::initVisibility
 */
void Filter::initAtomVisibility(int visibility) {

    for(unsigned int i = 0; i < this->atmCnt; i++) {
        this->atomVisibility[i] = visibility;
    }
    
}


/*
 * Filter::flagSolventAtoms 
 */
void Filter::flagSolventAtoms(const MolecularDataCall *mol) {
    
    unsigned int ch, at, res, m;
    
    this->solvAtmCnt = 0;
    this->isSolventAtom = new bool[mol->AtomCount()];
    
    // Init
    memset(this->isSolventAtom, 0, mol->AtomCount());

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
                        this->solvAtmCnt++;
                         
                    }
                }
            }
        }
    }
    
    this->protAtmCnt = this->atmCnt - this->solvAtmCnt;
}


/*
 * Filter::getProtAtoms
 */
void Filter::getProtAtoms(float *atomPos) {

    unsigned int at, protCnt = 0;
    
    for(at = 0; at < this->atmCnt; at++){
        if(!this->isSolventAtom[at]) {
            this->atmPosProt[protCnt*3] = atomPos[at*3];
            this->atmPosProt[protCnt*3+1] = atomPos[at*3+1];
            this->atmPosProt[protCnt*3+2] = atomPos[at*3+2];
            protCnt++;
        }
    }
}


/*
 * Filter::setHierarchicalVisibility
 */
void Filter::setHierarchicalVisibility(const MolecularDataCall *mol) {
    
     // TODO: this is ugly!
    
    unsigned int ch, at, res, m;
    
    // Visibility of one child implies the visibility of the parent
    if(this->hierarchyParam.Param<param::EnumParam>()->Value() == BOTTOMUP) {
        
        // Loop through all chains
        for(ch = 0; ch < mol->ChainCount(); ch++) {
    
            const_cast<MolecularDataCall::Chain*>(mol->Chains())[ch].SetFilter(0);
            
            // Loop through all molecules of this chain
            for(m = mol->Chains()[ch].FirstMoleculeIndex(); 
                m < mol->Chains()[ch].MoleculeCount() + mol->Chains()[ch].FirstMoleculeIndex(); 
                m++) {
                    
                const_cast<MolecularDataCall::Molecule*>(mol->Molecules())[m].SetFilter(0);
                
                // Loop through all residues of this molecule
                for(res = mol->Molecules()[m].FirstResidueIndex(); 
                    res < mol->Molecules()[m].ResidueCount() + mol->Molecules()[m].FirstResidueIndex(); 
                    res++) {
                        
                    const_cast<MolecularDataCall::Residue**>(mol->Residues())[res]->SetFilter(0);
                    
                    // Loop through all atoms of this residue
                    for(at = mol->Residues()[res]->FirstAtomIndex(); 
                        at < mol->Residues()[res]->AtomCount() + mol->Residues()[res]->FirstAtomIndex();
                        at++) {
                        
                        if(this->atomVisibility[at] == 1) {
                            const_cast<MolecularDataCall::Residue**>(mol->Residues())[res]->SetFilter(1);
                            break;
                        }
                    }
                    
                    if(mol->Residues()[res]->Filter() == 1) {
                        const_cast<MolecularDataCall::Molecule*>(mol->Molecules())[m].SetFilter(1);
                    }
                }
                
                if(mol->Molecules()[m].Filter() == 1) {
                    const_cast<MolecularDataCall::Chain*>(mol->Chains())[ch].SetFilter(1);
                }
            }
        }
    }
    // Visibility of the parent implies the visibility of all children
    else {
        
        // Loop through all chains
        for(ch = 0; ch < mol->ChainCount(); ch++) {
            
            const_cast<MolecularDataCall::Chain*>(mol->Chains())[ch].SetFilter(1);
            
            // Loop through all molecules of this chain
            for(m = mol->Chains()[ch].FirstMoleculeIndex(); 
                m < mol->Chains()[ch].MoleculeCount() + mol->Chains()[ch].FirstMoleculeIndex(); 
                m++) {
                    
                const_cast<MolecularDataCall::Molecule*>(mol->Molecules())[m].SetFilter(1);
                
                // Loop through all residues of this molecule
                for(res = mol->Molecules()[m].FirstResidueIndex(); 
                    res < mol->Molecules()[m].ResidueCount() + mol->Molecules()[m].FirstResidueIndex(); 
                    res++) {
                        
                    const_cast<MolecularDataCall::Residue**>(mol->Residues())[res]->SetFilter(1);
                    
                    // Loop through all atoms of this residue
                    for(at = mol->Residues()[res]->FirstAtomIndex(); 
                        at < mol->Residues()[res]->AtomCount() + mol->Residues()[res]->FirstAtomIndex();
                        at++) {
                        
                        if(this->atomVisibility[at] == 0) {
                            const_cast<MolecularDataCall::Residue**>(mol->Residues())[res]->SetFilter(0);
                            break;
                        }
                    }
                    
                    if(mol->Residues()[res]->Filter() == 0) {
                        const_cast<MolecularDataCall::Molecule*>(mol->Molecules())[m].SetFilter(0);
                    }
                }
                
                if(mol->Molecules()[m].Filter() == 0) {
                    const_cast<MolecularDataCall::Chain*>(mol->Chains())[ch].SetFilter(0);
                }
            }
        }
    }
}


/*
 * Filter::filerSolventAtoms
 */
void Filter::filterSolventAtomsAlt(float *atomPos) {
    
    using namespace vislib::sys;
    
#if (defined(WITH_CUDA) && (WITH_CUDA)) // GPU

    this->getProtAtoms(atomPos);

    // Get current atom positions
    cutilSafeCall(cudaMemcpy(this->atomPosD, atomPos, sizeof(float)*this->atmCnt*3, 
        cudaMemcpyHostToDevice));  
        
    cutilSafeCall(cudaMemcpy(this->atomPosProtD, this->atmPosProt, sizeof(float)*(this->atmCnt - this->solvAtmCnt)*3, 
        cudaMemcpyHostToDevice));
    
    // Calculate hash grid
    calcFilterHashGrid(this->gridAtomHashD, 
                       this->gridAtomIndexD,
                       this->atomPosProtD,
                       (this->atmCnt - this->solvAtmCnt));
    
        
    // Sort atoms based on hash
    //if(cudppSort(this->sortHandle, this->gridAtomHashD, this->gridAtomIndexD, 18, 
    //    (this->atmCnt - this->solvAtmCnt)) != CUDPP_SUCCESS) {
    //    
    //    Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, "Error performing CUDPPSort");
    //    exit(-1);
    //}
    sortParticles(this->gridAtomHashD, this->gridAtomIndexD, (this->atmCnt - this->solvAtmCnt));
    
        
    // Set all cells to empty
    cutilSafeCall(cudaMemset(this->cellStartD, 0xffffffff, this->params.numCells*sizeof(unsigned int))); 

    // Reorder position array into sorted order and find start and end of each cell
    reorderFilterData(this->cellStartD, 
                      this->cellEndD, 
                      this->gridAtomHashD,
                      this->gridAtomIndexD, 
                      this->atomPosProtD,
                      this->atomPosProtSortedD,
                      (this->atmCnt - this->solvAtmCnt));    
    
    // Set all atoms invisible
    cutilSafeCall(cudaMemset(this->atomVisibilityD, 0x00000000, this->atmCnt*sizeof(int)));
    
    // Calculate visibility of solvent atoms     
    calcSolventVisibilityAlt(this->cellStartD,
                          this->cellEndD,
                          this->atomPosD,
                          this->atomPosProtSortedD,
                          this->isSolventAtomD,
                          this->atomVisibilityD,
                          this->neighbourCellPosD,
                          this->atmCnt,
                          this->params.numNeighbours);
                                               
    
    // Copy visibility information from device to host
    cutilSafeCall(cudaMemcpy(this->atomVisibility, this->atomVisibilityD, 
        sizeof(int)*this->atmCnt, cudaMemcpyDeviceToHost));

#else // CPU

    unsigned int at, b;
    for(at = 0; at < this->atmCnt; at++) {
        
        if(this->isSolventAtom[at]) {
            
            // Check whether there are non-solvent atoms within the range
            this->atomVisibility[at] = 0;            
            for(b = 0; b < this->atmCnt; b++) {
                if(!this->isSolventAtom[b]) {
                    if(sqrt(pow(atomPos[3*at+0]-atomPos[3*b+0], 2)
                              + pow(atomPos[3*at+1]-atomPos[3*b+1], 2)
                              + pow(atomPos[3*at+2]-atomPos[3*b+2], 2)) 
                                <= this->solvRadiusParam.Param<param::FloatParam>()->Value()) {
                        
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

#endif // (defined(WITH_CUDA) && (WITH_CUDA))

}



/*
 * Filter::filerSolventAtoms
 */
void Filter::filterSolventAtoms(float *atomPos) {
    
    using namespace vislib::sys;
    
#if (defined(WITH_CUDA) && (WITH_CUDA)) // GPU

    this->getProtAtoms(atomPos);

    // Get current atom positions
    cutilSafeCall(cudaMemcpy(this->atomPosD, atomPos, sizeof(float)*this->atmCnt*3, 
        cudaMemcpyHostToDevice));  
        
    cutilSafeCall(cudaMemcpy(this->atomPosProtD, this->atmPosProt, sizeof(float)*(this->atmCnt - this->solvAtmCnt)*3, 
        cudaMemcpyHostToDevice));
    
    // Calculate hash grid
    calcFilterHashGrid(this->gridAtomHashD, 
                       this->gridAtomIndexD,
                       this->atomPosProtD,
                       (this->atmCnt - this->solvAtmCnt));
    
        
    // Sort atoms based on hash
    //if(cudppSort(this->sortHandle, this->gridAtomHashD, this->gridAtomIndexD, 18, 
    //    (this->atmCnt - this->solvAtmCnt)) != CUDPP_SUCCESS) {
    //    
    //    Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, "Error performing CUDPPSort");
    //    exit(-1);
    //}
    sortParticles(this->gridAtomHashD, this->gridAtomIndexD, (this->atmCnt - this->solvAtmCnt));
    
        
    // Set all cells to empty
    cutilSafeCall(cudaMemset(this->cellStartD, 0xffffffff, this->params.numCells*sizeof(unsigned int))); 

    // Reorder position array into sorted order and find start and end of each cell
    reorderFilterData(this->cellStartD, 
                      this->cellEndD, 
                      this->gridAtomHashD,
                      this->gridAtomIndexD, 
                      this->atomPosProtD,
                      this->atomPosProtSortedD,
                      (this->atmCnt - this->solvAtmCnt));    
    
    // Set all atoms invisible
    cutilSafeCall(cudaMemset(this->atomVisibilityD, 0x00000000, this->atmCnt*sizeof(int)));
    
    // Calculate visibility of solvent atoms     
    calcSolventVisibility(this->cellStartD,
                          this->cellEndD,
                          this->atomPosD,
                          this->atomPosProtSortedD,
                          this->isSolventAtomD,
                          this->atomVisibilityD,
                          this->atmCnt);
                                               
    
    // Copy visibility information from device to host
    cutilSafeCall(cudaMemcpy(this->atomVisibility, this->atomVisibilityD, 
        sizeof(int)*this->atmCnt, cudaMemcpyDeviceToHost));

#else // CPU

    unsigned int at, b;
    for(at = 0; at < this->atmCnt; at++) {
        
        if(this->isSolventAtom[at]) {
            
            // Check whether there are non-solvent atoms within the range
            this->atomVisibility[at] = 0;            
            for(b = 0; b < this->atmCnt; b++) {
                if(!this->isSolventAtom[b]) {
                    if(sqrt(pow(atomPos[3*at+0]-atomPos[3*b+0], 2)
                              + pow(atomPos[3*at+1]-atomPos[3*b+1], 2)
                              + pow(atomPos[3*at+2]-atomPos[3*b+2], 2)) 
                                <= this->solvRadiusParam.Param<param::FloatParam>()->Value()) {
                        
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

#endif // (defined(WITH_CUDA) && (WITH_CUDA))

}


