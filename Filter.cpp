/*
 * Filter.cpp
 *
 * Copyright (C) 2010 by University of Stuttgart (VISUS).
 * All rights reserved.
 */

#include "stdafx.h"
#include "CoreInstance.h"
#include "CalleeSlot.h"
#include "CallerSlot.h"
#include "MolecularDataCall.h"
#include <omp.h>
#include "Filter.h"

using namespace megamol;
using namespace megamol::core;
using namespace megamol::protein;

/*
 * Filter::Filter
 */
Filter::Filter(void) : core::Module(), 
    molDataCallerSlot("getData", "Connects the filter with molecule data storage"), 
    dataOutSlot("dataout", "The slot providing the filtered data"),
    calltimeOld(-1.0) {

    // Enable caller slot.
    this->molDataCallerSlot.SetCompatibleCall<MolecularDataCallDescription>();
    this->MakeSlotAvailable(&this->molDataCallerSlot);

    // Enable dataout slot.
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

}

/*
 * Filter::getData 
 */
bool Filter::getData(megamol::core::Call& call) {
    
    using vislib::sys::Log;

    int cnt;
    
    // Get a pointer to the outgoing data call.
    MolecularDataCall *molOut = this->molDataCallerSlot.CallAs<MolecularDataCall>();
    if(molOut == NULL) return false;;
    
    // Get a pointer to the incoming data call.
    MolecularDataCall *molIn = dynamic_cast<MolecularDataCall*>(&call);
    if(molIn == NULL) return false;
    
    // Calltime hasn't changed.
    if(molIn->Calltime() == this->calltimeOld) {

        // Set the frame ID, calltime and call data.
        molOut->SetFrameID(static_cast<int>(molIn->Calltime()));
        molOut->SetCalltime(molIn->Calltime());
        if(!(*molOut)(MolecularDataCall::CallForGetData)) return false;   

    }
    // Interpolate and filter if calltime has changed.
    else {
        
        // Get the second frame first.
        molOut->SetCalltime(molIn->Calltime());
        if(((static_cast<int>(molIn->Calltime()) + 1) < int(molOut->FrameCount()))) 
            molOut->SetFrameID(static_cast<int>(molIn->Calltime()) + 1);
        else
            molOut->SetFrameID(static_cast<int>(molIn->Calltime()));

        if(!(*molOut)(MolecularDataCall::CallForGetData)) return false;   
        
        // Get positions of the second frame.
        float *pos1 = new float[molOut->AtomCount() * 3];
        memcpy(pos1, molOut->AtomPositions(), molOut->AtomCount()*3*sizeof(float));
        
        // Unlock the second frame and get the first frame.
        molOut->Unlock();
        molOut->SetFrameID(static_cast<int>(molIn->Calltime()));
        
        if (!(*molOut)(MolecularDataCall::CallForGetData)) {
            delete[] pos1;
            return false;
        }
        
        float *pos0 = new float[molOut->AtomCount() * 3];
        memcpy(pos0, molOut->AtomPositions(), molOut->AtomCount() * 3 * sizeof( float));
    
        // Interpolate atom positions between frames.
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
        
        // ...
        
        this->calltimeOld = molIn->Calltime();
        
        delete[] pos0;
        delete[] pos1;
        delete[] posInter;

    }
    
    // Write filter information.
    
    // ... 

    // Transfer data from outgoing to incoming data call.
    *molIn = *molOut;

    return true;
}

/*
 * Filter::getExtent
 */
bool Filter::getExtent(core::Call& call) {
    
    using vislib::sys::Log;

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
}
