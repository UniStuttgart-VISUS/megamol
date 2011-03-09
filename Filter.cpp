/*
 * Filter.cpp
 *
 * Copyright (C) 2010 by University of Stuttgart (VISUS).
 * All rights reserved.
 */

#include "stdafx.h"
#include "CoreInstance.h"
#include "param/BoolParam.h"
#include "param/ParamSlot.h"
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
    interpolParam("posInterpolation", "Enable positional interpolation between frames" ) {

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
        
    // En-/disable positional interpolation.
    this->interpolParam.SetParameter(new param::BoolParam(true));
    this->MakeSlotAvailable(&this->interpolParam);
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

    // Set the frame ID, calltime and call data.
    molOut->SetFrameID(static_cast<int>(molIn->CallTime()));
    molOut->SetCallTime(molIn->CallTime());
    if(!(*molOut)(MolecularDataCall::CallForGetData)) return false;
    
    // Get positions of the first frame.
    float *pos0 = new float[molOut->AtomCount() * 3];
    memcpy(pos0, molOut->AtomPositions(), molOut->AtomCount()*3*sizeof(float));

    // Set next frame ID and get positions of the second frame.
    if(((static_cast<int>(molIn->CallTime()) + 1) < int(molOut->FrameCount())) &&
        this->interpolParam.Param<param::BoolParam>()->Value()) 
        molOut->SetFrameID(static_cast<int>(molIn->CallTime()) + 1);
    else
        molOut->SetFrameID(static_cast<int>(molIn->CallTime()));
    
    if (!(*molOut)(MolecularDataCall::CallForGetData)) {
        delete[] pos0;
        return false;
    }
    
    float *pos1 = new float[molOut->AtomCount() * 3];
    memcpy(pos1, molOut->AtomPositions(), molOut->AtomCount() * 3 * sizeof( float));

    // Interpolate atom positions between frames.
    this->atomPosInter.Clear();
    this->atomPosInter.SetCount(molOut->AtomCount() * 3);
    
    float inter = molIn->CallTime() - static_cast<float>(static_cast<int>(molIn->CallTime()));
    float threshold = vislib::math::Min(molOut->AccessBoundingBoxes().ObjectSpaceBBox().Width(),
        vislib::math::Min(molOut->AccessBoundingBoxes().ObjectSpaceBBox().Height(),
        molOut->AccessBoundingBoxes().ObjectSpaceBBox().Depth())) * 0.75f;
#pragma omp parallel for
    for(cnt = 0; cnt < int(molOut->AtomCount()); ++cnt ) {
        if( std::sqrt( std::pow( pos0[3*cnt+0] - pos1[3*cnt+0], 2) +
                std::pow( pos0[3*cnt+1] - pos1[3*cnt+1], 2) +
                std::pow( pos0[3*cnt+2] - pos1[3*cnt+2], 2) ) < threshold ) {
            this->atomPosInter[3*cnt+0] = (1.0f - inter) * pos0[3*cnt+0] + inter * pos1[3*cnt+0];
            this->atomPosInter[3*cnt+1] = (1.0f - inter) * pos0[3*cnt+1] + inter * pos1[3*cnt+1];
            this->atomPosInter[3*cnt+2] = (1.0f - inter) * pos0[3*cnt+2] + inter * pos1[3*cnt+2];
        } else if( inter < 0.5f ) {
            this->atomPosInter[3*cnt+0] = pos0[3*cnt+0];
            this->atomPosInter[3*cnt+1] = pos0[3*cnt+1];
            this->atomPosInter[3*cnt+2] = pos0[3*cnt+2];
        } else {
            this->atomPosInter[3*cnt+0] = pos1[3*cnt+0];
            this->atomPosInter[3*cnt+1] = pos1[3*cnt+1];
            this->atomPosInter[3*cnt+2] = pos1[3*cnt+2];
        }
    }
    

    // ...
    
    // Filter
    
    // ...


    molIn->SetUnlocker(new Filter::Unlocker(*molOut));

    // Transfer data from outgoing to incoming data call.
    // *molIn = *molOut; -> kopiert alles von a nach b (kein deep copy sondern die const-pointer)
    molIn->SetDataHash(this->datahash);

    molIn->SetAtoms(molOut->AtomCount(),
                    molOut->AtomTypeCount(),
                    molOut->AtomTypeIndices(),
                    this->atomPosInter.PeekElements(),
                    molOut->AtomTypes(),
                    molOut->AtomResidueIndices(),
                    molOut->AtomBFactors(),
                    molOut->AtomCharges(),
                    molOut->AtomOccupancies());

    molIn->SetBFactorRange(molOut->MinimumBFactor(),
                           molOut->MaximumBFactor());
    
    molIn->SetChargeRange(molOut->MinimumCharge(),
                          molOut->MaximumCharge());
    
    molIn->SetOccupancyRange(molOut->MinimumOccupancy(),
                             molOut->MaximumOccupancy());

    molIn->SetConnections(molOut->ConnectionCount(), molOut->Connection());
                
    molIn->SetResidues(molOut->ResidueCount(),
                       molOut->Residues());
      
    molIn->SetResidueTypeNames(molOut->ResidueTypeNameCount(),
                               molOut->ResidueTypeNames());
    
    molIn->SetMolecules(molOut->MoleculeCount(), molOut->Molecules());
    
    molIn->SetChains(molOut->ChainCount(), molOut->Chains());


    delete[] pos0;
    delete[] pos1;

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

    // TODO: Is this nessecary?
    molIn->SetDataHash(this->datahash);

    return true;
}
