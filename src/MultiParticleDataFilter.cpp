/*
 * MultiParticleDataFilter.cpp
 *
 * Copyright (C) 2013 by Universitaet Stuttgart (VISUS).
 * Author: Michael Krone
 * All rights reserved.
 */

#include "stdafx.h"
#include "MultiParticleDataFilter.h"
#include "mmcore/AbstractGetData3DCall.h"
#include "mmcore/param/FloatParam.h"


using namespace megamol;
using namespace megamol::core;
using namespace megamol::protein_cuda;
using namespace megamol::core::moldyn;


MultiParticleDataFilter::MultiParticleDataFilter(void) :
        getDataSlot("getData", "Calls particle data."),
        dataOutSlot("dataOut", "Provides filtered particle data"),
        thresholdParam("threshold", "The filter threshold.") {
    // the caller slot
    this->getDataSlot.SetCompatibleCall<MultiParticleDataCallDescription>();
    this->MakeSlotAvailable( &this->getDataSlot);

    
    // Multi stream particle data
    this->dataOutSlot.SetCallback(
            core::moldyn::MultiParticleDataCall::ClassName(),
            core::moldyn::MultiParticleDataCall::FunctionName(0),
            &MultiParticleDataFilter::getData);
    this->dataOutSlot.SetCallback(
            core::moldyn::MultiParticleDataCall::ClassName(),
            core::moldyn::MultiParticleDataCall::FunctionName(1),
            &MultiParticleDataFilter::getExtent);
    this->MakeSlotAvailable(&this->dataOutSlot);

    // the filter threshold parameter
    this->thresholdParam.SetParameter( new param::FloatParam( 0.0f));
    this->MakeSlotAvailable( &this->thresholdParam);
    
}


MultiParticleDataFilter::~MultiParticleDataFilter(void) {
}


/*
 * MultiParticleDataFilter::create
 */
bool MultiParticleDataFilter::create(void) {
    return true;
}


/*
 * MultiParticleDataFilter::release
 */
void MultiParticleDataFilter::release(void) {
}


/*
 * VTKLegacyDataLoaderUnstructuredGrid::getData
 */
bool MultiParticleDataFilter::getData(core::Call& call) {
    using namespace vislib::sys;

    // Try to get pointer to unstructured grid call
    core::moldyn::MultiParticleDataCall *mpdc =
        dynamic_cast<core::moldyn::MultiParticleDataCall*>(&call);

    if (mpdc != NULL) {        
        // get pointer to MultiParticleDataCall
        MultiParticleDataCall *data = this->getDataSlot.CallAs<MultiParticleDataCall>();
        if( data == NULL) return false;

        data->SetFrameID( mpdc->FrameID());
        
        if (!(*data)(0)) return false;
        
        mpdc->SetDataHash( data->DataHash());
        mpdc->SetParticleListCount( data->GetParticleListCount());
        // filter the data
        float threshold = this->thresholdParam.Param<param::FloatParam>()->Value();
        unsigned int colDataIdx = 0;
        unsigned int vertDataIdx = 0;
        this->vertexData.SetCount( data->GetParticleListCount());
        this->colorData.SetCount( data->GetParticleListCount());
        for(unsigned int i = 0; i < data->GetParticleListCount(); i++) {
            MultiParticleDataCall::Particles &parts = data->AccessParticles(i);
            this->vertexData[i].Clear();
            this->vertexData[i].AssertCapacity( parts.GetCount() * 3);
            this->colorData[i].Clear();
            this->colorData[i].AssertCapacity( parts.GetCount());
            // only handle COLDATA_FLOAT_I
            // TODO extend
            if( parts.GetColourDataType() != MultiParticleDataCall::Particles::COLDATA_FLOAT_I )
                continue;
            // only handle VERTDATA_FLOAT_XYZ
            // TODO extend
            if( parts.GetVertexDataType() != MultiParticleDataCall::Particles::VERTDATA_FLOAT_XYZ )
                continue;
            for(unsigned int pCnt = 0; pCnt < parts.GetCount(); pCnt++) {
                vertDataIdx = (parts.GetVertexDataStride() == 0) ? pCnt * 3 : pCnt * parts.GetVertexDataStride();
                colDataIdx = (parts.GetColourDataStride() == 0) ? pCnt : pCnt * parts.GetColourDataStride();
                if( ((float*)parts.GetColourData())[colDataIdx] > threshold ) {
                    this->vertexData[i].Add( ((float*)parts.GetVertexData())[vertDataIdx]);
                    this->vertexData[i].Add( ((float*)parts.GetVertexData())[vertDataIdx+1]);
                    this->vertexData[i].Add( ((float*)parts.GetVertexData())[vertDataIdx+2]);
                    this->colorData[i].Add( ((float*)parts.GetColourData())[colDataIdx]);
                }
            }
            // set values
            mpdc->AccessParticles(i).SetGlobalRadius( data->AccessParticles(i).GetGlobalRadius());
            mpdc->AccessParticles(i).SetCount( this->colorData[i].Count());
            mpdc->AccessParticles(i).SetGlobalType(0);
            mpdc->AccessParticles(i).SetColourData(
                core::moldyn::MultiParticleDataCall::Particles::COLDATA_FLOAT_I,
                (const void*)(this->colorData[i].PeekElements()));
            mpdc->AccessParticles(i).SetVertexData(
                core::moldyn::MultiParticleDataCall::Particles::VERTDATA_FLOAT_XYZ,
                (const void*)(this->vertexData[i].PeekElements()));
        }
        data->Unlock();

    } else {
        return false;
    }

    return true;
}


/*
 * VTKLegacyDataLoaderUnstructuredGrid::getExtent
 */
bool MultiParticleDataFilter::getExtent(core::Call& call) {
    AbstractGetData3DCall *cr3d = dynamic_cast<AbstractGetData3DCall *>(&call);
    if( cr3d == NULL ) return false;

    MultiParticleDataCall *data = this->getDataSlot.CallAs<MultiParticleDataCall>();
    if( data == NULL ) return false;
    if (!(*data)(1)) return false;

    cr3d->SetExtent( data->FrameCount(), data->AccessBoundingBoxes());

    return true;
}
