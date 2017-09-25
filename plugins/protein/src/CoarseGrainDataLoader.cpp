/*
 * CoarseGrainDataLoader.cpp
 *
 * Copyright (C) 2010 by University of Stuttgart (VISUS).
 * All rights reserved.
 */


#include "stdafx.h"
#include "CoarseGrainDataLoader.h"
#include "mmcore/param/FilePathParam.h"
#include "vislib/ArrayAllocator.h"
#include "vislib/sys/Log.h"
#include "vislib/math/mathfunctions.h"
#include "vislib/sys/MemmappedFile.h"
#include "vislib/SmartPtr.h"
#include "vislib/types.h"
#include "vislib/sys/sysfunctions.h"
#include "vislib/StringConverter.h"
#include "vislib/StringTokeniser.h"
#include "vislib/sys/ASCIIFileBuffer.h"
#include <ctime>
#include <iostream>

using namespace megamol;
using namespace megamol::core;
using namespace megamol::protein;

/*
 * protein::CoarseGrainDataLoader::CoarseGrainDataLoader
 */
CoarseGrainDataLoader::CoarseGrainDataLoader(void) : Module(),
		filenameSlot( "filename", "The path to the PDB data file to be loaded"),
        dataOutSlot( "dataout", "The slot providing the loaded data"),
        bbox(-1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f), datahash(0) {
    this->filenameSlot << new param::FilePathParam("");
    this->MakeSlotAvailable( &this->filenameSlot);

    this->dataOutSlot.SetCallback( SphereDataCall::ClassName(), SphereDataCall::FunctionName(SphereDataCall::CallForGetData), &CoarseGrainDataLoader::getData);
    this->dataOutSlot.SetCallback( SphereDataCall::ClassName(), SphereDataCall::FunctionName(SphereDataCall::CallForGetExtent), &CoarseGrainDataLoader::getExtent);
    this->MakeSlotAvailable( &this->dataOutSlot);
}


/*
 * protein::CoarseGrainDataLoader::~CoarseGrainDataLoader
 */
CoarseGrainDataLoader::~CoarseGrainDataLoader(void) {
    this->Release ();
}


/*
 * CoarseGrainDataLoader::create
 */
bool CoarseGrainDataLoader::create(void) {
    // intentionally empty
    return true;
}


/*
 * CoarseGrainDataLoader::getData
 */
bool CoarseGrainDataLoader::getData( core::Call& call) {
    SphereDataCall *dc = dynamic_cast<SphereDataCall*>( &call);
    if ( dc == NULL ) return false;

    if ( this->filenameSlot.IsDirty() ) {
        this->filenameSlot.ResetDirty();
        this->loadFile( this->filenameSlot.Param<core::param::FilePathParam>()->Value());
    }

    if ( dc->FrameID() >= this->data.Count() ) return false;

    dc->SetDataHash( this->datahash);

    // TODO: assign the data from the loader to the call
    dc->SetSpheres( this->sphereCount,
        (float*)this->data[dc->FrameID()].PeekElements(),
        (unsigned int*)this->sphereType.PeekElements(),
        (float*)this->sphereCharge[dc->FrameID()].PeekElements(),
        (unsigned char*)this->sphereColor.PeekElements() );
    dc->SetChargeRange( this->minCharge, this->maxCharge);

    dc->SetUnlocker( NULL);

    return true;
}


/*
 * CoarseGrainDataLoader::getExtent
 */
bool CoarseGrainDataLoader::getExtent( core::Call& call) {
    SphereDataCall *dc = dynamic_cast<SphereDataCall*>( &call);
    if ( dc == NULL ) return false;

    if ( this->filenameSlot.IsDirty() ) {
        this->filenameSlot.ResetDirty();
        this->loadFile( this->filenameSlot.Param<core::param::FilePathParam>()->Value());
    }

    dc->AccessBoundingBoxes().Clear();
    dc->AccessBoundingBoxes().SetObjectSpaceBBox( this->bbox);
    dc->AccessBoundingBoxes().SetObjectSpaceClipBox( this->bbox);

    dc->SetFrameCount( vislib::math::Max(1U, 
        static_cast<unsigned int>( this->data.Count())));

    dc->SetDataHash( this->datahash);

    return true;
}


/*
 * CoarseGrainDataLoader::release
 */
void CoarseGrainDataLoader::release(void) {
    for( unsigned int i = 0; i < this->data.Count(); ++i) {
        this->data[i].Clear();
    }
    this->data.Clear();
}


/*
 * CoarseGrainDataLoader::loadFile
 */
void CoarseGrainDataLoader::loadFile( const vislib::TString& filename) {
    using vislib::sys::Log;

    this->bbox.Set( 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f);
    this->sphereCount = 0;
    for( unsigned int i = 0; i < this->data.Count(); ++i ) {
        this->data[i].Clear();
    }
    this->data.Clear();
    this->datahash++;
    this->maxCharge = this->minCharge = 0.0f;

    time_t t = clock(); // DEBUG

    vislib::StringA line;
    unsigned int sphereCnt, frameCnt, lineCnt;
    
    t = clock(); // DEBUG

    vislib::sys::ASCIIFileBuffer file;

    // try to load the file
    if( file.LoadFile( T2A( filename) ) ) {
        // file successfully loaded, get number of spheres per frame
        if( file.Count() > 0 ) {
            this->sphereCount = vislib::CharTraitsA::ParseInt( file[0]);
            Log::DefaultLog.WriteMsg( Log::LEVEL_INFO, "Sphere count: %i", this->sphereCount ); // DEBUG
        } else {
            return;
        }
        // resize array according to number of spheres per frame
        this->frameCount = static_cast<unsigned int>( file.Count() - 1) / this->sphereCount;
        this->data.SetCount( this->frameCount);
        this->sphereCharge.SetCount( this->frameCount);
        this->sphereColor.SetCount( this->sphereCount * 3);
        this->sphereType.SetCount( this->sphereCount);
        lineCnt = 1;
        // read sphere data
        for( frameCnt = 0; frameCnt < this->frameCount; ++frameCnt ) {
            // assert capacity for the frame to store all spheres (x,y,z,rad)
            this->data[frameCnt].AssertCapacity( this->sphereCount * 4);
            // assert capacity for the frame to store the charge
            this->sphereCharge[frameCnt].AssertCapacity( this->sphereCount);
            // parse each sphere entry (i.e. line)
            for( sphereCnt = 0; sphereCnt < this->sphereCount; ++sphereCnt ) {
                line = file.Line( lineCnt);
                this->parseEntry( line, sphereCnt, frameCnt);
                lineCnt++;
            }
        }
    }

    Log::DefaultLog.WriteMsg( Log::LEVEL_INFO, "Time for loading file: %f", ( double( clock() - t) / double( CLOCKS_PER_SEC) )); // DEBUG

}

/*
 * parse one entry
 */
void CoarseGrainDataLoader::parseEntry( vislib::StringA &entry, unsigned int idx, 
    unsigned int frame) {

    vislib::Array<vislib::StringA> entries = vislib::StringTokeniser<vislib::CharTraitsA>::Split( entry, " ", true);

    // index / ID
    unsigned int id = atoi( entries[0]);
    // type
    unsigned int type = atoi( entries[1]);
    // position
    float posX = (float)atof( entries[2]);
    float posY = (float)atof( entries[3]);
    float posZ = (float)atof( entries[4]);
    // charge
    float charge = (float)atof( entries[5]);
    this->sphereCharge[frame].Add( charge);
    // radius
    float radius = (float)atof( entries[6]);

    // set the data
    this->data[frame].Add( posX);
    this->data[frame].Add( posY);
    this->data[frame].Add( posZ);
    this->data[frame].Add( radius);

    // set colors only for first frame
    if( frame == 0 ) {
        // set min and max charge to initial value
        if( idx == 0 ) 
            this->minCharge = this->maxCharge = charge;
        // set type
        this->sphereType[idx] = type;
        // set colors according to type
        if( type == 0 ) {
            this->sphereColor[3*idx+0] = 37;
            this->sphereColor[3*idx+1] = 136;
            this->sphereColor[3*idx+2] = 195;
        } else if( type == 2 ) {
            this->sphereColor[3*idx+0] = 90;
            this->sphereColor[3*idx+1] = 175;
            this->sphereColor[3*idx+2] = 50;
        } else if( type == 5 ) {
            this->sphereColor[3*idx+0] = 255;
            this->sphereColor[3*idx+1] = 128;
            this->sphereColor[3*idx+2] = 64;
        } else {
            this->sphereColor[3*idx+0] = 125;
            this->sphereColor[3*idx+1] = 125;
            this->sphereColor[3*idx+2] = 125;
        }
    }

    // set charge
    this->minCharge = vislib::math::Min( this->minCharge, charge);
    this->maxCharge = vislib::math::Max( this->maxCharge, charge);

    // update the bounding box
    vislib::math::Cuboid<float> sphereBBox( 
        posX - radius, 
        posY - radius, 
        posZ - radius, 
        posX + radius, 
        posY + radius, 
        posZ + radius);
    if( idx == 0 && frame == 0 ) {
        this->bbox = sphereBBox;
    } else {
        this->bbox.Union( sphereBBox);
    }
}
