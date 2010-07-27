/*
 * CCP4VolumeData.cpp
 *
 * Author: Michael Krone
 * Copyright (C) 2010 by Universitaet Stuttgart (VISUS).
 * All rights reserved.
 */


#include "stdafx.h"
#include "CCP4VolumeData.h"
#include "param/FilePathParam.h"
#include "vislib/MemmappedFile.h"
#include "vislib/Log.h"
#include "vislib/IllegalParamException.h"
#include "vislib/IllegalStateException.h"
#include "vislib/mathfunctions.h"
#include "vislib/OutOfRangeException.h"
#include "vislib/sysfunctions.h"
#include "vislib/StringTokeniser.h"
#include <string>
#include <iostream>
#include <fstream>

using namespace megamol;
using namespace megamol::core;


/*
 * protein::CCP4VolumeData::CCP4VolumeData
 */
protein::CCP4VolumeData::CCP4VolumeData(void) : Module (),
        volumeDataCalleeSlot( "providedata", "Connects the rendering with volume data storage"),
		filename( "filename", "The path to the CCP4 volume data file to load."),
        symmetry( 0), map( 0)
{
    CallVolumeDataDescription cpdd;
    this->volumeDataCalleeSlot.SetCallback(cpdd.ClassName(), "GetData", &CCP4VolumeData::VolumeDataCallback);
    this->MakeSlotAvailable(&this->volumeDataCalleeSlot);

    this->filename.SetParameter(new param::FilePathParam(""));
    this->MakeSlotAvailable(&this->filename);
	
}


/*
 * protein::CCP4VolumeData::~CCP4VolumeData
 */
protein::CCP4VolumeData::~CCP4VolumeData(void) {
    this->Release ();
}


/*
 * protein::CCP4VolumeData::VolumeDataCallback
 */
bool protein::CCP4VolumeData::VolumeDataCallback(Call& call) {
    // cast call
    protein::CallVolumeData *volcall = dynamic_cast<protein::CallVolumeData*>(&call);

    if( this->filename.IsDirty())  {
		// load the data.
		this->tryLoadFile();
		this->filename.ResetDirty();
    }

	if( volcall ) {
        // set the volume dimensions
        volcall->SetVolumeDimension( this->header.volDim[0], this->header.volDim[1], this->header.volDim[2]);
        //set the voxel map
        volcall->SetVoxelMapPointer( this->map);
        // get the dimensions and divide them by the number of cells
        vislib::math::Vector<float, 3> cellLength( this->header.cellDim);
        cellLength.SetX( cellLength.GetX() / float( this->header.volDim[0]));
        cellLength.SetY( cellLength.GetY() / float( this->header.volDim[1]));
        cellLength.SetZ( cellLength.GetZ() / float( this->header.volDim[2]));
        // compute the origin
        vislib::math::Vector<float, 3> origin(
            this->header.start[0] * cellLength.X() + this->header.offset[0],
            this->header.start[1] * cellLength.Y() + this->header.offset[1],
            this->header.start[2] * cellLength.Z() + this->header.offset[2]);
        // set bounding box
        volcall->SetBoundingBox( origin.X(), origin.Y(), origin.Z(),
            this->header.cellDim[0] + origin.X(),
            this->header.cellDim[1] + origin.Y(),
            this->header.cellDim[2] + origin.Z() );
        // set density values
        volcall->SetMinimumDensity( this->header.minDensity);
        volcall->SetMaximumDensity( this->header.maxDensity);
        volcall->SetMeanDensity( this->header.meanDensity);
	}

    return true;
}

/*
 *protein::CCP4VolumeData::create
 */
bool protein::CCP4VolumeData::create(void) {
    this->tryLoadFile();
    this->filename.ResetDirty();
    return true;
}


/*
 *protein::CCP4VolumeData::tryLoadFile
 */
bool protein::CCP4VolumeData::tryLoadFile(void) {
    using vislib::sys::Log;

	// clear all containers
	this->ClearData();



	// open file for reading
    vislib::StringA ccp4filename( this->filename.Param<param::FilePathParam>()->Value());
	const char *fn = ccp4filename.PeekBuffer();
    std::ifstream fin( fn, std::ios::binary | std::ios::in );

    // return if file open failed
    if( !fin ) {
        Log::DefaultLog.WriteMsg( Log::LEVEL_ERROR,
            "%s: Unable to open CCP4 input file \"%s\"", this->ClassName(), fn );
        return false;
    }

    // get length of file:
    fin.seekg( 0, std::ios::end);
    unsigned int fileLength = fin.tellg();
    fin.seekg( 0, std::ios::beg);

#define WORD 4

    // read header
    fin.read( (char*)&header, sizeof(header));

    // compute the number of bytes for the map
    unsigned int volBytes = header.volDim[0] * header.volDim[1] * header.volDim[2] * WORD;

    // read symmetry records and map
    if( this->header.mode == 2 ) {
        // check for symmetry records
        if( fileLength > ( sizeof( header) + this->header.nsymbyte + volBytes) ) {
            Log::DefaultLog.WriteMsg( Log::LEVEL_INFO,
                "%s: File is too large, assuming incorrectly set number of symmetry records.", 
                this->ClassName() );
            // compute correct number of bytes for symmetry records
            this->header.nsymbyte = fileLength - ( sizeof( this->header) + volBytes);
            // resize symmetry record array
            if( this->symmetry )
                delete[] this->symmetry;
            this->symmetry = new char[this->header.nsymbyte];
            // read symmetry records
            fin.read( this->symmetry, this->header.nsymbyte);
        } else if( this->header.nsymbyte > 0 ) {
            Log::DefaultLog.WriteMsg( Log::LEVEL_INFO,
                "%s: Reading symmetry records.", this->ClassName() );
            // resize symmetry record array
            if( this->symmetry )
                delete[] this->symmetry;
            this->symmetry = new char[this->header.nsymbyte];
            // read symmetry records
            fin.read( this->symmetry, this->header.nsymbyte);
        }
        // resize map
        if( this->map )
            delete[] this->map;
        this->map = new float[volBytes];
        // read map
        fin.read( (char*)map, volBytes);
    } else {
        Log::DefaultLog.WriteMsg( Log::LEVEL_ERROR,
            "%s: Mode %i not supported.", this->ClassName(), this->header.mode );
        return false;
    }

    // close file
	fin.close();
    Log::DefaultLog.WriteMsg(Log::LEVEL_INFO,
        "%s: File \"%s\" loaded successfully.\n",
        this->ClassName(), fn );

	// return 'true' (everything could be loaded)
	return true;
}


/*
 *protein::CCP4VolumeData::ClearData
 */
void protein::CCP4VolumeData::ClearData() {

}
