/*
* UncertaintyDataLoader.cpp
*
* Author: Matthias Braun
* Copyright (C) 2016 by Universitaet Stuttgart (VISUS).
* All rights reserved.
*
* This module is based on the source code of "BindingSiteData" in megamol protein plugin (svn revision 1500).
*
*/

#include "stdafx.h"

#include "UncertaintyDataLoader.h"

#include <math.h>
#include <string>

#include "mmcore/CoreInstance.h"
#include "mmcore/param/IntParam.h"
#include "mmcore/param/FilePathParam.h"
#include "mmcore/param/StringParam.h"

#include "vislib/sys/ASCIIFileBuffer.h"
#include "vislib/sys/BufferedFile.h"
#include "vislib/sys/sysfunctions.h"
#include "vislib/math/mathfunctions.h"

#include "UncertaintyDataCall.h"

#include <iostream>

using namespace megamol::core;
using namespace megamol::protein_uncertainty;


/*
 * UncertaintyDataLoader::UncertaintyDataLoader (CTOR)
 */
UncertaintyDataLoader::UncertaintyDataLoader( void ) : megamol::core::Module(),
													   dataOutSlot( "dataout", "The slot providing the uncertainty data"),
													   filenameSlot("filename (uid)", "The filename of the uncertainty input data file.") {
      
	this->filenameSlot << new param::FilePathParam("");
	this->MakeSlotAvailable(&this->filenameSlot);
    
	this->dataOutSlot.SetCallback(UncertaintyDataCall::ClassName(), UncertaintyDataCall::FunctionName(UncertaintyDataCall::CallForGetData), &UncertaintyDataLoader::getData);
    this->MakeSlotAvailable( &this->dataOutSlot);
    
}


/*
 * UncertaintyDataLoader::~UncertaintyDataLoader (DTOR)
 */
UncertaintyDataLoader::~UncertaintyDataLoader( void ) {
    this->Release();
}


/*
 * UncertaintyDataLoader::create
 */
bool UncertaintyDataLoader::create() {
    
    return true;
}


/*
 * UncertaintyDataLoader::release
 */
void UncertaintyDataLoader::release() {

}


/*
 * UncertaintyDataLoader::getData
 */
bool UncertaintyDataLoader::getData(Call& call) {
    using vislib::sys::Log;

	// Log::DefaultLog.WriteMsg(Log::LEVEL_INFO, "FUNC: UncertaintyDataLoader::getData - BEGIN"); // DEBUG

	// ...
	UncertaintyDataCall *udc = dynamic_cast<UncertaintyDataCall*>(&call);
    if ( !udc ) return false;

	// if new filename is set ... read new file and calculate uncertainty
	if (this->filenameSlot.IsDirty()) {
		this->filenameSlot.ResetDirty();
		this->readInputFile(this->filenameSlot.Param<core::param::FilePathParam>()->Value());
        // ...
	}
    
    // pass data to call, if available
    if( this->indexAminoAcidchainID.IsEmpty() ) {
        return false;
    } else {
        udc->SetDsspSecStructure(&this->dsspSecStructure);
        udc->SetStrideSecStructure(&this->strideSecStructure);
        udc->SetPdbSecStructure(&this->pdbSecStructure);
        udc->SetIndexAminoAcidchainID(&this->indexAminoAcidchainID);
        return true;
    }

	// Log::DefaultLog.WriteMsg(Log::LEVEL_INFO, "FUNC: UncertaintyDataLoader::getData - END"); // DEBUG
}


/*
* UncertaintyDataLoader::readInputFile
*/
void UncertaintyDataLoader::readInputFile(const vislib::TString& filename) {
	using vislib::sys::Log;

	Log::DefaultLog.WriteMsg(Log::LEVEL_INFO, "FUNC: UncertaintyDataLoader::readInputFile - BEGIN"); // DEBUG


	// temp variables
	unsigned int                 lineCnt;                               // line count of file
	vislib::sys::ASCIIFileBuffer file;                                  // ascii buffer of file
	vislib::StringA              line;                                  // current line of file
    const unsigned int           arrayCapacity = 1000;                  // default capacity of arrays (why 1000?)


	// reset data
	this->indexAminoAcidchainID.Clear();
    this->indexAminoAcidchainID.AssertCapacity(arrayCapacity);
	this->dsspSecStructure.Clear();
    this->dsspSecStructure.AssertCapacity(arrayCapacity);
	this->strideSecStructure.Clear();
    this->strideSecStructure.AssertCapacity(arrayCapacity);
	this->pdbSecStructure.Clear();
    this->pdbSecStructure.AssertCapacity(arrayCapacity);


	// try to load the file
	if (file.LoadFile( T2A(filename) )) {

        Log::DefaultLog.WriteMsg(Log::LEVEL_INFO, "FUNC: UncertaintyDataLoader::readInputFile - Opened file: \"%s\"", T2A(filename)); // DEBUG

		// file successfully loaded, read first frame
		lineCnt = 0;
		while (lineCnt < file.Count() && !line.StartsWith("END")) {
			// get the current DATA line from the file
			line = file.Line(lineCnt);
            
            if (line.StartsWith("DATA")) {
                
			    line = line.Substring(8); // cut tag so line indices match with column indices given in input file

                if (this->indexAminoAcidchainID.Count() == arrayCapacity) {
                    atomEntriesCapacity += 10000;
                    atomEntries.AssertCapacity(atomEntriesCapacity);
                }
                    
                this->indexAminoAcidchainID.Add(vislib::Pair<int, vislib::Pair<vislib::StringA, char>>());
                
                // PDB index of amino-acids is defined as number in columns 28 to 33 (range 6)
                this->indexAminoAcidchainID.Last().First() = std::atoi(line.Substring(32, 6)); // first parameter of substring is start, second parameter is range
                // PDB three letter code of amino-acids is given in columns 10,11 and 12
                this->indexAminoAcidchainID.Last().Second().First() = line.Substring(10, 3); 
                // PDB one letter chain id is defined at column 22
                this->indexAminoAcidchainID.Last().Second().Second() = line[22];
                
                // take DSSP one letter secondary structure summary which is defined at column 221
                this->dsspSecStructure.Add(line[228]);
                
                // STRIDE one letter secondary structure is defined at column 150
                this->strideSecStructure.Add(line[157]);
                
                // take first letter of PDB secondary structure definition at column 40
                this->pdbSecStructure.Add(line[44]);

                // std::cout << "Chain:" << this->indexAminoAcidchainID.Last().Second().Second() << "- Index:" << this->indexAminoAcidchainID.Last().Second().First() << "- Amino-acid:" << this->indexAminoAcidchainID.Last().First() << std::endl;
                // std::cout << "DSSP: " << this->dsspSecStructure.Last() << " - STRIDE: " << this->strideSecStructure.Last() << " - PDB: " << this->pdbSecStructure.Last() << std::endl;
            }

			// next line
			lineCnt++;
		}
		file.Clear();
		Log::DefaultLog.WriteMsg(Log::LEVEL_INFO, "FUNC: UncertaintyDataLoader::readInputFile - Read secondary structure for %i amino-acids.", indexAminoAcidchainID.Count()); // DEBUG
	}
	else {
		Log::DefaultLog.WriteMsg(Log::LEVEL_INFO, "FUNC: UncertaintyDataLoader::readInputFile - Coudn't find input file: \"%s\"", filename); // DEBUG
	}

	Log::DefaultLog.WriteMsg(Log::LEVEL_INFO, "FUNC: UncertaintyDataLoader::readInputFile - END"); // DEBUG
}

