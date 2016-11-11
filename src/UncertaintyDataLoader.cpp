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
													   pdbIDSlot("PDB-ID", "The PDB ID for which the uncertainty data should be generated.") {
      
	this->pdbIDSlot << new param::StringParam("");
	this->MakeSlotAvailable(&this->pdbIDSlot);
    
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

	// try to load run python script, if necessary
	if (this->pdbIDSlot.IsDirty()) {
		this->pdbIDSlot.ResetDirty();
		this->readInputFile(this->pdbIDSlot.Param<core::param::StringParam>()->Value());
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
void UncertaintyDataLoader::readInputFile(const vislib::StringA &pdbid) {
	using vislib::sys::Log;

	Log::DefaultLog.WriteMsg(Log::LEVEL_INFO, "FUNC: UncertaintyDataLoader::readInputFile - BEGIN"); // DEBUG

	// temp variables
	unsigned int lineCnt;
	vislib::sys::ASCIIFileBuffer file;
	vislib::StringA line;
	vislib::StringA tmpLine;
    const unsigned int arrayCapacity = 10000;

	vislib::StringA filename = this->fileLocation + pdbid + this->fileEnding;

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
	if (file.LoadFile(filename)) {

        Log::DefaultLog.WriteMsg(Log::LEVEL_INFO, "FUNC: UncertaintyDataLoader::readInputFile - Opened file: \"%s\"", filename); // DEBUG

		// file successfully loaded, read first frame
		lineCnt = 0;
		while (lineCnt < file.Count() && !line.StartsWith("END")) {
			// get the current DATA line from the file
			line = file.Line(lineCnt);
            
            if (line.StartsWith("DATA")) {
			    // extract data from line
			    tmpLine = line.Substring(8); // cut tag so column indices match with column numbers in input file

                this->indexAminoAcidchainID.Add(vislib::Pair<int, vislib::Pair<vislib::StringA, char>>());
                // PDB index of amino-acids is defined as number in columns 28 to 33 (range 6)
                this->indexAminoAcidchainID.Last().First() = std::atoi(tmpLine.Substring(28, 6)); // first parameter of substring is start, second parameter is range
                // PDB three letter code of amino-acids is given in columns 10,11 and 12
                this->indexAminoAcidchainID.Last().Second().First() = tmpLine.Substring(10, 3); 
                // PDB one letter chain id is defined at column 22
                this->indexAminoAcidchainID.Last().Second().Second() = tmpLine[22];
                // take DSSP one letter secondary structure summary which is defined at column 221
                this->dsspSecStructure.Add(tmpLine[221]);
                // STRIDE one letter secondary structure is defined at column 150
                this->strideSecStructure.Add(tmpLine[150]);
                // take first letter of PDB secondary structure definition at column 40
                this->pdbSecStructure.Add(tmpLine[40]);

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

