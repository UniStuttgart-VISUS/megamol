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

#include <iostream>


using namespace megamol::core;
using namespace megamol::protein_uncertainty;


/*
 * UncertaintyDataLoader::UncertaintyDataLoader (CTOR)
 */
UncertaintyDataLoader::UncertaintyDataLoader( void ) : megamol::core::Module(),
													   dataOutSlot( "dataout", "The slot providing the uncertainty data"),
													   filenameSlot("uidFilename", "The filename of the uncertainty input data file.") {
      
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

	// Get pointer to data call
	UncertaintyDataCall *udc = dynamic_cast<UncertaintyDataCall*>(&call);
    if ( !udc ) return false;

	// If new filename is set ... read new file and calculate uncertainty
	if (this->filenameSlot.IsDirty()) {
		this->filenameSlot.ResetDirty();
        
		this->readInputFile(this->filenameSlot.Param<core::param::FilePathParam>()->Value());   
	}
    
    // Pass data to call, if available
    if( this->indexAminoAcidchainID.IsEmpty() ) { // ... Assistant example
        return false;
    } else {
        udc->SetDsspSecStructure(&this->dsspSecStructure);
        udc->SetStrideSecStructure(&this->strideSecStructure);
        udc->SetPdbSecStructure(&this->pdbSecStructure);
        udc->SetIndexAminoAcidchainID(&this->indexAminoAcidchainID);
        // udc->SetSecStructProb(&this->secStructProb);
        return true;
    }
}


/*
* UncertaintyDataLoader::readInputFile
*/
void UncertaintyDataLoader::readInputFile(const vislib::TString& filename) {
	using vislib::sys::Log;

	//Log::DefaultLog.WriteMsg(Log::LEVEL_INFO, "BEGIN: readInputFile"); // DEBUG

	// Temp variables
	unsigned int                 lineCnt;              // line count of file
	vislib::sys::ASCIIFileBuffer file;                 // ascii buffer of file
	vislib::StringA              line;                 // current line of file

    // Reset data (or just if new file can be loaded?)
    this->indexAminoAcidchainID.Clear();
    this->dsspSecStructure.Clear();
    this->strideSecStructure.Clear();
    this->pdbSecStructure.Clear();
    // this->secStructProb.Clear();

        
	// Try to load the file
	if (file.LoadFile( T2A(filename) )) {

        Log::DefaultLog.WriteMsg(Log::LEVEL_INFO, "Opened uncertainty input data file: \"%s\"", T2A(filename)); // INFO

        // Reset array size
        // (maximum number of entries in data arrays is ~9 less than line count of file)
        this->pdbSecStructure.AssertCapacity(file.Count());
        this->strideSecStructure.AssertCapacity(file.Count());
        this->dsspSecStructure.AssertCapacity(file.Count());
        this->indexAminoAcidchainID.AssertCapacity(file.Count());
        // this->secStructProb.AssertCapacity(file.Count());

		// Run through file lines
		lineCnt = 0;
		while (lineCnt < file.Count() && !line.StartsWith("END")) {
            
			line = file.Line(lineCnt);
            
            // Just read lines beginning with DATA
            if (line.StartsWith("DATA")) {
                
                // Truncate line beginning (first 8 charachters), so character 
                // indices of line matches column indices given in input file
			    line = line.Substring(8); 

                // Add new empty element
                this->indexAminoAcidchainID.Add(vislib::Pair<int, vislib::Pair<vislib::StringA, char>>());
                // PDB index of amino-acids 
                this->indexAminoAcidchainID.Last().First() = std::atoi(line.Substring(32,6)); // first parameter of substring is start (beginning with 0), second parameter is range
                // PDB three letter code of amino-acids
                this->indexAminoAcidchainID.Last().Second().First() = line.Substring(10,3); 
                // PDB one letter chain id 
                this->indexAminoAcidchainID.Last().Second().Second() = line[22];
                
                // Take DSSP one letter secondary structure summary 
                this->dsspSecStructure.Add(line[228]);
                
                // STRIDE one letter secondary structure
                this->strideSecStructure.Add(line[157]);
                
                // Take first letter of PDB secondary structure definition
                this->pdbSecStructure.Add(line[44]);

                // DEBUG
                // std::cout << "Chain:" << this->indexAminoAcidchainID.Last().Second().Second() << "- Index:" << this->indexAminoAcidchainID.Last().First() << "- Amino-acid:" << this->indexAminoAcidchainID.Last().Second().First() << std::endl;
                // std::cout << "DSSP: " << this->dsspSecStructure.Last() << " - STRIDE: " << this->strideSecStructure.Last() << " - PDB: " << this->pdbSecStructure.Last() << std::endl;
            }

			// Next line
			lineCnt++;
		}
        //Clear ascii file buffer
		file.Clear();
		Log::DefaultLog.WriteMsg(Log::LEVEL_INFO, "Read secondary structure for %i amino-acids.", indexAminoAcidchainID.Count()); // INFO

        // recompute uncertainty for new data
        this->computeUncertainty();
	}
	else {
		Log::DefaultLog.WriteMsg(Log::LEVEL_INFO, "Coudn't find uncertainty input data file: \"%s\"", T2A(filename)); // INFO
	}
}


/*
* UncertaintyDataLoader::computeUncertainty
*/
void UncertaintyDataLoader::computeUncertainty(void) {
    using vislib::sys::Log;

    //Log::DefaultLog.WriteMsg(Log::LEVEL_INFO, "BEGIN: computeUncertainty"); // DEBUG
    /*
    if(this->secStructProb == NULL) {
        return;
    }
    */











    //Log::DefaultLog.WriteMsg(Log::LEVEL_INFO, "END: computeUncertainty"); // DEBUG
}