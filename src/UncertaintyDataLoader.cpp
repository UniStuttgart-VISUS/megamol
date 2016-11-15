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

#include <iostream> // DEBUG


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
    /** intentionally left empty ... */
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
        
		if(!this->readInputFile(this->filenameSlot.Param<core::param::FilePathParam>()->Value())) {
            return false;
        }
        if (!this->computeUncertainty()) {
            return false;
        }

        // DEBUG
        /*
        udc->SetDsspSecStructure(&this->dsspSecStructure);
        udc->SetStrideSecStructure(&this->strideSecStructure);
        udc->SetPdbSecStructure(&this->pdbSecStructure);
        udc->SetIndexAminoAcidchainID(&this->indexAminoAcidchainID);
        udc->SetSecStructUncertainty(&this->secStructUncertainty);

        for (int i = 0; i < this->indexAminoAcidchainID.Count(); i++) {
            std::cout << "DSSP: " << this->dsspSecStructure[i] << " - STRIDE: " << this->strideSecStructure[i] << " - PDB: " << this->pdbSecStructure[i] << std::endl;
            std::cout << "HELIX:   " << this->secStructUncertainty[i][UncertaintyDataCall::secStructure::HELIX] << std::endl;
            std::cout << "STRAND:  " << this->secStructUncertainty[i][UncertaintyDataCall::secStructure::STRAND] << std::endl;
            std::cout << "COIL:    " << this->secStructUncertainty[i][UncertaintyDataCall::secStructure::COIL] << std::endl;
            std::cout << "NOT DEF: " << this->secStructUncertainty[i][UncertaintyDataCall::secStructure::NOTDEFINED] << std::endl;
            std::cout << "MAX:     " << udc->GetMostLikelySecStructure(i).First() << " - " << udc->GetMostLikelySecStructure(i).Second() << std::endl;
        }
        */
    }
   

    // Pass secondary strucutre data to call, if available
    if( this->indexAminoAcidchainID.IsEmpty() ) { // ... Assistant example
        return false;
    } else {
        udc->SetDsspSecStructure(&this->dsspSecStructure);
        udc->SetStrideSecStructure(&this->strideSecStructure);
        udc->SetPdbSecStructure(&this->pdbSecStructure);
        udc->SetIndexAminoAcidchainID(&this->indexAminoAcidchainID);

        udc->SetSecStructUncertainty(&this->secStructUncertainty);

        return true;
    }
}


/*
* UncertaintyDataLoader::readInputFile
*/
bool UncertaintyDataLoader::readInputFile(const vislib::TString& filename) {
	using vislib::sys::Log;

	//Log::DefaultLog.WriteMsg(Log::LEVEL_INFO, "BEGIN: readInputFile"); // DEBUG

	// Temp variables
	unsigned int                 lineCnt;       // line count of file
	vislib::StringA              line;          // current line of file
    char                         tmpSecStruct;  // ...
    vislib::sys::ASCIIFileBuffer file;          // ascii buffer of file


    // Reset data (or just if new file can be loaded?)
    this->indexAminoAcidchainID.Clear();
    this->dsspSecStructure.Clear();
    this->strideSecStructure.Clear();
    this->pdbSecStructure.Clear();
    // this->secStructProb.Clear();

    // TODO: check if filename ends with '.uid' ...

	// Try to load the file
	if (file.LoadFile( T2A(filename) )) {

        // TODO: check is file contains valid data ...

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
                
                // Translate DSSP one letter secondary structure summary 
                tmpSecStruct = line[228];
                if ((tmpSecStruct == 'H') || (tmpSecStruct == 'G') || (tmpSecStruct == 'I')) {
                    this->dsspSecStructure.Add(UncertaintyDataCall::secStructure::HELIX);
                }
                else if (tmpSecStruct == 'E') {
                    this->dsspSecStructure.Add(UncertaintyDataCall::secStructure::STRAND);
                }
                else if ((tmpSecStruct == 'B') || (tmpSecStruct == 'T') || (tmpSecStruct == 'C')) {
                    this->dsspSecStructure.Add(UncertaintyDataCall::secStructure::COIL);
                }
                else {
                    this->dsspSecStructure.Add(UncertaintyDataCall::secStructure::NOTDEFINED);
                }

                // Translate STRIDE one letter secondary structure
                tmpSecStruct = line[157];
                if ((tmpSecStruct == 'H') || (tmpSecStruct == 'G') || (tmpSecStruct == 'I')) {
                    this->strideSecStructure.Add(UncertaintyDataCall::secStructure::HELIX);
                }
                else if (tmpSecStruct == 'E') {
                    this->strideSecStructure.Add(UncertaintyDataCall::secStructure::STRAND);
                }
                else if ((tmpSecStruct == 'B') || (tmpSecStruct == 'T') || (tmpSecStruct == 'C')) {
                    this->strideSecStructure.Add(UncertaintyDataCall::secStructure::COIL);
                }
                else {
                    this->strideSecStructure.Add(UncertaintyDataCall::secStructure::NOTDEFINED);
                }

                // Translate first letter of PDB secondary structure definition
                tmpSecStruct = line[44];
                if (tmpSecStruct == 'H') {
                    this->pdbSecStructure.Add(UncertaintyDataCall::secStructure::HELIX);
                }
                else if (tmpSecStruct == 'S') {
                    this->pdbSecStructure.Add(UncertaintyDataCall::secStructure::STRAND);
                }
                else if (tmpSecStruct == ' ') {
                    this->pdbSecStructure.Add(UncertaintyDataCall::secStructure::COIL);
                } 
                else {
                    this->pdbSecStructure.Add(UncertaintyDataCall::secStructure::NOTDEFINED);
                }

                // DEBUG
                // std::cout << "Chain:" << this->indexAminoAcidchainID.Last().Second().Second() << "- Index:" << this->indexAminoAcidchainID.Last().First() << "- Amino-acid:" << this->indexAminoAcidchainID.Last().Second().First() << std::endl;
                // std::cout << "DSSP: " << this->dsspSecStructure.Last() << " - STRIDE: " << this->strideSecStructure.Last() << " - PDB: " << this->pdbSecStructure.Last() << std::endl;
            }

			// Next line
			lineCnt++;
		}
        //Clear ascii file buffer
		file.Clear();
		Log::DefaultLog.WriteMsg(Log::LEVEL_INFO, "Read secondary structure for %i amino-acids.", this->indexAminoAcidchainID.Count()); // INFO
        return true;
	}
	else {
		Log::DefaultLog.WriteMsg(Log::LEVEL_INFO, "Coudn't find uncertainty input data file: \"%s\"", T2A(filename)); // INFO
        return false;
	}
}


/*
* UncertaintyDataLoader::computeUncertainty
*/
bool UncertaintyDataLoader::computeUncertainty(void) {
    using vislib::sys::Log;

    // Log::DefaultLog.WriteMsg(Log::LEVEL_INFO, "BEGIN: computeUncertainty"); // DEBUG

    // Reset uncertainty data 
    this->secStructUncertainty.Clear();

    if (this->indexAminoAcidchainID.Count() == 0) { // If no data is present ...
        return false;
    }
    this->secStructUncertainty.AssertCapacity(this->indexAminoAcidchainID.Count());

    // Initialize and calculate uncertainty data
    // Using secStructure (HELIX = 0, STRAND = 1,  COIL = 2, NOTDEFINED = 3) as index for vector:
    for (int i = 0; i < this->indexAminoAcidchainID.Count(); i++) {
        this->secStructUncertainty.Add(vislib::math::Vector<float, 4>(0.0f, 0.0f, 0.0f, 0.0f));

        this->secStructUncertainty[i][pdbSecStructure[i]] += 0.33333333f;
        this->secStructUncertainty[i][strideSecStructure[i]] += 0.33333333f;
        this->secStructUncertainty[i][dsspSecStructure[i]] += 0.33333333f;
    }

    // Log::DefaultLog.WriteMsg(Log::LEVEL_INFO, "END: computeUncertainty"); // DEBUG
    return true;
}