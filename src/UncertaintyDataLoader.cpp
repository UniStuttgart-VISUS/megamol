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
        for (int i = 0; i < this->pdbIndex.Count(); i++) {
            std::cout << "U: ";
            for (int j = 0; j < static_cast<int>(UncertaintyDataCall::secStructure::NoE); j++) {
                std::cout << this->secStructUncertainty[i][j] << " ";
            }
            std::cout << std::endl;
            std::cout << "S: ";
            for (int j = 0; j < static_cast<int>(UncertaintyDataCall::secStructure::NoE); j++) {
                std::cout << this->sortedSecStructUncertainty[i][j] << " ";
            }
            std::cout << std::endl;
        }
        */
    }

    // Pass secondary strucutre data to call, if available
    if( this->pdbIndex.IsEmpty() ) { // ... Assistant example
        return false;
    } else {
        udc->SetDsspSecStructure(&this->dsspSecStructure);
        udc->SetStrideSecStructure(&this->strideSecStructure);
        udc->SetPdbSecStructure(&this->pdbSecStructure);
        udc->SetPdbIndex(&this->pdbIndex);
        udc->SetAminoAcidName(&this->aminoAcidName);
        udc->SetChainID(&this->chainID);
        udc->SetMissingFlag(&this->missingFlag);
        udc->SetSecStructUncertainty(&this->secStructUncertainty);
        udc->SetSortedSecStructTypes(&this->sortedSecStructUncertainty);

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
    this->pdbIndex.Clear();
    this->chainID.Clear();
    this->aminoAcidName.Clear();
    this->missingFlag.Clear();
    this->dsspSecStructure.Clear();
    this->strideSecStructure.Clear();
    this->pdbSecStructure.Clear();

////////////////////////////////////////////////
// TODO: check if filename ends with '.uid' ...
////////////////////////////////////////////////

	// Try to load the file
	if (file.LoadFile( T2A(filename) )) {

////////////////////////////////////////////////
// TODO: check is file contains valid data ...
////////////////////////////////////////////////

        Log::DefaultLog.WriteMsg(Log::LEVEL_INFO, "Opened uncertainty input data file: \"%s\"", T2A(filename.PeekBuffer())); // INFO

        // Reset array size
        // (maximum number of entries in data arrays is ~9 less than line count of file)
        this->pdbSecStructure.AssertCapacity(file.Count());
        this->chainID.AssertCapacity(file.Count());
        this->aminoAcidName.AssertCapacity(file.Count());
        this->missingFlag.AssertCapacity(file.Count());
        this->strideSecStructure.AssertCapacity(file.Count());
        this->dsspSecStructure.AssertCapacity(file.Count());
        this->pdbIndex.AssertCapacity(file.Count());

		// Run through file lines
		lineCnt = 0;
		while (lineCnt < file.Count() && !line.StartsWith("END")) {
            
			line = file.Line(lineCnt);
            
            // Just read lines beginning with DATA
            if (line.StartsWith("DATA")) {
                
                // Truncate line beginning (first 8 charachters), so character 
                // indices of line matches column indices given in input file
			    line = line.Substring(8); 

                // PDB index of amino-acids 
                this->pdbIndex.Add(std::atoi(line.Substring(32,6))); // first parameter of substring is start (beginning with 0), second parameter is range
                // PDB three letter code of amino-acids
                this->aminoAcidName.Add(line.Substring(10,3)); 
                // PDB one letter chain id 
                this->chainID.Add(line[22]);
                // The Missing amino-acid flag
                if (line[26] == 'M')
                    this->missingFlag.Add(true);
                else
                    this->missingFlag.Add(false);
                
                // Translate DSSP one letter secondary structure summary 
                switch (line[228]) {
                    case 'H': this->dsspSecStructure.Add(UncertaintyDataCall::secStructure::H_ALPHA_HELIX); break;
                    case 'G': this->dsspSecStructure.Add(UncertaintyDataCall::secStructure::G_310_HELIX); break;
                    case 'I': this->dsspSecStructure.Add(UncertaintyDataCall::secStructure::I_PI_HELIX); break;
                    case 'E': this->dsspSecStructure.Add(UncertaintyDataCall::secStructure::E_EXT_STRAND); break;
                    case 'B': this->dsspSecStructure.Add(UncertaintyDataCall::secStructure::B_BRIDGE); break;
                    case 'T': this->dsspSecStructure.Add(UncertaintyDataCall::secStructure::T_H_TURN); break;
                    case 'S': this->dsspSecStructure.Add(UncertaintyDataCall::secStructure::S_BEND); break;
                    case ' ': this->dsspSecStructure.Add(UncertaintyDataCall::secStructure::NOTDEFINED); break;
                    default:  this->dsspSecStructure.Add(UncertaintyDataCall::secStructure::NOTDEFINED); break;
                }

                // Translate STRIDE one letter secondary structure
                switch (line[157]) {
                    case 'H': this->strideSecStructure.Add(UncertaintyDataCall::secStructure::H_ALPHA_HELIX); break;
                    case 'G': this->strideSecStructure.Add(UncertaintyDataCall::secStructure::G_310_HELIX); break;
                    case 'I': this->strideSecStructure.Add(UncertaintyDataCall::secStructure::I_PI_HELIX); break;
                    case 'E': this->strideSecStructure.Add(UncertaintyDataCall::secStructure::E_EXT_STRAND); break;
                    case 'B': this->strideSecStructure.Add(UncertaintyDataCall::secStructure::B_BRIDGE); break;
                    case 'b': this->strideSecStructure.Add(UncertaintyDataCall::secStructure::B_BRIDGE); break;
                    case 'T': this->strideSecStructure.Add(UncertaintyDataCall::secStructure::T_H_TURN); break;
                    case 'C': this->strideSecStructure.Add(UncertaintyDataCall::secStructure::C_COIL); break;
                    default:  this->strideSecStructure.Add(UncertaintyDataCall::secStructure::NOTDEFINED); break;
                }

                // Translate first letter of PDB secondary structure definition
                tmpSecStruct = line[44];
                if (tmpSecStruct == 'H') {
                    switch (line[82]) {
                        case '1': this->pdbSecStructure.Add(UncertaintyDataCall::secStructure::H_ALPHA_HELIX); break;  // right-handed-alpha
                        case '2': this->pdbSecStructure.Add(UncertaintyDataCall::secStructure::NOTDEFINED); break;     // right-handed omega
                        case '3': this->pdbSecStructure.Add(UncertaintyDataCall::secStructure::I_PI_HELIX); break;     // right-handed pi
                        case '4': this->pdbSecStructure.Add(UncertaintyDataCall::secStructure::NOTDEFINED); break;     // right-handed gamma
                        case '5': this->pdbSecStructure.Add(UncertaintyDataCall::secStructure::G_310_HELIX); break;    // right-handed 310
                        case '6': this->pdbSecStructure.Add(UncertaintyDataCall::secStructure::H_ALPHA_HELIX); break;  // left-handed alpha
                        case '7': this->pdbSecStructure.Add(UncertaintyDataCall::secStructure::NOTDEFINED); break;     // left-handed omega
                        case '8': this->pdbSecStructure.Add(UncertaintyDataCall::secStructure::NOTDEFINED); break;     // left-handed gamma
                        case '9': this->pdbSecStructure.Add(UncertaintyDataCall::secStructure::NOTDEFINED); break;     // 27 ribbon/helix
                        case '0': this->pdbSecStructure.Add(UncertaintyDataCall::secStructure::NOTDEFINED); break;     // Polyproline 
                        default:  this->pdbSecStructure.Add(UncertaintyDataCall::secStructure::NOTDEFINED); break;
                    }
                }
                else if (tmpSecStruct == 'S'){
                    this->pdbSecStructure.Add(UncertaintyDataCall::secStructure::E_EXT_STRAND);
                }
                else {
                    this->pdbSecStructure.Add(UncertaintyDataCall::secStructure::NOTDEFINED);
                }

                // DEBUG
                // std::cout << "Chain:" << this->chainID.Last() << " - Index:" << this->pdbIndex.Last() << " - Amino-acid:" << this->aminoAcidName.Last() << " - Missing: " << this->missingFlag.Last() << std::endl;
                // std::cout << "DSSP: " << this->dsspSecStructure.Last() << " - STRIDE: " << this->strideSecStructure.Last() << " - PDB: " << this->pdbSecStructure.Last() << std::endl;
            }
			// Next line
			lineCnt++;
		}
        //Clear ascii file buffer
		file.Clear();
		Log::DefaultLog.WriteMsg(Log::LEVEL_INFO, "Read secondary structure for %i amino-acids.", this->pdbIndex.Count()); // INFO
        return true;
	}
	else {
		Log::DefaultLog.WriteMsg(Log::LEVEL_INFO, "Coudn't find uncertainty input data file: \"%s\"", T2A(filename.PeekBuffer())); // INFO
        return false;
	}
}


/*
* UncertaintyDataLoader::computeUncertainty
*/
bool UncertaintyDataLoader::computeUncertainty(void) {
    using vislib::sys::Log;

    // Reset uncertainty data 
    this->secStructUncertainty.Clear();
    this->sortedSecStructUncertainty.Clear();

    if (this->pdbIndex.Count() == 0) { // If no data is present ...
        return false;
    }
    this->secStructUncertainty.AssertCapacity(this->pdbIndex.Count());
    this->sortedSecStructUncertainty.AssertCapacity(this->pdbIndex.Count());


    // initialize structure factors for all three methods
    float pdbStructFactor[UncertaintyDataCall::secStructure::NoE]    = { 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f };
    float strideStructFactor[UncertaintyDataCall::secStructure::NoE] = { 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f };
    float dsspStructFactor[UncertaintyDataCall::secStructure::NoE]   = { 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f };
                                                                      //  H     G     I     E     T     B     S     C     ND  

    // Initialize and calculate uncertainty data
    for (int i = 0; i < this->pdbIndex.Count(); i++) {

        // create new entry for amino-acid
        this->secStructUncertainty.Add(vislib::math::Vector<float, static_cast<int>(UncertaintyDataCall::secStructure::NoE)>());
        this->sortedSecStructUncertainty.Add(vislib::math::Vector<UncertaintyDataCall::secStructure, static_cast<int>(UncertaintyDataCall::secStructure::NoE)>());

        // loop over all possible secondary strucutres
        for (int j = 0; j < static_cast<int>(UncertaintyDataCall::secStructure::NoE); j++) {

            // initialising
            this->secStructUncertainty[i][j] = 0.0f;
            this->sortedSecStructUncertainty[i][j] = static_cast<UncertaintyDataCall::secStructure>(j);

            // apply individual factor
            // PDB

            if (this->pdbSecStructure[i] == static_cast<UncertaintyDataCall::secStructure>(j))
                this->secStructUncertainty[i][j] += pdbStructFactor[j];
            else
                this->secStructUncertainty[i][j] += ((1.0f - pdbStructFactor[j]) / (static_cast<float>(UncertaintyDataCall::secStructure::NoE) - 1.0f));

            // STRIDE
            if (this->strideSecStructure[i] == static_cast<UncertaintyDataCall::secStructure>(j))
                this->secStructUncertainty[i][j] += strideStructFactor[j];
            else
                this->secStructUncertainty[i][j] += ((1.0f - strideStructFactor[j]) / (static_cast<float>(UncertaintyDataCall::secStructure::NoE) - 1.0f));

            //DSSP
            if (this->dsspSecStructure[i] == static_cast<UncertaintyDataCall::secStructure>(j))
                this->secStructUncertainty[i][j] += dsspStructFactor[j];
            else
                this->secStructUncertainty[i][j] += ((1.0f - dsspStructFactor[j]) / (static_cast<float>(UncertaintyDataCall::secStructure::NoE) - 1.0f));

            // normalise
            this->secStructUncertainty[i][j] /= 3.0f; // because there are three methods
        }

        // using quicksort for sorting ...
        this->quickSortUncertainties(&(this->secStructUncertainty[i]), &(this->sortedSecStructUncertainty[i]), 0, (static_cast<int>(UncertaintyDataCall::secStructure::NoE)-1));
    }

    return true;
}


/*
* UncertaintyDataLoader::quickSortUncertainties
*/
void UncertaintyDataLoader::quickSortUncertainties(vislib::math::Vector<float, static_cast<int>(UncertaintyDataCall::secStructure::NoE)> *valueArr,
                                                   vislib::math::Vector<UncertaintyDataCall::secStructure, static_cast<int>(UncertaintyDataCall::secStructure::NoE)> *structArr,
                                                   int left, int right) {
    int i = left;
    int j = right;
    UncertaintyDataCall::secStructure tmpStruct;

    float pivot = valueArr->operator[](static_cast<int>(structArr->operator[]((int)(left + right) / 2)));

    // partition 
    while (i <= j) {

        while (valueArr->operator[](static_cast<int>(structArr->operator[](i))) < pivot)
            i++;
        while (valueArr->operator[](static_cast<int>(structArr->operator[](j))) > pivot)
            j--;
        if (i <= j) {
            // swap elements
            tmpStruct = structArr->operator[](i);
            structArr->operator[](i) = structArr->operator[](j);
            structArr->operator[](j) = tmpStruct;

            i++;
            j--;
        }
    }

    // recursion
    if (left < j) 
        this->quickSortUncertainties(valueArr, structArr, left, j);
        
    if (i < right)
        this->quickSortUncertainties(valueArr, structArr, i, right);
}
