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
//////////////////////////////////////////////////////////////////////////////////////////////
//
// TODO:
//
// - Sekundärstruktur: Länge
//
//////////////////////////////////////////////////////////////////////////////////////////////


#include "stdafx.h"

#include "UncertaintyDataLoader.h"

#include <math.h>
#include <string>

#include "mmcore/CoreInstance.h"
#include "mmcore/param/IntParam.h"
#include "mmcore/param/FilePathParam.h"
#include "mmcore/param/StringParam.h"
#include "mmcore/param/EnumParam.h"

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
													   filenameSlot("uidFilename", "The filename of the uncertainty input data file."),
                                                       methodSlot("calculationMethod", "Select a uncertainty calculation method."),
													   pdbAssignmentHelix(UncertaintyDataCall::pdbAssMethod::PDB_PROMOTIF),
													   pdbAssignmentSheet(UncertaintyDataCall::pdbAssMethod::PDB_PROMOTIF),
                                                       pdbID("") {
                                                           
	this->dataOutSlot.SetCallback(UncertaintyDataCall::ClassName(), UncertaintyDataCall::FunctionName(UncertaintyDataCall::CallForGetData), &UncertaintyDataLoader::getData);
    this->MakeSlotAvailable(&this->dataOutSlot);
    
	this->filenameSlot << new param::FilePathParam("");
	this->MakeSlotAvailable(&this->filenameSlot);
        
    this->currentMethod = AVERAGE;
    param::EnumParam *tmpEnum = new param::EnumParam(static_cast<int>(this->currentMethod));
    
    tmpEnum->SetTypePair(AVERAGE,    "AVERAGE");
    // tmpEnum->SetTypePair(..., "...");
    
    this->methodSlot << tmpEnum;
    this->MakeSlotAvailable(&this->methodSlot);    
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

    bool recalculate = false;
    
	// Get pointer to data call
	UncertaintyDataCall *udc = dynamic_cast<UncertaintyDataCall*>(&call);
    if ( !udc ) return false;

    // check if new method was chosen
	if (this->methodSlot.IsDirty()) {
        this->methodSlot.ResetDirty();  
        this->currentMethod = static_cast<calculationMethod>(this->methodSlot.Param<core::param::EnumParam>()->Value());
        recalculate = true;
    }

	// check if new filename is set 
	if (this->filenameSlot.IsDirty()) {
		this->filenameSlot.ResetDirty();
		if(!this->readInputFile(this->filenameSlot.Param<core::param::FilePathParam>()->Value())) {
            return false;
        }
        recalculate = true;
    }
    
    // calculate uncertainty if necessary
    if(recalculate) {
        switch(this->currentMethod) {
            case (AVERAGE): 
                if (!this->calculateUncertaintyAverage()) {
                    return false;
                }
                break;
                                   
            default: return false;
        }
        udc->SetRecalcFlag(true);
        
        // DEBUG
		/*
        for (int i = 0; i < this->pdbIndex.Count(); i++) {
            std::cout << "U: ";
            for (int j = 0; j < static_cast<int>(UncertaintyDataCall::secStructure::NOE); j++) {
                std::cout << this->secStructUncertainty[i][j] << " ";
            }
            std::cout << " - S: ";
            for (int j = 0; j < static_cast<int>(UncertaintyDataCall::secStructure::NOE); j++) {
                std::cout << this->sortedSecStructUncertainty[i][j] << " ";
            }
            std::cout << std::endl;
        }
        */
    }
    
    // pass secondary strucutre data to call, if available
    if( this->pdbIndex.IsEmpty() ) { 
        return false;
    } else {
        udc->SetSecStructure(&this->secStructAssignment);
        udc->SetPdbIndex(&this->pdbIndex);
        udc->SetAminoAcidName(&this->aminoAcidName);
        udc->SetChainID(&this->chainID);
        udc->SetResidueFlag(&this->residueFlag);
        udc->SetSecStructUncertainty(&this->secStructUncertainty);
        udc->SetSortedSecStructTypes(&this->sortedSecStructUncertainty);
        udc->SetPdbID(&this->pdbID);
        udc->SetPdbAssMethodHelix(&this->pdbAssignmentHelix);
        udc->SetPdbAssMethodSheet(&this->pdbAssignmentSheet);
		udc->SetChangeMeasure(&this->changeMeasure);
        return true;
    }
}


/*
* UncertaintyDataLoader::readInputFile
*/
bool UncertaintyDataLoader::readInputFile(const vislib::TString& filename) {
	using vislib::sys::Log;

	// temp variables
	unsigned int                 lineCnt;       // line count of file
	vislib::StringA              line;          // current line of file
    char                         tmpSecStruct;  
    vislib::sys::ASCIIFileBuffer file;          // ascii buffer of file
    vislib::StringA              filenameA = T2A(filename);
    vislib::StringA              tmpString;
	vislib::math::Vector<unsigned int, static_cast<unsigned int>(UncertaintyDataCall::assMethod::NOM)> tmpStructLength;

    // reset data (or just if new file can be loaded?)
    this->pdbIndex.Clear();
    this->chainID.Clear();
    this->aminoAcidName.Clear();
    this->residueFlag.Clear();
    for (unsigned int i = 0; i < secStructAssignment.Count(); i++) {
        this->secStructAssignment[i].Clear();
    }
    this->secStructAssignment.Clear();
	for (unsigned int i = 0; i < secStructLength.Count(); i++) {
		this->secStructLength[i].Clear();
	}
	this->secStructLength.Clear();

    // check if file ending matches ".uid"
    if(!filenameA.Contains(".uid")) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, "Wrong file ending detected, must be \".uid\": \"%s\"", filenameA.PeekBuffer()); // ERROR
        return false;
    }

	// Try to load the file
	if (file.LoadFile(filename)) {

        Log::DefaultLog.WriteMsg(Log::LEVEL_INFO, "Opened uncertainty input data file: \"%s\"", filenameA.PeekBuffer()); // INFO

        // Reset array size
        // (maximum number of entries in data arrays is ~9 less than line count of file)
		this->secStructLength.AssertCapacity(static_cast<unsigned int>(UncertaintyDataCall::assMethod::NOM));
		this->secStructAssignment.AssertCapacity(static_cast<unsigned int>(UncertaintyDataCall::assMethod::NOM));
        for (unsigned int i = 0; i < static_cast<unsigned int>(UncertaintyDataCall::assMethod::NOM); i++) {

            this->secStructAssignment.Add(vislib::Array<UncertaintyDataCall::secStructure>());
            this->secStructAssignment.Last().AssertCapacity(file.Count());

			this->secStructLength.Add(vislib::Array<unsigned int>());
			this->secStructLength.Last().AssertCapacity(file.Count());
        }
        this->chainID.AssertCapacity(file.Count());
        this->aminoAcidName.AssertCapacity(file.Count());
        this->residueFlag.AssertCapacity(file.Count());
        this->pdbIndex.AssertCapacity(file.Count());

		// Run through file lines
		lineCnt = 0;
		while (lineCnt < file.Count() && !line.StartsWith("END")) {
            
			line = file.Line(lineCnt);
            
            // get pdb id
            if(line.StartsWith("PDB")) {

                this->pdbID = line.Substring(9,4);
            }
			else if (line.StartsWith("METHOD")) {

				// parse assignment method for pdb
				// helix
				tmpString = line.Substring(42, 32);
				if (tmpString.Contains(" AUTHOR "))
					this->pdbAssignmentHelix = UncertaintyDataCall::pdbAssMethod::PDB_AUTHOR;
				else if (tmpString.Contains(" DSSP "))
					this->pdbAssignmentHelix = UncertaintyDataCall::pdbAssMethod::PDB_DSSP;
				else if (tmpString.Contains(" PROMOTIF "))
					this->pdbAssignmentHelix = UncertaintyDataCall::pdbAssMethod::PDB_PROMOTIF;
				else
					this->pdbAssignmentHelix = UncertaintyDataCall::pdbAssMethod::PDB_UNKNOWN;
				// sheet
				tmpString = line.Substring(105, 32);
				if (tmpString.Contains(" AUTHOR "))
					this->pdbAssignmentSheet = UncertaintyDataCall::pdbAssMethod::PDB_AUTHOR;
				else if (tmpString.Contains(" DSSP "))
					this->pdbAssignmentSheet = UncertaintyDataCall::pdbAssMethod::PDB_DSSP;
				else if (tmpString.Contains(" PROMOTIF "))
					this->pdbAssignmentSheet = UncertaintyDataCall::pdbAssMethod::PDB_PROMOTIF;
				else
					this->pdbAssignmentSheet = UncertaintyDataCall::pdbAssMethod::PDB_UNKNOWN;
			}
            // parsing lines beginning with DATA
            else if (line.StartsWith("DATA")) {
                
                // Truncate line beginning (first 8 charachters), so character 
                // indices of line matches column indices given in input file
			    line = line.Substring(8); 

                // PDB index of amino-acids 
                tmpString = line.Substring(32,6);
                // remove spaces
                tmpString.Remove(" ");
                this->pdbIndex.Add(tmpString.PeekBuffer()); // first parameter of substring is start (beginning with 0), second parameter is range
                
                // PDB three letter code of amino-acids
                this->aminoAcidName.Add(line.Substring(10,3)); 
                // PDB one letter chain id 
                this->chainID.Add(line[22]);
                // The Missing amino-acid flag
                if (line[26] == 'M')
                    this->residueFlag.Add(UncertaintyDataCall::addFlags::MISSING);
                else if (line[26] == 'H')
                    this->residueFlag.Add(UncertaintyDataCall::addFlags::HETEROGEN);
                else
                    this->residueFlag.Add(UncertaintyDataCall::addFlags::NOTHING);
                                   
                // Translate DSSP one letter secondary structure summary 
                switch (line[228]) {
                    case 'H': this->secStructAssignment[(int)UncertaintyDataCall::assMethod::DSSP].Add(UncertaintyDataCall::secStructure::H_ALPHA_HELIX); break;
					case 'G': this->secStructAssignment[(int)UncertaintyDataCall::assMethod::DSSP].Add(UncertaintyDataCall::secStructure::G_310_HELIX); break;
					case 'I': this->secStructAssignment[(int)UncertaintyDataCall::assMethod::DSSP].Add(UncertaintyDataCall::secStructure::I_PI_HELIX); break;
					case 'E': this->secStructAssignment[(int)UncertaintyDataCall::assMethod::DSSP].Add(UncertaintyDataCall::secStructure::E_EXT_STRAND); break;
					case 'B': this->secStructAssignment[(int)UncertaintyDataCall::assMethod::DSSP].Add(UncertaintyDataCall::secStructure::B_BRIDGE); break;
					case 'T': this->secStructAssignment[(int)UncertaintyDataCall::assMethod::DSSP].Add(UncertaintyDataCall::secStructure::T_H_TURN); break;
					case 'S': this->secStructAssignment[(int)UncertaintyDataCall::assMethod::DSSP].Add(UncertaintyDataCall::secStructure::S_BEND); break;
					case 'C': this->secStructAssignment[(int)UncertaintyDataCall::assMethod::DSSP].Add(UncertaintyDataCall::secStructure::C_COIL); break;
					default:  this->secStructAssignment[(int)UncertaintyDataCall::assMethod::DSSP].Add(UncertaintyDataCall::secStructure::NOTDEFINED); break;
                }

                // Translate STRIDE one letter secondary structure
                switch (line[157]) {
					case 'H': this->secStructAssignment[(int)UncertaintyDataCall::assMethod::STRIDE].Add(UncertaintyDataCall::secStructure::H_ALPHA_HELIX); break;
					case 'G': this->secStructAssignment[(int)UncertaintyDataCall::assMethod::STRIDE].Add(UncertaintyDataCall::secStructure::G_310_HELIX); break;
					case 'I': this->secStructAssignment[(int)UncertaintyDataCall::assMethod::STRIDE].Add(UncertaintyDataCall::secStructure::I_PI_HELIX); break;
					case 'E': this->secStructAssignment[(int)UncertaintyDataCall::assMethod::STRIDE].Add(UncertaintyDataCall::secStructure::E_EXT_STRAND); break;
					case 'B': this->secStructAssignment[(int)UncertaintyDataCall::assMethod::STRIDE].Add(UncertaintyDataCall::secStructure::B_BRIDGE); break;
					case 'b': this->secStructAssignment[(int)UncertaintyDataCall::assMethod::STRIDE].Add(UncertaintyDataCall::secStructure::B_BRIDGE); break;
					case 'T': this->secStructAssignment[(int)UncertaintyDataCall::assMethod::STRIDE].Add(UncertaintyDataCall::secStructure::T_H_TURN); break;
					case 'C': this->secStructAssignment[(int)UncertaintyDataCall::assMethod::STRIDE].Add(UncertaintyDataCall::secStructure::C_COIL); break;
					default:  this->secStructAssignment[(int)UncertaintyDataCall::assMethod::STRIDE].Add(UncertaintyDataCall::secStructure::NOTDEFINED); break;
                }

                // Translate first letter of PDB secondary structure definition
                tmpSecStruct = line[44];
                if (tmpSecStruct == 'H') {
                    switch (line[82]) {
						case '1': this->secStructAssignment[(int)UncertaintyDataCall::assMethod::PDB].Add(UncertaintyDataCall::secStructure::H_ALPHA_HELIX); break;  // right-handed-alpha
						case '2': this->secStructAssignment[(int)UncertaintyDataCall::assMethod::PDB].Add(UncertaintyDataCall::secStructure::NOTDEFINED); break;     // right-handed omega
						case '3': this->secStructAssignment[(int)UncertaintyDataCall::assMethod::PDB].Add(UncertaintyDataCall::secStructure::I_PI_HELIX); break;     // right-handed pi
						case '4': this->secStructAssignment[(int)UncertaintyDataCall::assMethod::PDB].Add(UncertaintyDataCall::secStructure::NOTDEFINED); break;     // right-handed gamma
						case '5': this->secStructAssignment[(int)UncertaintyDataCall::assMethod::PDB].Add(UncertaintyDataCall::secStructure::G_310_HELIX); break;    // right-handed 310
						case '6': this->secStructAssignment[(int)UncertaintyDataCall::assMethod::PDB].Add(UncertaintyDataCall::secStructure::H_ALPHA_HELIX); break;  // left-handed alpha
						case '7': this->secStructAssignment[(int)UncertaintyDataCall::assMethod::PDB].Add(UncertaintyDataCall::secStructure::NOTDEFINED); break;     // left-handed omega
						case '8': this->secStructAssignment[(int)UncertaintyDataCall::assMethod::PDB].Add(UncertaintyDataCall::secStructure::NOTDEFINED); break;     // left-handed gamma
						case '9': this->secStructAssignment[(int)UncertaintyDataCall::assMethod::PDB].Add(UncertaintyDataCall::secStructure::NOTDEFINED); break;     // 27 ribbon/helix
						case '0': this->secStructAssignment[(int)UncertaintyDataCall::assMethod::PDB].Add(UncertaintyDataCall::secStructure::NOTDEFINED); break;     // Polyproline 
						default:  this->secStructAssignment[(int)UncertaintyDataCall::assMethod::PDB].Add(UncertaintyDataCall::secStructure::NOTDEFINED); break;
                    }
                }
                else if (tmpSecStruct == 'S'){
					this->secStructAssignment[(int)UncertaintyDataCall::assMethod::PDB].Add(UncertaintyDataCall::secStructure::E_EXT_STRAND);
                }
                else {
					this->secStructAssignment[(int)UncertaintyDataCall::assMethod::PDB].Add(UncertaintyDataCall::secStructure::C_COIL);
                }
            }
			// Next line
			lineCnt++;
		}
		
		// calculate length of continuos secondary structure assignments
		for (int i = 0; i < this->pdbIndex.Count(); i++) {
			for (unsigned int j = 0; j < static_cast<unsigned int>(UncertaintyDataCall::assMethod::NOM); j++) {
				// init struct length
				if (i == 0) {
					tmpStructLength[j] = 1;
				}
				else if (i == this->pdbIndex.Count()-1) {
					if (this->secStructAssignment[j][i] != this->secStructAssignment[j][i - 1]) { // if last entry is different to previous
						for (unsigned int k = 0; k < tmpStructLength[j]; k++) {
							this->secStructLength[j].Add(tmpStructLength[j]);
						}
						tmpStructLength[j] = 1;
						this->secStructLength[j].Add(tmpStructLength[j]); // adding last entry (=1)
					}
					else { // las tentry is same as previous
						tmpStructLength[j]++;
						for (unsigned int k = 0; k < tmpStructLength[j]; k++) {
							this->secStructLength[j].Add(tmpStructLength[j]);
						}
					}
				}
				else {
					if (this->secStructAssignment[j][i] != this->secStructAssignment[j][i - 1]) {
						for (unsigned int k = 0; k < tmpStructLength[j]; k++) {
							this->secStructLength[j].Add(tmpStructLength[j]);
						}
						tmpStructLength[j] = 1;
					}
					else {
						tmpStructLength[j]++;
					}
				}
			}
		}

		// DEBUG
		/*
		for (int i = 0; i < this->pdbIndex.Count(); i++) {
			std::cout << this->secStructLength[0][i] << std::endl;
		}
		*/

        //Clear ascii file buffer
		file.Clear();
		Log::DefaultLog.WriteMsg(Log::LEVEL_INFO, "Read secondary structure for %i amino-acids.", this->pdbIndex.Count()); // INFO

        return true;
	}
	else {
		Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, "Coudn't find uncertainty input data file: \"%s\"", T2A(filename.PeekBuffer())); // ERROR
        return false;
	}
}


/*
* UncertaintyDataLoader::calculateUncertainty
*/
bool UncertaintyDataLoader::calculateUncertaintyAverage(void) {
    using vislib::sys::Log;

    float methodCount;
	float tmpChange;
    
    // Reset uncertainty data 
    this->secStructUncertainty.Clear();
    this->sortedSecStructUncertainty.Clear();
	this->changeMeasure.Clear();

    if (this->pdbIndex.IsEmpty()) { // return if no data is present ...
        return false;
    }
    this->secStructUncertainty.AssertCapacity(this->pdbIndex.Count());
    this->sortedSecStructUncertainty.AssertCapacity(this->pdbIndex.Count());
	this->changeMeasure.AssertCapacity(this->pdbIndex.Count());

	// initialize structure factors for all three methods with 1.0f
	vislib::Array<vislib::Array<float> > structFactor;
	for (unsigned int i = 0; i < static_cast<unsigned int>(UncertaintyDataCall::assMethod::NOM); i++) {
		structFactor.Add(vislib::Array<float>());
		for (unsigned int j = 0; j < static_cast<unsigned int>(UncertaintyDataCall::secStructure::NOE); j++) {
			structFactor[i].Add(1.0f);
		}
	}

    // Initialize and calculate uncertainty data
    for (int i = 0; i < this->pdbIndex.Count(); i++) {

        // create new entry for amino-acid
        this->secStructUncertainty.Add(vislib::math::Vector<float, static_cast<int>(UncertaintyDataCall::secStructure::NOE)>());
        this->sortedSecStructUncertainty.Add(vislib::math::Vector<UncertaintyDataCall::secStructure, static_cast<int>(UncertaintyDataCall::secStructure::NOE)>());

        methodCount = static_cast<float>(UncertaintyDataCall::assMethod::NOM);

        // loop over all possible secondary strucutres
        for (int j = 0; j < static_cast<int>(UncertaintyDataCall::secStructure::NOE); j++) {

            // initialising
            this->secStructUncertainty.Last()[j] = 0.0f;
			this->sortedSecStructUncertainty.Last()[j] = static_cast<UncertaintyDataCall::secStructure>(j);

            // loop over all methods
            for (unsigned int k = 0; k < static_cast<unsigned int>(UncertaintyDataCall::assMethod::NOM); k++) {
                
                if (this->secStructAssignment[static_cast<UncertaintyDataCall::assMethod>(k)][i] == static_cast<UncertaintyDataCall::secStructure>(j)) {
					// ignore NOTDEFINED structure type for uncerainty calculation
					if (static_cast<UncertaintyDataCall::secStructure>(j) == UncertaintyDataCall::secStructure::NOTDEFINED) {
						methodCount -= 1.0f;
					}
					else {
						this->secStructUncertainty.Last()[j] += structFactor[static_cast<UncertaintyDataCall::assMethod>(k)][j];
					}
                }
                else {
					this->secStructUncertainty.Last()[j] += ((1.0f - structFactor[static_cast<UncertaintyDataCall::assMethod>(k)][j]) / (static_cast<float>(UncertaintyDataCall::secStructure::NOE) - 1.0f));
                }
            }
        }
        
        // normalise
        for (int j = 0; j < static_cast<int>(UncertaintyDataCall::secStructure::NOE); j++) {
			this->secStructUncertainty.Last()[j] /= abs(methodCount);
        }
            
        // using quicksort for sorting ...
		this->quickSortUncertainties(&(this->secStructUncertainty.Last()), &(this->sortedSecStructUncertainty.Last()), 0, (static_cast<int>(UncertaintyDataCall::secStructure::NOE) - 1));

		// calculate change measure 
		tmpChange = 0.0f;
		for (unsigned int k = 0; k < static_cast<unsigned int>(UncertaintyDataCall::assMethod::NOM)-1; k++) {
			tmpChange += (this->secStructUncertainty.Last()[this->sortedSecStructUncertainty.Last()[k]] - this->secStructUncertainty.Last()[this->sortedSecStructUncertainty.Last()[k+1]]);
		}
		this->changeMeasure.Add(1.0f - tmpChange);

		// DEBUG
		// std::cout << this->changeMeasure.Last() << std::endl;
    }

    return true;
}


/*
* UncertaintyDataLoader::quickSortUncertainties
*/
void UncertaintyDataLoader::quickSortUncertainties(vislib::math::Vector<float, static_cast<int>(UncertaintyDataCall::secStructure::NOE)> *valueArr,
                                                   vislib::math::Vector<UncertaintyDataCall::secStructure, static_cast<int>(UncertaintyDataCall::secStructure::NOE)> *structArr,
                                                   int left, int right) {
    int i = left;
    int j = right;
    UncertaintyDataCall::secStructure tmpStruct;

    float pivot = valueArr->operator[](static_cast<int>(structArr->operator[]((int)(left + right) / 2)));

    // partition 
    while (i <= j) {

        while (valueArr->operator[](static_cast<int>(structArr->operator[](i))) > pivot)
            i++;
        while (valueArr->operator[](static_cast<int>(structArr->operator[](j))) < pivot)
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
