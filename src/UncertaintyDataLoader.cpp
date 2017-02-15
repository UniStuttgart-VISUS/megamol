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
// 
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

#define DATA_FLOAT_EPS 0.00001

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
	tmpEnum->SetTypePair(EXTENDED,   "EXTENDED");
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
		if(!this->ReadInputFile(this->filenameSlot.Param<core::param::FilePathParam>()->Value())) {
            return false;
        }
        recalculate = true;
    }
    
    // calculate uncertainty if necessary
    if(recalculate) {
        switch(this->currentMethod) {
            case (AVERAGE): 
                if (!this->CalculateUncertaintyAverage()) {
                    return false;
                }
                break;
			case (EXTENDED) :
				if (!this->CalculateUncertaintyExtended()) {
					return false;
				}
				break;
            default: return false;
        }
        udc->SetRecalcFlag(true);

        if (!this->CalculateStructureLength()) {
            return false;
        }
        
        // DEBUG - sorted structure assignments, secondary structure length and uncertainty
		unsigned int k = static_cast<unsigned int>(UncertaintyDataCall::assMethod::UNCERTAINTY);
        //for (unsigned int k = 0; k < static_cast<unsigned int>(UncertaintyDataCall::assMethod::NOM); k++) {
            for (int i = 0; i < this->pdbIndex.Count(); i++) {
                std::cout << "M: " << k << " - A: " << i << " - L: " << this->secStructLength[k][i] << " - Unc: " << this->uncertainty[i] << " - S: ";
                for (unsigned int n = 0; n < static_cast<unsigned int>(UncertaintyDataCall::secStructure::NOE); n++) {
                    std::cout << this->sortedSecStructAssignment[k][i][n] << "|";
                }
                std::cout << " - U: ";
                for (unsigned int n = 0; n < static_cast<unsigned int>(UncertaintyDataCall::secStructure::NOE); n++) {
                    std::cout << this->secStructUncertainty[k][i][n] << "|";
                }
                std::cout << std::endl;
            }
        //}
        
    }
    
    // pass secondary strucutre data to call, if available
    if( this->pdbIndex.IsEmpty() ) { 
        return false;
    } else {
        udc->SetPdbIndex(&this->pdbIndex);
        udc->SetAminoAcidName(&this->aminoAcidName);
        udc->SetChainID(&this->chainID);
        udc->SetResidueFlag(&this->residueFlag);
        udc->SetSecStructUncertainty(&this->secStructUncertainty);
        udc->SetSortedSecStructAssignment(&this->sortedSecStructAssignment);
        udc->SetPdbID(&this->pdbID);
        udc->SetPdbAssMethodHelix(&this->pdbAssignmentHelix);
        udc->SetPdbAssMethodSheet(&this->pdbAssignmentSheet);
		udc->SetUncertainty(&this->uncertainty);
        udc->SetStrideThreshold(&this->strideStructThreshold);
        udc->SetStrideEnergy(&this->strideStructEnergy);
        udc->SetDsspEnergy(&this->dsspStructEnergy);
        return true;
    }
}


/*
* UncertaintyDataLoader::ReadInputFile
*/
bool UncertaintyDataLoader::ReadInputFile(const vislib::TString& filename) {
	using vislib::sys::Log;

	// temp variables
	unsigned int                 lineCnt;       // line count of file
	vislib::StringA              line;          // current line of file
    char                         tmpSecStruct;  
    vislib::sys::ASCIIFileBuffer file;          // ascii buffer of file
    vislib::StringA              filenameA = T2A(filename);
    vislib::StringA              tmpString;

    // reset data (or just if new file can be loaded?)
    this->pdbIndex.Clear();
    this->chainID.Clear();
    this->aminoAcidName.Clear();
    this->residueFlag.Clear();
    
    // clear sortedSecStructAssignment
    for (unsigned int i = 0; i < sortedSecStructAssignment.Count(); i++) {
        this->sortedSecStructAssignment[i].Clear();
    }
    this->sortedSecStructAssignment.Clear();
    // clear secStructUncertainty
    for (unsigned int i = 0; i < secStructUncertainty.Count(); i++) {
        this->secStructUncertainty[i].Clear();
    }
    this->secStructUncertainty.Clear();

    this->strideStructThreshold.Clear();
    this->strideStructEnergy.Clear();
    this->dsspStructEnergy.Clear();

    // check if file ending matches ".uid"
    if(!filenameA.Contains(".uid")) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, "Wrong file ending detected, must be \".uid\": \"%s\"", filenameA.PeekBuffer()); // ERROR
        return false;
    }

	// Try to load the file
	if (file.LoadFile(filename)) {

        Log::DefaultLog.WriteMsg(Log::LEVEL_INFO, "Reading uncertainty input data file: \"%s\"", filenameA.PeekBuffer()); // INFO

        // Reset array size
        // (maximum number of entries in data arrays is ~9 less than line count of file)
        this->sortedSecStructAssignment.AssertCapacity(static_cast<unsigned int>(UncertaintyDataCall::assMethod::NOM));
        for (unsigned int i = 0; i < static_cast<unsigned int>(UncertaintyDataCall::assMethod::NOM); i++) {
            this->sortedSecStructAssignment.Add(vislib::Array<vislib::math::Vector<UncertaintyDataCall::secStructure, static_cast<int>(UncertaintyDataCall::secStructure::NOE)> >());
            this->sortedSecStructAssignment.Last().AssertCapacity(file.Count());
        }
        this->secStructUncertainty.AssertCapacity(static_cast<unsigned int>(UncertaintyDataCall::assMethod::NOM));
        for (unsigned int i = 0; i < static_cast<unsigned int>(UncertaintyDataCall::assMethod::NOM); i++) {
            this->secStructUncertainty.Add(vislib::Array<vislib::math::Vector<float, static_cast<int>(UncertaintyDataCall::secStructure::NOE)> >());
            this->secStructUncertainty.Last().AssertCapacity(file.Count());
        }

        this->strideStructThreshold.AssertCapacity(file.Count());
        this->strideStructEnergy.AssertCapacity(file.Count());
        this->dsspStructEnergy.AssertCapacity(file.Count());

        this->chainID.AssertCapacity(file.Count());
        this->aminoAcidName.AssertCapacity(file.Count());
        this->residueFlag.AssertCapacity(file.Count());
        this->pdbIndex.AssertCapacity(file.Count());


		// Run through file lines
		lineCnt = 0;
		while (lineCnt < file.Count() && !line.StartsWith("END")) {
            
			line = file.Line(lineCnt);
           
            if(line.StartsWith("PDB")) {                                // get pdb id 
                
                this->pdbID = line.Substring(9,4);
            }
			else if (line.StartsWith("METHOD")) {                       // parse assignment method for pdb

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
            else if (line.StartsWith("DATA")) {                         // parsing data lines
                
                // Truncate line beginning (first 8 charachters), so character 
                // indices of line matches column indices given in input file
			    line = line.Substring(8); 

                // PDB index of amino-acids 
                tmpString = line.Substring(32,6); // first parameter of substring is start (beginning with 0), second parameter is range
                // remove spaces
                tmpString.Remove(" ");
                this->pdbIndex.Add(tmpString.PeekBuffer()); 
                
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
                                   

                // INITIALISE UNCERTAINTY OF STRUCTURE ASSIGNMENTS 

                // tmp pointers
                vislib::Array<vislib::math::Vector<float, static_cast<int>(UncertaintyDataCall::secStructure::NOE)> > *tmpSSU;
                vislib::Array<vislib::math::Vector<UncertaintyDataCall::secStructure, static_cast<int>(UncertaintyDataCall::secStructure::NOE)> > *tmpSSSA;

                vislib::math::Vector<float, static_cast<int>(UncertaintyDataCall::secStructure::NOE)> defaultSSU;
                vislib::math::Vector<UncertaintyDataCall::secStructure, static_cast<int>(UncertaintyDataCall::secStructure::NOE)> defaultSSSA;
                // initialising default uncertainty and structure
                for (int j = 0; j < static_cast<int>(UncertaintyDataCall::secStructure::NOE); j++) {
                    defaultSSU[j]  = 0.0f;
                    defaultSSSA[j] = static_cast<UncertaintyDataCall::secStructure>(j);
                }


                // PDB
                tmpSSU = &this->secStructUncertainty[(int)UncertaintyDataCall::assMethod::PDB];
                tmpSSSA = &this->sortedSecStructAssignment[(int)UncertaintyDataCall::assMethod::PDB];
                tmpSSU->Add(defaultSSU);
                tmpSSSA->Add(defaultSSSA);
                // Translate first letter of PDB secondary structure definition
                tmpSecStruct = line[44];
                if (tmpSecStruct == 'H') {
                    switch (line[82]) {
                        case '1': tmpSSU->Last()[UncertaintyDataCall::secStructure::H_ALPHA_HELIX] = 1.0f; break;  // right-handed-alpha
						case '2': tmpSSU->Last()[UncertaintyDataCall::secStructure::NOTDEFINED] = 1.0f; break;     // right-handed omega
						case '3': tmpSSU->Last()[UncertaintyDataCall::secStructure::I_PI_HELIX] = 1.0f; break;     // right-handed pi
						case '4': tmpSSU->Last()[UncertaintyDataCall::secStructure::NOTDEFINED] = 1.0f; break;     // right-handed gamma
						case '5': tmpSSU->Last()[UncertaintyDataCall::secStructure::G_310_HELIX] = 1.0f; break;    // right-handed 310
						case '6': tmpSSU->Last()[UncertaintyDataCall::secStructure::H_ALPHA_HELIX] = 1.0f; break;  // left-handed alpha
						case '7': tmpSSU->Last()[UncertaintyDataCall::secStructure::NOTDEFINED] = 1.0f; break;     // left-handed omega
						case '8': tmpSSU->Last()[UncertaintyDataCall::secStructure::NOTDEFINED] = 1.0f; break;     // left-handed gamma
						case '9': tmpSSU->Last()[UncertaintyDataCall::secStructure::NOTDEFINED] = 1.0f; break;     // 27 ribbon/helix
						case '0': tmpSSU->Last()[UncertaintyDataCall::secStructure::NOTDEFINED] = 1.0f; break;     // Polyproline 
						default:  tmpSSU->Last()[UncertaintyDataCall::secStructure::NOTDEFINED] = 1.0f; break;
                    }
                }
                else if (tmpSecStruct == 'S'){
                    tmpSSU->Last()[UncertaintyDataCall::secStructure::E_EXT_STRAND] = 1.0f;
                }
                else {
                    tmpSSU->Last()[UncertaintyDataCall::secStructure::C_COIL] = 1.0f;
                }
                // sorting structure types
				// NE
                this->QuickSortUncertainties(&(tmpSSU->Last()), &(tmpSSSA->Last()), 0, (static_cast<int>(UncertaintyDataCall::secStructure::NOE) - 1));
                

                //STRIDE
                tmpSSU = &this->secStructUncertainty[(int)UncertaintyDataCall::assMethod::STRIDE];
                tmpSSSA = &this->sortedSecStructAssignment[(int)UncertaintyDataCall::assMethod::STRIDE];
                tmpSSU->Add(defaultSSU);
                tmpSSSA->Add(defaultSSSA);
                // Translate STRIDE one letter secondary structure
                switch (line[157]) {
					case 'H': tmpSSU->Last()[UncertaintyDataCall::secStructure::H_ALPHA_HELIX] = 1.0f; break;
					case 'G': tmpSSU->Last()[UncertaintyDataCall::secStructure::G_310_HELIX] = 1.0f; break;
					case 'I': tmpSSU->Last()[UncertaintyDataCall::secStructure::I_PI_HELIX] = 1.0f; break;
					case 'E': tmpSSU->Last()[UncertaintyDataCall::secStructure::E_EXT_STRAND] = 1.0f; break;
					case 'B': tmpSSU->Last()[UncertaintyDataCall::secStructure::B_BRIDGE] = 1.0f; break;
					case 'b': tmpSSU->Last()[UncertaintyDataCall::secStructure::B_BRIDGE] = 1.0f; break;
					case 'T': tmpSSU->Last()[UncertaintyDataCall::secStructure::T_H_TURN] = 1.0f; break;
                    case 't': tmpSSU->Last()[UncertaintyDataCall::secStructure::T_H_TURN] = 1.0f; break;
					case 'C': tmpSSU->Last()[UncertaintyDataCall::secStructure::C_COIL] = 1.0f; break;
					default:  tmpSSU->Last()[UncertaintyDataCall::secStructure::NOTDEFINED] = 1.0f; break;
                }
                // sorting structure types
                this->QuickSortUncertainties(&(tmpSSU->Last()), &(tmpSSSA->Last()), 0, (static_cast<int>(UncertaintyDataCall::secStructure::NOE) - 1));


                // DSSP
                tmpSSU = &this->secStructUncertainty[(int)UncertaintyDataCall::assMethod::DSSP];
                tmpSSSA = &this->sortedSecStructAssignment[(int)UncertaintyDataCall::assMethod::DSSP];
                tmpSSU->Add(defaultSSU);
                tmpSSSA->Add(defaultSSSA);
                // Translate DSSP one letter secondary structure summary 
                switch (line[305]) {
                    case 'H': tmpSSU->Last()[UncertaintyDataCall::secStructure::H_ALPHA_HELIX] = 1.0f; break;
					case 'G': tmpSSU->Last()[UncertaintyDataCall::secStructure::G_310_HELIX] = 1.0f; break;
					case 'I': tmpSSU->Last()[UncertaintyDataCall::secStructure::I_PI_HELIX] = 1.0f; break;
					case 'E': tmpSSU->Last()[UncertaintyDataCall::secStructure::E_EXT_STRAND] = 1.0f; break;
					case 'B': tmpSSU->Last()[UncertaintyDataCall::secStructure::B_BRIDGE] = 1.0f; break;
					case 'T': tmpSSU->Last()[UncertaintyDataCall::secStructure::T_H_TURN] = 1.0f; break;
					case 'S': tmpSSU->Last()[UncertaintyDataCall::secStructure::S_BEND] = 1.0f; break;
					case 'C': tmpSSU->Last()[UncertaintyDataCall::secStructure::C_COIL] = 1.0f; break;
					default:  tmpSSU->Last()[UncertaintyDataCall::secStructure::NOTDEFINED] = 1.0f; break;
                }
                // sorting structure types
                this->QuickSortUncertainties(&(tmpSSU->Last()), &(tmpSSSA->Last()), 0, (static_cast<int>(UncertaintyDataCall::secStructure::NOE) - 1));


                // UNCERTAINTY
                tmpSSU = &this->secStructUncertainty[(int)UncertaintyDataCall::assMethod::UNCERTAINTY];
                tmpSSSA = &this->sortedSecStructAssignment[(int)UncertaintyDataCall::assMethod::UNCERTAINTY];
                tmpSSU->Add(defaultSSU);
                tmpSSSA->Add(defaultSSSA);


                // Read threshold and energy values of STRIDE
                float T1a = (float)std::atof(line.Substring(204, 10));
                float T2a = (float)std::atof(line.Substring(215, 10));
                float T3a = (float)std::atof(line.Substring(226, 10));
                float T1b = (float)std::atof(line.Substring(237, 10));
                float T2b = (float)std::atof(line.Substring(248, 10));
                float HBEn1 = (float)std::atof(line.Substring(259, 10));
                float HBEn2 = (float)std::atof(line.Substring(207, 10));

                vislib::math::Vector<float, 5> tmpVec5;
                tmpVec5[0] = T1a;
                tmpVec5[1] = T2a;
                tmpVec5[2] = T3a;
                tmpVec5[3] = T1b;
                tmpVec5[4] = T2b;
                this->strideStructThreshold.Add(tmpVec5);

                vislib::math::Vector<float, 2> tmpVec2;
                tmpVec2[0] = HBEn1;
                tmpVec2[1] = HBEn2;
                this->strideStructEnergy.Add(tmpVec2);

                // Read threshold and energy values of DSSP
				float HBondAc0 = (float)std::atof(line.Substring(411, 8));
				float HBondAc1 = (float)std::atof(line.Substring(422, 8));
				float HBondDo0 = (float)std::atof(line.Substring(433, 8));
				float HBondDo1 = (float)std::atof(line.Substring(444, 8));

                vislib::math::Vector<float, 4> tmpVec4;
                tmpVec4[0] = HBondAc0;
                tmpVec4[1] = HBondDo0; 
                tmpVec4[2] = HBondAc1;
                tmpVec4[3] = HBondDo1;
                this->dsspStructEnergy.Add(tmpVec4);
            }
			// Next line
			lineCnt++;
		}

        //Clear ascii file buffer
		file.Clear();
		Log::DefaultLog.WriteMsg(Log::LEVEL_INFO, "Retrieved secondary structure assignments for %i amino-acids.", this->pdbIndex.Count()); // INFO

        return true;
	}
	else {
		Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, "Coudn't find uncertainty input data file: \"%s\"", T2A(filename.PeekBuffer())); // ERROR
        return false;
	}
}


/*
* UncertaintyDataLoader::CalculateStructureLength
*/
bool UncertaintyDataLoader::CalculateStructureLength(void) {

    const unsigned int methodCnt = static_cast<unsigned int>(UncertaintyDataCall::assMethod::NOM);

    vislib::math::Vector<unsigned int, methodCnt> tmpStructLength;

    // clear 
    for (unsigned int i = 0; i < secStructLength.Count(); i++) {
        this->secStructLength[i].Clear();
    }
    this->secStructLength.Clear();
    this->secStructLength.AssertCapacity(methodCnt);
    for (unsigned int i = 0; i < methodCnt; i++) {
        this->secStructLength.Add(vislib::Array<unsigned int>());
        this->secStructLength.Last().AssertCapacity(this->pdbIndex.Count());
    }

    for (unsigned int j = 0; j < methodCnt; j++) {
        for (unsigned int i = 0; i < this->pdbIndex.Count(); i++) {
            // init struct length
            if (i == 0) {
                tmpStructLength[j] = 1;
            }
            else if (i == this->pdbIndex.Count() - 1) {
                if (this->sortedSecStructAssignment[j][i][0] != this->sortedSecStructAssignment[j][i - 1][0]) { // if last entry is different to previous
                    for (unsigned int k = 0; k < tmpStructLength[j]; k++) {
                        this->secStructLength[j].Add(tmpStructLength[j]);
                    }
                    tmpStructLength[j] = 1;
                    this->secStructLength[j].Add(tmpStructLength[j]); // adding last entry (=1)
                }
                else { // last entry is same as previous
                    tmpStructLength[j]++;
                    for (unsigned int k = 0; k < tmpStructLength[j]; k++) {
                        this->secStructLength[j].Add(tmpStructLength[j]);
                    }
                }
            }
            else {
                if (this->sortedSecStructAssignment[j][i][0] != this->sortedSecStructAssignment[j][i - 1][0]) {
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

    return true;
}


/*
* UncertaintyDataLoader::CalculateUncertaintyExtended
*/
bool UncertaintyDataLoader::CalculateUncertaintyExtended(void) {
	using vislib::sys::Log;


    // return if no data is present ...
    if (this->pdbIndex.IsEmpty()) { 
        return false;
    }

    const unsigned int methodCnt    = static_cast<unsigned int>(UncertaintyDataCall::assMethod::NOM) - 1;  // UNCERTAINTY is not taken into account
    const unsigned int structCnt    = static_cast<unsigned int>(UncertaintyDataCall::secStructure::NOE);
    const unsigned int uncMethod    = static_cast<unsigned int>(UncertaintyDataCall::assMethod::UNCERTAINTY);
	const unsigned int consStrTypes = structCnt - 1; // NOTDEFINED is ignored
	 
    // Reset uncertainty data 
    this->secStructUncertainty[uncMethod].Clear();
    this->secStructUncertainty[uncMethod].AssertCapacity(this->pdbIndex.Count());
    this->sortedSecStructAssignment[uncMethod].Clear();    
    this->sortedSecStructAssignment[uncMethod].AssertCapacity(this->pdbIndex.Count());
    this->uncertainty.Clear();    
    this->uncertainty.AssertCapacity(this->pdbIndex.Count());
               
	// Create tmp structure type and structure uncertainty vectors for amino-acid
	vislib::math::Vector<float, structCnt>                             ssu; // uncertainty
	vislib::math::Vector<UncertaintyDataCall::secStructure, structCnt> ssa; // assignment


	// Distanz-Matrix
    const float M_SD[8][8] = { // STRIDE - DSSP
                  /*      I,      H,      G,      T,      S,      C,      B,      E */
            /* I */{   0.0f,  10.0f,  20.0f,  50.0f,  60.0f,  70.0f,  80.0f, 100.0f},               
            /* H */{  10.0f,   0.0f,  10.0f,  40.0f,  50.0f,  60.0f,  70.0f,  90.0f},
            /* G */{  20.0f,  10.0f,   0.0f,  30.0f,  40.0f,  50.0f,  60.0f,  80.0f},
            /* T */{  50.0f,  40.0f,  30.0f,   0.0f,  10.0f,  20.0f,  30.0f,  50.0f},
            /*(S)*/{  -1.0f,  -1.0f,  -1.0f,  -1.0f,  -1.0f,  -1.0f,  -1.0f,  -1.0f},
            /* C */{  70.0f,  60.0f,  50.0f,  20.0f,  10.0f,   0.0f,  10.0f,  30.0f},
            /* B */{  80.0f,  70.0f,  60.0f,  30.0f,  20.0f,  10.0f,   0.0f,  20.0f},
            /* E */{ 100.0f,  90.0f,  80.0f,  50.0f,  40.0f,  30.0f,  20.0f,   0.0f}};

    const float M_DP[8][8] = { // DSSP - PDB
	           	  /*      I,      H,      G,    (T),    (S),      C,    (B),      E */
            /* I */{   0.0f,  10.0f,  20.0f,  -1.0f,  -1.0f,  70.0f,  -1.0f, 100.0f},               
            /* H */{  10.0f,   0.0f,  10.0f,  -1.0f,  -1.0f,  60.0f,  -1.0f,  90.0f},
            /* G */{  20.0f,  10.0f,   0.0f,  -1.0f,  -1.0f,  50.0f,  -1.0f,  80.0f},
            /* T */{  50.0f,  40.0f,  30.0f,  -1.0f,  -1.0f,  20.0f,  -1.0f,  50.0f},
            /* S */{  60.0f,  50.0f,  40.0f,  -1.0f,  -1.0f,  10.0f,  -1.0f,  40.0f},
            /* C */{  70.0f,  60.0f,  50.0f,  -1.0f,  -1.0f,   0.0f,  -1.0f,  30.0f},
            /* B */{  80.0f,  70.0f,  60.0f,  -1.0f,  -1.0f,  10.0f,  -1.0f,  20.0f},
            /* E */{ 100.0f,  90.0f,  80.0f,  -1.0f,  -1.0f,  30.0f,  -1.0f,   0.0f}};
        
    const float M_SP[8][8] = { // STRIDE - PDB
		          /*      I,      H,      G,    (T),    (S),      C,    (B),      E */
            /* I */{   0.0f,  10.0f,  20.0f,  -1.0f,  -1.0f,  70.0f,  -1.0f, 100.0f},               
            /* H */{  10.0f,   0.0f,  10.0f,  -1.0f,  -1.0f,  60.0f,  -1.0f,  90.0f},
            /* G */{  20.0f,  10.0f,   0.0f,  -1.0f,  -1.0f,  50.0f,  -1.0f,  80.0f},
            /* T */{  50.0f,  40.0f,  30.0f,  -1.0f,  -1.0f,  20.0f,  -1.0f,  50.0f},
            /*(S)*/{  -1.0f,  -1.0f,  -1.0f,  -1.0f,  -1.0f,  -1.0f,  -1.0f,  -1.0f},
            /* C */{  70.0f,  60.0f,  50.0f,  -1.0f,  -1.0f,   0.0f,  -1.0f,  30.0f},
            /* B */{  80.0f,  70.0f,  60.0f,  -1.0f,  -1.0f,  10.0f,  -1.0f,  20.0f},
            /* E */{ 100.0f,  90.0f,  80.0f,  -1.0f,  -1.0f,  30.0f,  -1.0f,   0.0f}};


	// Looping over all amino-acids
	for (unsigned int a = 0; a < this->pdbIndex.Count(); a++) {

		// Calculate structure uncertainty per method =========================
		/*
		for (int mCnt = 0; mCnt < methodCnt; mCnt++) { // mCnt - The method loop.

			for (int s = 0; s < structCnt; s++) { // sCnt - The structure type loop.  
				UncertaintyDataCall::secStructure curStruct = static_cast<UncertaintyDataCall::secStructure>(s);
				// Init tmp structure type and structure uncertainty vectors
				ssu[s] = 0.0f;
				ssa[s] = curStruct;


			}

			// Assign values to arrays
			this->secStructUncertainty[uncMethod].Add(ssu);
			this->sortedSecStructAssignment[uncMethod].Add(ssa);
			this->uncertainty.Add(unc);
		}
		*/

		// Calculate structure uncertainty per amino-acid =====================
		float propSum = 0.0f;

		// Init structure type propability
		for (unsigned int sCntO = 0; sCntO < structCnt; sCntO++) {
			ssu[sCntO] = 0.0f;
			ssa[sCntO] = static_cast<UncertaintyDataCall::secStructure>(sCntO);
		}

		for (unsigned int sCntO = 0; sCntO < consStrTypes; sCntO++) {   // sCntO - The outer structure type loop.
			for (unsigned int mCntO = 0; mCntO < methodCnt; mCntO++) {  // mCntO - The outer method loop.
				for (unsigned int mCntI = 0; mCntI < methodCnt; mCntI++) {          // mCntI - The inner method loop.
					for (unsigned int sCntI = 0; sCntI < consStrTypes; sCntI++) {   // sCntI - The inner structure type loop.          

						UncertaintyDataCall::assMethod mPropO = static_cast<UncertaintyDataCall::assMethod>(mCntO);
						UncertaintyDataCall::assMethod mPropI = static_cast<UncertaintyDataCall::assMethod>(mCntI);
						float dist = 0.0;

						if ((mPropO == UncertaintyDataCall::assMethod::DSSP) && (mPropI == UncertaintyDataCall::assMethod::STRIDE)) {
							dist = M_SD[sCntI][sCntO];
						}
						else if ((mPropO == UncertaintyDataCall::assMethod::STRIDE) && (mPropI == UncertaintyDataCall::assMethod::DSSP)) {
							dist = M_SD[sCntO][sCntI];
						}
						else if ((mPropO == UncertaintyDataCall::assMethod::STRIDE) && (mPropI == UncertaintyDataCall::assMethod::PDB)) {
							dist = M_SP[sCntO][sCntI];
						}
						else if ((mPropO == UncertaintyDataCall::assMethod::PDB) && (mPropI == UncertaintyDataCall::assMethod::STRIDE)) {
							dist = M_SP[sCntI][sCntO];
						}
						else if ((mPropO == UncertaintyDataCall::assMethod::PDB) && (mPropI == UncertaintyDataCall::assMethod::DSSP)) {
							dist = M_DP[sCntI][sCntO];
						}
						else if ((mPropO == UncertaintyDataCall::assMethod::DSSP) && (mPropI == UncertaintyDataCall::assMethod::PDB)) {
							dist = M_DP[sCntO][sCntI];
						}

						ssu[sCntO] += ((1.0f - (dist / 100.0f)) * this->secStructUncertainty[mCntO][a][sCntO] * this->secStructUncertainty[mCntI][a][sCntI]);

					} // end: sCnt
				} // end: mCntI
			} // end: mCntO

			propSum += ssu[sCntO];

		} // end: sCntO


        // Normalizing structure propabilities to [0.0,1.0]
		for (unsigned int s = 0; s < consStrTypes; s++) {
			ssu[s] /= propSum;
        }
        
		// Sorting structure types by their propability
		this->QuickSortUncertainties(&(ssu), &(ssa), 0, (structCnt - 1));

		// TODO: Define all uncertainty values > (1.0-dUnc) as sure and set uncertainty to 1.0 ??? -  as Paramter!


		// Calculate reduced uncertainty ======================================

		float unc  = 0.0f;
		float mean = 1.0f / ((float)consStrTypes); 

		// Propability vector with maximum standard deviation
		vislib::math::Vector<float, consStrTypes> Pmax;
		for (unsigned int s = 0; s < consStrTypes; s++) {
			Pmax[s] = 0.0;

		}
		Pmax[0] = 1.0f;

		// Caluclating maximum standard deviation
		float variance  = 0.0f;
		for (unsigned int v = 0; v < consStrTypes; v++) {
			variance += ((Pmax[v] - mean)*(Pmax[v] - mean));
		}
		variance *= mean;
		float stdDevMax = (float)std::sqrt((double)(variance));

		// Calculating actual standard deviation
		variance = 0.0f;
		for (unsigned int v = 0; v < consStrTypes; v++) {
			variance += ((ssu[v] - mean)*(ssu[v] - mean));
		}
		variance *= mean;
		float stdDev   = (float)std::sqrt((double)(variance));

		// Calulating uncertainty
		unc = (1.0f - (stdDev / stdDevMax));
		// Just for rounding
		if (unc < 0.0f) {
			unc = 0.0f;
		}
		if (unc > 1.0f) {
			unc = 1.0f;
		}

        // Assign values to arrays
        this->secStructUncertainty[uncMethod].Add(ssu);
        this->sortedSecStructAssignment[uncMethod].Add(ssa);
        this->uncertainty.Add(unc);
    }
    
    Log::DefaultLog.WriteMsg(Log::LEVEL_INFO, "Calculated uncertainty for secondary structure.", this->pdbIndex.Count()); // INFO

    return true;
}


/*
* UncertaintyDataLoader::CalculateUncertaintyAverage
*/
bool UncertaintyDataLoader::CalculateUncertaintyAverage(void) {
    using vislib::sys::Log;

    // return if no data is present ...
    if (this->pdbIndex.IsEmpty()) { 
        return false;
    }

    const unsigned int methodCnt   = static_cast<unsigned int>(UncertaintyDataCall::assMethod::NOM) - 1;  // UNCERTAINTY is not taken into account
    const unsigned int structTypes = static_cast<unsigned int>(UncertaintyDataCall::secStructure::NOE);
    const unsigned int uncMethod   = static_cast<unsigned int>(UncertaintyDataCall::assMethod::UNCERTAINTY);
    
    // Reset uncertainty data 
    this->secStructUncertainty[uncMethod].Clear();
    this->secStructUncertainty[uncMethod].AssertCapacity(this->pdbIndex.Count());
    this->sortedSecStructAssignment[uncMethod].Clear();    
    this->sortedSecStructAssignment[uncMethod].AssertCapacity(this->pdbIndex.Count());
    this->uncertainty.Clear();    
    this->uncertainty.AssertCapacity(this->pdbIndex.Count());

    // Create tmp structure type and structure uncertainty vectors for amino-acid
    vislib::math::Vector<float, structTypes>                             ssu;
    vislib::math::Vector<UncertaintyDataCall::secStructure, structTypes> ssa;
    
	// Initialize structure factors for all three methods with 1.0f
	vislib::Array<vislib::Array<float> > structFactor;
	for (unsigned int i = 0; i < methodCnt; i++) {
		structFactor.Add(vislib::Array<float>());
		for (unsigned int j = 0; j < structTypes; j++) {
			structFactor[i].Add(1.0f);
		}
	}

    // Calculate uncertainty
    // Loop over all amino-acids
    for (int i = 0; i < this->pdbIndex.Count(); i++) {
        unsigned int consideredMethods = methodCnt;
        
        // Loop over all secondary strucutre types
        for (int j = 0; j < structTypes; j++) {
            UncertaintyDataCall::secStructure curStruct = static_cast<UncertaintyDataCall::secStructure>(j);
            
            // Init tmp structure type and structure uncertainty vectors
            ssu[j] = 0.0f;
            ssa[j] = curStruct;

            // Loop over all assignment methods 
            for (unsigned int k = 0; k < methodCnt; k++) {
                UncertaintyDataCall::assMethod curMethod = static_cast<UncertaintyDataCall::assMethod>(k);
                
                if (curStruct == this->sortedSecStructAssignment[curMethod][i][0]) {
					// Ignore NOTDEFINED structure type
					if (curStruct == UncertaintyDataCall::secStructure::NOTDEFINED) {
						consideredMethods -= 1;
					}
					else {
                        ssu[j] += structFactor[curMethod][j];
					}
                }
                else {
                    ssu[j] += ((1.0f - structFactor[curMethod][j]) / ((float)(structTypes - 1)));
                }
            }
        }
        
        // Normalise structure uncertainty to [0.0,1.0]
        for (unsigned int j = 0; j < structTypes; j++) {
            ssu[j] /= abs((float)consideredMethods);
        }
            
        // Sorting structure types by their uncertainty
        this->QuickSortUncertainties(&(ssu), &(ssa), 0, (structTypes-1));

		// Calculate reduced uncertainty value
		float unc = 0.0f;
        for (unsigned int k = 0; k < structTypes-1; k++) {
            unc += (ssu[ssa[k]] - ssu[ssa[k + 1]]);
		}
        unc = 1.0f - unc;
        
        // Assign values to arrays
        this->secStructUncertainty[uncMethod].Add(ssu);
        this->sortedSecStructAssignment[uncMethod].Add(ssa);
        this->uncertainty.Add(unc);
    }

    Log::DefaultLog.WriteMsg(Log::LEVEL_INFO, "Calculated AVERAGE uncertainty for secondary structure.", this->pdbIndex.Count()); // INFO

    return true;
}


/*
* UncertaintyDataLoader::QuickSortUncertainties
*/
void UncertaintyDataLoader::QuickSortUncertainties(vislib::math::Vector<float, static_cast<int>(UncertaintyDataCall::secStructure::NOE)> *valueArr,
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
        this->QuickSortUncertainties(valueArr, structArr, left, j);
        
    if (i < right)
        this->QuickSortUncertainties(valueArr, structArr, i, right);
}
