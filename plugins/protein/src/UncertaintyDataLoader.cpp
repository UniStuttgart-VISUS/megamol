/**
 * MegaMol
 * Copyright (c) 2021, MegaMol Dev Team
 * All rights reserved.
 */
//////////////////////////////////////////////////////////////////////////////////////////////
//
// TODO:
//  - Define all uncertainty values > (1.0-dUnc) as sure and set uncertainty to 1.0 ? -  as Paramter!
//  - Anpassung der reihenfolge der Strukturtypen in dataCall und den Matrizen !
//
//////////////////////////////////////////////////////////////////////////////////////////////


#include "stdafx.h"

#include "UncertaintyDataLoader.h"

#include <algorithm>
#include <math.h>
#include <string>

#include "mmcore/CoreInstance.h"
#include "mmcore/param/EnumParam.h"
#include "mmcore/param/FilePathParam.h"
#include "mmcore/param/IntParam.h"
#include "mmcore/param/StringParam.h"
#include "mmcore/utility/log/Log.h"
#include "mmcore/utility/sys/ASCIIFileBuffer.h"

#include "vislib/math/mathfunctions.h"
#include "vislib/sys/BufferedFile.h"
#include "vislib/sys/sysfunctions.h"

#include <iomanip>  // DEBUG
#include <iostream> // DEBUG

#define DATA_FLOAT_EPS 0.00001f

#define minimum(a, b) (((a) < (b)) ? (a) : (b))
#define maximum(a, b) (((a) > (b)) ? (a) : (b))

using namespace megamol::core;
using namespace megamol::protein;
using namespace megamol::protein_calls;

/*
 * UncertaintyDataLoader::UncertaintyDataLoader (CTOR)
 */
UncertaintyDataLoader::UncertaintyDataLoader(void)
        : megamol::core::Module()
        , dataOutSlot("dataout", "The slot providing the uncertainty data")
        , filenameSlot("uidFilename", "The filename of the uncertainty input data file.")
        , methodSlot("calculationMethod", "Select a uncertainty calculation method.")
        , pdbAssignmentHelix(UncertaintyDataCall::pdbAssMethod::PDB_PROMOTIF)
        , pdbAssignmentSheet(UncertaintyDataCall::pdbAssMethod::PDB_PROMOTIF)
        , pdbID("") {

    this->dataOutSlot.SetCallback(UncertaintyDataCall::ClassName(),
        UncertaintyDataCall::FunctionName(UncertaintyDataCall::CallForGetData), &UncertaintyDataLoader::getData);
    this->MakeSlotAvailable(&this->dataOutSlot);

    this->filenameSlot << new param::FilePathParam("");
    this->MakeSlotAvailable(&this->filenameSlot);

    this->currentMethod = EXTENDED;
    param::EnumParam* tmpEnum = new param::EnumParam(static_cast<int>(this->currentMethod));
    tmpEnum->SetTypePair(AVERAGE, "AVERAGE");
    tmpEnum->SetTypePair(EXTENDED, "EXTENDED");
    this->methodSlot << tmpEnum;
    this->MakeSlotAvailable(&this->methodSlot);
}


/*
 * UncertaintyDataLoader::~UncertaintyDataLoader (DTOR)
 */
UncertaintyDataLoader::~UncertaintyDataLoader(void) {
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
    using megamol::core::utility::log::Log;

    bool recalculate = false;

    // Get pointer to data call
    UncertaintyDataCall* udc = dynamic_cast<UncertaintyDataCall*>(&call);
    if (!udc)
        return false;

    // check if new method was chosen
    if (this->methodSlot.IsDirty()) {
        this->methodSlot.ResetDirty();
        this->currentMethod = static_cast<calculationMethod>(this->methodSlot.Param<core::param::EnumParam>()->Value());
        recalculate = true;
    }

    // check if new filename is set
    if (this->filenameSlot.IsDirty()) {
        this->filenameSlot.ResetDirty();
        if (!this->ReadInputFile(this->filenameSlot.Param<core::param::FilePathParam>()->Value())) {
            return false;
        }
        recalculate = true;
    }

    // calculate uncertainty if necessary
    if (recalculate) {
        switch (this->currentMethod) {
        case (AVERAGE):
            if (!this->CalculateUncertaintyAverage()) {
                return false;
            }
            break;
        case (EXTENDED):
            if (!this->CalculateUncertaintyExtended()) {
                return false;
            }
            break;
        default:
            return false;
        }
        udc->SetRecalcFlag(true);

        if (!this->CalculateStructureLength()) {
            return false;
        }

        // DEBUG - sorted structure assignments, secondary structure length and uncertainty
        /*
        unsigned int w = 5;
        unsigned int k = static_cast<unsigned int>(UncertaintyDataCall::assMethod::UNCERTAINTY);

        k = static_cast<unsigned int>(UncertaintyDataCall::assMethod::STRIDE);
// for (unsigned int k = 0; k < static_cast<unsigned int>(UncertaintyDataCall::assMethod::NOM); k++) {

    for (int i = 0; i < this->pdbIndex.Count(); i++) {
                        std::cout << std::setprecision(3) << "M: " << std::setw(w) << k << " - A: " << std::setw(w) << i
<< " - L: " << std::setw(w) << this->secStructLength[k][i];
                        // std::cout << " - Unc: " << std::setw(w) << this->uncertainty[i];
                        std::cout << " - S: ";
        for (unsigned int n = 0; n < static_cast<unsigned int>(UncertaintyDataCall::secStructure::NOE); n++) {
                                std::cout << std::setprecision(3) << std::setw(w) <<
this->sortedSecStructAssignment[k][i][n] << "|";
        }

                        std::cout << " - U: ";
        for (unsigned int n = 0; n < static_cast<unsigned int>(UncertaintyDataCall::secStructure::NOE); n++) {
                                std::cout << std::setprecision(3) << std::setw(w) << this->secStructUncertainty[k][i][n]
<< "|";
        }

        std::cout << std::endl;
    }
// }
*/
    }

    // pass secondary strucutre data to call, if available
    if (this->pdbIndex.IsEmpty()) {
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
        udc->SetDsspEnergy(&this->dsspStructEnergy);
        udc->SetProsignThreshold(&this->prosignStructThreshold);
        return true;
    }
}


/*
 * UncertaintyDataLoader::ReadInputFile
 */
bool UncertaintyDataLoader::ReadInputFile(const std::filesystem::path& filename) {
    using megamol::core::utility::log::Log;

    // temp variables
    unsigned int lineCnt; // line count of file
    vislib::StringA line; // current line of file
    char tmpSecStruct;
    vislib::sys::ASCIIFileBuffer file; // ascii buffer of file
    vislib::StringA filenameA(filename.c_str());
    vislib::StringA tmpString;

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
    this->dsspStructEnergy.Clear();
    this->prosignStructThreshold.Clear();

    // check if file ending matches ".uid"
    if (!filenameA.Contains(".uid")) {
        Log::DefaultLog.WriteMsg(
            Log::LEVEL_ERROR, "Wrong file ending detected, must be \".uid\": \"%s\"", filenameA.PeekBuffer()); // ERROR
        return false;
    }

    // Try to load the file
    if (file.LoadFile(filename.c_str())) {

        Log::DefaultLog.WriteMsg(
            Log::LEVEL_INFO, "Reading uncertainty input data file: \"%s\"", filenameA.PeekBuffer()); // INFO

        // Reset array size
        // (maximum number of entries in data arrays is ~9 less than line count of file)
        this->sortedSecStructAssignment.AssertCapacity(static_cast<unsigned int>(UncertaintyDataCall::assMethod::NOM));
        for (unsigned int i = 0; i < static_cast<unsigned int>(UncertaintyDataCall::assMethod::NOM); i++) {
            this->sortedSecStructAssignment.Add(vislib::Array<vislib::math::Vector<UncertaintyDataCall::secStructure,
                    static_cast<int>(UncertaintyDataCall::secStructure::NOE)>>());
            this->sortedSecStructAssignment.Last().AssertCapacity(file.Count());
        }
        this->secStructUncertainty.AssertCapacity(static_cast<unsigned int>(UncertaintyDataCall::assMethod::NOM));
        for (unsigned int i = 0; i < static_cast<unsigned int>(UncertaintyDataCall::assMethod::NOM); i++) {
            this->secStructUncertainty.Add(
                vislib::Array<vislib::math::Vector<float, static_cast<int>(UncertaintyDataCall::secStructure::NOE)>>());
            this->secStructUncertainty.Last().AssertCapacity(file.Count());
        }

        this->strideStructThreshold.AssertCapacity(file.Count());
        this->dsspStructEnergy.AssertCapacity(file.Count());
        this->prosignStructThreshold.AssertCapacity(file.Count());

        this->chainID.AssertCapacity(file.Count());
        this->aminoAcidName.AssertCapacity(file.Count());
        this->residueFlag.AssertCapacity(file.Count());
        this->pdbIndex.AssertCapacity(file.Count());


        // Run through file lines
        lineCnt = 0;
        char LastChainID = ' ';
        char currentChainID = ' ';
        bool skip = false;

        while (lineCnt < file.Count() && !line.StartsWith("END")) {

            line = file.Line(lineCnt);

            if (line.StartsWith("PDB")) { // get pdb id

                this->pdbID = line.Substring(9, 4);
            } else if (line.StartsWith("METHOD")) { // parse assignment method for pdb
                line = line.Substring(8);

                // helix
                tmpString = line.Substring(34, 35);
                if (tmpString.Contains("AUTHOR") || tmpString.Contains("DEPOSITOR"))
                    this->pdbAssignmentHelix = UncertaintyDataCall::pdbAssMethod::PDB_AUTHOR;
                else if (tmpString.Contains("DSSP") || tmpString.Contains("KABSCH"))
                    this->pdbAssignmentHelix = UncertaintyDataCall::pdbAssMethod::PDB_DSSP;
                else if (tmpString.Contains("PROMOTIF"))
                    this->pdbAssignmentHelix = UncertaintyDataCall::pdbAssMethod::PDB_PROMOTIF;
                else
                    this->pdbAssignmentHelix = UncertaintyDataCall::pdbAssMethod::PDB_UNKNOWN;

                // sheet
                tmpString = line.Substring(97, 35);
                if (tmpString.Contains("AUTHOR") || tmpString.Contains("DEPOSITOR"))
                    this->pdbAssignmentSheet = UncertaintyDataCall::pdbAssMethod::PDB_AUTHOR;
                else if (tmpString.Contains("DSSP") || tmpString.Contains("KABSCH"))
                    this->pdbAssignmentSheet = UncertaintyDataCall::pdbAssMethod::PDB_DSSP;
                else if (tmpString.Contains("PROMOTIF"))
                    this->pdbAssignmentSheet = UncertaintyDataCall::pdbAssMethod::PDB_PROMOTIF;
                else
                    this->pdbAssignmentSheet = UncertaintyDataCall::pdbAssMethod::PDB_UNKNOWN;
            } else if (line.StartsWith("DATA")) { // parsing data lines

                // Truncate line beginning (first 8 charachters), so character
                // indices of line matches column indices given in input file
                line = line.Substring(8);

                // PDB one letter chain id
                currentChainID = line[22];

                // Ignore HETATM at the end with repeating chain IDs,
                // add each chain just once!
                if (LastChainID != currentChainID) {
                    skip = false;
                    LastChainID = currentChainID;
                    for (unsigned int c = 0; c < this->chainID.Count(); c++) {
                        if (this->chainID[c] == currentChainID) {
                            skip = true;
                            break;
                        }
                    }
                }

                // skip line if chain id repeats after break (cut HETATM at the end)
                if (!skip) {

                    // PDB index of amino-acids
                    tmpString = line.Substring(
                        32, 6); // first parameter of substring is start (beginning with 0), second parameter is range
                    // remove spaces
                    tmpString.Remove(" ");
                    this->pdbIndex.Add(tmpString.PeekBuffer());

                    // PDB three letter code of amino-acids
                    this->aminoAcidName.Add(line.Substring(10, 3));

                    // adding chain id
                    this->chainID.Add(currentChainID);

                    // The Missing amino-acid flag
                    if (line[26] == 'M')
                        this->residueFlag.Add(UncertaintyDataCall::addFlags::MISSING);
                    else if (line[26] == 'H')
                        this->residueFlag.Add(UncertaintyDataCall::addFlags::HETEROGEN);
                    else
                        this->residueFlag.Add(UncertaintyDataCall::addFlags::NOTHING);


                    // INITIALISE UNCERTAINTY OF STRUCTURE ASSIGNMENTS

                    // tmp pointers
                    vislib::Array<
                        vislib::math::Vector<float, static_cast<int>(UncertaintyDataCall::secStructure::NOE)>>* tmpSSU;
                    vislib::Array<vislib::math::Vector<UncertaintyDataCall::secStructure,
                        static_cast<int>(UncertaintyDataCall::secStructure::NOE)>>* tmpSSSA;

                    vislib::math::Vector<float, static_cast<int>(UncertaintyDataCall::secStructure::NOE)> defaultSSU;
                    vislib::math::Vector<UncertaintyDataCall::secStructure,
                        static_cast<int>(UncertaintyDataCall::secStructure::NOE)>
                        defaultSSSA;
                    // initialising default uncertainty and structure
                    for (int j = 0; j < static_cast<int>(UncertaintyDataCall::secStructure::NOE); j++) {
                        defaultSSU[j] = 0.0f;
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
                        case '1':
                            tmpSSU->Last()[UncertaintyDataCall::secStructure::H_ALPHA_HELIX] = 1.0f;
                            break; // right-handed-alpha
                        case '2':
                            tmpSSU->Last()[UncertaintyDataCall::secStructure::NOTDEFINED] = 1.0f;
                            break; // right-handed omega
                        case '3':
                            tmpSSU->Last()[UncertaintyDataCall::secStructure::I_PI_HELIX] = 1.0f;
                            break; // right-handed pi
                        case '4':
                            tmpSSU->Last()[UncertaintyDataCall::secStructure::NOTDEFINED] = 1.0f;
                            break; // right-handed gamma
                        case '5':
                            tmpSSU->Last()[UncertaintyDataCall::secStructure::G_310_HELIX] = 1.0f;
                            break; // right-handed 310
                        case '6':
                            tmpSSU->Last()[UncertaintyDataCall::secStructure::H_ALPHA_HELIX] = 1.0f;
                            break; // left-handed alpha
                        case '7':
                            tmpSSU->Last()[UncertaintyDataCall::secStructure::NOTDEFINED] = 1.0f;
                            break; // left-handed omega
                        case '8':
                            tmpSSU->Last()[UncertaintyDataCall::secStructure::NOTDEFINED] = 1.0f;
                            break; // left-handed gamma
                        case '9':
                            tmpSSU->Last()[UncertaintyDataCall::secStructure::NOTDEFINED] = 1.0f;
                            break; // 27 ribbon/helix
                        case '0':
                            tmpSSU->Last()[UncertaintyDataCall::secStructure::NOTDEFINED] = 1.0f;
                            break; // Polyproline
                        default:
                            tmpSSU->Last()[UncertaintyDataCall::secStructure::NOTDEFINED] = 1.0f;
                            break;
                        }
                    } else if (tmpSecStruct == 'S') {
                        tmpSSU->Last()[UncertaintyDataCall::secStructure::E_EXT_STRAND] = 1.0f;
                    } else {
                        tmpSSU->Last()[UncertaintyDataCall::secStructure::C_COIL] = 1.0f;
                    }
                    // sorting structure types
                    this->QuickSortUncertainties(&(tmpSSU->Last()), &(tmpSSSA->Last()), 0,
                        (static_cast<int>(UncertaintyDataCall::secStructure::NOE) - 1));


                    // STRIDE
                    tmpSSU = &this->secStructUncertainty[(int)UncertaintyDataCall::assMethod::STRIDE];
                    tmpSSSA = &this->sortedSecStructAssignment[(int)UncertaintyDataCall::assMethod::STRIDE];
                    tmpSSU->Add(defaultSSU);
                    tmpSSSA->Add(defaultSSSA);
                    // Translate STRIDE one letter secondary structure
                    switch (line[157]) {
                    case 'H':
                        tmpSSU->Last()[UncertaintyDataCall::secStructure::H_ALPHA_HELIX] = 1.0f;
                        break;
                    case 'G':
                        tmpSSU->Last()[UncertaintyDataCall::secStructure::G_310_HELIX] = 1.0f;
                        break;
                    case 'I':
                        tmpSSU->Last()[UncertaintyDataCall::secStructure::I_PI_HELIX] = 1.0f;
                        break;
                    case 'E':
                        tmpSSU->Last()[UncertaintyDataCall::secStructure::E_EXT_STRAND] = 1.0f;
                        break;
                    case 'B':
                        tmpSSU->Last()[UncertaintyDataCall::secStructure::B_BRIDGE] = 1.0f;
                        break;
                    case 'b':
                        tmpSSU->Last()[UncertaintyDataCall::secStructure::B_BRIDGE] = 1.0f;
                        break;
                    case 'T':
                        tmpSSU->Last()[UncertaintyDataCall::secStructure::T_H_TURN] = 1.0f;
                        break;
                    case 't':
                        tmpSSU->Last()[UncertaintyDataCall::secStructure::T_H_TURN] = 1.0f;
                        break;
                    case 'C':
                        tmpSSU->Last()[UncertaintyDataCall::secStructure::C_COIL] = 1.0f;
                        break;
                    default:
                        tmpSSU->Last()[UncertaintyDataCall::secStructure::NOTDEFINED] = 1.0f;
                        break;
                    }
                    // sorting structure types
                    this->QuickSortUncertainties(&(tmpSSU->Last()), &(tmpSSSA->Last()), 0,
                        (static_cast<int>(UncertaintyDataCall::secStructure::NOE) - 1));


                    // DSSP
                    tmpSSU = &this->secStructUncertainty[(int)UncertaintyDataCall::assMethod::DSSP];
                    tmpSSSA = &this->sortedSecStructAssignment[(int)UncertaintyDataCall::assMethod::DSSP];
                    tmpSSU->Add(defaultSSU);
                    tmpSSSA->Add(defaultSSSA);
                    // Translate DSSP one letter secondary structure summary
                    switch (line[305]) {
                    case 'H':
                        tmpSSU->Last()[UncertaintyDataCall::secStructure::H_ALPHA_HELIX] = 1.0f;
                        break;
                    case 'G':
                        tmpSSU->Last()[UncertaintyDataCall::secStructure::G_310_HELIX] = 1.0f;
                        break;
                    case 'I':
                        tmpSSU->Last()[UncertaintyDataCall::secStructure::I_PI_HELIX] = 1.0f;
                        break;
                    case 'E':
                        tmpSSU->Last()[UncertaintyDataCall::secStructure::E_EXT_STRAND] = 1.0f;
                        break;
                    case 'B':
                        tmpSSU->Last()[UncertaintyDataCall::secStructure::B_BRIDGE] = 1.0f;
                        break;
                    case 'T':
                        tmpSSU->Last()[UncertaintyDataCall::secStructure::T_H_TURN] = 1.0f;
                        break;
                    case 'S':
                        tmpSSU->Last()[UncertaintyDataCall::secStructure::S_BEND] = 1.0f;
                        break;
                    case 'C':
                        tmpSSU->Last()[UncertaintyDataCall::secStructure::C_COIL] = 1.0f;
                        break;
                    default:
                        tmpSSU->Last()[UncertaintyDataCall::secStructure::NOTDEFINED] = 1.0f;
                        break;
                    }
                    // sorting structure types
                    this->QuickSortUncertainties(&(tmpSSU->Last()), &(tmpSSSA->Last()), 0,
                        (static_cast<int>(UncertaintyDataCall::secStructure::NOE) - 1));


                    // PROSIGN
                    tmpSSU = &this->secStructUncertainty[(int)UncertaintyDataCall::assMethod::PROSIGN];
                    tmpSSSA = &this->sortedSecStructAssignment[(int)UncertaintyDataCall::assMethod::PROSIGN];
                    tmpSSU->Add(defaultSSU);
                    tmpSSSA->Add(defaultSSSA);
                    // Translate DSSP one letter secondary structure summary
                    switch (line[480]) {
                    case 'H':
                        tmpSSU->Last()[UncertaintyDataCall::secStructure::H_ALPHA_HELIX] = 1.0f;
                        break;
                    case 'G':
                        tmpSSU->Last()[UncertaintyDataCall::secStructure::G_310_HELIX] = 1.0f;
                        break;
                    case 'I':
                        tmpSSU->Last()[UncertaintyDataCall::secStructure::I_PI_HELIX] = 1.0f;
                        break;
                    case 'E':
                        tmpSSU->Last()[UncertaintyDataCall::secStructure::E_EXT_STRAND] = 1.0f;
                        break;
                    case 'B':
                        tmpSSU->Last()[UncertaintyDataCall::secStructure::NOTDEFINED] = 1.0f;
                        break;
                    case 'T':
                        tmpSSU->Last()[UncertaintyDataCall::secStructure::NOTDEFINED] = 1.0f;
                        break;
                    case 'S':
                        tmpSSU->Last()[UncertaintyDataCall::secStructure::NOTDEFINED] = 1.0f;
                        break;
                    case 'C':
                        tmpSSU->Last()[UncertaintyDataCall::secStructure::C_COIL] = 1.0f;
                        break;
                    default:
                        tmpSSU->Last()[UncertaintyDataCall::secStructure::NOTDEFINED] = 1.0f;
                        break;
                    }
                    // sorting structure types
                    this->QuickSortUncertainties(&(tmpSSU->Last()), &(tmpSSSA->Last()), 0,
                        (static_cast<int>(UncertaintyDataCall::secStructure::NOE) - 1));

                    // UNCERTAINTY
                    tmpSSU = &this->secStructUncertainty[(int)UncertaintyDataCall::assMethod::UNCERTAINTY];
                    tmpSSSA = &this->sortedSecStructAssignment[(int)UncertaintyDataCall::assMethod::UNCERTAINTY];
                    tmpSSU->Add(defaultSSU);
                    tmpSSSA->Add(defaultSSSA);


                    // Read threshold and energy values of STRIDE
                    // std::atof converts "inf" to "inf"
                    // std::atof converts " "   to "0.0"
                    float Th1 = (float)std::atof(line.Substring(204, 10));
                    float Th3 = (float)std::atof(line.Substring(215, 10));
                    float Th4 = (float)std::atof(line.Substring(226, 10));
                    float Tb1p = (float)std::atof(line.Substring(237, 10));
                    float Tb2p = (float)std::atof(line.Substring(248, 10));
                    float Tb1a = (float)std::atof(line.Substring(259, 10));
                    float Tb2a = (float)std::atof(line.Substring(207, 10));

                    vislib::math::Vector<float, 7> tmpVec7;
                    tmpVec7[0] = (Th1 > 1.0E38f) ? (1.0E38f) : (Th1);
                    tmpVec7[1] = (Th3 > 1.0E38f) ? (1.0E38f) : (Th3);
                    tmpVec7[2] = (Th4 > 1.0E38f) ? (1.0E38f) : (Th4);
                    tmpVec7[3] = (Tb1p > 1.0E38f) ? (1.0E38f) : (Tb1p);
                    tmpVec7[4] = (Tb2p > 1.0E38f) ? (1.0E38f) : (Tb2p);
                    tmpVec7[5] = (Tb1a > 1.0E38f) ? (1.0E38f) : (Tb1a);
                    tmpVec7[6] = (Tb2a > 1.0E38f) ? (1.0E38f) : (Tb2a);
                    this->strideStructThreshold.Add(tmpVec7);

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

                    // Read threshold values of PROSIGN
                    std::string bla = line.Substring(498, 10).PeekBuffer();
                    float alphaValue = (float)std::atof(line.Substring(498, 10));
                    float threeTenValue = (float)std::atof(line.Substring(511, 9));
                    float piValue = (float)std::atof(line.Substring(523, 7));
                    float betaValue = (float)std::atof(line.Substring(533, 9));
                    float helixThr = (float)std::atof(line.Substring(545, 11));
                    float betaThr = (float)std::atof(line.Substring(559, 10));

                    vislib::math::Vector<float, 6> tmpVec6;
                    tmpVec6[0] = (alphaValue > 1.0E38f) ? (1.0E38f) : alphaValue;
                    tmpVec6[1] = (threeTenValue > 1.0E38f) ? (1.0E38f) : threeTenValue;
                    tmpVec6[2] = (piValue > 1.0E38f) ? (1.0E38f) : piValue;
                    tmpVec6[3] = (betaValue > 1.0E38f) ? (1.0E38f) : betaValue;
                    tmpVec6[4] = (helixThr > 1.0E38f) ? (1.0E38f) : helixThr;
                    tmpVec6[5] = (betaThr > 1.0E38f) ? (1.0E38f) : betaThr;
                    this->prosignStructThreshold.Add(tmpVec6);
                }
            }
            // Next line
            lineCnt++;
        }

        // Clear ascii file buffer
        file.Clear();
        Log::DefaultLog.WriteMsg(Log::LEVEL_INFO, "Retrieved secondary structure assignments for %i amino-acids.",
            this->pdbIndex.Count()); // INFO

        return true;
    } else {
        Log::DefaultLog.WriteMsg(
            Log::LEVEL_ERROR, "Coudn't find uncertainty input data file: \"%s\"", T2A(filename.c_str())); // ERROR
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
            } else if (i == this->pdbIndex.Count() - 1) {
                if (this->sortedSecStructAssignment[j][i][0] !=
                    this->sortedSecStructAssignment[j][i - 1][0]) { // if last entry is different to previous
                    for (unsigned int k = 0; k < tmpStructLength[j]; k++) {
                        this->secStructLength[j].Add(tmpStructLength[j]);
                    }
                    tmpStructLength[j] = 1;
                    this->secStructLength[j].Add(tmpStructLength[j]); // adding last entry (=1)
                } else {                                              // last entry is same as previous
                    tmpStructLength[j]++;
                    for (unsigned int k = 0; k < tmpStructLength[j]; k++) {
                        this->secStructLength[j].Add(tmpStructLength[j]);
                    }
                }
            } else {
                if (this->sortedSecStructAssignment[j][i][0] != this->sortedSecStructAssignment[j][i - 1][0]) {
                    for (unsigned int k = 0; k < tmpStructLength[j]; k++) {
                        this->secStructLength[j].Add(tmpStructLength[j]);
                    }
                    tmpStructLength[j] = 1;
                } else {
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
    using megamol::core::utility::log::Log;

    // return if no data is present ...
    if (this->pdbIndex.IsEmpty()) {
        return false;
    }

    const unsigned int methodCnt =
        static_cast<unsigned int>(UncertaintyDataCall::assMethod::NOM) - 1; // UNCERTAINTY is not taken into account
    const unsigned int structCnt = static_cast<unsigned int>(UncertaintyDataCall::secStructure::NOE);
    const unsigned int consStrTypes = structCnt - 1; // NOTDEFINED is ignored
    const unsigned int dsspMethod = static_cast<unsigned int>(UncertaintyDataCall::assMethod::DSSP);
    const unsigned int strideMethod = static_cast<unsigned int>(UncertaintyDataCall::assMethod::STRIDE);
    const unsigned int pdbMethod = static_cast<unsigned int>(UncertaintyDataCall::assMethod::PDB);
    const unsigned int uncMethod = static_cast<unsigned int>(UncertaintyDataCall::assMethod::UNCERTAINTY);
    const unsigned int prosignMethod = static_cast<unsigned int>(UncertaintyDataCall::assMethod::PROSIGN);

    // Reset uncertainty data
    this->secStructUncertainty[uncMethod].Clear();
    this->secStructUncertainty[uncMethod].AssertCapacity(this->pdbIndex.Count());
    this->sortedSecStructAssignment[uncMethod].Clear();
    this->sortedSecStructAssignment[uncMethod].AssertCapacity(this->pdbIndex.Count());
    this->uncertainty.Clear();
    this->uncertainty.AssertCapacity(this->pdbIndex.Count());

    // Create tmp structure type and structure uncertainty vectors for amino-acid
    vislib::math::Vector<float, structCnt> ssu;                             // uncertainty
    vislib::math::Vector<UncertaintyDataCall::secStructure, structCnt> ssa; // assignment


    // Distanz-Matrix
    const float M_SD[8][8] = {// STRIDE - DSSP
                              /*      G,      T,      H,      I,      S,      C,      B,      E */
        /* G */ {5.0f, 35.0f, 55.0f, 85.0f, 5.0f, 105.0f, 135.0f, 145.0f},
        /* T */ {35.0f, 15.0f, 35.0f, 65.0f, 5.0f, 85.0f, 115.0f, 125.0f},
        /* H */ {55.0f, 35.0f, 10.0f, 40.0f, 5.0f, 60.0f, 90.0f, 100.0f},
        /* I */ {85.0f, 65.0f, 40.0f, 5.0f, 5.0f, 25.0f, 55.0f, 65.0f},
        /*(S)*/ {-1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f},
        /* C */ {105.0f, 85.0f, 60.0f, 25.0f, 15.0f, 0.0f, 30.0f, 40.0f},
        /* B */ {135.0f, 115.0f, 90.0f, 55.0f, 35.0f, 30.0f, 10.0f, 20.0f},
        /* E */ {145.0f, 125.0f, 100.0f, 65.0f, 45.0f, 40.0f, 20.0f, 0.0f}};

    const float M_DA[8][8] = {// DSSP - AUTHOR
                              /*      G,      T,      H,      I,      S,      C,      B,      E */
        /* G */ {0.0f, -1.0f, 150.0f, 150.0f, -1.0f, 150.0f, -1.0f, 150.0f},
        /* T */ {150.0f, -1.0f, 150.0f, 150.0f, -1.0f, 150.0f, -1.0f, 150.0f},
        /* H */ {150.0f, -1.0f, 0.0f, 150.0f, -1.0f, 150.0f, -1.0f, 150.0f},
        /* I */ {150.0f, -1.0f, 150.0f, 0.0f, -1.0f, 150.0f, -1.0f, 150.0f},
        /* S */ {150.0f, -1.0f, 150.0f, 150.0f, -1.0f, 150.0f, -1.0f, 150.0f},
        /* C */ {150.0f, -1.0f, 150.0f, 150.0f, -1.0f, 0.0f, -1.0f, 150.0f},
        /* B */ {150.0f, -1.0f, 150.0f, 150.0f, -1.0f, 150.0f, -1.0f, 150.0f},
        /* E */ {150.0f, -1.0f, 150.0f, 150.0f, -1.0f, 150.0f, -1.0f, 0.0f}};

    const float M_SA[8][8] = {// STRIDE - AUTHOR
                              /*      G,      T,      H,      I,      S,      C,      B,      E */
        /* G */ {0.0f, -1.0f, 150.0f, 150.0f, -1.0f, 150.0f, -1.0f, 150.0f},
        /* T */ {150.0f, -1.0f, 150.0f, 150.0f, -1.0f, 150.0f, -1.0f, 150.0f},
        /* H */ {150.0f, -1.0f, 0.0f, 150.0f, -1.0f, 150.0f, -1.0f, 150.0f},
        /* I */ {150.0f, -1.0f, 150.0f, 0.0f, -1.0f, 150.0f, -1.0f, 150.0f},
        /* S */ {-1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f},
        /* C */ {150.0f, -1.0f, 150.0f, 150.0f, -1.0f, 0.0f, -1.0f, 150.0f},
        /* B */ {150.0f, -1.0f, 150.0f, 150.0f, -1.0f, 150.0f, -1.0f, 150.0f},
        /* E */ {150.0f, -1.0f, 150.0f, 150.0f, -1.0f, 150.0f, -1.0f, 0.0f}};


    const float M_SP[8][8] = {// STRIDE - PROMOTIF
                              /*      G,      T,      H,      I,      S,      C,      B,      E */
        /* G */ {5.0f, -1.0f, 40.0f, 70.0f, -1.0f, 90.0f, -1.0f, 130.0f},
        /* T */ {20.0f, -1.0f, 20.0f, 50.0f, -1.0f, 70.0f, -1.0f, 110.0f},
        /* H */ {40.0f, -1.0f, 10.0f, 40.0f, -1.0f, 60.0f, -1.0f, 100.0f},
        /* I */ {70.0f, -1.0f, 40.0f, 5.0f, -1.0f, 25.0f, -1.0f, 65.0f},
        /* S */ {-1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f},
        /* C */ {90.0f, -1.0f, 60.0f, 25.0f, -1.0f, 0.0f, -1.0f, 40.0f},
        /* B */ {120.0f, -1.0f, 90.0f, 55.0f, -1.0f, 30.0f, -1.0f, 20.0f},
        /* E */ {130.0f, -1.0f, 100.0f, 65.0f, -1.0f, 40.0f, -1.0f, 0.0f}};


    const float M_DP[8][8] = {// DSSP - PROMOTIF
                              /*      G,      T,      H,      I,      S,      C,      B,      E */
        /* G */ {5.0f, -1.0f, 50.0f, 80.0f, -1.0f, 100.0f, -1.0f, 135.0f},
        /* T */ {35.0f, -1.0f, 30.0f, 60.0f, -1.0f, 80.0f, -1.0f, 115.0f},
        /* H */ {50.0f, -1.0f, 5.0f, 35.0f, -1.0f, 55.0f, -1.0f, 90.0f},
        /* I */ {80.0f, -1.0f, 35.0f, 5.0f, -1.0f, 25.0f, -1.0f, 60.0f},
        /* S */ {5.0f, -1.0f, 5.0f, 5.0f, -1.0f, 15.0f, -1.0f, 50.0f},
        /* C */ {100.0f, -1.0f, 55.0f, 25.0f, -1.0f, 0.0f, -1.0f, 35.0f},
        /* B */ {125.0f, -1.0f, 80.0f, 50.0f, -1.0f, 25.0f, -1.0f, 15.0f},
        /* E */ {135.0f, -1.0f, 90.0f, 60.0f, -1.0f, 35.0f, -1.0f, 0.0f}};

    // TODO Better values for the PROSIGN matrices

    const float M_PrD[8][8] = {// PROSIGN - DSSP
                               /*      G,      T,      H,      I,      S,      C,      B,      E */
        /* G */ {5.0f, 35.0f, 55.0f, 85.0f, 5.0f, 105.0f, 135.0f, 145.0f},
        /*(T)*/ {-1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f},
        /* H */ {55.0f, 35.0f, 10.0f, 40.0f, 5.0f, 60.0f, 90.0f, 100.0f},
        /* I */ {85.0f, 65.0f, 40.0f, 5.0f, 5.0f, 25.0f, 55.0f, 65.0f},
        /*(S)*/ {-1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f},
        /* C */ {105.0f, 85.0f, 60.0f, 25.0f, 15.0f, 0.0f, 30.0f, 40.0f},
        /*(B)*/ {-1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f},
        /* E */ {145.0f, 125.0f, 100.0f, 65.0f, 45.0f, 40.0f, 20.0f, 0.0f}};


    const float M_PrA[8][8] = {// PROSIGN - Author
                               /*      G,    (T),      H,      I,      (S),      C,      (B),      E */
        /* G */ {5.0f, -1.0f, 55.0f, 85.0f, -1.0f, 105.0f, -1.0f, 145.0f},
        /*(T)*/ {-1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f},
        /* H */ {55.0f, -1.0f, 10.0f, 40.0f, -1.0f, 60.0f, -1.0f, 100.0f},
        /* I */ {85.0f, -1.0f, 40.0f, 5.0f, -1.0f, 25.0f, -1.0f, 65.0f},
        /*(S)*/ {-1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f},
        /* C */ {105.0f, -1.0f, 60.0f, 25.0f, -1.0f, 0.0f, -1.0f, 40.0f},
        /*(B)*/ {-1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f},
        /* E */ {145.0f, -1.0f, 100.0f, 65.0f, -1.0f, 40.0f, -1.0f, 0.0f}};


    const float M_PrP[8][8] = {// PROSIGN - PROMOTIF
                               /*      G,    (T),      H,      I,      (S),      C,      (B),      E */
        /* G */ {5.0f, -1.0f, 55.0f, 85.0f, -1.0f, 105.0f, -1.0f, 145.0f},
        /*(T)*/ {-1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f},
        /* H */ {55.0f, -1.0f, 10.0f, 40.0f, -1.0f, 60.0f, -1.0f, 100.0f},
        /* I */ {85.0f, -1.0f, 40.0f, 5.0f, -1.0f, 25.0f, -1.0f, 65.0f},
        /*(S)*/ {-1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f},
        /* C */ {105.0f, -1.0f, 60.0f, 25.0f, -1.0f, 0.0f, -1.0f, 40.0f},
        /*(B)*/ {-1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f},
        /* E */ {145.0f, -1.0f, 100.0f, 65.0f, -1.0f, 40.0f, -1.0f, 0.0f}};


    const float M_PrS[8][8] = {// PROSIGN - STRIDE
                               /*      G,      T,      H,      I,      S,      C,      (B),      E */
        /* G */ {5.0f, 35.0f, 55.0f, 85.0f, 5.0f, 105.0f, -1.0f, 145.0f},
        /*(T)*/ {-1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f},
        /* H */ {55.0f, 35.0f, 10.0f, 40.0f, 5.0f, 60.0f, -1.0f, 100.0f},
        /* I */ {85.0f, 65.0f, 40.0f, 5.0f, 5.0f, 25.0f, -1.0f, 65.0f},
        /*(S)*/ {-1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f},
        /* C */ {105.0f, 85.0f, 60.0f, 25.0f, 15.0f, 0.0f, -1.0f, 40.0f},
        /*(B)*/ {-1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f},
        /* E */ {145.0f, 125.0f, 100.0f, 65.0f, 45.0f, 40.0f, -1.0f, 0.0f}};

    const float distMax = 150.0f;


    // Calculate structure uncertainty for STRIDE =========================
    /*
            STRIDE_THRESHOLDH1 - 230.00f
            STRIDE_THRESHOLDH3     0.12f
            STRIDE_THRESHOLDH4     0.06f
            STRIDE_THRESHOLDE1 - 240.00f
            STRIDE_THRESHOLDE2 - 310.00f

            T1    -410.435 -230.000  -49.565  180.435
            T3      -0.094    0.120    0.214    0.094
            T4      -0.047    0.060    0.107    0.047
            T1[p] -553.195 -310.000 - 66.805  243.195
            T2[p] -553.195 -310.000 - 66.805  243.195
            T1[a] -428.279 -240.000 - 51.721  188.279
            T2[a] -428.279 -240.000 - 51.721  188.279
    */

    float DeltaTh1 = 180.435f;
    float DeltaTh3 = 0.094f;
    float DeltaTh4 = 0.047f;
    float DeltaTbp = 243.195f;
    float DeltaTba = 188.279f;

    float Th1 = STRIDE_THRESHOLDH1;
    float Th3 = STRIDE_THRESHOLDH3;
    float Th4 = STRIDE_THRESHOLDH4;
    float Tbp = STRIDE_THRESHOLDE2;
    float Tba = STRIDE_THRESHOLDE1;

    // Looping over all amino-acids
    for (unsigned int a = 0; a < this->pdbIndex.Count(); a++) {

        // Skip MISSING
        if (this->residueFlag[a] != UncertaintyDataCall::addFlags::MISSING) {

            float tah1 = this->strideStructThreshold[a][0];
            float tah3 = this->strideStructThreshold[a][1];
            float tah4 = this->strideStructThreshold[a][2];

            // Check HELIX
            unsigned int helixStr = (unsigned int)UncertaintyDataCall::secStructure::H_ALPHA_HELIX;
            if (a + 1 < this->pdbIndex.Count()) {

                float taah1 = this->strideStructThreshold[a + 1][0];
                if ((tah1 <= (Th1 + DeltaTh1)) && (taah1 <= (Th1 + DeltaTh1))) {

                    // a
                    float propHelixA = 0.0f;
                    if (tah1 < (Th1 - DeltaTh1)) {
                        propHelixA = 1.0f;
                    } else if (tah1 < Th1) { // tah1 >= Th1 - DeltaTh1

                        propHelixA = 1.0f - 0.5f * (1.0f - std::abs(tah1 - Th1) / DeltaTh1);
                    } else { // Th1 <= tah1 <= (Th1 + DeltaTh1)

                        propHelixA = 0.5f * (1.0f - std::abs(tah1 - Th1) / DeltaTh1);
                    }

                    // a+1
                    float propHelixAA = 0.0f;
                    if (taah1 < (Th1 - DeltaTh1)) {
                        propHelixAA = 1.0f;
                    } else if (taah1 < Th1) { // taah1 >= Th1 - DeltaTh1

                        propHelixAA = 1.0f - 0.5f * (1.0f - std::abs(taah1 - Th1) / DeltaTh1);
                    } else { // Th1 <= taah1 <= (Th1 + DeltaTh1)

                        propHelixAA = 0.5f * (1.0f - std::abs(taah1 - Th1) / DeltaTh1);
                    }

                    // set propHelixA to minimum of both values
                    propHelixA = minimum(propHelixAA, propHelixA);

                    // assign probability to amino-acids a ... a+4
                    for (unsigned int aCnt = a; aCnt < a + 4; aCnt++) {

                        if (aCnt >= this->pdbIndex.Count()) {
                            break;
                        }

                        // take maximum probability for helix
                        this->secStructUncertainty[strideMethod][aCnt][helixStr] =
                            maximum(propHelixA, this->secStructUncertainty[strideMethod][aCnt][helixStr]);

                        // search for other structure with probability > 0
                        int strN = -1;
                        for (unsigned int sCnt = 0; sCnt < structCnt; sCnt++) {
                            if ((sCnt != helixStr) && (this->secStructUncertainty[strideMethod][aCnt][sCnt] > 0.0f)) {
                                strN = sCnt;
                                break;
                            }
                        }
                        if (strN == -1) { // no structure with probability > 0 ... take COIL
                            this->secStructUncertainty[strideMethod][aCnt]
                                                      [(unsigned int)UncertaintyDataCall::secStructure::C_COIL] =
                                1.0f - this->secStructUncertainty[strideMethod][aCnt][helixStr];
                        } else {
                            this->secStructUncertainty[strideMethod][aCnt][strN] =
                                1.0f - this->secStructUncertainty[strideMethod][aCnt][helixStr];
                        }

                        this->QuickSortUncertainties(&(this->secStructUncertainty[strideMethod][aCnt]),
                            &(this->sortedSecStructAssignment[strideMethod][aCnt]), 0, (structCnt - 1));


                        // DEBUG
                        /*
                        std::cout << "HELIX: " << a << " - " << aCnt << " | ";
                        for (unsigned int n = 0; n < static_cast<unsigned int>(UncertaintyDataCall::secStructure::NOE);
                        n++) { std::cout << std::setprecision(3) << std::setw(8) <<
                        this->secStructUncertainty[strideMethod][aCnt][n] << " | ";
                        }
                        std::cout << std::endl;
                        */
                    }

                    // Check borders with Th3 and Th4
                    // a-1
                    if (a >= 1) {

                        float tah3 = this->strideStructThreshold[a - 1][1];
                        float propHelixB = 0.0f;

                        if ((Th3 + DeltaTh3) <= tah3) {
                            propHelixB = 1.0f;
                        } else if (Th3 < tah3) {
                            propHelixB = 1.0f - 0.5f * (1.0f - std::abs(tah3 - Th3) / DeltaTh3);
                        } else if ((Th3 - DeltaTh3) < tah3) {
                            propHelixB = 0.5f * (1.0f - std::abs(tah3 - Th3) / DeltaTh3);
                        } else {
                            propHelixB = 0.0f;
                        }

                        propHelixB *= propHelixA;
                        this->secStructUncertainty[strideMethod][a - 1][helixStr] =
                            maximum(propHelixB, this->secStructUncertainty[strideMethod][a - 1][helixStr]);

                        // search for other structure with probability > 0
                        int strN = -1;
                        for (unsigned int sCnt = 0; sCnt < structCnt; sCnt++) {
                            if ((sCnt != helixStr) && (this->secStructUncertainty[strideMethod][a - 1][sCnt] > 0.0f)) {
                                strN = sCnt;
                                break;
                            }
                        }
                        if (strN == -1) { // no structure with probability > 0 ... take COIL
                            this->secStructUncertainty[strideMethod][a - 1]
                                                      [(unsigned int)UncertaintyDataCall::secStructure::C_COIL] =
                                1.0f - this->secStructUncertainty[strideMethod][a - 1][helixStr];
                        } else {
                            this->secStructUncertainty[strideMethod][a - 1][strN] =
                                1.0f - this->secStructUncertainty[strideMethod][a - 1][helixStr];
                        }

                        this->QuickSortUncertainties(&(this->secStructUncertainty[strideMethod][a - 1]),
                            &(this->sortedSecStructAssignment[strideMethod][a - 1]), 0, (structCnt - 1));
                    }

                    // a+4
                    if ((a + 4) < this->pdbIndex.Count()) {
                        float tah4 = this->strideStructThreshold[a + 4][2];

                        float propHelixB = 0.0f;

                        if ((Th4 + DeltaTh4) <= tah4) {
                            propHelixB = 1.0f;
                        } else if (Th4 < tah4) {
                            propHelixB = 1.0f - 0.5f * (1.0f - std::abs(tah4 - Th4) / DeltaTh4);
                        } else if ((Th4 - DeltaTh4) < tah4) {
                            propHelixB = 0.5f * (1.0f - std::abs(tah4 - Th4) / DeltaTh4);
                        } else {
                            propHelixB = 0.0f;
                        }

                        propHelixB *= propHelixA;
                        this->secStructUncertainty[strideMethod][a + 4][helixStr] =
                            maximum(propHelixB, this->secStructUncertainty[strideMethod][a + 4][helixStr]);

                        // search for other structure with probability > 0
                        int strN = -1;
                        for (unsigned int sCnt = 0; sCnt < structCnt; sCnt++) {
                            if ((sCnt != helixStr) && (this->secStructUncertainty[strideMethod][a + 4][sCnt] > 0.0f)) {
                                strN = sCnt;
                                break;
                            }
                        }
                        if (strN == -1) { // no structure with probability > 0 ... take COIL
                            this->secStructUncertainty[strideMethod][a + 4]
                                                      [(unsigned int)UncertaintyDataCall::secStructure::C_COIL] =
                                1.0f - this->secStructUncertainty[strideMethod][a + 4][helixStr];
                        } else {
                            this->secStructUncertainty[strideMethod][a + 4][strN] =
                                1.0f - this->secStructUncertainty[strideMethod][a + 4][helixStr];
                        }

                        this->QuickSortUncertainties(&(this->secStructUncertainty[strideMethod][a + 4]),
                            &(this->sortedSecStructAssignment[strideMethod][a + 4]), 0, (structCnt - 1));
                    }
                }
            }
            /////

            // Check SHEET

            float tab1p = this->strideStructThreshold[a][3];
            float tab2p = this->strideStructThreshold[a][4];
            float tab1a = this->strideStructThreshold[a][5];
            float tab2a = this->strideStructThreshold[a][6];

            unsigned int bridgeStr = (unsigned int)UncertaintyDataCall::secStructure::B_BRIDGE;
            unsigned int sheetStr = (unsigned int)UncertaintyDataCall::secStructure::E_EXT_STRAND;

            // paralllel
            if ((tab1p <= (Tbp + DeltaTbp)) && (tab2p <= (Tbp + DeltaTbp))) {

                float sheetProp1 = 0.0;
                if (tab1p < (Tbp - DeltaTbp)) {
                    sheetProp1 = 1.0f;
                } else if (tab1p < Tbp) {
                    sheetProp1 = 1.0f - 0.5f * (1.0f - std::abs(tab1p - Tbp) / DeltaTbp);
                } else { // tab1p < Tbp + DeltaTbp
                    sheetProp1 = 0.5f * (1.0f - std::abs(tab1p - Tbp) / DeltaTbp);
                }

                float sheetProp2 = 0.0;
                if (tab2p < (Tbp - DeltaTbp)) {
                    sheetProp2 = 1.0f;
                } else if (tab2p < Tbp) {
                    sheetProp2 = 1.0f - 0.5f * (1.0f - std::abs(tab2p - Tbp) / DeltaTbp);
                } else { // tab2p < Tbp + DeltaTbp
                    sheetProp2 = 0.5f * (1.0f - std::abs(tab2p - Tbp) / DeltaTbp);
                }


                float tmpProp = maximum(this->secStructUncertainty[strideMethod][a][bridgeStr],
                    this->secStructUncertainty[strideMethod][a][sheetStr]);

                this->secStructUncertainty[strideMethod][a][bridgeStr] =
                    maximum(tmpProp, minimum(sheetProp1, sheetProp2));
                this->secStructUncertainty[strideMethod][a][sheetStr] = 0.0f;

                // search for other structure with probability > 0
                int strN = -1;
                for (unsigned int sCnt = 0; sCnt < structCnt; sCnt++) {
                    if ((sCnt != bridgeStr) && (this->secStructUncertainty[strideMethod][a][sCnt] > 0.0f)) {
                        strN = sCnt;
                        break;
                    }
                }
                if (strN == -1) { // no structure with probability > 0 ... take COIL
                    this->secStructUncertainty[strideMethod][a]
                                              [(unsigned int)UncertaintyDataCall::secStructure::C_COIL] =
                        1.0f - this->secStructUncertainty[strideMethod][a][bridgeStr];
                } else {
                    this->secStructUncertainty[strideMethod][a][strN] =
                        1.0f - this->secStructUncertainty[strideMethod][a][bridgeStr];
                }


                // Check neighbours
                bool flip = false;

                if ((a > 0) && (this->secStructUncertainty[strideMethod][a - 1][sheetStr] > 0.0f)) {
                    flip = true;
                } else if ((a + 1 < this->pdbIndex.Count()) &&
                           (this->secStructUncertainty[strideMethod][a + 1][sheetStr] > 0.0f)) {
                    flip = true;
                }

                if ((a > 0) && (this->secStructUncertainty[strideMethod][a - 1][bridgeStr] > 0.0f)) {
                    flip = true;
                    this->secStructUncertainty[strideMethod][a - 1][sheetStr] =
                        this->secStructUncertainty[strideMethod][a - 1][bridgeStr];
                    this->secStructUncertainty[strideMethod][a - 1][bridgeStr] = 0.0f;
                    this->QuickSortUncertainties(&(this->secStructUncertainty[strideMethod][a - 1]),
                        &(this->sortedSecStructAssignment[strideMethod][a - 1]), 0, (structCnt - 1));
                }

                if ((a + 1 < this->pdbIndex.Count()) &&
                    (this->secStructUncertainty[strideMethod][a + 1][bridgeStr] > 0.0f)) {
                    flip = true;
                    this->secStructUncertainty[strideMethod][a + 1][sheetStr] =
                        this->secStructUncertainty[strideMethod][a + 1][bridgeStr];
                    this->secStructUncertainty[strideMethod][a + 1][bridgeStr] = 0.0f;
                    this->QuickSortUncertainties(&(this->secStructUncertainty[strideMethod][a + 1]),
                        &(this->sortedSecStructAssignment[strideMethod][a + 1]), 0, (structCnt - 1));
                }

                if (flip) {
                    this->secStructUncertainty[strideMethod][a][sheetStr] =
                        this->secStructUncertainty[strideMethod][a][bridgeStr];
                    this->secStructUncertainty[strideMethod][a][bridgeStr] = 0.0f;
                }

                this->QuickSortUncertainties(&(this->secStructUncertainty[strideMethod][a]),
                    &(this->sortedSecStructAssignment[strideMethod][a]), 0, (structCnt - 1));
            }


            // antiparallel
            if ((tab1a <= (Tba + DeltaTba)) && (tab2a <= (Tba + DeltaTba))) {

                float sheetProp1 = 0.0;
                if (tab1a < (Tba - DeltaTba)) {
                    sheetProp1 = 1.0f;
                } else if (tab1a < Tba) {
                    sheetProp1 = 1.0f - 0.5f * (1.0f - std::abs(tab1a - Tba) / DeltaTba);
                } else { // tab1a < Tba + DeltaTba
                    sheetProp1 = 0.5f * (1.0f - std::abs(tab1a - Tba) / DeltaTba);
                }

                float sheetProp2 = 0.0;
                if (tab2a < (Tba - DeltaTba)) {
                    sheetProp2 = 1.0f;
                } else if (tab2a < Tba) {
                    sheetProp2 = 1.0f - 0.5f * (1.0f - std::abs(tab2a - Tba) / DeltaTba);
                } else { // tab2a < Tba + DeltaTba
                    sheetProp2 = 0.5f * (1.0f - std::abs(tab2a - Tba) / DeltaTba);
                }

                float tmpProp = maximum(this->secStructUncertainty[strideMethod][a][bridgeStr],
                    this->secStructUncertainty[strideMethod][a][sheetStr]);

                this->secStructUncertainty[strideMethod][a][bridgeStr] =
                    maximum(tmpProp, minimum(sheetProp1, sheetProp2));
                this->secStructUncertainty[strideMethod][a][sheetStr] = 0.0f;

                // search for other structure with probability > 0
                int strN = -1;
                for (unsigned int sCnt = 0; sCnt < structCnt; sCnt++) {
                    if ((sCnt != bridgeStr) && (this->secStructUncertainty[strideMethod][a][sCnt] > 0.0f)) {
                        strN = sCnt;
                        break;
                    }
                }
                if (strN == -1) { // no structure with probability > 0 ... take COIL
                    this->secStructUncertainty[strideMethod][a]
                                              [(unsigned int)UncertaintyDataCall::secStructure::C_COIL] =
                        1.0f - this->secStructUncertainty[strideMethod][a][bridgeStr];
                } else {
                    this->secStructUncertainty[strideMethod][a][strN] =
                        1.0f - this->secStructUncertainty[strideMethod][a][bridgeStr];
                }

                // Check neighbours
                bool flip = false;

                if ((a > 0) && (this->secStructUncertainty[strideMethod][a - 1][sheetStr] > 0.0f)) {
                    flip = true;
                } else if ((a + 1 < this->pdbIndex.Count()) &&
                           (this->secStructUncertainty[strideMethod][a + 1][sheetStr] > 0.0f)) {
                    flip = true;
                }

                if ((a > 0) && (this->secStructUncertainty[strideMethod][a - 1][bridgeStr] > 0.0f)) {
                    flip = true;
                    this->secStructUncertainty[strideMethod][a - 1][sheetStr] =
                        this->secStructUncertainty[strideMethod][a - 1][bridgeStr];
                    this->secStructUncertainty[strideMethod][a - 1][bridgeStr] = 0.0f;
                    this->QuickSortUncertainties(&(this->secStructUncertainty[strideMethod][a - 1]),
                        &(this->sortedSecStructAssignment[strideMethod][a - 1]), 0, (structCnt - 1));
                }

                if ((a + 1 < this->pdbIndex.Count()) &&
                    (this->secStructUncertainty[strideMethod][a + 1][bridgeStr] > 0.0f)) {
                    flip = true;
                    this->secStructUncertainty[strideMethod][a + 1][sheetStr] =
                        this->secStructUncertainty[strideMethod][a + 1][bridgeStr];
                    this->secStructUncertainty[strideMethod][a + 1][bridgeStr] = 0.0f;
                    this->QuickSortUncertainties(&(this->secStructUncertainty[strideMethod][a + 1]),
                        &(this->sortedSecStructAssignment[strideMethod][a + 1]), 0, (structCnt - 1));
                }

                if (flip) {
                    this->secStructUncertainty[strideMethod][a][sheetStr] =
                        this->secStructUncertainty[strideMethod][a][bridgeStr];
                    this->secStructUncertainty[strideMethod][a][bridgeStr] = 0.0f;
                }

                this->QuickSortUncertainties(&(this->secStructUncertainty[strideMethod][a]),
                    &(this->sortedSecStructAssignment[strideMethod][a]), 0, (structCnt - 1));
            }

            /////

        } // missing
    }     // a


    /*for (unsigned int i = 0; i < this->secStructUncertainty.Count(); i++) {
            for (unsigned int j = 0; j < this->secStructUncertainty[i].Count(); j++) {
                    for (unsigned int k = 0; k < this->secStructUncertainty[i][j].Length(); k++) {
                            printf("%u %u %u %f\n", i, j, k, this->secStructUncertainty[i][j][k]);
                    }
            }
    }*/

    // Calculate structure uncertainty per amino-acid =====================

    for (unsigned int a = 0; a < this->pdbIndex.Count(); a++) {
        // Init structure type propability
        for (unsigned int sCntO = 0; sCntO < structCnt; sCntO++) {
            ssu[sCntO] = 0.0f;
            ssa[sCntO] = static_cast<UncertaintyDataCall::secStructure>(sCntO);
        }

        float unc = 0.0f;
        float propSum = 0.0f;

        // Skip MISSING
        if (this->residueFlag[a] != UncertaintyDataCall::addFlags::MISSING) {

            for (unsigned int sCntO = 0; sCntO < consStrTypes; sCntO++) { // sCntO - The outer structure type loop.

                for (unsigned int mCntO = 0; mCntO < methodCnt; mCntO++) { // mCntO - The outer method loop.

                    UncertaintyDataCall::assMethod mCurO = static_cast<UncertaintyDataCall::assMethod>(mCntO);

                    for (unsigned int mCntI = 0; mCntI < methodCnt; mCntI++) { // mCntI - The inner method loop.

                        float dist = 0.0f;
                        UncertaintyDataCall::assMethod mCurI = static_cast<UncertaintyDataCall::assMethod>(mCntI);

                        for (unsigned int sCntI = 0; sCntI < consStrTypes;
                             sCntI++) { // sCntI - The inner structure type loop
                                        // Compare each method just with different methods, not with itself
                            if (mCntO != mCntI) {
                                // Get distance of structure types
                                if ((mCurO == strideMethod) && (mCurI == dsspMethod)) {
                                    dist = M_SD[sCntO][sCntI];
                                } else if ((mCurO == dsspMethod) && (mCurI == strideMethod)) {
                                    dist = M_SD[sCntI][sCntO];
                                } else if ((mCurO == prosignMethod) && (mCurI == dsspMethod)) {
                                    dist = M_PrD[sCntO][sCntI];
                                } else if ((mCurO == dsspMethod) && (mCurI == prosignMethod)) {
                                    dist = M_PrD[sCntI][sCntO];
                                } else if ((mCurO == prosignMethod) && (mCurI == strideMethod)) {
                                    dist = M_PrS[sCntO][sCntI];
                                } else if ((mCurO == strideMethod) && (mCurI == prosignMethod)) {
                                    dist = M_PrS[sCntI][sCntO];
                                } else if (mCurO == pdbMethod) {
                                    if ((sCntO == (unsigned int)UncertaintyDataCall::secStructure::G_310_HELIX) ||
                                        (sCntO == (unsigned int)UncertaintyDataCall::secStructure::H_ALPHA_HELIX) ||
                                        (sCntO == (unsigned int)UncertaintyDataCall::secStructure::I_PI_HELIX)) {

                                        if (pdbAssignmentHelix == UncertaintyDataCall::pdbAssMethod::PDB_DSSP) {
                                            if (mCurI == dsspMethod) {
                                                (dist = 0.0f);
                                            }
                                            if (mCurI == strideMethod) {
                                                (dist = M_SD[sCntI][sCntO]);
                                            }
                                            if (mCurI == prosignMethod) {
                                                (dist = M_PrD[sCntI][sCntO]);
                                            }
                                        } else if (pdbAssignmentHelix ==
                                                   UncertaintyDataCall::pdbAssMethod::PDB_AUTHOR) {
                                            if (mCurI == dsspMethod) {
                                                (dist = M_DA[sCntI][sCntO]);
                                            }
                                            if (mCurI == strideMethod) {
                                                (dist = M_SA[sCntI][sCntO]);
                                            }
                                            if (mCurI == prosignMethod) {
                                                (dist = M_PrA[sCntI][sCntO]);
                                            }
                                        } else { // (pdbAssignmentHelix == PDB_PROMOTIF)
                                            if (mCurI == dsspMethod) {
                                                (dist = M_DP[sCntI][sCntO]);
                                            }
                                            if (mCurI == strideMethod) {
                                                (dist = M_SP[sCntI][sCntO]);
                                            }
                                            if (mCurI == prosignMethod) {
                                                (dist = M_PrP[sCntI][sCntO]);
                                            }
                                        }
                                    } else if (sCntO == (unsigned int)UncertaintyDataCall::secStructure::E_EXT_STRAND) {
                                        if (pdbAssignmentSheet == UncertaintyDataCall::pdbAssMethod::PDB_DSSP) {
                                            if (mCurI == dsspMethod) {
                                                (dist = 0.0f);
                                            }
                                            if (mCurI == strideMethod) {
                                                (dist = M_SD[sCntI][sCntO]);
                                            }
                                            if (mCurI == prosignMethod) {
                                                (dist = M_PrD[sCntI][sCntO]);
                                            }
                                        } else if (pdbAssignmentSheet ==
                                                   UncertaintyDataCall::pdbAssMethod::PDB_AUTHOR) {
                                            if (mCurI == dsspMethod) {
                                                (dist = M_DA[sCntI][sCntO]);
                                            }
                                            if (mCurI == strideMethod) {
                                                (dist = M_SA[sCntI][sCntO]);
                                            }
                                            if (mCurI == prosignMethod) {
                                                (dist = M_PrA[sCntI][sCntO]);
                                            }
                                        } else { // (pdbAssignmentSheet == PDB_PROMOTIF)
                                            if (mCurI == dsspMethod) {
                                                (dist = M_DP[sCntI][sCntO]);
                                            }
                                            if (mCurI == strideMethod) {
                                                (dist = M_SP[sCntI][sCntO]);
                                            }
                                            if (mCurI == prosignMethod) {
                                                (dist = M_PrP[sCntI][sCntO]);
                                            }
                                        }
                                    } else { // COIL
                                        if (mCurI == dsspMethod) {
                                            (dist = M_DP[sCntI][sCntO]);
                                        }
                                        if (mCurI == strideMethod) {
                                            (dist = M_SP[sCntI][sCntO]);
                                        }
                                        if (mCurI == prosignMethod) {
                                            (dist = M_PrP[sCntI][sCntO]);
                                        }
                                    }
                                } else if (mCurI == pdbMethod) {
                                    if ((sCntI == (unsigned int)UncertaintyDataCall::secStructure::G_310_HELIX) ||
                                        (sCntI == (unsigned int)UncertaintyDataCall::secStructure::H_ALPHA_HELIX) ||
                                        (sCntI == (unsigned int)UncertaintyDataCall::secStructure::I_PI_HELIX)) {
                                        if (pdbAssignmentHelix == UncertaintyDataCall::pdbAssMethod::PDB_DSSP) {
                                            if (mCurO == dsspMethod) {
                                                (dist = 0.0f);
                                            }
                                            if (mCurO == strideMethod) {
                                                (dist = M_SD[sCntO][sCntI]);
                                            }
                                            if (mCurO == prosignMethod) {
                                                (dist = M_PrD[sCntO][sCntI]);
                                            }
                                        } else if (pdbAssignmentHelix ==
                                                   UncertaintyDataCall::pdbAssMethod::PDB_AUTHOR) {
                                            if (mCurO == dsspMethod) {
                                                (dist = M_DA[sCntO][sCntI]);
                                            }
                                            if (mCurO == strideMethod) {
                                                (dist = M_SA[sCntO][sCntI]);
                                            }
                                            if (mCurO == prosignMethod) {
                                                (dist = M_PrA[sCntO][sCntI]);
                                            }
                                        } else { // (pdbAssignmentHelix == PDB_PROMOTIF)
                                            if (mCurO == dsspMethod) {
                                                (dist = M_DP[sCntO][sCntI]);
                                            }
                                            if (mCurO == strideMethod) {
                                                (dist = M_SP[sCntO][sCntI]);
                                            }
                                            if (mCurO == prosignMethod) {
                                                (dist = M_PrP[sCntO][sCntI]);
                                            }
                                        }
                                    } else if (sCntI == (unsigned int)UncertaintyDataCall::secStructure::E_EXT_STRAND) {
                                        if (pdbAssignmentSheet == UncertaintyDataCall::pdbAssMethod::PDB_DSSP) {
                                            if (mCurO == dsspMethod) {
                                                (dist = 0.0f);
                                            }
                                            if (mCurO == strideMethod) {
                                                (dist = M_SD[sCntO][sCntI]);
                                            }
                                            if (mCurO == prosignMethod) {
                                                (dist = M_PrD[sCntO][sCntI]);
                                            }
                                        } else if (pdbAssignmentSheet ==
                                                   UncertaintyDataCall::pdbAssMethod::PDB_AUTHOR) {
                                            if (mCurO == dsspMethod) {
                                                (dist = M_DA[sCntO][sCntI]);
                                            }
                                            if (mCurO == strideMethod) {
                                                (dist = M_SA[sCntO][sCntI]);
                                            }
                                            if (mCurO == prosignMethod) {
                                                (dist = M_PrA[sCntO][sCntI]);
                                            }
                                        } else { // (pdbAssignmentSheet == PDB_PROMOTIF)
                                            if (mCurO == dsspMethod) {
                                                (dist = M_DP[sCntO][sCntI]);
                                            }
                                            if (mCurO == strideMethod) {
                                                (dist = M_SP[sCntO][sCntI]);
                                            }
                                            if (mCurO == prosignMethod) {
                                                (dist = M_PrP[sCntO][sCntI]);
                                            }
                                        }
                                    } else { // COIL
                                        if (mCurO == dsspMethod) {
                                            (dist = M_DP[sCntO][sCntI]);
                                        }
                                        if (mCurO == strideMethod) {
                                            (dist = M_SP[sCntO][sCntI]);
                                        }
                                        if (mCurO == prosignMethod) {
                                            (dist = M_PrP[sCntO][sCntI]);
                                        }
                                    }
                                }
                                // Consider only valid structure types for each method
                                if (dist > -1.0f) {
                                    ssu[sCntO] += (this->secStructUncertainty[mCntO][a][sCntO] *
                                                      this->secStructUncertainty[mCntI][a][sCntI]) /
                                                  ((1 + distMax - dist) * (1 + distMax - dist));
                                }
                            } //  mCntO != mCntI
                        }     // sCnt
                    }         // mCntI
                }             // mCntO
                propSum += ssu[sCntO];
            } // sCntO

            // Normalizing structure propabilities to [0.0,1.0]
            for (unsigned int s = 0; s < consStrTypes; s++) {
                if (propSum > FLT_EPSILON) {
                    ssu[s] /= propSum; // (propSum > 0.0f)?(propSum):(1.0f);
                }
            }

            // Sorting structure types by their propability
            this->QuickSortUncertainties(&(ssu), &(ssa), 0, (structCnt - 1));

            // Calculate reduced uncertainty ======================================

            unc = 0.0f;
            float mean = 1.0f / ((float)consStrTypes);

            // Propability vector with maximum standard deviation
            vislib::math::Vector<float, consStrTypes> Pmax;
            for (unsigned int s = 0; s < consStrTypes; s++) {
                Pmax[s] = 0.0;
            }
            Pmax[0] = 1.0f;

            // Caluclating maximum standard deviation
            float variance = 0.0f;
            for (unsigned int v = 0; v < consStrTypes; v++) {
                variance += ((Pmax[v] - mean) * (Pmax[v] - mean));
            }
            variance *= mean;
            float stdDevMax = (float)std::sqrt((double)(variance));

            // Calculating actual standard deviation
            variance = 0.0f;
            for (unsigned int v = 0; v < consStrTypes; v++) {
                variance += ((ssu[v] - mean) * (ssu[v] - mean));
            }
            variance *= mean;
            float stdDev = (float)std::sqrt((double)(variance));

            // Normalizing standard deviation and calulating uncertainty
            unc = (1.0f - (stdDev / stdDevMax));
            // Just for rounding
            if (unc < 0.0f) {
                unc = 0.0f;
            }
            if (unc > 1.0f) {
                unc = 1.0f;
            }
        }

        // printf("%f\n", unc);

        // Assign values to arrays
        this->secStructUncertainty[uncMethod].Add(ssu);
        this->sortedSecStructAssignment[uncMethod].Add(ssa);
        this->uncertainty.Add(unc);
    }
    Log::DefaultLog.WriteMsg(
        Log::LEVEL_INFO, "Calculated uncertainty for secondary structure.", this->pdbIndex.Count()); // INFO
    return true;
}


/*
 * UncertaintyDataLoader::CalculateUncertaintyAverage
 */
bool UncertaintyDataLoader::CalculateUncertaintyAverage(void) {
    using megamol::core::utility::log::Log;

    // return if no data is present ...
    if (this->pdbIndex.IsEmpty()) {
        return false;
    }

    const unsigned int methodCnt =
        static_cast<unsigned int>(UncertaintyDataCall::assMethod::NOM) - 1; // UNCERTAINTY is not taken into account
    const unsigned int structTypes = static_cast<unsigned int>(UncertaintyDataCall::secStructure::NOE);
    const unsigned int uncMethod = static_cast<unsigned int>(UncertaintyDataCall::assMethod::UNCERTAINTY);

    // Reset uncertainty data
    this->secStructUncertainty[uncMethod].Clear();
    this->secStructUncertainty[uncMethod].AssertCapacity(this->pdbIndex.Count());
    this->sortedSecStructAssignment[uncMethod].Clear();
    this->sortedSecStructAssignment[uncMethod].AssertCapacity(this->pdbIndex.Count());
    this->uncertainty.Clear();
    this->uncertainty.AssertCapacity(this->pdbIndex.Count());

    // Create tmp structure type and structure uncertainty vectors for amino-acid
    vislib::math::Vector<float, structTypes> ssu;
    vislib::math::Vector<UncertaintyDataCall::secStructure, structTypes> ssa;

    // Initialize structure factors for all three methods with 1.0f
    vislib::Array<vislib::Array<float>> structFactor;
    for (unsigned int i = 0; i < methodCnt; i++) {
        structFactor.Add(vislib::Array<float>());
        for (unsigned int j = 0; j < structTypes; j++) {
            structFactor[i].Add(1.0f);
        }
    }

    // Calculate uncertainty
    // Loop over all amino-acids
    for (int i = 0; i < static_cast<int>(this->pdbIndex.Count()); i++) {
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
                    } else {
                        ssu[j] += structFactor[curMethod][j];
                    }
                } else {
                    ssu[j] += ((1.0f - structFactor[curMethod][j]) / ((float)(structTypes - 1)));
                }
            }
        }

        // Normalise structure uncertainty to [0.0,1.0]
        for (unsigned int j = 0; j < structTypes; j++) {
            ssu[j] /= abs((float)consideredMethods);
        }

        // Sorting structure types by their uncertainty
        this->QuickSortUncertainties(&(ssu), &(ssa), 0, (structTypes - 1));

        // Calculate reduced uncertainty value
        float unc = 0.0f;
        for (unsigned int k = 0; k < structTypes - 1; k++) {
            unc += (ssu[ssa[k]] - ssu[ssa[k + 1]]);
        }
        unc = 1.0f - unc;

        // Assign values to arrays
        this->secStructUncertainty[uncMethod].Add(ssu);
        this->sortedSecStructAssignment[uncMethod].Add(ssa);
        this->uncertainty.Add(unc);
    }

    Log::DefaultLog.WriteMsg(
        Log::LEVEL_INFO, "Calculated AVERAGE uncertainty for secondary structure.", this->pdbIndex.Count()); // INFO

    return true;
}


/*
 * UncertaintyDataLoader::QuickSortUncertainties
 */
void UncertaintyDataLoader::QuickSortUncertainties(
    vislib::math::Vector<float, static_cast<int>(UncertaintyDataCall::secStructure::NOE)>* valueArr,
    vislib::math::Vector<UncertaintyDataCall::secStructure, static_cast<int>(UncertaintyDataCall::secStructure::NOE)>*
        structArr,
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
