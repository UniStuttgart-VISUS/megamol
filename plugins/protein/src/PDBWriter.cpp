//
// PDBWriter.h
//
// Copyright (C) 2013 by University of Stuttgart (VISUS).
// All rights reserved.
//
// Created on: Apr 19, 2013
//     Author: scharnkn
//

#include "PDBWriter.h"
#include "mmcore/param/BoolParam.h"
#include "mmcore/param/ButtonParam.h"
#include "mmcore/param/IntParam.h"
#include "mmcore/param/StringParam.h"
#include "mmcore/utility/log/Log.h"
#include "sys/stat.h"
#include "vislib/sys/File.h"
#include <cmath>
#include <fstream>
#include <sstream>
#include <string>

#if defined(_WIN32)
#include <direct.h>
#endif

using namespace megamol;
using namespace megamol::protein;
using namespace megamol::protein_calls;
using namespace megamol::core::utility::log;

typedef unsigned int uint;

// TODO
// + Error handling if not enough space on device
// + Make folder creating routine plattform independent
// + handle file system errors
// + Check whether file exists before overwriting it

/*
 * PDBWriter::PDBWriter
 */
PDBWriter::PDBWriter()
        : AbstractJob()
        , Module()
        , dataCallerSlot("getdata", "Connects the writer module with the data source.")
        , writePQRSlot("writePQR", "Parameter to trigger writing of *.pqr files instead of *.pdb")
        , includeSolventAtomsSlot(
              "includeSolventAtoms", "Parameter to determine whether the solvent should be included")
        , writeSepFilesSlot(
              "writeSepFiles", "Parameter to determine whether all frames should be written into separate files")
        , minFrameSlot("minFrame", "Parameter to determine the first frame to be written")
        , nFramesSlot("nFrames", "Parameter to determine the number of frames to be written")
        , strideSlot("stride", "Parameter to determine the stride used when writing frames")
        , filenamePrefixSlot("filenamePrefix", "Parameter for the filename prefix")
        , outDirSlot("outputFolder", "Parameter for the output folder")
        , triggerButtonSlot("trigger", "Starts the pdb writing process")
        , rescaleBFactorSlot("rescaleBFactor", "If set, the BFactor is rescaled to a range from 0 to 100.")
        , jobDone(false)
        , filenameDigits(0)
        , useModelRecord(false) {

    // Make data caller slot available
    this->dataCallerSlot.SetCompatibleCall<MolecularDataCallDescription>();
    this->MakeSlotAvailable(&this->dataCallerSlot);

    // Parameter slot to trigger writing of *.pqr instead of *.pdb
    this->writePQRSlot << new core::param::BoolParam(false);
    this->MakeSlotAvailable(&this->writePQRSlot);

    // Parameter to include/exclude solvent when writing the file
    this->includeSolventAtomsSlot << new core::param::BoolParam(true);
    this->MakeSlotAvailable(&this->includeSolventAtomsSlot);

    // Parameter to write separate files
    this->writeSepFilesSlot << new core::param::BoolParam(false);
    this->MakeSlotAvailable(&this->writeSepFilesSlot);

    // Parameter to determine the first frame to be written
    this->minFrameSlot << new core::param::IntParam(0);
    this->MakeSlotAvailable(&this->minFrameSlot);

    // Parameter to determine the number of frames to be written
    this->nFramesSlot << new core::param::IntParam(1);
    this->MakeSlotAvailable(&this->nFramesSlot);

    // Parameter to determine the stride used when writing frames
    this->strideSlot << new core::param::IntParam(1);
    this->MakeSlotAvailable(&this->strideSlot);

    // Parameter for the filename prefix
    this->filenamePrefixSlot << new core::param::StringParam("out");
    this->MakeSlotAvailable(&this->filenamePrefixSlot);

    // Parameter for the output folder
    this->outDirSlot << new core::param::StringParam(".");
    this->MakeSlotAvailable(&this->outDirSlot);

    // Parameter for the trigger button
    this->triggerButtonSlot << new core::param::ButtonParam(core::view::Key::KEY_P);
    this->triggerButtonSlot.SetUpdateCallback(&PDBWriter::buttonCallback);
    this->MakeSlotAvailable(&this->triggerButtonSlot);

    // Parameter for the rescaling bool
    this->rescaleBFactorSlot << new core::param::BoolParam(false);
    this->MakeSlotAvailable(&this->rescaleBFactorSlot);
}


/*
 * PDBWriter::~PDBWriter
 */
PDBWriter::~PDBWriter() {
    this->Release();
}


/*
 * PDBWriter::IsRunning
 */
bool PDBWriter::IsRunning(void) const {
    return (!(this->jobDone));
}

/*
 * PDBWriter::buttonCallback
 */
bool PDBWriter::buttonCallback(core::param::ParamSlot& slot) {
    return this->Start();
}


/*
 * PDBWriter::Start
 */
bool PDBWriter::Start(void) {

    uint frameCnt;

    MolecularDataCall* mol = this->dataCallerSlot.CallAs<MolecularDataCall>();

    // Get extent of the data set
    if (mol == NULL) {
        this->jobDone = true;
        return false;
    }


    if (!(*mol)(MolecularDataCall::CallForGetExtent)) {
        return false;
    }
    frameCnt = mol->FrameCount();

    Log::DefaultLog.WriteInfo("%s: Number of frames %u", this->ClassName(), frameCnt);

    // Determine maximum frame to be written
    uint maxFrame = this->minFrameSlot.Param<core::param::IntParam>()->Value() +
                    this->strideSlot.Param<core::param::IntParam>()->Value() *
                        (this->nFramesSlot.Param<core::param::IntParam>()->Value() - 1);

    // Check whether the selected frames are valid
    if (maxFrame >= frameCnt) {
        Log::DefaultLog.WriteError("%s: Invalid frame selection (max frame is %u, but number of frames is %u",
            this->ClassName(), maxFrame, frameCnt);
        this->jobDone = true;
        return false;
    }

    // Determine number of digits to be used in generated filenames
    unsigned int counter = 0;
    this->filenameDigits = 0;
#if defined(_WIN32)
    this->filenameDigits = 6; // TODO Do not hardcode number of digits
#else
    this->filenameDigits = 6; // TODO Do not hardcode number of digits
#endif

    // Create output directories if necessary
    if (!this->createDirectories(this->outDirSlot.Param<core::param::StringParam>()->Value().c_str())) {
        this->jobDone = true;
        return false;
    }

    // Loop through all the selected frames
    for (int fr = this->minFrameSlot.Param<core::param::IntParam>()->Value(); fr <= static_cast<int>(maxFrame);
         fr += this->strideSlot.Param<core::param::IntParam>()->Value()) {

        this->useModelRecord = false;

        if (mol != NULL) {
            mol->SetFrameID(fr, true); // Set 'force' flag
            if (!(*mol)(MolecularDataCall::CallForGetData)) {
                this->jobDone = true;
                return false;
            }
        }

        if (mol != NULL) {
            if (this->writePQRSlot.Param<core::param::BoolParam>()->Value()) {
                // Write to pqr file
                if (!this->writePQR(mol)) {
                    this->jobDone = true;
                    return false;
                }
            } else {
                // Write to pdb file
                if (!this->writePDB(mol)) {
                    this->jobDone = true;
                    return false;
                }
            }
        }

        // Unlock frame
        if (mol != NULL) {
            mol->Unlock();
        }
    }

    this->jobDone = true;
    return true;
}


/*
 * PDBWriter::Terminate
 */
bool PDBWriter::Terminate(void) {
    return true; // TODO What does this do?
}


/*
 * PDBWriter::create
 */
bool PDBWriter::create(void) {
    return true;
}


/*
 * PDBWriter::release
 */
void PDBWriter::release(void) {}


/**
 * PDBWriter::createDirectories
 */
bool PDBWriter::createDirectories(vislib::StringA folder) {
    using namespace vislib;
    using namespace vislib::sys;
    if (File::IsDirectory(folder)) {
        return true;
    } else {
        if (folder.Contains("/")) {
            if (this->createDirectories(folder.Substring(0, folder.FindLast("/")))) {
#if defined(_WIN32)
                _mkdir(folder.PeekBuffer());
#else
                mkdir(folder, 777);
#endif
            }
        } else {
#if defined(_WIN32)
            _mkdir(folder.PeekBuffer());
#else
            mkdir(folder, 777);
#endif
        }
    }

    return true;
}


/*
 * PDBWriter::writePDB
 */
bool PDBWriter::writePDB(MolecularDataCall* mol) {
    using namespace vislib;

    uint modelCounter = 1;

    std::string filename;
    // If writing to seperate files: generate filename based on frame number
    if (this->writeSepFilesSlot.Param<core::param::BoolParam>()->Value()) {
        std::stringstream ss;
        ss.width(this->filenameDigits);
        ss.fill('0');
        std::string digits;
        ss << mol->FrameID();
        filename.append(
            StringA(this->outDirSlot.Param<core::param::StringParam>()->Value().c_str())); // Set output folder
        filename.append("/");
        filename.append(StringA(this->filenamePrefixSlot.Param<core::param::StringParam>()->Value().c_str())
                            .PeekBuffer()); // Set prefix
        filename.append(".");
        filename.append((ss.str()).c_str());
        filename.append(".pdb");
    } else { // Otherwise, use filename prefix and suffix
        filename.append(
            StringA(this->outDirSlot.Param<core::param::StringParam>()->Value().c_str())); // Set output folder
        filename.append("/");
        filename.append(StringA(this->filenamePrefixSlot.Param<core::param::StringParam>()->Value().c_str())
                            .PeekBuffer()); // Set prefix
        filename.append(".pdb");
    }

    if (mol->AtomCount() > 99999) {
        this->useModelRecord = true;
    }

    Log::DefaultLog.WriteInfo("%s: Writing frame %u to file '%s'", this->ClassName(), mol->FrameID(), filename.data());

    // Try to open the output file
    std::ofstream outfile;
    if (this->writeSepFilesSlot.Param<core::param::BoolParam>()->Value()) {
        outfile.open(filename.data(), std::ios::out | std::ios::binary);
    } else {
        // When using a single file, open with 'append' flag
        outfile.open(filename.data(), std::ios::out | std::ios::binary | std::ios::app);
    }
    if (!outfile.good()) {
        Log::DefaultLog.WriteError("%s: Unable to open file '%s'\n", this->ClassName(), filename.data());
        return false;
    }

    outfile.fill(' ');

    /* Remark about origin of the file */

    outfile << "REMARK    This file was created using MegaMol" << std::endl;


    /* Write bounding box extent into 'CRYST1' record */

    outfile << "REMARK    THESE ARE THE EXTENTS OF THE BBOX" << std::endl;
    outfile << "CRYST1  ";
    outfile.width(5);
    outfile.precision(2);
    outfile << std::fixed << mol->AccessBoundingBoxes().ObjectSpaceBBox().Width() << "  ";
    outfile.width(5);
    outfile.precision(2);
    outfile << std::fixed << mol->AccessBoundingBoxes().ObjectSpaceBBox().Height() << "  ";
    outfile.width(5);
    outfile.precision(2);
    outfile << std::fixed << mol->AccessBoundingBoxes().ObjectSpaceBBox().Depth();
    outfile << "  90.00  90.00  90.00 P 1           1" << std::endl;

    uint ch, m, res, at;

    float minBfactor = mol->MinimumBFactor();
    float maxBfactor = mol->MaximumBFactor();
    bool changeBfactor = this->rescaleBFactorSlot.Param<core::param::BoolParam>()->Value();

    /* Write 'ATOM' records for all atoms */

    // Loop through all chains
    for (ch = 0; ch < mol->ChainCount(); ++ch) {

        // Chain contains solvent atoms
        if ((mol->Chains()[ch].Type() == MolecularDataCall::Chain::SOLVENT) &&
            (!this->includeSolventAtomsSlot.Param<core::param::BoolParam>()->Value())) {
            continue; // Skip solvent molecules if wanted
        }

        // Loop through all molecules of this chain
        for (m = mol->Chains()[ch].FirstMoleculeIndex();
             m < mol->Chains()[ch].MoleculeCount() + mol->Chains()[ch].FirstMoleculeIndex(); ++m) {
            // Loop through all residues of this molecule
            for (res = mol->Molecules()[m].FirstResidueIndex();
                 res < mol->Molecules()[m].ResidueCount() + mol->Molecules()[m].FirstResidueIndex(); ++res) {
                // Loop through all atoms of this residue
                for (at = mol->Residues()[res]->FirstAtomIndex();
                     at < mol->Residues()[res]->AtomCount() + mol->Residues()[res]->FirstAtomIndex(); ++at) {

                    if ((at % 99999 == 0) && this->useModelRecord) {
                        outfile << "MODEL     ";
                        outfile.width(4);
                        outfile << std::right << modelCounter << std::endl;
                        modelCounter += 1;
                    }

                    outfile << "ATOM";

                    // Write atom number
                    outfile.width(7);
                    outfile << std::right << at % 99999 + 1;

                    // Write atom name
                    std::string atomName = std::string(mol->AtomTypes()[mol->AtomTypeIndices()[at]].Name());
                    outfile << " ";
                    if (atomName.length() <= 3) {
                        outfile << " ";
                        outfile.width(3);
                        outfile << std::left << atomName;
                    } else {
                        outfile << atomName;
                    }
                    outfile << " ";

                    // Write amino acid/residue name
                    outfile << mol->ResidueTypeNames()[mol->Residues()[res]->Type()];
                    outfile << " ";

                    // Write chain id
                    outfile << char(ch % 26 + 65);
                    //outfile << "X";

                    // Write residue number (using hex system to save space)
                    outfile.width(4);
                    //outfile << std::right << std::hex << res << std::dec;
                    outfile << std::right << (res + 1);
                    outfile << "    ";

                    // Write x coordinate
                    outfile.width(8);
                    outfile.precision(3);
                    outfile << std::fixed << std::right << mol->AtomPositions()[at * 3 + 0];

                    // Write y coordinate
                    outfile.width(8);
                    outfile.precision(3);
                    outfile << std::fixed << std::right << mol->AtomPositions()[at * 3 + 1];

                    // Write z coordinate
                    outfile.width(8);
                    outfile.precision(3);
                    outfile << std::fixed << std::right << mol->AtomPositions()[at * 3 + 2];

                    // Write occupancy
                    outfile.width(6);
                    outfile.precision(2);
                    outfile << std::fixed << std::right << mol->AtomOccupancies()[at];

                    // Write temperature factor
                    outfile.width(6);
                    outfile.precision(2);
                    if (changeBfactor) {
                        outfile << std::fixed << std::right
                                << (mol->AtomBFactors()[at] - minBfactor) * 100.0f / maxBfactor;
                    } else {
                        outfile << std::fixed << std::right << mol->AtomBFactors()[at];
                    }

                    outfile << "            " << std::endl;

                    if (this->useModelRecord && ((at % 99999 == 99998) || (at == mol->AtomCount() - 1))) {
                        outfile << "ENDMDL" << std::endl;
                    }
                }
            }
        }
    }

    // Write 'END' record
    outfile << "END" << std::endl;

    // Close the output file
    outfile.close();

    return true;
}


/*
 * PDBWriter::writePQR
 */
bool PDBWriter::writePQR(MolecularDataCall* mol) {
    using namespace vislib;

    uint modelCounter = 1;

    std::string filename;
    // If writing to seperate files: generate filename based on frame number
    if (this->writeSepFilesSlot.Param<core::param::BoolParam>()->Value()) {
        std::stringstream ss;
        ss.width(this->filenameDigits);
        ss.fill('0');
        std::string digits;
        ss << mol->FrameID();
        filename.append(
            StringA(this->outDirSlot.Param<core::param::StringParam>()->Value().c_str())); // Set output folder
        filename.append("/");
        filename.append(
            StringA(this->filenamePrefixSlot.Param<core::param::StringParam>()->Value().c_str())); // Set prefix
        filename.append(".");
        filename.append((ss.str()).c_str());
        filename.append(".pqr");
    } else { // Otherwise, use filename prefix and suffix
        filename.append(
            StringA(this->outDirSlot.Param<core::param::StringParam>()->Value().c_str())); // Set output folder
        filename.append("/");
        filename.append(
            StringA(this->filenamePrefixSlot.Param<core::param::StringParam>()->Value().c_str())); // Set prefix
        filename.append(".pqr");
    }

    Log::DefaultLog.WriteInfo("%s: Writing frame %u to file '%s'", this->ClassName(), mol->FrameID(), filename.data());

    // Try to open the output file
    std::ofstream outfile;
    if (this->writeSepFilesSlot.Param<core::param::BoolParam>()->Value()) {
        outfile.open(filename.data(), std::ios::out | std::ios::binary);
    } else {
        // When using a single file, open with 'append' flag
        outfile.open(filename.data(), std::ios::out | std::ios::binary | std::ios::app);
    }
    if (!outfile.good()) {
        Log::DefaultLog.WriteError("%s: Unable to open file '%s'\n", this->ClassName(), filename.data());
        return false;
    }

    outfile.fill(' ');

    /* Remark about origin of the file */

    outfile << "REMARK    This file was created using MegaMol" << std::endl;

    /* PQR File spedification */

    outfile << "REMARK    The B-factors in this file hold atomic charges" << std::endl;
    outfile << "REMARK    The occupancy in this file hold atomic radii" << std::endl;


    /* Write bounding box extent into 'CRYST1' record */

    outfile << "REMARK    THESE ARE THE EXTENTS OF THE BBOX" << std::endl;
    outfile << "CRYST1  ";
    outfile.width(5);
    outfile.precision(2);
    outfile << std::fixed << mol->AccessBoundingBoxes().ObjectSpaceBBox().Width() << "  ";
    outfile.width(5);
    outfile.precision(2);
    outfile << std::fixed << mol->AccessBoundingBoxes().ObjectSpaceBBox().Height() << "  ";
    outfile.width(5);
    outfile.precision(2);
    outfile << std::fixed << mol->AccessBoundingBoxes().ObjectSpaceBBox().Depth();
    outfile << "  90.00  90.00  90.00 P 1           1" << std::endl;

    uint ch, m, res, at;

    float minBfactor = mol->MinimumBFactor();
    float maxBfactor = mol->MaximumBFactor();
    bool changeBfactor = this->rescaleBFactorSlot.Param<core::param::BoolParam>()->Value();

    /* Write 'ATOM' records for all atoms */

    // Loop through all chains
    for (ch = 0; ch < mol->ChainCount(); ++ch) {

        // Chain contains solvent atoms
        if ((mol->Chains()[ch].Type() == MolecularDataCall::Chain::SOLVENT) &&
            (!this->includeSolventAtomsSlot.Param<core::param::BoolParam>()->Value())) {
            continue; // Skip solvent molecules if wanted
        }

        // Loop through all molecules of this chain
        for (m = mol->Chains()[ch].FirstMoleculeIndex();
             m < mol->Chains()[ch].MoleculeCount() + mol->Chains()[ch].FirstMoleculeIndex(); ++m) {
            // Loop through all residues of this molecule
            for (res = mol->Molecules()[m].FirstResidueIndex();
                 res < mol->Molecules()[m].ResidueCount() + mol->Molecules()[m].FirstResidueIndex(); ++res) {
                // Loop through all atoms of this residue
                for (at = mol->Residues()[res]->FirstAtomIndex();
                     at < mol->Residues()[res]->AtomCount() + mol->Residues()[res]->FirstAtomIndex(); ++at) {

                    if ((at % 99999 == 0) && this->useModelRecord) {
                        outfile << "MODEL     ";
                        outfile.width(4);
                        outfile << std::right << modelCounter << std::endl;
                        modelCounter += 1;
                    }

                    outfile << "ATOM";

                    // Write atom number
                    outfile.width(7);
                    outfile << std::right << at % 99999 + 1;

                    // Write atom name
                    std::string atomName = std::string(mol->AtomTypes()[mol->AtomTypeIndices()[at]].Name());
                    outfile << " ";
                    if (atomName.length() <= 3) {
                        outfile << " ";
                        outfile.width(3);
                        outfile << std::left << atomName;
                    } else {
                        outfile << atomName;
                    }
                    outfile << " ";

                    // Write amino acid/residue name
                    outfile << mol->ResidueTypeNames()[mol->Residues()[res]->Type()];
                    outfile << " ";

                    // Write chain id
                    outfile << char(ch % 26 + 65);

                    // Write residue number (using hex system to save space)
                    outfile.width(4);
                    //outfile << std::right << std::hex << res << std::dec;
                    outfile << std::right << res + 1;
                    printf("RES %u\n", res + 1);
                    outfile << "    ";

                    // Write x coordinate
                    outfile.width(8);
                    outfile.precision(3);
                    outfile << std::fixed << std::right << mol->AtomPositions()[at * 3 + 0];

                    // Write y coordinate
                    outfile.width(8);
                    outfile.precision(3);
                    outfile << std::fixed << std::right << mol->AtomPositions()[at * 3 + 1];

                    // Write z coordinate
                    outfile.width(8);
                    outfile.precision(3);
                    outfile << std::fixed << std::right << mol->AtomPositions()[at * 3 + 2];

                    // Write atom charges (note: are stored in occupancies array)
                    outfile.width(6);
                    outfile.precision(2);
                    outfile << std::fixed << std::right << mol->AtomOccupancies()[at];

                    // Write atom radii (note: are stored in B factor array)
                    outfile.width(6);
                    outfile.precision(2);
                    if (changeBfactor) {
                        outfile << std::fixed << std::right
                                << (mol->AtomBFactors()[at] - minBfactor) * 100.0f / maxBfactor;
                    } else {
                        outfile << std::fixed << std::right << mol->AtomBFactors()[at];
                    }

                    outfile << std::endl;

                    if (this->useModelRecord && ((at % 99999 == 99998) || (at == mol->AtomCount() - 1))) {
                        outfile << "ENDMDL" << std::endl;
                    }
                }
            }
        }
    }

    // Write 'END' record
    outfile << "END" << std::endl;

    // Close the output file
    outfile.close();

    return true;
}
