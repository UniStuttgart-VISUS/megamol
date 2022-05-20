/*
 * SolventHydroBondGenerator.cpp
 *
 * Copyright (C) 2011 by University of Stuttgart (VISUS).
 * All rights reserved.
 */


#include "SolventHydroBondGenerator.h"
#include "mmcore/param/BoolParam.h"
#include "mmcore/param/FilePathParam.h"
#include "mmcore/param/FloatParam.h"
#include "mmcore/param/IntParam.h"
#include "mmcore/param/StringParam.h"
#include "mmcore/utility/log/Log.h"
#include "mmcore/utility/sys/ASCIIFileBuffer.h"
#include "mmcore/utility/sys/MemmappedFile.h"
#include "vislib/ArrayAllocator.h"
#include "vislib/SmartPtr.h"
#include "vislib/StringConverter.h"
#include "vislib/StringTokeniser.h"
#include "vislib/math/ShallowPoint.h"
#include "vislib/math/mathfunctions.h"
#include "vislib/sys/PerformanceCounter.h"
#include "vislib/sys/sysfunctions.h"
#include "vislib/types.h"
#include <fstream>
#include <iostream>
#include <omp.h>

using namespace megamol;
using namespace megamol::core;
using namespace megamol::protein;
using namespace megamol::protein_calls;


megamol::protein::SolventHydroBondGenerator::SolventHydroBondGenerator()
        : dataOutSlot("dataout", "The slot providing the generated solvent data")
        , molDataInputCallerSlot("getInputData", "molecular data source (usually the PDB loader)")
        ,
        //hBondDataFile( "hBondDataFile", "file to store hydrogen bond data"),
        hBondDistance("hBondDistance", "distance for hydrogen bonds (angstroem?)")
        , hBondDonorAcceptorDistance(
              "hBondDonorAcceptorDistance", "distance between donor and acceptor of the hydrogen bonds")
        , hBondDonorAcceptorAngle(
              "hBondDonorAcceptorAngle", "angle between donor-acceptor and donor-hydrogen in degrees")
        , showMiddlePositions("showMiddlePositions", "show the middle of all atom positions over time") {
    this->molDataInputCallerSlot.SetCompatibleCall<MolecularDataCallDescription>();
    this->MakeSlotAvailable(&this->molDataInputCallerSlot);

    this->dataOutSlot.SetCallback(MolecularDataCall::ClassName(),
        MolecularDataCall::FunctionName(MolecularDataCall::CallForGetData), &SolventHydroBondGenerator::getData);
    this->dataOutSlot.SetCallback(MolecularDataCall::ClassName(),
        MolecularDataCall::FunctionName(MolecularDataCall::CallForGetExtent), &SolventHydroBondGenerator::getExtent);
    this->MakeSlotAvailable(&this->dataOutSlot);

    // distance for hydrogen bonds
    this->hBondDistance.SetParameter(new param::FloatParam(1.9f, 0.0f));
    this->MakeSlotAvailable(&this->hBondDistance);

    // distance between donor and acceptor of the hydrogen bonds
    this->hBondDonorAcceptorDistance.SetParameter(new param::FloatParam(3.5f, 0.0f));
    this->MakeSlotAvailable(&this->hBondDonorAcceptorDistance);

    // angle between donor-acceptor and donor-hydrogen in degrees
    this->hBondDonorAcceptorAngle.SetParameter(new param::FloatParam(30.0f, 0.0f));
    this->MakeSlotAvailable(&this->hBondDonorAcceptorAngle);

    //this->hBondDataFile.SetParameter(new param::StringParam("hbond.dat"));
    //this->MakeSlotAvailable( &this->hBondDataFile);

    this->showMiddlePositions.SetParameter(new param::BoolParam(false));
    this->MakeSlotAvailable(&this->showMiddlePositions);

    for (int i = 0; i < HYDROGEN_BOND_IN_CORE; i++)
        curHBondFrame[i] = -1;

    this->maxOMPThreads = omp_get_max_threads();
    this->neighbourIndices = new vislib::Array<unsigned int>[this->maxOMPThreads];
    //this->neighbHydrogenIndices = new vislib::Array<unsigned int>[this->maxOMPThreads];
}

megamol::protein::SolventHydroBondGenerator::~SolventHydroBondGenerator() {
    delete[] this->neighbourIndices;
    this->Release();
}

bool megamol::protein::SolventHydroBondGenerator::create(void) {
    // hier alle initialisierungen rein, die fehlschlagen können
    return true;
}

void megamol::protein::SolventHydroBondGenerator::release(void) {
    // hier alles freigeben was in create() initialisiert wird!
}

/**
 * -> preprocessing step
 *
 *- Aufenthaltswahrscheinlichkeit/-dauer (einzelne Molekuele oder Molekueltypen ueber komplette Trajektorie
 * berechnen & als Farbe auf statische Molekueloberflueche mappen) *
 * -> das geht so net! das laeuft auf ne volumen-akkumulation hinaus ...
 */
void megamol::protein::SolventHydroBondGenerator::calcSpatialProbabilities(
    MolecularDataCall* src, MolecularDataCall* dst) {
    int nFrames = src->FrameCount();
    int nAtoms = src->AtomCount();

    if ((int)this->middleAtomPos.Count() < nAtoms * 3) {
        this->middleAtomPos.SetCount(nAtoms * 3);
    }
    memset(&this->middleAtomPos[0], 0, this->middleAtomPos.Count() * sizeof(float));

    float* middlePosPtr = &this->middleAtomPos[0];
    const float* atomPositions = src->AtomPositions();

    //#pragma omp parallel for private( ??? )
    for (int i = 0; i < nFrames; i++) {
        src->SetFrameID(i, true);
        if (!(*src)(MolecularDataCall::CallForGetData))
            continue; // return false;

#pragma omp parallel for
        for (int aIdx = 0; aIdx < nAtoms * 3; aIdx += 3) {
            middlePosPtr[aIdx] += atomPositions[aIdx];
            middlePosPtr[aIdx + 1] += atomPositions[aIdx + 1];
            middlePosPtr[aIdx + 2] += atomPositions[aIdx + 2];
        }
    }

    float normalize = 1.0f / nFrames;

#pragma omp parallel for
    for (int aIdx = 0; aIdx < nAtoms * 3; aIdx++)
        middlePosPtr[aIdx] *= normalize;

    //molDest->SetAtomPositions(middleAtomPos.PeekElements());
}

/*
 *
 *
void megamol::protein::SolventDataGenerator::findDonors(MolecularDataCall *data) {
        for( int i = 0; i < data->ResidueCount(; i++ ) {
                MolecularDataCall::Residue *residue = data->Residues()[i];
        }
}
*/


#if 0
class HbondIO {
private:
    vislib::sys::File *readHandle, *writeHandle;

    /*vislib::sys::File::FileSize*/
    unsigned int dataStartOffset, atomCount, frameCount, frameSizeInBytes;

public:
    HbondIO(unsigned int atomCount, unsigned int frameCount, const vislib::StringA& fname, bool read)
        : dataStartOffset(0), atomCount(atomCount), frameCount(frameCount) {
        if (read) {
            readHandle = new vislib::sys::MemmappedFile();
            if (!readHandle->Open(fname, vislib::sys::File::READ_ONLY, vislib::sys::File::SHARE_READ, vislib::sys::File::OPEN_ONLY)) {
                delete readHandle;
                readHandle = 0;
            }
            writeHandle = 0;
        } else {
            writeHandle = new vislib::sys::MemmappedFile();
            if (!writeHandle->Open(fname, vislib::sys::File::WRITE_ONLY, vislib::sys::File::SHARE_READ, vislib::sys::File::CREATE_OVERWRITE)) {
                delete writeHandle;
                writeHandle = 0;
            }
            readHandle = 0;
        }
    }
    ~HbondIO() {
        delete readHandle;
        delete writeHandle;
    }

    bool writeFrame(int *frame, unsigned int frameId) {
        if (!writeHandle)
            return false;

        // write header at first
        if ( dataStartOffset == 0) {
            dataStartOffset = 4*sizeof(dataStartOffset); // store 4 ineteger entries as header for now ...
            frameSizeInBytes = atomCount * sizeof(int); // 1 signed integer per atom specifying the hbond connection to another atom
            writeHandle->Write(&dataStartOffset, sizeof(dataStartOffset));
            writeHandle->Write(&atomCount, sizeof(atomCount));
            writeHandle->Write(&frameCount, sizeof(frameCount));
            writeHandle->Write(&frameSizeInBytes, sizeof(frameSizeInBytes));
            ASSERT(writeHandle->Tell()==dataStartOffset);
            if (writeHandle->Tell()!=dataStartOffset)
                return false;
        }

        ASSERT(frameId < frameCount);
        writeHandle->Seek(dataStartOffset + frameId*frameSizeInBytes);
        if (writeHandle->Write(frame, frameSizeInBytes) != frameSizeInBytes)
            return false;
        return true;
    }

    bool readFrame(int *frame, unsigned int frameId) {
        if (!readHandle)
            return false;

        // read header at first
        if ( dataStartOffset == 0) {
            readHandle->Read(&dataStartOffset, sizeof(dataStartOffset));
            readHandle->Read(&atomCount, sizeof(atomCount));
            readHandle->Read(&frameCount, sizeof(frameCount));
            readHandle->Read(&frameSizeInBytes, sizeof(frameSizeInBytes));
            ASSERT(frameSizeInBytes==atomCount * sizeof(int)); // this is some sort of file type/integrity check ...
            if (frameSizeInBytes!=atomCount * sizeof(int))
                return false;
        }

        ASSERT(frameId < frameCount);
        readHandle->Seek(dataStartOffset + frameId*frameSizeInBytes);
        if (readHandle->Read(frame, frameSizeInBytes) != frameSizeInBytes)
            return false;
        return true;
    }
};
#endif


void megamol::protein::SolventHydroBondGenerator::calcHydroBondsForCurFrame(
    MolecularDataCall* data, const float* atomPositions, int* atomHydroBondsIndicesPtr) {
    //const float *atomPositions = data->AtomPositions();
    const MolecularDataCall::AtomType* atomTypes = data->AtomTypes();
    const unsigned int* atomTypeIndices = data->AtomTypeIndices();
    const int* atomResidueIndices = data->AtomResidueIndices();

    // set all entries to "not connected"
    memset(atomHydroBondsIndicesPtr, -1, sizeof(int) * data->AtomCount());

    if (reverseConnection.Count() < data->AtomCount())
        reverseConnection.SetCount(data->AtomCount());
    int* reverseConnectionPtr = &reverseConnection[0];
    memset(reverseConnectionPtr, -1, reverseConnection.Count() * sizeof(int));

    vislib::sys::PerformanceCounter timer(true);
    //timer.SetMark();

#if 0
    float hbondDist = hBondDistance.Param<param::FloatParam>()->Value();
    neighbourFinder.SetPointData(atomPositions, data->AtomCount(), data->AccessBoundingBoxes().ObjectSpaceBBox(), hbondDist);

    // looping over residues may not be a good idea?! (index-traversal?) loop over all possible acceptors ...
#pragma omp parallel for
    for( int rIdx = 0; rIdx < data->ResidueCount(); rIdx++ ) {
        const MolecularDataCall::Residue *residue = data->Residues()[rIdx];

        // we're only interested in hydrogen bonds between polymer/protein molecule and surounding solvent
        if (data->IsSolvent(residue))
            continue;

        // find possible acceptor atoms in the current residuum (for now just O, N, C(?)

        // vorerst nur Sauerstoff und Stickstoff als Akzeptor/Donator (N, O)
        int lastAtomIdx = residue->FirstAtomIndex()+residue->AtomCount();
        for(int aIdx = residue->FirstAtomIndex(); aIdx < lastAtomIdx; aIdx++) {
            // is this atom already connected?
            if (reverseConnectionPtr[aIdx] != -1)
                continue;

            const MolecularDataCall::AtomType& t = atomTypes[atomTypeIndices[aIdx]];
            const vislib::StringA& name = t.Name();
            char element = name[0];

/*
JW: ich fuerchte fuer eine allgemeine Deffinition der Wasserstoffbruecken muss man ueber die Bindungsenergien gehen und diese berechnen.
Fuer meine Simulationen und alle Bio-Geschichten reicht die Annahme, dass Sauerstoff, Stickstoff und Fluor (was fast nie vorkommt)
Wasserstoffbruecken bilden und dabei als Donor und Aktzeptor dienen koenne. Dabei ist der Wasserstoff am Donor gebunden und bildet die Bruecke zum Akzeptor.
*/
            if (element=='N' || element=='O' /*|| element=='F' || element=='C'??*/) {
                int ompThreadID = omp_get_thread_num();
                neighbourIndices[ompThreadID].Clear(); // clear, keep capacity ...
                neighbourIndices[ompThreadID].SetCapacityIncrement( 100); // set capacity increment
                neighbourFinder.FindNeighboursInRange(&atomPositions[aIdx*3], hbondDist, neighbourIndices[ompThreadID]);
                for(int nIdx = 0; nIdx<neighbourIndices[ompThreadID].Count(); nIdx++) {
                    int neighbIndex = neighbourIndices[ompThreadID][nIdx];
                    // atom from the current residue?
                    if (atomResidueIndices[neighbIndex]==rIdx)
                        continue;
                    // check if a H-atom is in range and add a h-bond...
                    if (atomTypes[atomTypeIndices[neighbIndex]].Name()[0]=='H') {
                        atomHydroBondsIndicesPtr[aIdx] = neighbIndex;
                        // avoid double checks - only one hydrogen bond per atom?!
                        reverseConnectionPtr[neighbIndex] = aIdx;
                        // TODO: maybe mark double time? or double with negative index?
                    }
                }
            } else if (element=='H') {
                // TODO !?
            }
        }
    }
#else

    /* create hydrogen connections */
    if (this->hydrogenConnections.Count() <= data->AtomCount() * MAX_HYDROGENS_PER_ATOM) {
        this->hydrogenConnections.SetCount(data->AtomCount() * MAX_HYDROGENS_PER_ATOM);
        memset(&this->hydrogenConnections[0], -1, this->hydrogenConnections.Count() * sizeof(int));
        int count = data->ConnectionCount();
        for (int i = 0; i < count; i++) {
            int idx0 = data->Connection()[2 * i];
            int idx1 = data->Connection()[2 * i + 1];
            char element0 = atomTypes[atomTypeIndices[idx0]].Name()[0];
            char element1 = atomTypes[atomTypeIndices[idx1]].Name()[0];

            /* make sure the hydrogen atom is 'idx1' */
            if (element0 == 'H') {
                vislib::math::Swap(idx0, idx1);
                vislib::math::Swap(element0, element1);
            }

            // check if we have a possible donor/acceptor here ...
            if (element0 != 'O' && element0 != 'N')
                continue;

            // add hydrogen connection if present ...
            if (element1 == 'H') {
                int hydrogenConnIdx = idx0 * MAX_HYDROGENS_PER_ATOM;
                for (int j = 0; j < MAX_HYDROGENS_PER_ATOM; j++) {
                    if (hydrogenConnections[hydrogenConnIdx] == -1) {
                        hydrogenConnections[hydrogenConnIdx] = idx1;
                        break;
                    }
                    hydrogenConnIdx++;
                }
            }
        }

        this->donorAcceptors.SetCount(data->AtomCount());
        memset(&this->donorAcceptors[0], -1, this->donorAcceptors.Count() * sizeof(int));
        for (unsigned int i = 0; i < data->AtomCount(); i++) {
            char element = atomTypes[atomTypeIndices[i]].Name()[0];
            if (element == 'O' || element == 'N')
                donorAcceptors[i] = 1;
        }
    }

    // only fill in donors/acceptors into the neighbour finder grid ...
    float hbondDonorAcceptorDist = hBondDonorAcceptorDistance.Param<param::FloatParam>()->Value();
    float hbondDonorAcceptorAngle = hBondDonorAcceptorAngle.Param<param::FloatParam>()->Value() *
                                    static_cast<float>(vislib::math::PI_DOUBLE / 180.0);
    neighbourFinder.SetPointData(atomPositions, data->AtomCount(), data->AccessBoundingBoxes().ObjectSpaceBBox(),
        hbondDonorAcceptorDist, &donorAcceptors[0]);

    const int* hydrogenConnectionsPtr = hydrogenConnections.PeekElements();

    // looping over residues may not be a good idea?! (index-traversal?) loop over all possible acceptors ...
#pragma omp parallel for
    for (int rIdx = 0; rIdx < static_cast<int>(data->ResidueCount()); rIdx++) {
        const MolecularDataCall::Residue* residue = data->Residues()[rIdx];

        // we're only interested in hydrogen bonds between polymer/protein molecule and surounding solvent
        //#error großes Problem: es werden so nicht alle Wasserstoffbrücken gefunden?!
        if (data->IsSolvent(residue))
            continue;

        // find possible acceptor atoms in the current residuum (for now just O, N, C(?)

        // vorerst nur Sauerstoff und Stickstoff als Akzeptor/Donator (N, O)
        unsigned int lastAtomIdx = residue->FirstAtomIndex() + residue->AtomCount();
        for (unsigned int atomIndex = residue->FirstAtomIndex(); atomIndex < lastAtomIdx; atomIndex++) {
            // is this atom already connected?
            //if (reverseConnectionPtr[atomIndex] >= 0 continue;

            /*
            JW: ich fuerchte fuer eine allgemeine Deffinition der Wasserstoffbruecken muss man ueber die Bindungsenergien gehen und diese berechnen.
            Fuer meine Simulationen und alle Bio-Geschichten reicht die Annahme, dass Sauerstoff, Stickstoff und Fluor (was fast nie vorkommt)
            Wasserstoffbruecken bilden und dabei als Donor und Aktzeptor dienen koenne. Dabei ist der Wasserstoff am Donor gebunden und bildet die Bruecke zum Akzeptor.
            */

            //#error poly->solv, solv->poly! und poly->poly! solv->solv auf keine fall auf der oberfläche! (zumindest unterscheiden)

            // nitrogen and oxygen can be donors and acceptors here ...
            if (donorAcceptors[atomIndex] != -1 /*element=='N' || element=='O'*/) {
                // access a private array for this parallel thread ...
                vislib::Array<unsigned int>& privateNeighbourIndices = neighbourIndices[omp_get_thread_num()];
                privateNeighbourIndices.Clear();                   // clear, keep capacity ...
                privateNeighbourIndices.SetCapacityIncrement(100); // set capacity increment
                neighbourFinder.FindNeighboursInRange(
                    &atomPositions[atomIndex * 3], hbondDonorAcceptorDist, privateNeighbourIndices);

                for (int nIdx = 0; nIdx < (int)privateNeighbourIndices.Count(); nIdx++) {
                    int neighbIndex = privateNeighbourIndices[nIdx];
                    //char elementNeighb = atomTypes[atomTypeIndices[neighbIndex]].Name()[0];

                    // atom from the current residue?
                    if (atomResidueIndices[neighbIndex] == rIdx)
                        continue;

                    //ASSERT(donorAcceptors[neighbIndex] != -1);
                    //if ( elementNeighb=='O' || elementNeighb=='N' ) { ... }


                    // DEBUG
                    //atomHydroBondsIndicesPtr[atomIndex] = neighbIndex;
                    //atomHydroBondsIndicesPtr[neighbIndex] = atomIndex;

                    // check for other acceptor/donor - all atoms inside 'neighbourFinder' only consist of donor/acceptor atoms ..
                    // loop over hydrogen atoms from donor 'atomIndex'-  - 'neighbIndex' is the acceptor
                    int hydrogenConnIdx = atomIndex * MAX_HYDROGENS_PER_ATOM;
                    for (int j = 0; j < MAX_HYDROGENS_PER_ATOM; j++) {
                        int hydrogenAtomIdx = hydrogenConnectionsPtr[hydrogenConnIdx];
                        if (hydrogenAtomIdx != -1 && validHydrogenBond(atomIndex, hydrogenAtomIdx, neighbIndex,
                                                         atomPositions, hbondDonorAcceptorAngle)) {
                            atomHydroBondsIndicesPtr[neighbIndex] = hydrogenAtomIdx;
                            // mark this donor/acceptor pair as already connetected with a hydrogen bond
                            //reverseConnectionPtr[atomIndex] = neighbIndex;
                            // TODO: maybe mark double time? or double with negative index?
                            break;
                        }
                        hydrogenConnIdx++;
                    }
                    // loop over hydrogen atoms from donor 'neighbIndex' - 'atomIndex' is the acceptor
                    hydrogenConnIdx = neighbIndex * MAX_HYDROGENS_PER_ATOM;
                    for (int j = 0; j < MAX_HYDROGENS_PER_ATOM; j++) {
                        int hydrogenAtomIdx = hydrogenConnectionsPtr[hydrogenConnIdx];
                        if (hydrogenAtomIdx != -1 && validHydrogenBond(neighbIndex, hydrogenAtomIdx, atomIndex,
                                                         atomPositions, hbondDonorAcceptorAngle)) {
                            atomHydroBondsIndicesPtr[atomIndex] = hydrogenAtomIdx;
                            // mark this donor/acceptor pair as already connetected with a hydrogen bond
                            //reverseConnectionPtr[neighbIndex] = atomIndex;
                            // TODO: maybe mark double time? or double with negative index?
                            break;
                        }
                        hydrogenConnIdx++;
                    }
                }
            }
        }
    }
#endif

    std::cout << "Hydrogen bonds computed in " << std::fixed << timer.ToMillis(timer.Difference()) << " ms."
              << std::endl;
}


#if 1
bool megamol::protein::SolventHydroBondGenerator::calcHydrogenBondStatistics(
    MolecularDataCall* dataTarget, MolecularDataCall* dataSource) {
    int savedSrcFrameId = dataSource->FrameID();
    int savedTargetFrameId = dataTarget->FrameID();

    int solvResCount = dataSource->AtomSolventResidueCount();
    const unsigned int* solventResidueIndices = dataSource->SolventResidueIndices();

    if (this->hydrogenBondStatistics.Count() < solvResCount * dataSource->AtomCount()) {
        this->hydrogenBondStatistics.SetCount(solvResCount * dataSource->AtomCount());
        memset(&this->hydrogenBondStatistics[0], 0, hydrogenBondStatistics.Count() * sizeof(unsigned int));
    }
    unsigned int* hydrogenBondStatisticsPtr = &this->hydrogenBondStatistics[0];

    unsigned int solventAtoms = 0;
    unsigned int polymerAtoms = 0;

    for (unsigned int frameId = 0; frameId < dataSource->FrameCount(); frameId++) {
        dataSource->SetFrameID(frameId);
        dataTarget->SetFrameID(frameId);
        if (!(*dataSource)(MolecularDataCall::CallForGetData))
            return false;
        if (!this->getHBonds(dataTarget, dataSource))
            return false;
        int atomCnt = dataTarget->AtomCount();
        const int* hydrogenBonds = dataTarget->AtomHydrogenBondIndices();
        const int* residueIndices = dataTarget->AtomResidueIndices();
        /*#pragma omp parallel for
                        for(int atomIdx = 0; atomIdx < atomCnt; atomIdx++) {
                                int residueIdx = residueIndices[atomIdx];
                                if (dataTarget->IsSolvent(dataTarget->Residues()[residueIdx]))
                                        continue;
                        }*/

        //#pragma omp parallel for
        for (unsigned int rIdx = 0; rIdx < dataSource->ResidueCount(); rIdx++) {
            const MolecularDataCall::Residue* residue = dataSource->Residues()[rIdx];

            //if (dataSource->IsSolvent(residue)) continue;
            bool isSolvent = dataSource->IsSolvent(residue);

            if (frameId == 0) {
                if (isSolvent)
                    solventAtoms += residue->AtomCount();
                else
                    polymerAtoms += residue->AtomCount();
            }

            // vorerst nur Sauerstoff und Stickstoff als Akzeptor/Donator (N, O)
            int lastAtomIdx = residue->FirstAtomIndex() + residue->AtomCount();
            for (int atomIndex = residue->FirstAtomIndex(); atomIndex < lastAtomIdx; atomIndex++) {
                if (hydrogenBonds[atomIndex] == -1)
                    continue;
                int otherAtomIndex = hydrogenBonds[atomIndex];
                int otherResidueIdx = residueIndices[otherAtomIndex];
                const MolecularDataCall::Residue* otherResidue = dataSource->Residues()[otherResidueIdx];
                bool isOtherSolvent = dataTarget->IsSolvent(otherResidue);
#if 1

                if (isSolvent != isOtherSolvent) {
                    // polymer/solvent HBond ...
                    if (!isSolvent) {
                        for (int srIdx = 0; srIdx < solvResCount; srIdx++) {
                            if (solventResidueIndices[srIdx] == otherResidue->Type())
                                hydrogenBondStatisticsPtr[solvResCount * atomIndex + srIdx]++;
                        }
                        //hydrogenBondStatisticsPtr[solvResCount*atomIndex + (rIdx%this->solvResCount)]++;
                    } else /*if (!isOtherSolvent)*/ {
                        for (int srIdx = 0; srIdx < solvResCount; srIdx++) {
                            if (solventResidueIndices[srIdx] == residue->Type())
                                hydrogenBondStatisticsPtr[solvResCount * otherAtomIndex + srIdx]++;
                        }
                        //hydrogenBondStatisticsPtr[solvResCount*otherAtomIndex + (otherResidueIdx%this->solvResCount)]++;
                    }
                } else {
                    // both atoms solvent or both polymer?
                }
#else
                if (!isSolvent)
                    hydrogenBondStatisticsPtr[atomIndex]++;
                if (!isOtherSolvent)
                    hydrogenBondStatisticsPtr[otherAtomIndex]++;
#endif
            }
        }
    }

    /*dataSource*/ dataTarget->SetAtomHydrogenBondStatistics(hydrogenBondStatisticsPtr /*, this->solvResCount*/);
    dataSource->SetFrameID(savedSrcFrameId);
    dataTarget->SetFrameID(savedTargetFrameId);

    std::cout << "solvent atoms: " << solventAtoms << " polymer atoms: " << polymerAtoms << std::endl;

    return true;
}
#endif

bool megamol::protein::SolventHydroBondGenerator::getHBonds(
    MolecularDataCall* dataTarget, MolecularDataCall* dataSource) {
    int reqFrame = dataTarget->FrameID();
    int cacheIndex = reqFrame % HYDROGEN_BOND_IN_CORE;
    float hbondDist = hBondDistance.Param<param::FloatParam>()->Value();

    if (curHBondFrame[cacheIndex] == reqFrame) {
        // recalc hbonds if 'hBondDistance' has changed?!
        //if( this->hBondDistance.IsDirty() )
        dataTarget->SetAtomHydrogenBondIndices(atomHydroBondsIndices[cacheIndex].PeekElements());
        dataTarget->SetAtomHydrogenBondDistance(hbondDist);
        return true;
    }

    vislib::Array<int>& atomHydroBonds = this->atomHydroBondsIndices[cacheIndex];
    if (atomHydroBonds.Count() < dataSource->AtomCount())
        atomHydroBonds.SetCount(dataSource->AtomCount());

#if 0
    const vislib::TString fileName = hBondDataFile.Param<param::StringParam>()->Value();
    if (fileName.IsEmpty())
        return false;

    if (vislib::sys::File::Exists(fileName)) {
        HbondIO input(dataSource->AtomCount(), dataSource->FrameCount(), fileName, true);
        if (input.readFrame(&atomHydroBonds[0], reqFrame)) {
            curHBondFrame[cacheIndex] = reqFrame;
            dataTarget->SetAtomHydrogenBondIndices(atomHydroBonds.PeekElements());
            dataTarget->SetAtomHydrogenBondDistance(hbondDist);
            return true;
        }
        return false;
    }

    HbondIO *output = new HbondIO(data->AtomCount(), data->FrameCount(), fileName, false);

    int *tmpArray = new int[data->AtomCount()];

    // calculate hydrogen bounds for all frames and store them in a file ...
    for(int frameId = 0; frameId < data->FrameCount(); frameId++) {
        int *atomHydroBondsIndicesPtr = (frameId == reqFrame ? &atomHydroBonds[0] : tmpArray);

#if 0
        // workaround to avoid recursion ...
        {
            int tmp = curHBondFrame[frameId % HYDROGEN_BOND_IN_CORE];
            curHBondFrame[frameId % HYDROGEN_BOND_IN_CORE] = frameId;
            data->SetFrameID(frameId);
            if( !(*dataTarget)(MolecularDataCall::CallForGetData))
                return false;
            curHBondFrame[frameId % HYDROGEN_BOND_IN_CORE] = tmp;
            calcHydroBondsForCurFrame(dataTarget, atomHydroBondsIndicesPtr);
        }
#else
        dataSource->SetFrameId(frameId);
        if( !(*dataSource)(MolecularDataCall::CallForGetData))
            return false;
        calcHydroBondsForCurFrame(dataSource, atomHydroBondsIndicesPtr);
#endif

        output->writeFrame(atomHydroBondsIndicesPtr, frameId);
    }
    delete output;
    delete tmpArray;

#else
    dataSource->SetFrameID(reqFrame);
    if (!(*dataSource)(MolecularDataCall::CallForGetData))
        return false;
    calcHydroBondsForCurFrame(dataSource, &atomHydroBonds[0]);
#endif

    curHBondFrame[cacheIndex] = reqFrame;
    dataTarget->SetFrameID(reqFrame);
    dataTarget->SetAtomHydrogenBondDistance(hbondDist);
    dataTarget->SetAtomHydrogenBondIndices(atomHydroBonds.PeekElements());

    return true;
}

bool megamol::protein::SolventHydroBondGenerator::getExtent(core::Call& call) {
    MolecularDataCall* molDest = dynamic_cast<MolecularDataCall*>(&call); // dataOutSlot ??
    MolecularDataCall* molSource = this->molDataInputCallerSlot.CallAs<MolecularDataCall>();

    if (!molDest || !molSource)
        return false;

    if (!(*molSource)(MolecularDataCall::CallForGetExtent))
        return false;

    // forward data ...
    molDest->AccessBoundingBoxes().Clear();
    molDest->AccessBoundingBoxes().SetObjectSpaceBBox(molSource->AccessBoundingBoxes().ObjectSpaceBBox());
    molDest->AccessBoundingBoxes().SetObjectSpaceClipBox(molSource->AccessBoundingBoxes().ObjectSpaceClipBox());
    molDest->SetFrameCount(molSource->FrameCount());
    molDest->SetDataHash(molSource->DataHash());
    return true;
}

bool megamol::protein::SolventHydroBondGenerator::getData(core::Call& call) {
    MolecularDataCall* molDest = dynamic_cast<MolecularDataCall*>(&call); // dataOutSlot ??
    MolecularDataCall* molSource = this->molDataInputCallerSlot.CallAs<MolecularDataCall>();

    if (!molDest || !molSource)
        return false;

    molSource->SetFrameID(molDest->FrameID()); // forward frame request
    if (!(*molSource)(MolecularDataCall::CallForGetData))
        return false;

    *molDest = *molSource;

    // testing?
    if (!this->hydrogenBondStatistics.Count()) {
        this->calcHydrogenBondStatistics(molDest, molSource);
    } else {
        molDest->SetAtomHydrogenBondStatistics(this->hydrogenBondStatistics.PeekElements());
    }

    if (this->showMiddlePositions.Param<param::BoolParam>()->Value()) {
        if (!middleAtomPos.Count()) {
            calcSpatialProbabilities(molSource, molDest);

            // calc hydrogen bonds for middle positions
            if (middleAtomPosHBonds.Count() < molSource->AtomCount())
                middleAtomPosHBonds.SetCount(molSource->AtomCount());
            calcHydroBondsForCurFrame(molSource, middleAtomPos.PeekElements(), &middleAtomPosHBonds[0]);
        }
        molDest->SetAtomPositions(middleAtomPos.PeekElements());
        molDest->SetAtomHydrogenBondDistance(hBondDistance.Param<param::FloatParam>()->Value());
        molDest->SetAtomHydrogenBondIndices(middleAtomPosHBonds.PeekElements());
        molDest->SetDataHash(molSource->DataHash() * 666);
    } else {

        // reset all hbond data if this parameter changes ...
        if (this->hBondDistance.IsDirty() || this->hBondDonorAcceptorDistance.IsDirty() ||
            this->hBondDonorAcceptorAngle.IsDirty()) {
            this->hBondDistance.ResetDirty();
            this->hBondDonorAcceptorDistance.ResetDirty();
            this->hBondDonorAcceptorAngle.ResetDirty();
            for (int i = 0; i < HYDROGEN_BOND_IN_CORE; i++)
                this->curHBondFrame[i] = -1;
            molDest->SetDataHash(molSource->DataHash() * 666); // hacky ?
        }

        getHBonds(molDest, molSource);
    }


    molDest->SetUnlocker(new SolventHydroBondGenerator::Unlocker(*molSource), false);
    return true;
}
