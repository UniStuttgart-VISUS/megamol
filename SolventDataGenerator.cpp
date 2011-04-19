/*
 * SolventDataGenerator.cpp
 *
 * Copyright (C) 2011 by University of Stuttgart (VISUS).
 * All rights reserved.
 */


#include "stdafx.h"
#include "SolventDataGenerator.h"
#include "param/FilePathParam.h"
#include "param/IntParam.h"
#include "param/BoolParam.h"
#include "param/StringParam.h"
#include "param/FloatParam.h"
#include "vislib/ArrayAllocator.h"
#include "vislib/Log.h"
#include "vislib/mathfunctions.h"
#include "vislib/MemmappedFile.h"
#include "vislib/SmartPtr.h"
#include "vislib/types.h"
#include "vislib/sysfunctions.h"
#include "vislib/StringConverter.h"
#include "vislib/StringTokeniser.h"
#include "vislib/ASCIIFileBuffer.h"
#include "vislib/ShallowPoint.h"
#include <ctime>
#include <iostream>
#include <fstream>
#include <omp.h>

using namespace megamol;
using namespace megamol::core;
using namespace megamol::protein;


megamol::protein::SolventDataGenerator::SolventDataGenerator() :
		dataOutSlot( "dataout", "The slot providing the generated solvent data"),
		molDataInputCallerSlot( "getInputData", "molecular data source (usually the PDB loader)"),
		hBondDataFile( "hBondDataFile", "file to store hydrogen bond data"),
		hBondDistance("hBondDistance", "distance for hydrogen bonds (angstroem?)"),
		showMiddlePositions("showMiddlePositions", "show the middle of all atom positions over time")
{
	this->molDataInputCallerSlot.SetCompatibleCall<MolecularDataCallDescription>();
	this->MakeSlotAvailable( &this->molDataInputCallerSlot);

	this->dataOutSlot.SetCallback( MolecularDataCall::ClassName(), MolecularDataCall::FunctionName(MolecularDataCall::CallForGetData), &SolventDataGenerator::getData);
	this->dataOutSlot.SetCallback( MolecularDataCall::ClassName(), MolecularDataCall::FunctionName(MolecularDataCall::CallForGetExtent), &SolventDataGenerator::getExtent);
	this->MakeSlotAvailable( &this->dataOutSlot);

	// distance for hydrogen bonds
	this->hBondDistance.SetParameter(new param::FloatParam(1.9f, 0.0f));
	this->MakeSlotAvailable( &this->hBondDistance);

	this->hBondDataFile.SetParameter(new param::StringParam("hbond.dat"));
	this->MakeSlotAvailable( &this->hBondDataFile);

	this->showMiddlePositions.SetParameter(new param::BoolParam(false));
	this->MakeSlotAvailable( &this->showMiddlePositions);

	for(int i = 0; i < HYDROGEN_BOND_IN_CORE; i++)
		curHBondFrame[i] = -1;

	this->maxOMPThreads = omp_get_max_threads();
	this->neighbourIndices = new vislib::Array<unsigned int>[this->maxOMPThreads];
}

megamol::protein::SolventDataGenerator::~SolventDataGenerator() {
	delete[] this->neighbourIndices;
}

bool megamol::protein::SolventDataGenerator::create(void) {
	return true;
}

/**
 * -> preprocessing step
 *
 *- Aufenthaltswahrscheinlichkeit/-dauer (einzelne Molek�le oder Molek�ltypen �ber komplette Trajektorie berechnen & als Farbe auf statische Molek�loberfl�che mappen) *
 * -> das geht so net! das l�uft auf ne volumen-akkumulation hinaus ...
 */
void megamol::protein::SolventDataGenerator::calcSpatialProbabilities(MolecularDataCall *src, MolecularDataCall *dst) {
	int nFrames = src->FrameCount();
	int nAtoms = src->AtomCount();

	//middleAtomPos.AssertCapacity(nAtoms*3);
	middleAtomPos.SetCount(nAtoms*3);

	float *middlePosPtr = &middleAtomPos.First();
	const float *atomPositions = src->AtomPositions();
	memset(middlePosPtr, 0, sizeof(float)*3*nAtoms);

//#pragma omp parallel for private( ??? )
	for(int i = 0; i < nFrames; i++) {
		src->SetFrameID(i,true);
		if( !(*src)(MolecularDataCall::CallForGetData))
			continue; // return false;

		#pragma omp parallel for
		for(int aIdx = 0; aIdx < nAtoms*3; aIdx+=3) {
			middlePosPtr[aIdx] += atomPositions[aIdx];
			middlePosPtr[aIdx+1] += atomPositions[aIdx+1];
			middlePosPtr[aIdx+2] += atomPositions[aIdx+2];
		}
	}

	float normalize = 1.0f / nFrames;

	#pragma omp parallel for
	for(int aIdx = 0; aIdx < nAtoms*3; aIdx++)
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


void megamol::protein::SolventDataGenerator::calcHydroBondsForCurFrame(MolecularDataCall *data, const float *atomPositions, int *atomHydroBondsIndicesPtr) {
	float hbondDist = hBondDistance.Param<param::FloatParam>()->Value();
	//const float *atomPositions = data->AtomPositions();
	const MolecularDataCall::AtomType *atomTypes = data->AtomTypes();
	const unsigned int *atomTypeIndices = data->AtomTypeIndices();
	const int *atomResidueIndices = data->AtomResidueIndices();

	neighbourFinder.SetPointData(atomPositions, data->AtomCount(), data->AccessBoundingBoxes().ObjectSpaceBBox(), hbondDist);

    // set all entries to "not connected"
	memset(atomHydroBondsIndicesPtr, -1, sizeof(int)*data->AtomCount());

	if (reverseConnection.Count() < data->AtomCount())
		reverseConnection.SetCount(data->AtomCount());
	int *reverseConnectionPtr = &reverseConnection[0];
	memset(reverseConnectionPtr, -1, sizeof(int)*data->AtomCount());

	time_t t = clock();

	// looping over residues may not be a good idea?! (index-traversal?) loop over all possible acceptors ...
#pragma omp parallel for num_threads(maxOMPThreads)
	for( int rIdx = 0; rIdx < data->ResidueCount(); rIdx++ ) {
		const MolecularDataCall::Residue *residue = data->Residues()[rIdx];

		// we're only interested in hydrogen bonds between polymer/protein molecule and surounding solvent
		int idx = residue->MoleculeIndex();
		const MolecularDataCall::Molecule& molecule = data->Molecules()[idx];
		idx = molecule.ChainIndex();
		const MolecularDataCall::Chain& chain = data->Chains()[idx];
		if (chain.Type() == MolecularDataCall::Chain::SOLVENT)
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
JW: ich f�rchte f�r eine allgemeine Deffinition der Wasserstoffbr�cken mu� man �ber die Bindungsenergien gehen und diese berechnen.
F�r meine Simulationen und alle Bio-Geschichten reicht die Annahme, dass Sauerstoff, Stickstoff und Fluor (was fast nie vorkommt)
Wasserstoffbr�cken bilden und dabei als Donor und Aktzeptor dienen k�nne. Dabei ist der Wasserstoff am Donor gebunden und bildet die Br�cke zum Akzeptor.
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
			}
		}
	}

    std::cout << "Hydrogen bonds computed in " << ( double( clock() - t) / double( CLOCKS_PER_SEC) ) << " seconds." << std::endl;

}


bool megamol::protein::SolventDataGenerator::getHBonds(MolecularDataCall *dataTarget, MolecularDataCall *dataSource) {
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
		HbondIO input(data->AtomCount(), data->FrameCount(), fileName, true);
		if (input.readFrame(&atomHydroBonds[0], reqFrame)) {
			curHBondFrame[cacheIndex] = reqFrame;
			data->SetAtomHydrogenBondIndices(atomHydroBonds.PeekElements());
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
	if( !(*dataSource)(MolecularDataCall::CallForGetData))
		return false;
	calcHydroBondsForCurFrame(dataSource, &atomHydroBonds[0]);
#endif

	curHBondFrame[cacheIndex] = reqFrame;
	dataTarget->SetFrameID(reqFrame);
	dataTarget->SetAtomHydrogenBondDistance(hbondDist);
	dataTarget->SetAtomHydrogenBondIndices(atomHydroBonds.PeekElements());

	return true;
}

bool megamol::protein::SolventDataGenerator::getExtent(core::Call& call) {
	MolecularDataCall *molDest = dynamic_cast<MolecularDataCall*>( &call); // dataOutSlot ??
	MolecularDataCall *molSource = this->molDataInputCallerSlot.CallAs<MolecularDataCall>();

	if (!molDest || !molSource)
		return false;

	if( !(*molSource)(MolecularDataCall::CallForGetExtent))
		return false;

	// forward data ...
	molDest->AccessBoundingBoxes().Clear();
	molDest->AccessBoundingBoxes().SetObjectSpaceBBox( molSource->AccessBoundingBoxes().ObjectSpaceBBox() );
	molDest->AccessBoundingBoxes().SetObjectSpaceClipBox( molSource->AccessBoundingBoxes().ObjectSpaceClipBox() );
	molDest->SetFrameCount( molSource->FrameCount() );
	molDest->SetDataHash( molSource->DataHash() );
	return true;
}

bool megamol::protein::SolventDataGenerator::getData(core::Call& call) {
	MolecularDataCall *molDest = dynamic_cast<MolecularDataCall*>( &call); // dataOutSlot ??
	MolecularDataCall *molSource = this->molDataInputCallerSlot.CallAs<MolecularDataCall>();

	if (!molDest || !molSource)
		return false;

	molSource->SetFrameID( molDest->FrameID() ); // forward frame request
	if( !(*molSource)(MolecularDataCall::CallForGetData))
		return false;

	*molDest = *molSource;

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
		molDest->SetDataHash(molSource->DataHash()*666);
	} else {

		// reset all hbond data if this parameter changes ...
		if (hBondDistance.IsDirty()) {
			hBondDistance.ResetDirty();
			for(int i = 0; i < HYDROGEN_BOND_IN_CORE; i++)
				this->curHBondFrame[i] = -1;
			molDest->SetDataHash(molSource->DataHash()*666); // hacky ?
		}

		// test: only compute hydrogen bounds once at startup ... (this is not suficcient for trajectories)
		getHBonds(molDest, molSource);
	}

	return true;
}

void megamol::protein::SolventDataGenerator::release(void) {
}
