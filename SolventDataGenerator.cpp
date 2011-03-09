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
#include <ctime>
#include <iostream>
#include <fstream>

using namespace megamol;
using namespace megamol::core;
using namespace megamol::protein;


megamol::protein::SolventDataGenerator::SolventDataGenerator() :
		dataOutSlot( "dataout", "The slot providing the generated solvent data"),
		molDataInputCallerSlot( "getInputData", "molecular data source (usually the PDB loader)"),
		hBondDistance("hBondDistance", "distance for hydrogen bonds (angstroem?)")
{
	this->molDataInputCallerSlot.SetCompatibleCall<MolecularDataCallDescription>();
	this->MakeSlotAvailable( &this->molDataInputCallerSlot);

	this->dataOutSlot.SetCallback( MolecularDataCall::ClassName(), MolecularDataCall::FunctionName(MolecularDataCall::CallForGetData), &SolventDataGenerator::getData);
	this->dataOutSlot.SetCallback( MolecularDataCall::ClassName(), MolecularDataCall::FunctionName(MolecularDataCall::CallForGetExtent), &SolventDataGenerator::getExtent);
	this->MakeSlotAvailable( &this->dataOutSlot);

	// distance for hydrogen bonds
	this->hBondDistance.SetParameter(new param::FloatParam( 1.9f, 0.0f));
	this->MakeSlotAvailable( &this->hBondDistance);
}

megamol::protein::SolventDataGenerator::~SolventDataGenerator() {
}

bool megamol::protein::SolventDataGenerator::create(void) {
	return true;
}

/**
 * -> preprocessing step
 *
 *- Aufenthaltswahrscheinlichkeit/-dauer (einzelne Moleküle oder Molekültypen über komplette Trajektorie berechnen & als Farbe auf statische Moleküloberfläche mappen) *
 */
void megamol::protein::SolventDataGenerator::calcSpatialProbabilities(MolecularDataCall *src, MolecularDataCall *dst) {
	int nFrames = src->FrameCount();
	int nAtoms = src->AtomCount();

	middleAtomPos.AssertCapacity(nAtoms*3);
	middleAtomPos.SetCount(nAtoms*3);

	float *middlePosPtr = &middleAtomPos.First(); // TODO: hier gibts ne exception!
	memset(middlePosPtr, 0, sizeof(float)*3*nAtoms);

//#pragma omp parallel for private( ??? )
	for(int i = 0; i < nFrames; i++) {
		src->SetFrameID(i,true);
		if( !(*src)(MolecularDataCall::CallForGetData))
			continue; // return false;

		#pragma omp parallel for
		for(int aIdx = 0; aIdx < nAtoms; aIdx+=3) {
			middlePosPtr[aIdx] += src->AtomPositions()[aIdx];
			middlePosPtr[aIdx+1] += src->AtomPositions()[aIdx+1];
			middlePosPtr[aIdx+2] += src->AtomPositions()[aIdx+2];
		}
	}

	float normalize = 1.0f / nFrames;

	#pragma omp parallel for
	for(int aIdx = 0; aIdx < nAtoms*3; aIdx++)
		middlePosPtr[aIdx] *= normalize;

	dst->SetAtoms(src->AtomCount(), src->AtomTypeCount(), src->AtomTypeIndices(), /*src->AtomPositions()*/middlePosPtr,
		src->AtomTypes(), src->AtomResidueIndices(), src->AtomBFactors(), src->AtomCharges(), src->AtomOccupancies());
}

/*
 *
 *
void megamol::protein::SolventDataGenerator::findDonors(MolecularDataCall *data) {
	int nResidues = data->ResidueCount();

	for( int i = 0; i < nResidues; i++ ) {
		MolecularDataCall::Residue *residue = data->Residues()[i];
	}
}
*/

/*
JW: ich fürchte für eine allgemeine Deffinition der Wasserstoffbrücken muß man über die Bindungsenergien gehen und diese berechnen.
Für meine Simulationen und alle Bio-Geschichten reicht die Annahme, dass Sauerstoff, Stickstoff und Fluor (was fast nie vorkommt)
Wasserstoffbrücken bilden und dabei als Donor und Aktzeptor dienen könne. Dabei ist der Wasserstoff am Donor gebunden und bildet die Brücke zum Akzeptor.
* /
void megamol::protein::SolventDataGenerator::calcHBonds(MolecularDataCall *data) {
//	findDonors();
//	findAcceptors();
	int nResidues = data->ResidueCount();
	float hbondDist = hBondDistance.Param<param::FloatParam>()->Value();

	// looping over residues may not be a good idea?! (index-traversal?) loop over all possible acceptors ...
	for( int i = 0; i < nResidues; i++ ) {
		MolecularDataCall::Residue *residue = data->Residues()[i];

		// find possible acceptor atoms in the current residuum (for now just O, N, C(?)
		Atom O, N, C;
		O = residue->FindAtom('O');
		N = residue->FindAtom('N');
		C = residue->FindAtom('C');

		if (O || N || C) {
			find_neighbours_in_range(0, 'H', hbondDist);
		}
	}
}*/

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

	calcSpatialProbabilities(molSource, molDest);

	return true;
}

void megamol::protein::SolventDataGenerator::release(void) {
}
