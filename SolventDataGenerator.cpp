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
		molDataInputCallerSlot( "getInputData", "molecular data source (usually the PDB loader)")
{
	this->molDataInputCallerSlot.SetCompatibleCall<MolecularDataCallDescription>();
	this->MakeSlotAvailable( &this->molDataInputCallerSlot);

	this->dataOutSlot.SetCallback( MolecularDataCall::ClassName(), MolecularDataCall::FunctionName(MolecularDataCall::CallForGetData), &SolventDataGenerator::getData);
	this->dataOutSlot.SetCallback( MolecularDataCall::ClassName(), MolecularDataCall::FunctionName(MolecularDataCall::CallForGetExtent), &SolventDataGenerator::getExtent);
	this->MakeSlotAvailable( &this->dataOutSlot);
}

megamol::protein::SolventDataGenerator::~SolventDataGenerator() {
}

bool megamol::protein::SolventDataGenerator::create(void) {
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

	// TODO: operator= for MolecularDataCall ?

	molSource->SetFrameID( molDest->FrameID() ); // forward frame request
	if( !(*molSource)(MolecularDataCall::CallForGetData))
		return false;
	
	*molDest = *molSource;

	return true;
}

void megamol::protein::SolventDataGenerator::release(void) {
}
