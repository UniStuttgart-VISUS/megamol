/*
 *	HydroBondGenerator.cpp
 *	
 *	Copyright (C) 2016 by University of Stuttgart (VISUS).
 *	All rights reserved.
 */

#include "stdafx.h"
#include "HydroBondGenerator.h"

#include "protein_calls/MolecularDataCall.h"

#include "mmcore/param/FloatParam.h"
#include "mmcore/param/BoolParam.h"

using namespace megamol;
using namespace megamol::core;
using namespace megamol::protein;
using namespace megamol::protein_calls;

/*
 *	HydroBondGenerator::~HydroBondGenerator
 */
HydroBondGenerator::HydroBondGenerator() : Module(),
	inDataSlot("dataIn", "Molecular data source (usually PDBLoader)"),
	outDataSlot("dataOut", "The slot providing the molecular data including the generated hydrogen bonds"),
	hBondDistance("hBondDistance", "Maximal distance of a hydrogen bond in anstrom"),
	hBondDonorAcceptorDistance("hBondDonorAcceptorDistance", "Maximal distance between donor and acceptor of a hydrogen bond"),
	hBondDonorAcceptorAngle("hBondDonorAcceptorAngle", "Maximal angle between donor-acceptor and donor-hydrogen in a hydrogen bond"),
	alphaHelixHBonds("alphaHelixHBonds", "Shall the H-Bonds inside the alpha helices be computed?"),
	betaSheetHBonds("betaSheetHBonds", "Shall the H-Bonds between two beta sheets be computed?"),
	otherHBonds("otherHBonds", "Shall all other H-Bonds be computed?") {

	this->inDataSlot.SetCompatibleCall<MolecularDataCallDescription>();
	this->MakeSlotAvailable(&this->inDataSlot);

	this->outDataSlot.SetCallback(MolecularDataCall::ClassName(), MolecularDataCall::FunctionName(MolecularDataCall::CallForGetData), &HydroBondGenerator::getData);
	this->outDataSlot.SetCallback(MolecularDataCall::ClassName(), MolecularDataCall::FunctionName(MolecularDataCall::CallForGetExtent), &HydroBondGenerator::getExtent);
	this->MakeSlotAvailable(&this->outDataSlot);

	// distance for hydrogen bonds
	this->hBondDistance.SetParameter(new param::FloatParam(1.9f, 0.0f));
	this->MakeSlotAvailable(&this->hBondDistance);

	// distance between donor and acceptor of the hydrogen bonds
	this->hBondDonorAcceptorDistance.SetParameter(new param::FloatParam(3.5f, 0.0f));
	this->MakeSlotAvailable(&this->hBondDonorAcceptorDistance);

	// angle between donor-acceptor and donor-hydrogen in degrees
	this->hBondDonorAcceptorAngle.SetParameter(new param::FloatParam(30.0f, 0.0f));
	this->MakeSlotAvailable(&this->hBondDonorAcceptorAngle);

	this->alphaHelixHBonds.SetParameter(new param::BoolParam(true));
	this->MakeSlotAvailable(&this->alphaHelixHBonds);

	this->betaSheetHBonds.SetParameter(new param::BoolParam(true));
	this->MakeSlotAvailable(&this->betaSheetHBonds);

	this->otherHBonds.SetParameter(new param::BoolParam(true));
	this->MakeSlotAvailable(&this->otherHBonds);

	this->lastDataHash = 0;
	this->dataHashOffset = 0;
}

/*
 *	HydroBondGenerator::~HydroBondGenerator
 */
HydroBondGenerator::~HydroBondGenerator(void) {
	this->Release();
}

/*
 *	HydroBondGenerator::create
 */
bool HydroBondGenerator::create(void) {
	return true;
}

/*
 *	HydroBondGenerator::getData
 */
bool HydroBondGenerator::getData(Call& call) {
	MolecularDataCall * outCall = dynamic_cast<MolecularDataCall*>(&call);
	MolecularDataCall * inCall = this->inDataSlot.CallAs<MolecularDataCall>();

	if (!outCall || !inCall) {
		return false;
	}
	inCall->SetFrameID(outCall->FrameID());
	if (!(*inCall)(MolecularDataCall::CallForGetData)) {
		return false;
	}

	*outCall = *inCall; // deep copy

	return true;
}

/*
 *	HydroBondGenerator::getExtent
 */
bool HydroBondGenerator::getExtent(Call& call) {
	MolecularDataCall * outCall = dynamic_cast<MolecularDataCall*>(&call);
	MolecularDataCall * inCall = this->inDataSlot.CallAs<MolecularDataCall>();

	if (!outCall || !inCall) {
		return false;
	}

	if (!(*inCall)(MolecularDataCall::CallForGetExtent))
		return false;

	// increment local data hash if data will change
	if (this->hBondDistance.IsDirty() || this->hBondDonorAcceptorAngle.IsDirty() || this->hBondDonorAcceptorDistance.IsDirty() ||
		this->alphaHelixHBonds.IsDirty() || this->betaSheetHBonds.IsDirty() || this->otherHBonds.IsDirty()) {
		this->dataHashOffset++;
	}

	outCall->AccessBoundingBoxes().Clear();
	outCall->AccessBoundingBoxes().SetObjectSpaceBBox(inCall->AccessBoundingBoxes().ObjectSpaceBBox());
	outCall->AccessBoundingBoxes().SetObjectSpaceClipBox(inCall->AccessBoundingBoxes().ObjectSpaceClipBox());
	outCall->SetFrameCount(inCall->FrameCount());
	outCall->SetDataHash(inCall->DataHash() + dataHashOffset);
	return true;
}

/*
 *	HydroBondGenerator::release
 */
void HydroBondGenerator::release(void) {
}