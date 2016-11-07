/*
 *	SecStructFlattener.cpp
 *
 *	Copyright (C) 2016 by Universitaet Stuttgart (VISUS).
 *	All rights reserved
 */

#include "stdafx.h"
#include "SecStructFlattener.h"

#include "mmcore/param/BoolParam.h"
#include "mmcore/param/ButtonParam.h"
#include "mmcore/param/EnumParam.h"

using namespace megamol;
using namespace megamol::core;
using namespace megamol::protein_cuda;
using namespace megamol::protein_calls;

/*
 *	SecStructFlattener::SecStructFlattener
 */
SecStructFlattener::SecStructFlattener(void) :
	Module(),
	getDataSlot("getData", "Calls molecular data"),
	dataOutSlot("dataOut", "Provides the flattened molecular data"),
	playParam("animation::play", "Should the animation be played?"),
	playButtonParam("animation::playButton", "Button to toggle animation"),
	flatPlaneMode("flatPlaneMode", "The plane the protein gets flattened to") {

	// caller slot
	this->getDataSlot.SetCompatibleCall<MolecularDataCallDescription>();
	this->MakeSlotAvailable(&this->getDataSlot);

	// callee slot
	this->dataOutSlot.SetCallback(MolecularDataCall::ClassName(), MolecularDataCall::FunctionName(0), &SecStructFlattener::getData);
	this->dataOutSlot.SetCallback(MolecularDataCall::ClassName(), MolecularDataCall::FunctionName(1), &SecStructFlattener::getExtent);
	this->MakeSlotAvailable(&this->dataOutSlot);

	this->playParam.SetParameter(new param::BoolParam(false));
	this->MakeSlotAvailable(&this->playParam);

	this->playButtonParam << new param::ButtonParam('p');
	this->playButtonParam.SetUpdateCallback(this, &SecStructFlattener::onPlayToggleButton);
	this->MakeSlotAvailable(&this->playButtonParam);

	param::EnumParam * fpParam = new param::EnumParam(int(FlatPlane::XY_PLANE));
	FlatPlane fp;
	for (int i = 0; i < getFlatPlaneModeNumber(); i++) {
		fp = getFlatPlaneByIndex(i);
		fpParam->SetTypePair(fp, getFlatPlaneName(fp).c_str());
	}
	this->flatPlaneMode << fpParam;
	this->MakeSlotAvailable(&this->flatPlaneMode);

	this->atomPositions = NULL;
	this->atomPositionsSize = 0;
}

/*
 *	SecStructFlattener::~SecStructFlattener
 */
SecStructFlattener::~SecStructFlattener(void) {
	this->Release();
}

/*
 *	SecStructFlattener::create
 */
bool SecStructFlattener::create(void) {
	return true;
}

/*
 *	SecStructFlattener::release
 */
void SecStructFlattener::release(void) {
	if (this->atomPositions != NULL) {
		delete[] this->atomPositions;
		this->atomPositions = NULL;
		this->atomPositionsSize = 0;
	}
}

/*
 *	SecStructFlattener::flatten
 */
void SecStructFlattener::flatten(void) {

	auto bbCenter = this->boundingBox.CalcCenter();

	switch (this->flatPlaneMode.Param<param::EnumParam>()->Value()) {
	case XY_PLANE: 
		for (unsigned int i = 0; i < this->atomPositionsSize / 3; i++) {
			this->atomPositions[i * 3 + 2] = bbCenter.GetZ();
		}
		break;
	case XZ_PLANE: 
		for (unsigned int i = 0; i < this->atomPositionsSize / 3; i++) {
			this->atomPositions[i * 3 + 1] = bbCenter.GetY();
		}
		break;
	case YZ_PLANE: 
		for (unsigned int i = 0; i < this->atomPositionsSize / 3; i++) {
			this->atomPositions[i * 3 + 0] = bbCenter.GetX();
		}
		break;
	case LEAST_COMMON: 
		break;
	case ARBITRARY: 
		break;
	default: 
		break;
	}
}

/*
 *	SecStructFlattener::getFlatPlaneByIndex
 */
SecStructFlattener::FlatPlane SecStructFlattener::getFlatPlaneByIndex(unsigned int idx) {
	switch (idx) {
		case 0:		return FlatPlane::XY_PLANE;
		case 1:		return FlatPlane::XZ_PLANE;
		case 2:		return FlatPlane::YZ_PLANE;
		case 3:		return FlatPlane::LEAST_COMMON;
		case 4:		return FlatPlane::ARBITRARY;
		default:	return FlatPlane::XY_PLANE;
	}
}

/*
 *	SecStructFlattener::getFlatPlaneModeNumber
 */
int SecStructFlattener::getFlatPlaneModeNumber(void) {
	return 5;
}

/*
 *	SecStructFlattener::getFlatPlaneName
 */
std::string SecStructFlattener::getFlatPlaneName(SecStructFlattener::FlatPlane fp) {
	switch (fp) {
		case XY_PLANE		: return "XY Plane";
		case XZ_PLANE		: return "XZ Plane";
		case YZ_PLANE		: return "YZ Plane";
		case LEAST_COMMON	: return "Least Common";
		case ARBITRARY		: return "Arbitrary";
		default				: return "";
	}
}

/*
 *	SecStructFlattener::getData
 */
bool SecStructFlattener::getData(core::Call& call) {
	MolecularDataCall * outCall = dynamic_cast<MolecularDataCall*>(&call);
	if (outCall == NULL) return false;

	flatten();

	outCall->SetAtomPositions(this->atomPositions);
	return true;
}

/*
 *	SecStructFlattener::getExtent
 */
bool SecStructFlattener::getExtent(core::Call& call) {
	
	MolecularDataCall * agdc = dynamic_cast<MolecularDataCall*>(&call);
	if (agdc == NULL) return false;

	MolecularDataCall *mdc = this->getDataSlot.CallAs<MolecularDataCall>();
	if (mdc == NULL) return false;
	mdc->SetCalltime(agdc->Calltime());
	if (!(*mdc)(1)) return false;
	if (!(*mdc)(0)) return false;

	this->boundingBox = mdc->AccessBoundingBoxes().ObjectSpaceBBox();

	agdc->operator=(*mdc); // deep copy

	// copy the atom positions to the array used here
	if (atomPositions != NULL) {
		delete[] this->atomPositions;
		this->atomPositions = NULL;
		this->atomPositionsSize = 0;
	}

	atomPositions = new float[mdc->AtomCount() * 3];
	atomPositionsSize = mdc->AtomCount() * 3;

	for (unsigned int i = 0; i < mdc->AtomCount(); i++) {
		atomPositions[i * 3 + 0] = mdc->AtomPositions()[i * 3 + 0];
		atomPositions[i * 3 + 1] = mdc->AtomPositions()[i * 3 + 1];
		atomPositions[i * 3 + 2] = mdc->AtomPositions()[i * 3 + 2];
	}

	return true;
}

/*
 *	SecStructFlattener::onPlayToggleButton
 */
bool SecStructFlattener::onPlayToggleButton(param::ParamSlot& p) {
	param::BoolParam *bp = this->playParam.Param<param::BoolParam>();
	bp->SetValue(!bp->Value());
	bool play = bp->Value();
	return true;
}