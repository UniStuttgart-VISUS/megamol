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
	playButtonParam("animation::playButton", "Button to toggle animation") {

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
 *	SecStructFlattener::getData
 */
bool SecStructFlattener::getData(core::Call& call) {
	MolecularDataCall * outCall = dynamic_cast<MolecularDataCall*>(&call);
	if (outCall == NULL) return false;

	//outCall->SetAtomPositions(this->atomPositions);
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

	agdc->operator=(*mdc); // deep copy

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