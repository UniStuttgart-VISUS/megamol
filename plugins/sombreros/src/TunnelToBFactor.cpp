/*
 * TunnelToBFactor.cpp
 * Copyright (C) 2006-2017 by MegaMol Team
 * Alle Rechte vorbehalten.
 */
#include "stdafx.h"
#include "TunnelToBFactor.h"

#include "TunnelResidueDataCall.h"
#include "protein_calls/MolecularDataCall.h"


using namespace megamol;
using namespace megamol::core;
using namespace megamol::sombreros;
using namespace megamol::protein_calls;

/* 
 * TunnelToBFactor::TunnelToBFactor
 */
TunnelToBFactor::TunnelToBFactor(void) : Module(), 
		dataOutSlot("dataOut", "Output slot for the output molecular data"),
		molInSlot("moleculeIn", "Input slot for the molecular data"),
		tunnelInSlot("tunnelIn", "Input slot for the tunnel data") {

	// caller slots
	this->molInSlot.SetCompatibleCall<MolecularDataCallDescription>();
	this->MakeSlotAvailable(&this->molInSlot);

	this->tunnelInSlot.SetCompatibleCall<TunnelResidueDataCallDescription>();
	this->MakeSlotAvailable(&this->tunnelInSlot);

	// callee slot
	this->dataOutSlot.SetCallback(MolecularDataCall::ClassName(), MolecularDataCall::FunctionName(0), &TunnelToBFactor::getData);
	this->dataOutSlot.SetCallback(MolecularDataCall::ClassName(), MolecularDataCall::FunctionName(1), &TunnelToBFactor::getExtent);
	this->MakeSlotAvailable(&this->dataOutSlot);

	// parameters

}

/* 
 * TunnelToBFactor::~TunnelToBFactor
 */
TunnelToBFactor::~TunnelToBFactor(void) {
	this->Release();
}

/* 
 * TunnelToBFactor::create
 */
bool TunnelToBFactor::create(void) {
	return true;
}

/* 
 * TunnelToBFactor::release
 */
void TunnelToBFactor::release(void) {
}

/* 
 * TunnelToBFactor::getData
 */
bool TunnelToBFactor::getData(Call& call) {
	MolecularDataCall * outCall = dynamic_cast<MolecularDataCall*>(&call);
	if (outCall == nullptr) return false;

	MolecularDataCall * mdc = this->molInSlot.CallAs<MolecularDataCall>();
	if (mdc == nullptr) return false;

	mdc->SetCalltime(outCall->Calltime());
	if (!(*mdc)(0)) return false;

	outCall->operator=(*mdc);

	applyBFactor(outCall, mdc);

	return true;
}

/* 
 * TunnelToBFactor::getExtent
 */
bool TunnelToBFactor::getExtent(Call& call) {
	MolecularDataCall * outCall = dynamic_cast<MolecularDataCall*>(&call);
	if (outCall == nullptr) return false;

	MolecularDataCall * mdc = this->molInSlot.CallAs<MolecularDataCall>();
	if (mdc == nullptr) return false;

	mdc->SetCalltime(outCall->Calltime());
	if (!(*mdc)(1)) return false;

	outCall->operator=(*mdc); // deep copy

	return true;
}

/*
 * TunnelToBFactor::applyBFactor
 */
void TunnelToBFactor::applyBFactor(MolecularDataCall * outCall, MolecularDataCall * inCall) {
	auto numFactors = inCall->AtomCount();
	this->bFactors.resize(numFactors, 0.0f);

	for (unsigned int i = 0; i < numFactors / 2; i++) {
		this->bFactors[i] = 1.0f;
	}
}