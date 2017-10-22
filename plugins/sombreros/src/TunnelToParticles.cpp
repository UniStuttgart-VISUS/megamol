/*
 * TunnelToParticles.cpp
 * Copyright (C) 2006-2017 by MegaMol Team
 * Alle Rechte vorbehalten.
 */
#include "stdafx.h"
#include "TunnelToParticles.h"

#include "TunnelResidueDataCall.h"

using namespace megamol;
using namespace megamol::core;
using namespace megamol::sombreros;
using namespace megamol::core::moldyn;

/*
 * TunnelToParticles::TunnelToParticles
 */
TunnelToParticles::TunnelToParticles(void) : Module(),
		dataOutSlot("getData", "Slot providing the tunnel data as particles"),
		tunnelInSlot("tunnelIn", "Slot taking the tunnel data as input"){

	// caller slot
	this->tunnelInSlot.SetCompatibleCall<TunnelResidueDataCallDescription>();
	this->MakeSlotAvailable(&this->tunnelInSlot);

	// callee slot
	this->dataOutSlot.SetCallback(MultiParticleDataCall::ClassName(), MultiParticleDataCall::FunctionName(0), &TunnelToParticles::getData);
	this->dataOutSlot.SetCallback(MultiParticleDataCall::ClassName(), MultiParticleDataCall::FunctionName(1), &TunnelToParticles::getExtent);
	this->MakeSlotAvailable(&this->dataOutSlot);

	// parameters
}

/*
 * TunnelToParticles::~TunnelToParticles
 */
TunnelToParticles::~TunnelToParticles(void) {
	this->Release();
}

/*
 * TunnelToParticles::create
 */
bool TunnelToParticles::create(void) {
	return true;
}

/*
 * TunnelToParticles::release
 */
void TunnelToParticles::release(void) {
}

/*
 * TunnelToParticles::getData
 */
bool TunnelToParticles::getData(Call& call) {
	MultiParticleDataCall * mpdc = dynamic_cast<MultiParticleDataCall*>(&call);
	if (mpdc == nullptr) return false;

	TunnelResidueDataCall * trdc = this->tunnelInSlot.CallAs<TunnelResidueDataCall>();
	if (trdc == nullptr) return false;

	if (!(*trdc)(0)) return false;

	// TODO copy data

	return true;
}

/*
 * TunnelToParticles::getExtent
 */
bool TunnelToParticles::getExtent(Call& call) {
	MultiParticleDataCall * mpdc = dynamic_cast<MultiParticleDataCall*>(&call);
	if (mpdc==nullptr) return false;

	TunnelResidueDataCall * trdc = this->tunnelInSlot.CallAs<TunnelResidueDataCall>();
	if (trdc==nullptr) return false;

	if (!(*trdc)(1)) return false;
	if (!(*trdc)(0)) return false;

	// TODO compute bounding box

	return true;
}