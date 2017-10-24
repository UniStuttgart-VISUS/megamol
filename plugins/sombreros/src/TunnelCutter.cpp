/*
 * TunnelCutter.h
 * Copyright (C) 2006-2017 by MegaMol Team
 * Alle Rechte vorbehalten.
 */
#include "stdafx.h"
#include "TunnelCutter.h"

#include "mmstd_trisoup/CallTriMeshData.h"
#include "TunnelResidueDataCall.h"

using namespace megamol;
using namespace megamol::core;
using namespace megamol::trisoup;
using namespace megamol::sombreros;

/*
 * TunnelCutter::TunnelCutter
 */
TunnelCutter::TunnelCutter(void) : Module(),
		meshInSlot("dataIn", "Receives the input mesh"),
		cutMeshOutSlot("getData", "Returns the mesh data of the wanted area"),
		tunnelInSlot("tunnelIn", "Receives the input tunnel data"){

	// Callee slot
	this->cutMeshOutSlot.SetCallback(CallTriMeshData::ClassName(), CallTriMeshData::FunctionName(0), &TunnelCutter::getData);
	this->cutMeshOutSlot.SetCallback(CallTriMeshData::ClassName(), CallTriMeshData::FunctionName(1), &TunnelCutter::getExtent);
	this->MakeSlotAvailable(&this->cutMeshOutSlot);

	// Caller slots
	this->meshInSlot.SetCompatibleCall<CallTriMeshDataDescription>();
	this->MakeSlotAvailable(&this->meshInSlot);

	this->tunnelInSlot.SetCompatibleCall<TunnelResidueDataCallDescription>();
	this->MakeSlotAvailable(&this->tunnelInSlot);

	this->lastDataHash = 0;
	this->hashOffset = 0;
}

/*
 * TunnelCutter::~TunnelCutter
 */
TunnelCutter::~TunnelCutter(void) {
	this->Release();
}

/*
 * TunnelCutter::create
 */
bool TunnelCutter::create(void) {
	return true;
}

/*
 * TunnelCutter::release
 */
void TunnelCutter::release(void) {
}

/*
 * TunnelCutter::getData
 */
bool TunnelCutter::getData(Call& call) {
	CallTriMeshData * outCall = dynamic_cast<CallTriMeshData*>(&call);
	if (outCall == nullptr) return false;

	CallTriMeshData * inCall = this->meshInSlot.CallAs<CallTriMeshData>();
	if (inCall == nullptr) return false;

	TunnelResidueDataCall * tc = this->tunnelInSlot.CallAs<TunnelResidueDataCall>();
	if (tc == nullptr) return false;

	inCall->SetFrameID(outCall->FrameID());
	tc->SetFrameID(outCall->FrameID());

	if (!(*inCall)(0)) return false;
	if (!(*tc)(0)) return false;

	// TODO right now: only data copying
	//outCall->operator=(*inCall);

	// something happened with the input data, we have to recopy it
	if (lastDataHash != inCall->DataHash()) {
		lastDataHash = inCall->DataHash();

		this->meshVector.resize(inCall->Count());
		for (int i = 0; i < inCall->Count(); i++) {
			this->meshVector[i] = inCall->Objects()[i];
		}
	}

	outCall->SetObjects(this->meshVector.size(), this->meshVector.data());
	this->meshVector[0].GetVertexAttribDataType();

	return true;
}

/*
 * TunnelCutter::getExtent
 */
bool TunnelCutter::getExtent(Call& call) {
	CallTriMeshData * outCall = dynamic_cast<CallTriMeshData*>(&call);
	if (outCall == nullptr) return false;

	CallTriMeshData * inCall = this->meshInSlot.CallAs<CallTriMeshData>();
	if (inCall == nullptr) return false;

	TunnelResidueDataCall * tc = this->tunnelInSlot.CallAs<TunnelResidueDataCall>();
	if (tc == nullptr) return false;

	inCall->SetFrameID(outCall->FrameID());
	tc->SetFrameID(outCall->FrameID());

	if (!(*inCall)(1)) return false;
	if (!(*tc)(1)) return false;

	outCall->SetDataHash(inCall->DataHash() + this->hashOffset);
	outCall->SetFrameCount(inCall->FrameCount());
	outCall->SetExtent(inCall->FrameCount(), inCall->AccessBoundingBoxes());

	return true;
}