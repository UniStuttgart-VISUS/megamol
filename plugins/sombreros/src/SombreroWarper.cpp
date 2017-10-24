/*
 * TunnelCutter.h
 * Copyright (C) 2006-2017 by MegaMol Team
 * Alle Rechte vorbehalten.
 */
#include "stdafx.h"
#include "SombreroWarper.h"

#include "mmstd_trisoup/CallTriMeshData.h"
#include "TunnelResidueDataCall.h"

using namespace megamol;
using namespace megamol::core;
using namespace megamol::trisoup;
using namespace megamol::sombreros;

/*
 * SombreroWarper::SombreroWarper
 */
SombreroWarper::SombreroWarper(void) : Module(),
		meshInSlot("dataIn", "Receives the input mesh"),
		warpedMeshOutSlot("getData", "Returns the mesh data of the wanted area") {

	// Callee slot
	this->warpedMeshOutSlot.SetCallback(CallTriMeshData::ClassName(), CallTriMeshData::FunctionName(0), &SombreroWarper::getData);
	this->warpedMeshOutSlot.SetCallback(CallTriMeshData::ClassName(), CallTriMeshData::FunctionName(1), &SombreroWarper::getExtent);
	this->MakeSlotAvailable(&this->warpedMeshOutSlot);

	// Caller slots
	this->meshInSlot.SetCompatibleCall<CallTriMeshDataDescription>();
	this->MakeSlotAvailable(&this->meshInSlot);

	this->lastDataHash = 0;
	this->hashOffset = 0;
}

/*
 * SombreroWarper::~SombreroWarper
 */
SombreroWarper::~SombreroWarper(void) {
	this->Release();
}

/*
 * SombreroWarper::create
 */
bool SombreroWarper::create(void) {
	return true;
}

/*
 * SombreroWarper::release
 */
void SombreroWarper::release(void) {
}

/*
 * SombreroWarper::getData
 */
bool SombreroWarper::getData(Call& call) {
	CallTriMeshData * outCall = dynamic_cast<CallTriMeshData*>(&call);
	if (outCall == nullptr) return false;

	CallTriMeshData * inCall = this->meshInSlot.CallAs<CallTriMeshData>();
	if (inCall == nullptr) return false;

	inCall->SetFrameID(outCall->FrameID());

	if (!(*inCall)(0)) return false;

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

	return true;
}

/*
 * TunnelCutter::getExtent
 */
bool SombreroWarper::getExtent(Call& call) {
	CallTriMeshData * outCall = dynamic_cast<CallTriMeshData*>(&call);
	if (outCall == nullptr) return false;

	CallTriMeshData * inCall = this->meshInSlot.CallAs<CallTriMeshData>();
	if (inCall == nullptr) return false;

	inCall->SetFrameID(outCall->FrameID());

	if (!(*inCall)(1)) return false;

	outCall->SetDataHash(inCall->DataHash() + this->hashOffset);
	outCall->SetFrameCount(inCall->FrameCount());
	outCall->SetExtent(inCall->FrameCount(), inCall->AccessBoundingBoxes());

	return true;
}