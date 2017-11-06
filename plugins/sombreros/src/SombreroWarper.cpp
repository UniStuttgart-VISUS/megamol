/*
 * TunnelCutter.h
 * Copyright (C) 2006-2017 by MegaMol Team
 * Alle Rechte vorbehalten.
 */
#include "stdafx.h"
#include "SombreroWarper.h"

#include "mmstd_trisoup/CallTriMeshData.h"
#include "TunnelResidueDataCall.h"
#include <climits>

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
	this->dirtyFlag = false;
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

	// something happened with the input data, we have to recopy it
	if ((lastDataHash != inCall->DataHash()) || dirtyFlag) {
		lastDataHash = inCall->DataHash();
		dirtyFlag = false;

		copyMeshData(*inCall);

		//this->meshVector.resize(inCall->Count());
		//for (int i = 0; i < inCall->Count(); i++) {
		//	// TODO change
		//	this->meshVector[i] = inCall->Objects()[i];
		//}

		//bool borderResult = this->findSombreroBorder();
		//if (!borderResult) return false;


	}

	outCall->SetObjects(static_cast<uint>(this->meshVector.size()), this->meshVector.data());

	return true;
}

/*
 * SombreroWarper::getExtent
 */
bool SombreroWarper::getExtent(Call& call) {
	CallTriMeshData * outCall = dynamic_cast<CallTriMeshData*>(&call);
	if (outCall == nullptr) return false;

	CallTriMeshData * inCall = this->meshInSlot.CallAs<CallTriMeshData>();
	if (inCall == nullptr) return false;

	this->checkParameters();
	
	if (dirtyFlag) {
		this->hashOffset++;
	}

	inCall->SetFrameID(outCall->FrameID());

	if (!(*inCall)(1)) return false;

	outCall->SetDataHash(inCall->DataHash() + this->hashOffset);
	outCall->SetFrameCount(inCall->FrameCount());
	outCall->SetExtent(inCall->FrameCount(), inCall->AccessBoundingBoxes());

	return true;
}

/*
 * SombreroWarper::checkParameters
 */
void SombreroWarper::checkParameters(void) {
	// TODO
}

/*
 * SombreroWarper::copyMeshData
 */
bool SombreroWarper::copyMeshData(CallTriMeshData& ctmd) {
	this->meshVector.clear();
	this->meshVector.resize(ctmd.Count());
	
	this->vertices.clear();
	this->vertices.resize(ctmd.Count());

	this->normals.clear();
	this->normals.resize(ctmd.Count());

	this->colors.clear();
	this->colors.resize(ctmd.Count());

	this->atomIndexAttachment.clear();
	this->atomIndexAttachment.resize(ctmd.Count());

	this->vertexLevelAttachment.clear();
	this->vertexLevelAttachment.resize(ctmd.Count());

	this->faces.clear();
	this->faces.resize(ctmd.Count());

	for (uint i = 0; i < ctmd.Count(); i++) {
		uint vertCount = ctmd.Objects()[i].GetVertexCount();
		uint triCount = ctmd.Objects()[i].GetTriCount();
		uint attribCount = ctmd.Objects()[i].GetVertexAttribCount();

		uint atomIndexAttrib = UINT_MAX;
		uint vertexLvlAttrib = UINT_MAX;
		if (attribCount < 2) {
			vislib::sys::Log::DefaultLog.WriteError("Too few vertex attributes detected. The input mesh for the Sombrero warper needs at least two UINT32 vertex attributes.");
			return false;
		}
		// determine the location of the needed attributes
		for (uint i = 0; i < attribCount; i++) {
			auto dt = ctmd.Objects()[i].GetVertexAttribDataType(i);
			if (atomIndexAttrib == UINT_MAX && dt == ctmd.Objects()[i].DT_UINT32) {
				atomIndexAttrib = i;
			}
			else if (vertexLvlAttrib == UINT_MAX && dt == ctmd.Objects()[i].DT_UINT32) {
				vertexLvlAttrib = i;
			}
		}
		if (atomIndexAttrib == UINT_MAX || vertexLvlAttrib == UINT_MAX) {
			vislib::sys::Log::DefaultLog.WriteError("Not enough UINT32 vertex attributes detected. The input mesh for the Sombrero warper needs at least two UINT32 vertex attributes.");
			return false;
		}

		this->vertices[i].resize(vertCount * 3);
		this->normals[i].resize(vertCount * 3);
		this->colors[i].resize(vertCount * 3);
		this->atomIndexAttachment[i].resize(vertCount);
		this->vertexLevelAttachment[i].resize(vertCount);
		this->faces[i].resize(triCount * 3);

		std::memcpy(this->vertices[i].data(), ctmd.Objects()[i].GetVertexPointerFloat(), vertCount * 3 * sizeof(float));
		std::memcpy(this->normals[i].data(), ctmd.Objects()[i].GetNormalPointerFloat(), vertCount * 3 * sizeof(float));
		std::memcpy(this->colors[i].data(), ctmd.Objects()[i].GetColourPointerByte(), vertCount * 3 * sizeof(unsigned char));
		std::memcpy(this->atomIndexAttachment[i].data(), ctmd.Objects()[i].GetVertexAttribPointerUInt32(0), vertCount * sizeof(uint));
		std::memcpy(this->vertexLevelAttachment[i].data(), ctmd.Objects()[i].GetVertexAttribPointerUInt32(1), vertCount * sizeof(uint));
		std::memcpy(this->faces[i].data(), ctmd.Objects()[i].GetTriIndexPointerUInt32(), triCount * 3 * sizeof(uint));

		this->meshVector[i].SetVertexData(vertCount, this->vertices[i].data(), this->normals[i].data(), this->colors[i].data(), nullptr, false);
		this->meshVector[i].SetTriangleData(triCount, this->faces[i].data(), false);
		this->meshVector[i].SetMaterial(nullptr);
		this->meshVector[i].AddVertexAttribPointer(this->atomIndexAttachment[i].data());
		this->meshVector[i].AddVertexAttribPointer(this->vertexLevelAttachment[i].data());
	}
	return true;
}

/*
 * SombreroWarper::findSombreroBorder
 */
bool SombreroWarper::findSombreroBorder(void) {
	for (auto mesh : this->meshVector) {
		uint attribCnt = mesh.GetVertexAttribCount();
		// we need 2 unsigned int attributes
		uint atomIndexAttrib = UINT_MAX;
		uint vertexLvlAttrib = UINT_MAX;
		if (attribCnt < 2) {
			vislib::sys::Log::DefaultLog.WriteError("Too few vertex attributes detected. The input mesh for the Sombrero warper needs at least two UINT32 vertex attributes.");
			return false;
		}
		// determine the location of the needed attributes
		for (uint i = 0; i < attribCnt; i++) {
			auto dt = mesh.GetVertexAttribDataType(i);
			if (atomIndexAttrib == UINT_MAX && dt == mesh.DT_UINT32) {
				atomIndexAttrib = i;
			} else if (vertexLvlAttrib == UINT_MAX && dt == mesh.DT_UINT32) {
				vertexLvlAttrib = i;
			}
		}
		if (atomIndexAttrib == UINT_MAX || vertexLvlAttrib == UINT_MAX) {
			vislib::sys::Log::DefaultLog.WriteError("Not enough UINT32 vertex attributes detected. The input mesh for the Sombrero warper needs at least two UINT32 vertex attributes.");
			return false;
		}
#if 1
		// adjust the color
		uint vCnt = mesh.GetVertexCount();
		for (uint i = 0; i < vCnt; i++) {
			
		}
#endif
	}

	return true;
}