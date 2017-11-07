/*
 * TunnelCutter.h
 * Copyright (C) 2006-2017 by MegaMol Team
 * Alle Rechte vorbehalten.
 */
#include "stdafx.h"
#include "SombreroWarper.h"

#include "mmcore/param/IntParam.h"
#include "mmcore/param/FloatParam.h"
#include "mmstd_trisoup/CallTriMeshData.h"
#include "TunnelResidueDataCall.h"
#include <climits>
#include <iostream>

using namespace megamol;
using namespace megamol::core;
using namespace megamol::trisoup;
using namespace megamol::sombreros;

/*
 * SombreroWarper::SombreroWarper
 */
SombreroWarper::SombreroWarper(void) : Module(),
		meshInSlot("dataIn", "Receives the input mesh"),
		warpedMeshOutSlot("getData", "Returns the mesh data of the wanted area"),
		minBrimLevelParam("minBrimLevel", "Minimal vertex level to count as brim."),
		maxBrimLevelParam("maxBrimLevel", "Maximal vertex level to count as brim. A value of -1 sets the value to the maximal available level") {

	// Callee slot
	this->warpedMeshOutSlot.SetCallback(CallTriMeshData::ClassName(), CallTriMeshData::FunctionName(0), &SombreroWarper::getData);
	this->warpedMeshOutSlot.SetCallback(CallTriMeshData::ClassName(), CallTriMeshData::FunctionName(1), &SombreroWarper::getExtent);
	this->MakeSlotAvailable(&this->warpedMeshOutSlot);

	// Caller slots
	this->meshInSlot.SetCompatibleCall<CallTriMeshDataDescription>();
	this->MakeSlotAvailable(&this->meshInSlot);

	// Param slots
	this->minBrimLevelParam.SetParameter(new param::IntParam(1, 0, 100));
	this->MakeSlotAvailable(&this->minBrimLevelParam);

	this->maxBrimLevelParam.SetParameter(new param::IntParam(-1, -1, 100));
	this->MakeSlotAvailable(&this->maxBrimLevelParam);

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

	// something happened with the input data, we have to recompute it
	if ((lastDataHash != inCall->DataHash()) || dirtyFlag) {
		lastDataHash = inCall->DataHash();
		dirtyFlag = false;

		// copy
		if (!this->copyMeshData(*inCall)) return false;

		// search the sombrero border
		if (!this->findSombreroBorder()) return false;
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
	if (this->minBrimLevelParam.IsDirty()) {
		this->minBrimLevelParam.ResetDirty();
		this->dirtyFlag = true;
	}
	if (this->maxBrimLevelParam.IsDirty()) {
		this->maxBrimLevelParam.ResetDirty();
		this->dirtyFlag = true;
	}
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

	this->edgesForward.clear();
	this->edgesForward.resize(ctmd.Count());

	this->edgesReverse.clear();
	this->edgesReverse.resize(ctmd.Count());

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
		for (uint j = 0; j < attribCount; j++) {
			auto dt = ctmd.Objects()[i].GetVertexAttribDataType(j);
			if (atomIndexAttrib == UINT_MAX && dt == ctmd.Objects()[i].DT_UINT32) {
				atomIndexAttrib = j;
			}
			else if (vertexLvlAttrib == UINT_MAX && dt == ctmd.Objects()[i].DT_UINT32) {
				vertexLvlAttrib = j;
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

		// copy the edges
		this->edgesForward[i].clear();
		this->edgesReverse[i].clear();
		for (uint j = 0; j < triCount; j++) {
			uint vert1 = this->faces[i][j * 3 + 0];
			uint vert2 = this->faces[i][j * 3 + 1];
			uint vert3 = this->faces[i][j * 3 + 2];

			edgesForward[i].push_back(std::pair<uint, uint>(vert1, vert2));
			edgesForward[i].push_back(std::pair<uint, uint>(vert2, vert3));
			edgesForward[i].push_back(std::pair<uint, uint>(vert3, vert1));
			edgesReverse[i].push_back(std::pair<uint, uint>(vert2, vert1));
			edgesReverse[i].push_back(std::pair<uint, uint>(vert3, vert2));
			edgesReverse[i].push_back(std::pair<uint, uint>(vert1, vert3));
		}
		// sort the search structures
		std::sort(edgesForward[i].begin(), edgesForward[i].end(), [](const std::pair<unsigned int, unsigned int> &left, const std::pair<unsigned int, unsigned int> &right) {
			return left.first < right.first;
		});
		std::sort(edgesReverse[i].begin(), edgesReverse[i].end(), [](const std::pair<unsigned int, unsigned int> &left, const std::pair<unsigned int, unsigned int> &right) {
			return left.first < right.first;
		});
	}
	return true;
}

/*
 * SombreroWarper::findSombreroBorder
 */
bool SombreroWarper::findSombreroBorder(void) {
	this->borderVertices.clear();
	this->borderVertices.resize(this->meshVector.size());
	this->brimFlags.clear();
	this->brimFlags.resize(this->meshVector.size());
	for (uint i = 0; i < static_cast<uint>(this->meshVector.size()); i++) {
		CallTriMeshData::Mesh& mesh = this->meshVector[i];

		uint vCnt = mesh.GetVertexCount();
		uint fCnt = mesh.GetTriCount();

		// NOTE: the direct manipulation of the vectors and not the meshes only works because the mesh does not own 
		// the data storage. If this is changed, the code below has to be changed, too.

		// adjust the color
		uint maxVal = 0;
		for (uint j = 0; j < vCnt; j++) {
			uint atVal = this->vertexLevelAttachment[i][j];
			if (atVal > maxVal) maxVal = atVal;
		}
#if 0 // color the vertices corresponding to their level
		float mvf = static_cast<float>(maxVal);
		for (uint j = 0; j < vCnt; j++) {
			float atVal = static_cast<float>(this->vertexLevelAttachment[i][j]);
			float factor = atVal / mvf;
			vislib::math::Vector<float, 3> resCol;
			if (factor < 0.5f) {
				vislib::math::Vector<float, 3> blue(0.0f, 0.0f, 1.0f);
				vislib::math::Vector<float, 3> white(1.0f, 1.0f, 1.0f);
				resCol = (factor * 2.0f) * white + (1.0f - (factor * 2.0f)) * blue;

			} else {
				vislib::math::Vector<float, 3> red(1.0f, 0.0f, 0.0f);
				vislib::math::Vector<float, 3> white(1.0f, 1.0f, 1.0f);
				resCol = ((factor - 0.5f) * 2.0f) * red + (1.0f - ((factor - 0.5f) * 2.0f)) * white;
			}
			this->colors[i][j * 3 + 0] = static_cast<unsigned char>(resCol.X() * 255.0f);
			this->colors[i][j * 3 + 1] = static_cast<unsigned char>(resCol.Y() * 255.0f);
			this->colors[i][j * 3 + 2] = static_cast<unsigned char>(resCol.Z() * 255.0f);
		}
#endif
		// we need at least 1 border vertex with level > 0 to start an outer border and brim
		if (maxVal < 1) {
			vislib::sys::Log::DefaultLog.WriteError("No region growing was performed and therefore no brim can be specified");
			return false;
		}

		int maxBrimVal = this->maxBrimLevelParam.Param<param::IntParam>()->Value();
		unsigned int minBrim = this->minBrimLevelParam.Param<param::IntParam>()->Value();
		unsigned int maxBrim = this->maxBrimLevelParam.Param<param::IntParam>()->Value();
		if (maxBrimVal < 0) maxBrim = maxVal;

		if (minBrim > maxBrim) {
			vislib::sys::Log::DefaultLog.WriteError("The minBrim value is larger than the maxBrim value");
			return false;
		}

		// count the number of vertices we have to process
		uint maxValueVertexCount = static_cast<uint>(std::count(this->vertexLevelAttachment[i].begin(), this->vertexLevelAttachment[i].end(), maxBrim));

		this->borderVertices[i].clear();
		std::vector<std::set<uint>> localBorder;
		uint setIndex = 0;
		std::set<uint> candidates; // set containing all border candidates
		for (size_t j = 0; j < this->vertexLevelAttachment[i].size(); j++) {
			if (this->vertexLevelAttachment[i][j] == maxBrim) {
				candidates.insert(static_cast<uint>(j));
			}
		}
		
		while (!candidates.empty()) {
			auto start = static_cast<uint>(*candidates.begin()); // take the first element as starting point
			candidates.erase(start);
			localBorder.push_back(std::set<uint>());
			localBorder[setIndex].insert(start);
			std::set<uint> localCandidates;
			localCandidates.insert(start);

			while (!localCandidates.empty()) {
				auto current = static_cast<uint>(*localCandidates.begin());
				localCandidates.erase(current);

				auto forward = std::lower_bound(edgesForward[i].begin(), edgesForward[i].end(), current, [](std::pair<unsigned int, unsigned int> &x, unsigned int val) {
					return x.first < val;
				});
				auto reverse = std::lower_bound(edgesReverse[i].begin(), edgesReverse[i].end(), current, [](std::pair<unsigned int, unsigned int> &x, unsigned int val) {
					return x.first < val;
				});

				// go through all forward edges
				while ((*forward).first == current && forward != edgesForward[i].end()) {
					auto target = (*forward).second;
					if (this->vertexLevelAttachment[i][target] == maxBrim && localCandidates.count(target) == 0 && localBorder[setIndex].count(target) == 0) {
						// when we have found an edge target which is not yet known, add it as local candidate and to the border set
						localBorder[setIndex].insert(target);
						localCandidates.insert(target);
						candidates.erase(target);
					}
					forward++;
				}

				// go through all backward edges
				while ((*reverse).first == current && reverse != edgesReverse[i].end()) {
					auto target = (*reverse).second;
					if (this->vertexLevelAttachment[i][target] == maxBrim && localCandidates.count(target) == 0 && localBorder[setIndex].count(target) == 0) {
						// when we have found an edge target which is not yet known, add it as local candidate and to the border set
						localBorder[setIndex].insert(target);
						localCandidates.insert(target);
						candidates.erase(target);
					}
					reverse++;
				}
			}
			setIndex++;
		}
		
		// all borders have been located
		// find the longest one
		size_t maxSize = 0;
		size_t maxIndex = 0;
		for (size_t j = 0; j < localBorder.size(); j++) {
			if (localBorder[j].size() > maxSize) {
				maxSize = localBorder[j].size();
				maxIndex = j;
			}
		}
		// write all indices to the storage
		this->borderVertices[i] = localBorder[maxIndex];

#if 1 // color the border vertices red, the rest white
		vislib::math::Vector<float, 3> red(1.0f, 0.0f, 0.0f);
		vislib::math::Vector<float, 3> white(1.0f, 1.0f, 1.0f);
		for (uint j = 0; j < vCnt; j++) {
			if (this->borderVertices[i].count(j) > 0) {
				this->colors[i][j * 3 + 0] = static_cast<unsigned char>(red.X() * 255.0f);
				this->colors[i][j * 3 + 1] = static_cast<unsigned char>(red.Y() * 255.0f);
				this->colors[i][j * 3 + 2] = static_cast<unsigned char>(red.Z() * 255.0f);
			} else {
				this->colors[i][j * 3 + 0] = static_cast<unsigned char>(white.X() * 255.0f);
				this->colors[i][j * 3 + 1] = static_cast<unsigned char>(white.Y() * 255.0f);
				this->colors[i][j * 3 + 2] = static_cast<unsigned char>(white.Z() * 255.0f);
			}
		}
#endif
	}

	return true;
}