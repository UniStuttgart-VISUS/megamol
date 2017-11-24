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
#include "protein_calls/BindingSiteCall.h"
#include "TunnelResidueDataCall.h"
#include "tesselator.h"
#include <climits>
#include <iostream>

using namespace megamol;
using namespace megamol::core;
using namespace megamol::trisoup;
using namespace megamol::sombreros;
using namespace megamol::protein_calls;

/*
 * SombreroWarper::SombreroWarper
 */
SombreroWarper::SombreroWarper(void) : Module(),
		meshInSlot("dataIn", "Receives the input mesh"),
		tunnelInSlot("tunnelIn", "Receives the tunnel data"),
		warpedMeshOutSlot("getData", "Returns the mesh data of the wanted area"),
		minBrimLevelParam("minBrimLevel", "Minimal vertex level to count as brim."),
		maxBrimLevelParam("maxBrimLevel", "Maximal vertex level to count as brim. A value of -1 sets the value to the maximal available level"),
		liftingTargetDistance("meshDeformation::liftingTargetDistance", "The distance that is applied to a vertex during the lifting process."),
		maxAllowedLiftingDistance("meshDeformation::maxAllowedDistance", "The maximum allowed distance before vertex lifting is performed.") {

	// Callee slot
	this->warpedMeshOutSlot.SetCallback(CallTriMeshData::ClassName(), CallTriMeshData::FunctionName(0), &SombreroWarper::getData);
	this->warpedMeshOutSlot.SetCallback(CallTriMeshData::ClassName(), CallTriMeshData::FunctionName(1), &SombreroWarper::getExtent);
	this->MakeSlotAvailable(&this->warpedMeshOutSlot);

	// Caller slots
	this->meshInSlot.SetCompatibleCall<CallTriMeshDataDescription>();
	this->MakeSlotAvailable(&this->meshInSlot);

	this->tunnelInSlot.SetCompatibleCall<TunnelResidueDataCallDescription>();
	this->MakeSlotAvailable(&this->tunnelInSlot);

	// Param slots
	this->minBrimLevelParam.SetParameter(new param::IntParam(1, 1, 100));
	this->MakeSlotAvailable(&this->minBrimLevelParam);

	this->maxBrimLevelParam.SetParameter(new param::IntParam(-1, -1, 100));
	this->MakeSlotAvailable(&this->maxBrimLevelParam);

	this->liftingTargetDistance.SetParameter(new param::IntParam(2, 2, 10));
	this->MakeSlotAvailable(&this->liftingTargetDistance);

	this->maxAllowedLiftingDistance.SetParameter(new param::IntParam(2, 2, 10));
	this->MakeSlotAvailable(&this->maxAllowedLiftingDistance);

	this->lastDataHash = 0;
	this->hashOffset = 0;
	this->dirtyFlag = false;

	this->cuda_kernels = std::unique_ptr<CUDAKernels>(new CUDAKernels());
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

	TunnelResidueDataCall * tunnelCall = this->tunnelInSlot.CallAs<TunnelResidueDataCall>();
	if (tunnelCall == nullptr) return false;

	inCall->SetFrameID(outCall->FrameID());
	tunnelCall->SetFrameID(outCall->FrameID());

	if (!(*inCall)(0)) return false;
	if (!(*tunnelCall)(0)) return false;

	// something happened with the input data, we have to recompute it
	if ((lastDataHash != inCall->DataHash()) || dirtyFlag) {
		lastDataHash = inCall->DataHash();
		dirtyFlag = false;

		// copy
		if (!this->copyMeshData(*inCall)) return false;

		// search the sombrero border
		if (!this->findSombreroBorder()) return false;

		// fill the holes of the mesh
		if (!this->fillMeshHoles()) return false;

		// recompute the broken vertex distances
		if (!this->recomputeVertexDistances()) return false;

		if (!this->computeVertexAngles(*tunnelCall)) return false;

		// warp the mesh in the correct position
		if (!this->warpMesh(*tunnelCall)) return false;

		// reset the mesh vector
		for (uint i = 0; i < static_cast<uint>(this->meshVector.size()); i++) {
			this->meshVector[i].SetVertexData(static_cast<uint>(this->vertices[i].size() / 3), this->vertices[i].data(), this->normals[i].data(), this->colors[i].data(), nullptr, false);
			this->meshVector[i].SetTriangleData(static_cast<uint>(this->faces[i].size() / 3), this->faces[i].data(), false);
			this->meshVector[i].SetMaterial(nullptr);
			this->meshVector[i].AddVertexAttribPointer(this->atomIndexAttachment[i].data());
			this->meshVector[i].AddVertexAttribPointer(this->vertexLevelAttachment[i].data());
			this->meshVector[i].AddVertexAttribPointer(this->bsDistanceAttachment[i].data());
		}
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

	TunnelResidueDataCall * tunnelCall = this->tunnelInSlot.CallAs<TunnelResidueDataCall>();
	if (tunnelCall == nullptr) return false;

	this->checkParameters();
	
	if (dirtyFlag) {
		this->hashOffset++;
	}

	inCall->SetFrameID(outCall->FrameID());
	tunnelCall->SetFrameID(outCall->FrameID());

	if (!(*inCall)(1)) return false;
	if (!(*tunnelCall)(1)) return false;

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
	if (this->maxAllowedLiftingDistance.IsDirty()) {
		this->maxAllowedLiftingDistance.ResetDirty();
		this->dirtyFlag = true;
	}
	if (this->liftingTargetDistance.IsDirty()) {
		this->liftingTargetDistance.ResetDirty();
		this->dirtyFlag = true;
	}
}

/*
 * SombreroWarper::copyMeshData
 */
bool SombreroWarper::copyMeshData(CallTriMeshData& ctmd) {
	this->meshVector.clear();
	this->meshVector.resize(ctmd.Count());
	this->meshVector.shrink_to_fit();
	
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

	this->bsDistanceAttachment.clear();
	this->bsDistanceAttachment.resize(ctmd.Count());

	this->edgesForward.clear();
	this->edgesForward.resize(ctmd.Count());

	this->edgesReverse.clear();
	this->edgesReverse.resize(ctmd.Count());

	this->vertexEdgeOffsets.clear();
	this->vertexEdgeOffsets.resize(ctmd.Count());

	this->faces.clear();
	this->faces.resize(ctmd.Count());

	for (uint i = 0; i < ctmd.Count(); i++) {
		uint vertCount = ctmd.Objects()[i].GetVertexCount();
		uint triCount = ctmd.Objects()[i].GetTriCount();
		uint attribCount = ctmd.Objects()[i].GetVertexAttribCount();

		uint atomIndexAttrib = UINT_MAX;
		uint vertexLvlAttrib = UINT_MAX;
		uint bsDistAttrib = UINT_MAX;
		if (attribCount < 3) {
			vislib::sys::Log::DefaultLog.WriteError("Too few vertex attributes detected. The input mesh for the Sombrero warper needs at least three UINT32 vertex attributes.");
			return false;
		}
		// determine the location of the needed attributes
		for (uint j = 0; j < attribCount; j++) {
			auto dt = ctmd.Objects()[i].GetVertexAttribDataType(j);
			if (atomIndexAttrib == UINT_MAX && dt == ctmd.Objects()[i].DT_UINT32) {
				atomIndexAttrib = j;
			} else if (vertexLvlAttrib == UINT_MAX && dt == ctmd.Objects()[i].DT_UINT32) {
				vertexLvlAttrib = j;
			} else if (bsDistAttrib == UINT_MAX && dt == ctmd.Objects()[i].DT_UINT32) {
				bsDistAttrib = j;
			}
		}
		if (atomIndexAttrib == UINT_MAX || vertexLvlAttrib == UINT_MAX || bsDistAttrib == UINT_MAX) {
			vislib::sys::Log::DefaultLog.WriteError("Not enough UINT32 vertex attributes detected. The input mesh for the Sombrero warper needs at least three UINT32 vertex attributes.");
			return false;
		}

		this->vertices[i].resize(vertCount * 3);
		this->normals[i].resize(vertCount * 3);
		this->colors[i].resize(vertCount * 3);
		this->atomIndexAttachment[i].resize(vertCount);
		this->vertexLevelAttachment[i].resize(vertCount);
		this->bsDistanceAttachment[i].resize(vertCount);
		this->faces[i].resize(triCount * 3);

		std::memcpy(this->vertices[i].data(), ctmd.Objects()[i].GetVertexPointerFloat(), vertCount * 3 * sizeof(float));
		std::memcpy(this->normals[i].data(), ctmd.Objects()[i].GetNormalPointerFloat(), vertCount * 3 * sizeof(float));
		std::memcpy(this->colors[i].data(), ctmd.Objects()[i].GetColourPointerByte(), vertCount * 3 * sizeof(unsigned char));
		std::memcpy(this->atomIndexAttachment[i].data(), ctmd.Objects()[i].GetVertexAttribPointerUInt32(atomIndexAttrib), vertCount * sizeof(uint));
		std::memcpy(this->vertexLevelAttachment[i].data(), ctmd.Objects()[i].GetVertexAttribPointerUInt32(vertexLvlAttrib), vertCount * sizeof(uint));
		std::memcpy(this->bsDistanceAttachment[i].data(), ctmd.Objects()[i].GetVertexAttribPointerUInt32(bsDistAttrib), vertCount * sizeof(uint));
		std::memcpy(this->faces[i].data(), ctmd.Objects()[i].GetTriIndexPointerUInt32(), triCount * 3 * sizeof(uint));

		this->meshVector[i].SetVertexData(vertCount, this->vertices[i].data(), this->normals[i].data(), this->colors[i].data(), nullptr, false);
		this->meshVector[i].SetTriangleData(triCount, this->faces[i].data(), false);
		this->meshVector[i].SetMaterial(nullptr);
		this->meshVector[i].AddVertexAttribPointer(this->atomIndexAttachment[i].data());
		this->meshVector[i].AddVertexAttribPointer(this->vertexLevelAttachment[i].data());
		this->meshVector[i].AddVertexAttribPointer(this->bsDistanceAttachment[i].data());

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
		reconstructEdgeSearchStructures(i, vertCount);
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
	this->cutVertices.clear();
	this->cutVertices.resize(this->meshVector.size());
	this->brimIndices.clear();
	this->brimIndices.resize(this->meshVector.size());
	for (uint i = 0; i < static_cast<uint>(this->meshVector.size()); i++) {
		CallTriMeshData::Mesh& mesh = this->meshVector[i];

		uint vCnt = mesh.GetVertexCount();
		uint fCnt = mesh.GetTriCount();

		// NOTE: the direct manipulation of the vectors and not the meshes only works because the mesh does not own 
		// the data storage. If this is changed, the code below has to be changed, too.

		// build triangle search structures
		std::vector<Triangle> firstOrder;
		std::vector<Triangle> secondOrder;
		std::vector<Triangle> thirdOrder;
		firstOrder.resize(fCnt);
		for (uint j = 0; j < fCnt; j++) {
			firstOrder[j] = Triangle(this->faces[i][j * 3 + 0], this->faces[i][j * 3 + 1], this->faces[i][j * 3 + 2]);
		}
		secondOrder = firstOrder;
		thirdOrder = firstOrder;
		std::sort(firstOrder.begin(), firstOrder.end(), [](const Triangle& a, const Triangle& b) {
			return a.v1 < b.v1;
		});
		std::sort(secondOrder.begin(), secondOrder.end(), [](const Triangle& a, const Triangle& b) {
			return a.v2 < b.v2;
		});
		std::sort(thirdOrder.begin(), thirdOrder.end(), [](const Triangle& a, const Triangle& b) {
			return a.v3 < b.v3;
		});

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

				auto forward = edgesForward[i].begin() + this->vertexEdgeOffsets[i][current].first;
				auto reverse = edgesReverse[i].begin() + this->vertexEdgeOffsets[i][current].second;

				// go through all forward edges
				while (forward != edgesForward[i].end() && (*forward).first == current) {
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
				while (reverse != edgesReverse[i].end() && (*reverse).first == current) {
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

		if (localBorder.size() < 1) {
			vislib::sys::Log::DefaultLog.WriteError("No brim border found. No calculation possible!");
			return false;
		}

		// clean the border
		std::set<uint> newLocalBorder;
		for (auto it = localBorder[maxIndex].begin(); it != localBorder[maxIndex].end(); it++) {
			auto vIdx = *it;
			// we iterate over all outgoing triangles and add up the angles they produce
			// if the resulting angle is not very close to 360°, it is a cut vertex
			auto firstIt = std::lower_bound(firstOrder.begin(), firstOrder.end(), vIdx, [](const Triangle& t, uint s) {
				return t.v1 < s;
			});
			auto secondIt = std::lower_bound(secondOrder.begin(), secondOrder.end(), vIdx, [](const Triangle& t, uint s) {
				return t.v2 < s;
			});
			auto thirdIt = std::lower_bound(thirdOrder.begin(), thirdOrder.end(), vIdx, [](const Triangle& t, uint s) {
				return t.v3 < s;
			});

			std::set<uint> vertexSet;
			uint triCount = 0;
			while (firstIt != firstOrder.end() && (*firstIt).v1 == vIdx) {
				vertexSet.insert((*firstIt).v2);
				vertexSet.insert((*firstIt).v3);
				triCount++;
				firstIt++;
			}
			while (secondIt != secondOrder.end() && (*secondIt).v2 == vIdx) {
				vertexSet.insert((*secondIt).v1);
				vertexSet.insert((*secondIt).v3);
				triCount++;
				secondIt++;
			}
			while (thirdIt != thirdOrder.end() && (*thirdIt).v3 == vIdx) {
				vertexSet.insert((*thirdIt).v2);
				vertexSet.insert((*thirdIt).v1);
				triCount++;
				thirdIt++;
			}
			// if we have more adjacent vertices than adjacent triangles, we have a cut vertex
			if (vertexSet.size() > triCount) {
				newLocalBorder.insert(vIdx);
			}
		}
		localBorder[maxIndex] = newLocalBorder;

		// write all indices to the storage
		this->borderVertices[i] = localBorder[maxIndex];
#if 0 // color all found vertices blue
		for (uint j = 0; j < localBorder.size(); j++) {
			for (uint k = 0; k < vCnt; k++) {
				if (localBorder[j].count(k) > 0) {
					this->colors[i][k * 3 + 0] = 0;
					this->colors[i][k * 3 + 1] = 0;
					this->colors[i][k * 3 + 2] = 255;
				} else {
					this->colors[i][k * 3 + 0] = 255;
					this->colors[i][k * 3 + 1] = 255;
					this->colors[i][k * 3 + 2] = 255;
				}
			}
		}
#endif

		/**
		 *	At this point, the border is found, we now try to extend it to form the brim
		 */

#if 0 // color the border vertices red, the rest white
		vislib::math::Vector<float, 3> red(1.0f, 0.0f, 0.0f);
		vislib::math::Vector<float, 3> white(1.0f, 1.0f, 1.0f);
		for (uint j = 0; j < vCnt; j++) {
			if (this->borderVertices[i].count(j) > 0) {
				this->colors[i][j * 3 + 0] = static_cast<unsigned char>(red.X() * 255.0f);
				this->colors[i][j * 3 + 1] = static_cast<unsigned char>(red.Y() * 255.0f);
				this->colors[i][j * 3 + 2] = static_cast<unsigned char>(red.Z() * 255.0f);
			} else {
				//this->colors[i][j * 3 + 0] = static_cast<unsigned char>(white.X() * 255.0f);
				//this->colors[i][j * 3 + 1] = static_cast<unsigned char>(white.Y() * 255.0f);
				//this->colors[i][j * 3 + 2] = static_cast<unsigned char>(white.Z() * 255.0f);
			}
		}
#endif
		std::vector<uint> vertexLevels(vCnt, UINT_MAX);
		this->brimFlags[i].resize(vCnt, false);
		this->brimIndices[i].clear();
		for (size_t j = 0; j < vCnt; j++) {
			if (this->borderVertices[i].count(static_cast<uint>(j)) > 0) {
				this->brimFlags[i][j] = true;
				this->brimIndices[i].push_back(static_cast<uint>(j));
				vertexLevels[j] = 0;
			}
		}

		// perform a region growing starting from the found border
		std::set<uint> brimCandidates = this->borderVertices[i];
		while (!brimCandidates.empty()) {
			uint current = static_cast<uint>(*brimCandidates.begin());
			brimCandidates.erase(current);

			// search for the start indices in both edge lists
			auto forward = edgesForward[i].begin() + this->vertexEdgeOffsets[i][current].first;
			auto reverse = edgesReverse[i].begin() + this->vertexEdgeOffsets[i][current].second;

			while (forward != edgesForward[i].end() && (*forward).first == current) {
				auto target = (*forward).second;
				if (vertexLevels[target] > vertexLevels[current] + 1) {
					vertexLevels[target] = vertexLevels[current] + 1;
					brimCandidates.insert(target);
				}
				forward++;
			}
			while (reverse != edgesReverse[i].end() && (*reverse).first == current) {
				auto target = (*reverse).second;
				if (vertexLevels[target] > vertexLevels[current] + 1) {
					vertexLevels[target] = vertexLevels[current] + 1;
					brimCandidates.insert(target);
				}
				reverse++;
			}
		}

		// go through all vertices. Where the level is <= than maxBrim - minBrim, assign the brim
		for (size_t j = 0; j < vCnt; j++) {
			if (vertexLevels[j] <= maxBrim - minBrim) {
				this->brimFlags[i][j] = true;
#if 0 // coloring of brim vertices
				if (vertexLevels[j] != 0) {
					this->colors[i][j * 3 + 0] = 0;
					this->colors[i][j * 3 + 1] = 255;
					this->colors[i][j * 3 + 2] = 0;
				}
#endif
			}
		}

		this->cutVertices[i].resize(localBorder.size() - 1);

		uint myIndex = 0;
		// identify all real cut vertices
		for (size_t j = 0; j < localBorder.size(); j++) {
			// iterate over all identified vertices
			for (auto vIdx : localBorder[j]) {
				// we iterate over all outgoing triangles and add up the angles they produce
				// if the resulting angle is not very close to 360°, it is a cut vertex
				auto firstIt = std::lower_bound(firstOrder.begin(), firstOrder.end(), vIdx, [](const Triangle& t, uint s) {
					return t.v1 < s;
				});
				auto secondIt = std::lower_bound(secondOrder.begin(), secondOrder.end(), vIdx, [](const Triangle& t, uint s) {
					return t.v2 < s;
				});
				auto thirdIt = std::lower_bound(thirdOrder.begin(), thirdOrder.end(), vIdx, [](const Triangle& t, uint s) {
					return t.v3 < s;
				});

				std::set<uint> vertexSet;
				uint triCount = 0;
				while (firstIt != firstOrder.end() && (*firstIt).v1 == vIdx) {
					vertexSet.insert((*firstIt).v2);
					vertexSet.insert((*firstIt).v3);
					triCount++;
					firstIt++;
				}
				while (secondIt != secondOrder.end() && (*secondIt).v2 == vIdx) {
					vertexSet.insert((*secondIt).v1);
					vertexSet.insert((*secondIt).v3);
					triCount++;
					secondIt++;
				}
				while (thirdIt != thirdOrder.end() && (*thirdIt).v3 == vIdx) {
					vertexSet.insert((*thirdIt).v2);
					vertexSet.insert((*thirdIt).v1);
					triCount++;
					thirdIt++;
				}
				// if we have more adjacent vertices than adjacent triangles, we have a cut vertex
				if (vertexSet.size() > triCount) {
					if (j != maxIndex) {
						this->cutVertices[i][myIndex].insert(vIdx);
					}
				}
			}
			if (j != maxIndex) {
				myIndex++;
			}
		}

#if 0 // color all cut vertices
		for (uint j = 0; j < vCnt; j++) {
			bool colored = false;
			for (uint s = 0; s < this->cutVertices[i].size(); s++) {
				if (this->cutVertices[i][s].count(j) > 0) {
					this->colors[i][j * 3 + 0] = 0;
					this->colors[i][j * 3 + 1] = 0;
					this->colors[i][j * 3 + 2] = 255;
					colored = true;
				}
			}
			if (!colored) {
				this->colors[i][j * 3 + 0] = 255;
				this->colors[i][j * 3 + 1] = 255;
				this->colors[i][j * 3 + 2] = 255;
			}
		}
#endif
		for (size_t j = 0; j < this->cutVertices[i].size(); j++) {
			if (this->cutVertices[i][j].size() == 0) {
				this->cutVertices[i].erase(this->cutVertices[i].begin() + j);
				j--;
			}
		}
	}

	return true;
}

/*
 * SombreroWarper::warpMesh
 */
bool SombreroWarper::warpMesh(TunnelResidueDataCall& tunnelCall) {
	
	this->newBsDistances = this->bsDistanceAttachment;
	this->sombreroLength.clear();
	this->sombreroLength.resize(this->meshVector.size());
	this->sombreroRadius.clear();
	this->sombreroRadius.resize(this->meshVector.size());
	this->brimWidth.clear();
	this->brimWidth.resize(this->meshVector.size());

	for (size_t i = 0; i < this->meshVector.size(); i++) {
		uint vCnt = static_cast<uint>(this->vertices[i].size() / 3);
		uint fCnt = static_cast<uint>(this->faces[i].size() / 3);

		// find the index of the binding site vertex
		auto bsIt = std::find(this->bsDistanceAttachment[i].begin(), this->bsDistanceAttachment[i].end(), 0);
		uint bsVertex = 0; // index of binding site vertex
		if (bsIt != this->bsDistanceAttachment[i].end()) {
			bsVertex = static_cast<uint>(bsIt - this->bsDistanceAttachment[i].begin());
		} else {
			vislib::sys::Log::DefaultLog.WriteError("No binding site vertex present. No computation possible!");
			return false;
		}

		
#if 0 
		/*
		 * step 1: vertex level computation
		 */
		bool liftResult = this->liftVertices();
		if (!liftResult) {
			return false;
		}
#endif
	

		/*
		 * step 2: compute the necessary parameters
		 */
		// the length of the sombrero is the length of the longest tunnel
		float longestLength = 0.0f;
		for (uint j = 0; j < static_cast<uint>(tunnelCall.getTunnelNumber()); j++) {
			auto& tunnel = tunnelCall.getTunnelDescriptions()[j];
			float localLength = 0.0f;
			vislib::math::Vector<float, 3> first, second;
			for (uint k = 4; k < static_cast<uint>(tunnel.coordinates.size()); k += 4) {
				first = vislib::math::Vector<float, 3>(&tunnel.coordinates[k - 4]);
				second = vislib::math::Vector<float, 3>(&tunnel.coordinates[k]);
				localLength += (second - first).Length();
			}
			if (localLength > longestLength) {
				longestLength = localLength;
			}
		}
		this->sombreroLength[i] = longestLength;

		// the inner radius is the median of the sphere radii
		std::vector<float> radii;
		for (uint j = 0; j < static_cast<uint>(tunnelCall.getTunnelNumber()); j++) {
			auto& tunnel = tunnelCall.getTunnelDescriptions()[j];
			for (uint k = 3; k < static_cast<uint>(tunnel.coordinates.size()); k += 4) {
				radii.push_back(tunnel.coordinates[k]);
			}
		}
		std::sort(radii.begin(), radii.end());
		this->sombreroRadius[i] = radii[static_cast<uint>(radii.size() / 2)];

		// the brim radius is the average closest distance between the brim border vertices and the kink
		uint minLevel = static_cast<uint>(this->minBrimLevelParam.Param<param::IntParam>()->Value());
		float avg = 0.0f;
		for (uint j = 0; j < static_cast<uint>(this->brimIndices.size()); j++) {
			uint level = this->vertexLevelAttachment[i][this->brimIndices[i][j]];
			uint current = this->brimIndices[i][j];
			float dist = 0.0f;
			while (level > minLevel) {
				auto forward = edgesForward[i].begin() + this->vertexEdgeOffsets[i][current].first;
				auto reverse = edgesReverse[i].begin() + this->vertexEdgeOffsets[i][current].second;

				bool found = false;
				while (forward != edgesForward[i].end() && (*forward).first == current && !found) {
					auto target = (*forward).second;
					if (this->vertexLevelAttachment[i][target] < level) {
						level = this->vertexLevelAttachment[i][target];
						found = true;
						// compute the distance between the two vertices
						vislib::math::Vector<float, 3> curPos(&this->vertices[i][current * 3]);
						vislib::math::Vector<float, 3> targetPos(&this->vertices[i][target * 3]);
						dist += (targetPos - curPos).Length();
						current = target;
					}
					forward++;
				}
				while (reverse != edgesReverse[i].end() && (*reverse).first == current && !found) {
					auto target = (*reverse).second;
					if (this->vertexLevelAttachment[i][target] < level) {	
						level = this->vertexLevelAttachment[i][target];
						found = true;
						// compute the distance between the two vertices
						vislib::math::Vector<float, 3> curPos(&this->vertices[i][current * 3]);
						vislib::math::Vector<float, 3> targetPos(&this->vertices[i][target * 3]);
						dist += (targetPos - curPos).Length();
						current = target;
					}
					reverse++;
				}
			}
			avg += dist;
		}
		this->brimWidth[i] = avg / static_cast<float>(this->brimIndices.size());

		/**
		 * step 3: mesh deformation
		 */
		bool yResult = this->computeHeightPerVertex(bsVertex);
		if (!yResult) return false;

		bool xzResult = this->computeXZCoordinatePerVertex();
		if (!xzResult) return false;
	}

	return true;
}

/*
 * libtessAlloc
 */
void * libtessAlloc(void * userData, unsigned int size) {
	int * allocated = (int*)userData;
	TESS_NOTUSED(userData);
	*allocated += static_cast<int>(size);
	return malloc(size);
}

/*
 * libtessFree
 */
void libtessFree(void * userData, void * ptr) {
	TESS_NOTUSED(userData);
	free(ptr);
}

/*
 * SombreroWarper::fillMeshHoles
 */
bool SombreroWarper::fillMeshHoles(void) {
	// for each mesh
	for (uint i = 0; i < this->meshVector.size(); i++) {
		// for each hole in the mesh

		std::vector<std::vector<uint>> sortedCuts;
		sortedCuts.resize(this->cutVertices[i].size());
		// we have to sort the mesh vertices to be consecutive along the border
		for (uint j = 0; j < this->cutVertices[i].size(); j++) {
			sortedCuts[j].resize(this->cutVertices[i][j].size());

			auto localSet = this->cutVertices[i][j];
			uint current = *localSet.begin();
			uint setsize = static_cast<uint>(this->cutVertices[i][j].size());
			localSet.erase(current);
			sortedCuts[j][0] = current;
			uint k = 0;
			while (!localSet.empty()) {
				auto forward = edgesForward[i].begin() + this->vertexEdgeOffsets[i][current].first;
				auto reverse = edgesReverse[i].begin() + this->vertexEdgeOffsets[i][current].second;
				bool found = false;
				while (forward != edgesForward[i].end() && (*forward).first == current) {
					auto target = (*forward).second;
					if (this->cutVertices[i][j].count(target) > 0) {
						if (localSet.count(target) > 0 || k == setsize - 1) {
							if (k == 0) {
								// the direction does not matter with k == 0
								localSet.erase(target);
								sortedCuts[j][k + 1] = target;
								found = true;
							} else {
								if (sortedCuts[j][k - 1] != target) {
									localSet.erase(target);
									if (k != setsize - 1) {
										sortedCuts[j][k + 1] = target;
									}
									found = true;
								}
							}
						}
					}
					forward++;
				}
				if (!found) {
					while (reverse != edgesReverse[i].end() && (*reverse).first == current) {
						auto target = (*reverse).second;
						if (this->cutVertices[i][j].count(target) > 0) {
							if (localSet.count(target) > 0 || k == setsize - 1) {
								if (k == 0) {
									// the direction does not matter with k == 0
									localSet.erase(target);
									sortedCuts[j][k + 1] = target;
									found = true;
								}
								else {
									if (sortedCuts[j][k - 1] != target) {
										localSet.erase(target);
										if (k != setsize - 1) {
											sortedCuts[j][k + 1] = target;
										}
										found = true;
									}
								}
							}
						}
						reverse++;
					}
				}
				if (k != setsize - 1) {
					current = sortedCuts[j][k + 1];
				}
				k++;
			}
		}


		for (uint j = 0; j < this->cutVertices[i].size(); j++) {
#if 0 
			// allocate everythin necessary for libtess
			TESSalloc ma;
			TESStesselator* tess = nullptr;
			int allocated = 0;
			memset(&ma, 0, sizeof(ma));
			ma.memalloc = libtessAlloc;
			ma.memfree = libtessFree;
			ma.userData = (void*)&allocated;
			ma.extraVertices = 256;

			tess = tessNewTess(&ma);
			if (tess == nullptr) {
				vislib::sys::Log::DefaultLog.WriteError("No tessellator could be acquired!");
				return false;
			}

			// build input arrays for libtess
			std::vector<float> vertexPositions(this->cutVertices[i][j].size() * 3);
			std::vector<float> vertexNormals(this->cutVertices[i][j].size() * 3);
			std::vector<uint> indexMap(this->cutVertices[i][j].size());
			uint k = 0;
			for (uint v : sortedCuts[j]) {
				vertexPositions[k * 3 + 0] = this->vertices[i][v * 3 + 0];
				vertexPositions[k * 3 + 1] = this->vertices[i][v * 3 + 1];
				vertexPositions[k * 3 + 2] = this->vertices[i][v * 3 + 2];
				vertexNormals[k * 3 + 0] = this->normals[i][v * 3 + 0];
				vertexNormals[k * 3 + 1] = this->normals[i][v * 3 + 1];
				vertexNormals[k * 3 + 2] = this->normals[i][v * 3 + 2];
				indexMap[k] = v;
				k++;
			}
			tessAddContour(tess, 3, vertexPositions.data(), sizeof(float) * 3, static_cast<int>(vertexPositions.size() / 3));
			if (!tessTesselate(tess, TESS_WINDING_NONZERO, TESS_POLYGONS, 3, 3, vertexNormals.data())) {
				vislib::sys::Log::DefaultLog.WriteError("The tessellation of a mesh hole failed!");
				return false;
			}

			const float * verts = tessGetVertices(tess);
			const int * vinds = tessGetVertexIndices(tess);
			const int * elems = tessGetElements(tess);
			const int nverts = tessGetVertexCount(tess);
			const int nelems = tessGetElementCount(tess);

			std::vector<uint> newVertices;
			uint oldVCount = static_cast<uint>(this->atomIndexAttachment[i].size());
			for (uint k = 0; k < static_cast<uint>(nverts); k++) {
				if (vinds[k] == TESS_UNDEF) {
					newVertices.push_back(k);
					this->vertices[i].push_back(verts[3 * k + 0]);
					this->vertices[i].push_back(verts[3 * k + 1]);
					this->vertices[i].push_back(verts[3 * k + 2]);
					this->normals[i].push_back(0.0f);
					this->normals[i].push_back(0.0f);
					this->normals[i].push_back(0.0f);
					this->colors[i].push_back(255);
					this->colors[i].push_back(0);
					this->colors[i].push_back(0);
					// unsure about the following ones
					this->vertexLevelAttachment[i].push_back(0);
					this->bsDistanceAttachment[i].push_back(UINT_MAX);
					this->atomIndexAttachment[i].push_back(0);
				}
			}

			for (int k = 0; k < nelems; k++) {
				uint x = elems[k * 3 + 0];
				uint y = elems[k * 3 + 1];
				uint z = elems[k * 3 + 2];

				if (vinds[x] != TESS_UNDEF) {
					x = indexMap[vinds[x]];
				}
				else {
					auto it = std::find(newVertices.begin(), newVertices.end(), x);
					uint index = static_cast<uint>(it - newVertices.begin());
					x = oldVCount + index;
				}
				if (vinds[y] != TESS_UNDEF) {
					y = indexMap[vinds[y]];
				}
				else {
					auto it = std::find(newVertices.begin(), newVertices.end(), y);
					uint index = static_cast<uint>(it - newVertices.begin());
					y = oldVCount + index;
				}
				if (vinds[z] != TESS_UNDEF) {
					z = indexMap[vinds[z]];
				}
				else {
					auto it = std::find(newVertices.begin(), newVertices.end(), z);
					uint index = static_cast<uint>(it - newVertices.begin());
					z = oldVCount + index;
				}
				this->faces[i].push_back(x);
				this->faces[i].push_back(y);
				this->faces[i].push_back(z);

				this->edgesForward[i].push_back(std::pair<uint, uint>(x, y));
				this->edgesForward[i].push_back(std::pair<uint, uint>(y, z));
				this->edgesForward[i].push_back(std::pair<uint, uint>(z, x));
				this->edgesReverse[i].push_back(std::pair<uint, uint>(y, x));
				this->edgesReverse[i].push_back(std::pair<uint, uint>(z, y));
				this->edgesReverse[i].push_back(std::pair<uint, uint>(x, z));
			}
#else 
			vislib::math::Vector<float, 3> avgPos(0.0f, 0.0f, 0.0f);
			vislib::math::Vector<float, 3> avgNormal(0.0f, 0.0f, 0.0f);
			vislib::math::Vector<float, 3> avgColor(0.0f, 0.0f, 0.0f);

			for (auto v : this->cutVertices[i][j]) {
				vislib::math::Vector<float, 3> pos(&this->vertices[i][v * 3]);
				vislib::math::Vector<float, 3> normal(&this->normals[i][v * 3]);
				vislib::math::Vector<float, 3> color(static_cast<float>(this->colors[i][v * 3]),
					static_cast<float>(this->colors[i][v * 3 + 1]), static_cast<float>(this->colors[i][v * 3 + 2]));
				avgPos += pos;
				avgNormal += normal;
				avgColor += color;
			}
			avgPos /= static_cast<float>(this->cutVertices[i][j].size());
			avgColor /= static_cast<float>(this->cutVertices[i][j].size());
			avgNormal.Normalise();

			this->vertices[i].push_back(avgPos[0]);
			this->vertices[i].push_back(avgPos[1]);
			this->vertices[i].push_back(avgPos[2]);
			this->normals[i].push_back(avgNormal[0]);
			this->normals[i].push_back(avgNormal[1]);
			this->normals[i].push_back(avgNormal[2]);
			this->colors[i].push_back(static_cast<unsigned char>(avgColor[0]));
			this->colors[i].push_back(static_cast<unsigned char>(avgColor[1]));
			this->colors[i].push_back(static_cast<unsigned char>(avgColor[2]));
			this->atomIndexAttachment[i].push_back(0);
			this->bsDistanceAttachment[i].push_back(UINT_MAX);
			this->vertexLevelAttachment[i].push_back(0);

			// vertex was added, now add all triangles
			uint siz = static_cast<uint>(sortedCuts[j].size());
			for (uint k = 0; k < siz; k++) {
				uint x = static_cast<uint>(this->atomIndexAttachment[i].size() - 1);
				uint y = sortedCuts[j][k];
				uint z = sortedCuts[j][(k+1) % siz];
				this->faces[i].push_back(x);
				this->faces[i].push_back(y);
				this->faces[i].push_back(z);

				this->edgesForward[i].push_back(std::pair<uint, uint>(x, y));
				this->edgesForward[i].push_back(std::pair<uint, uint>(y, z));
				this->edgesForward[i].push_back(std::pair<uint, uint>(z, x));
				this->edgesReverse[i].push_back(std::pair<uint, uint>(y, x));
				this->edgesReverse[i].push_back(std::pair<uint, uint>(z, y));
				this->edgesReverse[i].push_back(std::pair<uint, uint>(x, z));
			}
#endif
		}
		// resort the search structures
		std::sort(edgesForward[i].begin(), edgesForward[i].end(), [](const std::pair<unsigned int, unsigned int> &left, const std::pair<unsigned int, unsigned int> &right) {
			return left.first < right.first;
		});
		std::sort(edgesReverse[i].begin(), edgesReverse[i].end(), [](const std::pair<unsigned int, unsigned int> &left, const std::pair<unsigned int, unsigned int> &right) {
			return left.first < right.first;
		});
		// remove edge duplicates
		edgesForward[i].erase(std::unique(edgesForward[i].begin(), edgesForward[i].end()), edgesForward[i].end());
		edgesReverse[i].erase(std::unique(edgesReverse[i].begin(), edgesReverse[i].end()), edgesReverse[i].end());
		reconstructEdgeSearchStructures(i, static_cast<uint>(this->vertices[i].size() / 3));
	}

	return true;
}

/*
 * SombreroWarper::recomputeVertexDistances
 */
bool SombreroWarper::recomputeVertexDistances(void) {
	for (uint i = 0; i < static_cast<uint>(this->meshVector.size()); i++) {
		auto it = std::find(this->bsDistanceAttachment[i].begin(), this->bsDistanceAttachment[i].end(), 0);
		uint bsIndex = UINT_MAX;
		if (it != this->bsDistanceAttachment[i].end()) {
			bsIndex = static_cast<uint>(it - this->bsDistanceAttachment[i].begin());
		} else {
			vislib::sys::Log::DefaultLog.WriteError("No binding site index found!");
			return false;
		}

		std::set<uint> allowedVerticesSet;
		std::set<uint> newset;
		allowedVerticesSet.insert(bsIndex);
		this->bsDistanceAttachment[i][bsIndex] = 0;
		while (!allowedVerticesSet.empty()) {
			newset.clear();
			// for each currently allowed vertex
			for (auto element : allowedVerticesSet) {
				// search for the start indices in both edge lists
				auto forward = edgesForward[i].begin() + this->vertexEdgeOffsets[i][element].first;
				auto reverse = edgesReverse[i].begin() + this->vertexEdgeOffsets[i][element].second;
				// go through all forward edges starting with the vertex
				while (forward != edgesForward[i].end() && (*forward).first == element) {
					auto val = (*forward).second;
					if (this->bsDistanceAttachment[i][val] > this->bsDistanceAttachment[i][element] + 1) {
						this->bsDistanceAttachment[i][val] = this->bsDistanceAttachment[i][element] + 1;
						newset.insert(val);
					}
					forward++;
				}
				// do the same thing for all reverse edges
				while (reverse != edgesReverse[i].end() && (*reverse).first == element) {
					auto val = (*reverse).second;
					// check whether the endpoint is valid
					if (this->bsDistanceAttachment[i][val] > this->bsDistanceAttachment[i][element] + 1) {
						this->bsDistanceAttachment[i][val] = this->bsDistanceAttachment[i][element] + 1;
						newset.insert(val);
					}
					reverse++;
				}
			}
			allowedVerticesSet = newset;
		}
	}
	return true;
}

/*
 * SombreroWarper::computeVertexAngles
 */
bool SombreroWarper::computeVertexAngles(TunnelResidueDataCall& tunnelCall) {

	this->rahiAngles.resize(this->meshVector.size());
	this->rahiAngles.shrink_to_fit();

	for (uint i = 0; i < static_cast<uint>(this->meshVector.size()); i++) {
		// first: find the meridian

		// startpoint: the binding site vertex
		auto it = std::find(this->bsDistanceAttachment[i].begin(), this->bsDistanceAttachment[i].end(), 0);
		uint startIndex = UINT_MAX;
		if (it != this->bsDistanceAttachment[i].end()) {
			startIndex = static_cast<uint>(it - this->bsDistanceAttachment[i].begin());
		}
		else {
			vislib::sys::Log::DefaultLog.WriteError("No start binding site index found!");
			return false;
		}

		// end-point the brim vertex with the lowest level
		uint lowestLevel = UINT_MAX;
		uint endIndex = UINT_MAX;
		std::sort(this->brimIndices[i].begin(), this->brimIndices[i].end());
		for (auto v : this->brimIndices[i]) {
			if (this->bsDistanceAttachment[i][v] < lowestLevel) {
				lowestLevel = this->bsDistanceAttachment[i][v];
				endIndex = v;
			}
		}

		std::vector<uint> meridian;
		meridian.push_back(endIndex);

		uint current = endIndex;
		for (uint j = 0; j < lowestLevel - 1; j++) {

			uint mylevel = lowestLevel - j;

			// search for the start indices in both edge lists
			auto forward = edgesForward[i].begin() + this->vertexEdgeOffsets[i][current].first;
			auto reverse = edgesReverse[i].begin() + this->vertexEdgeOffsets[i][current].second;

			bool found = false;
			while (forward != edgesForward[i].end() && (*forward).first == current) {
				auto target = (*forward).second;
				if (this->bsDistanceAttachment[i][target] == mylevel - 1) {
					meridian.push_back(target);
					current = target;
					found = true;
					break;
				}
				forward++;
			}
			if (found) continue;
			while (reverse != edgesReverse[i].end() && (*reverse).first == current) {
				auto target = (*reverse).second;
				if (this->bsDistanceAttachment[i][target] == mylevel - 1) {
					meridian.push_back(target);
					current = target;
					found = true;
					break;
				}
				reverse++;
			}
		}

		meridian.push_back(startIndex);

#if 0 // color meridian vertices
		for (auto v : meridian) {
			this->colors[i][3 * v + 0] = 255;
			this->colors[i][3 * v + 1] = 0;
			this->colors[i][3 * v + 2] = 0;
		}
#endif
		/**
		 * The indices mean the following:
		 * -1: polar vertex (only assigned for the binding site vertex
		 * 1 : meridian vertex
		 * 2 : vertex adjacent to the meridian to the left
		 * 3 : vertex adjacent to the meridian to the right
		 * 0 : all other vertices
		 */
		std::vector<int> vTypes(this->vertexLevelAttachment[i].size(), 0);
		vTypes.shrink_to_fit();
		for (auto v : meridian) {
			vTypes[v] = 1;
		}
		vTypes[startIndex] = -1;

#if 0 // color brim vertices
		for (auto v : this->brimIndices[i]) {
			this->colors[i][3 * v + 0] = 255;
			this->colors[i][3 * v + 1] = 0;
			this->colors[i][3 * v + 2] = 0;
		}
#endif

		// have to sort the brim indices in a circular manner
		std::vector<uint> sortedBrim;
		std::set<uint> brimTest(this->brimIndices[i].begin(), this->brimIndices[i].end());
		std::set<uint> readySet;
		sortedBrim.push_back(endIndex);
		readySet.insert(endIndex);
		bool isClockwise = false;
		uint k = 0;
		while (sortedBrim.size() != brimTest.size()) {
			current = sortedBrim[sortedBrim.size() - 1];
			auto forward = edgesForward[i].begin() + this->vertexEdgeOffsets[i][current].first;
			auto reverse = edgesReverse[i].begin() + this->vertexEdgeOffsets[i][current].second;
			bool found = false;
			while (forward != edgesForward[i].end() && (*forward).first == current) {
				auto target = (*forward).second;
				if (brimTest.count(target) > 0 && readySet.count(target) == 0) {
					sortedBrim.push_back(target);
					readySet.insert(target);
					found = true;
					break;
				}
				forward++;
			}
			if (!found) {
				while (reverse != edgesReverse[i].end() && (*reverse).first == current) {
					auto target = (*reverse).second;
					if (brimTest.count(target) > 0 && readySet.count(target) == 0) {
						sortedBrim.push_back(target);
						readySet.insert(target);
						found = true;
						break;
					}
					reverse++;
				}
			}
			k++;
			if (!found) {
				vislib::sys::Log::DefaultLog.WriteError("The brim of the sombrero is not continous. Aborting...");
				return false;
			}
		}
#if 0
		vislib::math::Vector<float, 3> red(255.0f, 0.0f, 0.0f);
		float factor = 1.0f / static_cast<float>(sortedBrim.size());
		int f = 0;
		for (auto v : sortedBrim) {
			this->colors[i][3 * v + 0] = static_cast<unsigned char>(f * factor * red[0]);
			this->colors[i][3 * v + 1] = static_cast<unsigned char>(f * factor * red[1]);
			this->colors[i][3 * v + 2] = static_cast<unsigned char>(f * factor * red[2]);
			f++;
		}
#endif

		// the brim is sorted, now we can estimate the directions
		// we assume that the endIndex is in the front
		vislib::math::Vector<float, 3> endVertex(&this->vertices[i][3 * endIndex]);
		vislib::math::Vector<float, 3> startVertex(&this->vertices[i][3 * startIndex]);
		vislib::math::Vector<float, 3> v1, v2;
		uint v1Idx = static_cast<uint>(sortedBrim.size() / 4);
		uint v2Idx = static_cast<uint>(sortedBrim.size() / 2);
		if (endIndex == v1Idx || endIndex == v2Idx || v1Idx == v2Idx) {
			vislib::sys::Log::DefaultLog.WriteError("The brim is too small to compute further");
			return false;
		}
		v1 = vislib::math::Vector<float, 3>(&this->vertices[i][3 * v1Idx]);
		v2 = vislib::math::Vector<float, 3>(&this->vertices[i][3 * v2Idx]);

		// compute the average brim center
		vislib::math::Vector<float, 3> centerVertex(0.0f, 0.0f, 0.0f);
		for (auto v : sortedBrim) {
			centerVertex[0] += this->vertices[i][3 * v + 0];
			centerVertex[1] += this->vertices[i][3 * v + 1];
			centerVertex[2] += this->vertices[i][3 * v + 2];
		}
		centerVertex /= static_cast<float>(sortedBrim.size());

		auto dir = centerVertex - startVertex;
		dir.Normalise();

		// project v1 and v2 onto the plane by centerVertex and dir
		v1 = v1 - (v1 - centerVertex).Dot(dir) * dir;
		v2 = v2 - (v2 - centerVertex).Dot(dir) * dir;

		vislib::math::Vector<float, 3> d1 = v1 - endVertex;
		vislib::math::Vector<float, 3> d2 = v2 - endVertex;
		d1.Normalise();
		d2.Normalise();
		auto normal = d1.Cross(d2);
		normal.Normalise();

		uint left, right;
		if (normal.Dot(dir) >= 0) {
			// the brim index 1 vertex is left of the end vertex
			vTypes[sortedBrim[1]] = 2;
			vTypes[sortedBrim[sortedBrim.size() - 1]] = 3;
			left = sortedBrim[1];
			right = sortedBrim[sortedBrim.size() - 1];
			isClockwise = true;
		} else {
			// the brim index 1 vertex is right of the end vertex
			vTypes[sortedBrim[1]] = 3;
			vTypes[sortedBrim[sortedBrim.size() - 1]] = 2;
			right = sortedBrim[1];
			left = sortedBrim[sortedBrim.size() - 1];
			isClockwise = false;
		}

		// determine candidate vertices
		for (auto current : meridian) {
			if (current == startIndex) continue;
			auto forward = edgesForward[i].begin() + this->vertexEdgeOffsets[i][current].first;
			auto reverse = edgesReverse[i].begin() + this->vertexEdgeOffsets[i][current].second;

			while (forward != edgesForward[i].end() && (*forward).first == current) {
				auto target = (*forward).second;
				if (vTypes[target] == 0) {
					vTypes[target] = 4;
				}
				forward++;
			}
			while (reverse != edgesReverse[i].end() && (*reverse).first == current) {
				auto target = (*reverse).second;
				if (vTypes[target] == 0) {
					vTypes[target] = 4;
				}
				reverse++;
			}
		}

		// propagate the values
		// on the left
		current = left;
		while (current != UINT_MAX) {
			auto forward = edgesForward[i].begin() + this->vertexEdgeOffsets[i][current].first;
			auto reverse = edgesReverse[i].begin() + this->vertexEdgeOffsets[i][current].second;
			bool found = false;
			while (forward != edgesForward[i].end() && (*forward).first == current) {
				auto target = (*forward).second;
				if (vTypes[target] == 4) {
					vTypes[target] = 2;
					current = target;
					found = true;
					break;
				} else if (vTypes[target] == -1) {
					current = UINT_MAX;
					found = true;
					break;
				}
				forward++;
			}
			if (found) continue;
			while (reverse != edgesReverse[i].end() && (*reverse).first == current) {
				auto target = (*reverse).second;
				if (vTypes[target] == 4) {
					vTypes[target] = 2;
					current = target;
					found = true;
					break;
				} else if (vTypes[target] == -1) {
					current = UINT_MAX;
					found = true;
					break;
				}
				reverse++;
			}
		}
		// on the right
		current = right;
		while (current != UINT_MAX) {
			auto forward = edgesForward[i].begin() + this->vertexEdgeOffsets[i][current].first;
			auto reverse = edgesReverse[i].begin() + this->vertexEdgeOffsets[i][current].second;
			bool found = false;
			while (forward != edgesForward[i].end() && (*forward).first == current) {
				auto target = (*forward).second;
				if (vTypes[target] == 4) {
					vTypes[target] = 3;
					current = target;
					found = true;
					break;
				} else if (vTypes[target] == -1) {
					current = UINT_MAX;
					found = true;
					break;
				}
				forward++;
			}
			if (found) continue;
			while (reverse != edgesReverse[i].end() && (*reverse).first == current) {
				auto target = (*reverse).second;
				if (vTypes[target] == 4) {
					vTypes[target] = 3;
					current = target;
					found = true;
					break;
				} else if (vTypes[target] == -1) {
					current = UINT_MAX;
					found = true;
					break;
				}
				reverse++;
			}
		}
		if (meridian.size() < 2) {
			vislib::sys::Log::DefaultLog.WriteError("The meridian is not long enough to proceed with the computation");
			return false;
		}

#if 0 // color meridian vertices
		for (uint v = 0; v < this->vertexLevelAttachment[i].size(); v++) {
			if (vTypes[v] == 1 || vTypes[v] == -1) { // red middle
				this->colors[i][3 * v + 0] = 255;
				this->colors[i][3 * v + 1] = 0;
				this->colors[i][3 * v + 2] = 0;
			} else if (vTypes[v] == 2) { // green left
				this->colors[i][3 * v + 0] = 0;
				this->colors[i][3 * v + 1] = 255;
				this->colors[i][3 * v + 2] = 0;
			} else if (vTypes[v] == 3) { // blue right
				this->colors[i][3 * v + 0] = 0;
				this->colors[i][3 * v + 1] = 0;
				this->colors[i][3 * v + 2] = 255;
			} else if(vTypes[v] == 4) { // candidates yellow
				this->colors[i][3 * v + 0] = 255;
				this->colors[i][3 * v + 1] = 255;
				this->colors[i][3 * v + 2] = 0;
			} else {
				this->colors[i][3 * v + 0] = 255;
				this->colors[i][3 * v + 1] = 255;
				this->colors[i][3 * v + 2] = 255;
			}
		}
#endif

#if 0
		const float thePi = 3.14159265358979f;
		// initialize the angle values of the circumpolar vertices
		// for the brim
		if (isClockwise) {
			for (uint j = 0; j < static_cast<uint>(sortedBrim.size()); j++) {
				this->rahiAngles[i][sortedBrim[j]] = (2.0f * thePi) - (2.0f * thePi * (static_cast<float>(j) / static_cast<float>(sortedBrim.size())));
			}
		} else {
			for (uint j = 0; j < static_cast<uint>(sortedBrim.size()); j++) {
				this->rahiAngles[i][sortedBrim[j]] = 2.0f * thePi * (static_cast<float>(j) / static_cast<float>(sortedBrim.size()));
			}
		}

		// for the vertices around the binding site vertex
		std::set<uint> bsVertices;
		// search for the vertex right of the first meridian vertex
		auto forward = edgesForward[i].begin() + this->vertexEdgeOffsets[i][startIndex].first;
		auto reverse = edgesReverse[i].begin() + this->vertexEdgeOffsets[i][startIndex].second;
		while (forward != edgesForward[i].end() && (*forward).first == startIndex) {
			auto target = (*forward).second;
			bsVertices.insert(target);
			forward++;
		}
		while (reverse != edgesReverse[i].end() && (*reverse).first == startIndex) {
			auto target = (*reverse).second;
			bsVertices.insert(target);
			reverse++;
		}
		if (bsVertices.size() < 3) {
			vislib::sys::Log::DefaultLog.WriteError("The binding site vertex lies in a degenerate region. Aborting...");
			return false;
		}
		std::vector<uint> bsVertexCircle;
		bsVertexCircle.push_back(meridian[meridian.size() - 2]);
		for (auto v : bsVertices) {
			if (vTypes[v] == 3) {
				bsVertexCircle.push_back(v);
			}
		}
		if (bsVertexCircle.size() != 2) {
			vislib::sys::Log::DefaultLog.WriteError("Something went wrong during the circle computation. Aborting...");
			return false;
		}
		std::set<uint> doneset = std::set<uint>(bsVertexCircle.begin(), bsVertexCircle.end());
		while (bsVertexCircle.size() != bsVertices.size()) {
			current = bsVertexCircle[bsVertexCircle.size() - 1];
			auto forward = edgesForward[i].begin() + this->vertexEdgeOffsets[i][current].first;
			auto reverse = edgesReverse[i].begin() + this->vertexEdgeOffsets[i][current].second;
			bool found = false;
			while (forward != edgesForward[i].end() && (*forward).first == current && !found) {
				auto target = (*forward).second;
				if (doneset.count(target) == 0 && bsVertices.count(target) > 0) {
					doneset.insert(target);
					bsVertexCircle.push_back(target);
					found = true;
				}
				forward++;
			}
			while (reverse != edgesReverse[i].end() && (*reverse).first == current && !found) {
				auto target = (*reverse).second;
				if (doneset.count(target) == 0 && bsVertices.count(target) > 0) {
					doneset.insert(target);
					bsVertexCircle.push_back(target);
					found = true;
				}
				reverse++;
			}
		}
		for (uint j = 0; j < static_cast<uint>(bsVertexCircle.size()); j++) {
			this->rahiAngles[i][bsVertexCircle[j]] = 2.0f * thePi * (static_cast<float>(j) / static_cast<float>(bsVertexCircle.size()));
		}
#endif

		// add north pole vertex to the mesh to be able to use the old code
		uint newIndex = static_cast<uint>(this->vertexLevelAttachment[i].size());
		this->vertices[i].push_back(0.0f);
		this->vertices[i].push_back(0.0f);
		this->vertices[i].push_back(0.0f);
		this->colors[i].push_back(0);
		this->colors[i].push_back(0);
		this->colors[i].push_back(0);
		this->normals[i].push_back(0.0f);
		this->normals[i].push_back(1.0f);
		this->normals[i].push_back(0.0f);
		this->vertexLevelAttachment[i].push_back(UINT_MAX);
		this->atomIndexAttachment[i].push_back(UINT_MAX);
		this->bsDistanceAttachment[i].push_back(UINT_MAX);

		vTypes.push_back(-1);
		// add a face for each neighbor of the new vertex
		for (uint j = 0; j < static_cast<uint>(sortedBrim.size()); j++) {
			this->faces[i].push_back(newIndex);
			this->faces[i].push_back(sortedBrim[j]);
			this->faces[i].push_back(sortedBrim[(j + 1) % sortedBrim.size()]);
			this->edgesForward[i].push_back(std::pair<uint, uint>(newIndex, sortedBrim[j]));
			this->edgesReverse[i].push_back(std::pair<uint, uint>(sortedBrim[j], newIndex));
			this->edgesForward[i].push_back(std::pair<uint, uint>(newIndex, sortedBrim[(j + 1) % sortedBrim.size()]));
			this->edgesReverse[i].push_back(std::pair<uint, uint>(sortedBrim[(j + 1) % sortedBrim.size()], newIndex));
			// the other two edges should exist already
		}
		reconstructEdgeSearchStructures(i, newIndex + 1);

		this->rahiAngles[i].resize(this->atomIndexAttachment[i].size(), 0.0f);
		this->rahiAngles[i].shrink_to_fit();
		// compute valid vertex vector
		std::vector<bool> validVertices(this->vertexLevelAttachment[i].size(), true);
		validVertices.shrink_to_fit();
		validVertices[startIndex] = false;
		validVertices[endIndex] = false;
		for (auto c : meridian) {
			validVertices[c] = false;
		}
		// vertex edge offset vector
		std::vector<std::vector<CUDAKernels::Edge>> vertex_edge_offset_local(this->vertexLevelAttachment[i].size());
		std::vector<uint> offsetDepth(this->vertexLevelAttachment[i].size(), 0);
		vertex_edge_offset_local.shrink_to_fit();
		for (uint j = 0; j < static_cast<uint>(vertex_edge_offset_local.size()); j++) {
			auto forward = edgesForward[i].begin() + this->vertexEdgeOffsets[i][j].first;
			auto reverse = edgesReverse[i].begin() + this->vertexEdgeOffsets[i][j].second;
			while (forward != edgesForward[i].end() && (*forward).first == j) {
				auto target = (*forward).second;
				CUDAKernels::Edge edge;
				edge.vertex_id_0 = j;
				edge.vertex_id_1 = target;
				vertex_edge_offset_local[j].push_back(edge);
				forward++;
			}
			while (reverse != edgesReverse[i].end() && (*reverse).first == j) {
				auto target = (*reverse).second;
				CUDAKernels::Edge edge;
				edge.vertex_id_0 = j;
				edge.vertex_id_1 = target;
				vertex_edge_offset_local[j].push_back(edge);
				reverse++;
			}
		}
		for (auto e : edgesForward[i]) { // compute the number of adjacent edges
			offsetDepth[e.first]++;
			offsetDepth[e.second]++;
		}
		uint sum = offsetDepth[0];
		offsetDepth[0] = 0;
		for (uint j = 1; j < offsetDepth.size(); j++) {
			uint oldVal = offsetDepth[j];
			offsetDepth[j] = sum;
			sum += oldVal;
		}

		bool ret = this->cuda_kernels->CreatePhiValues(0.01f, this->rahiAngles[i], validVertices, vertex_edge_offset_local, offsetDepth, vTypes);
		if (!ret) {
			vislib::sys::Log::DefaultLog.WriteError("The CUDA angle diffusion failed");
			return false;
		}

		// remove the added vertex
		this->vertices[i].erase(this->vertices[i].end() - 3, this->vertices[i].end());
		this->colors[i].erase(this->colors[i].end() - 3, this->colors[i].end());
		this->normals[i].erase(this->normals[i].end() - 3, this->normals[i].end());
		this->vertexLevelAttachment[i].pop_back();
		this->atomIndexAttachment[i].pop_back();
		this->bsDistanceAttachment[i].pop_back();
		uint addedFaceValues = static_cast<uint>(sortedBrim.size() * 3);
		this->faces[i].erase(this->faces[i].end() - addedFaceValues, this->faces[i].end());
		uint oldIndex = static_cast<uint>(this->vertexLevelAttachment[i].size());
		auto rit = std::remove_if(this->edgesForward[i].begin(), this->edgesForward[i].end(), [oldIndex](const std::pair<uint, uint>& e) {
			return ((e.first == oldIndex) || (e.second == oldIndex));
		});
		this->edgesForward[i].erase(rit, this->edgesForward[i].end());
		rit = std::remove_if(this->edgesReverse[i].begin(), this->edgesReverse[i].end(), [oldIndex](const std::pair<uint, uint>& e) {
			return ((e.first == oldIndex) || (e.second == oldIndex));
		});
		this->reconstructEdgeSearchStructures(i, oldIndex);

#if 0 // color by angle
		for (uint v = 0; v < this->vertexLevelAttachment[i].size(); v++) {
			this->colors[i][3 * v + 0] = 255 * (this->rahiAngles[i][v] / (2.0f * 3.14159265358979f));
			this->colors[i][3 * v + 1] = 0;
			this->colors[i][3 * v + 2] = 0;
		}
#endif
	}

	return true;
}

/*
 * SombreroWarper::reconstructEdgeSearchStructures
 */
void SombreroWarper::reconstructEdgeSearchStructures(uint index, uint vertex_count) {
	// sort the array
	std::sort(edgesForward[index].begin(), edgesForward[index].end(), [](const std::pair<unsigned int, unsigned int> &left, const std::pair<unsigned int, unsigned int> &right) {
		return left.first < right.first;
	});
	std::sort(edgesReverse[index].begin(), edgesReverse[index].end(), [](const std::pair<unsigned int, unsigned int> &left, const std::pair<unsigned int, unsigned int> &right) {
		return left.first < right.first;
	});
	// construct the offset array
	this->vertexEdgeOffsets[index] = std::vector<std::pair<uint, uint>>(vertex_count, std::pair<uint, uint>(UINT_MAX, UINT_MAX));
	for (uint i = 0; i < edgesForward[index].size(); i++) {
		uint jj = edgesForward[index][i].first;
		if (this->vertexEdgeOffsets[index][jj].first == UINT_MAX) {
			this->vertexEdgeOffsets[index][jj].first = i;
		}
	}
	for (uint i = 0; i < edgesReverse[index].size(); i++) {
		uint jj = edgesReverse[index][i].first;
		if (this->vertexEdgeOffsets[index][jj].second == UINT_MAX) {
			this->vertexEdgeOffsets[index][jj].second = i;
		}
	}
	// repair unset entries
	for (uint i = 0; i < vertex_count; i++) {
		if (this->vertexEdgeOffsets[index][i].first == UINT_MAX) {
			uint j = i + 1;
			while (j < vertex_count) {
				if (this->vertexEdgeOffsets[index][j].first != UINT_MAX) {
					this->vertexEdgeOffsets[index][i].first = this->vertexEdgeOffsets[index][j].first;
					break;
				}
				j++;
			}
			if (j >= vertex_count) {
				this->vertexEdgeOffsets[index][i].first = static_cast<uint>(this->edgesForward[index].size() - 1);
			}
		}
		if (this->vertexEdgeOffsets[index][i].second == UINT_MAX) {
			uint j = i + 1;
			while (j < vertex_count) {
				if (this->vertexEdgeOffsets[index][j].second != UINT_MAX) {
					this->vertexEdgeOffsets[index][i].second = this->vertexEdgeOffsets[index][j].second;
					break;
				}
				j++;
			}
			if (j >= vertex_count) {
				this->vertexEdgeOffsets[index][i].second = static_cast<uint>(this->edgesReverse[index].size() - 1);
			}
		}
	}
}

/**
 * SombreroWarper::liftVertices
 */
bool SombreroWarper::liftVertices(void) {
	// TODO currently not implemented, maybe needed later on
	return true;
}

/**
 * SombreroWarper::computeHeightPerVertex
 */
bool SombreroWarper::computeHeightPerVertex(uint bsVertex) {

	for (uint i = 0; i < static_cast<uint>(this->meshVector.size()); i++) {
		float maxHeight = this->sombreroLength[i] / 2.0f;
		float minHeight = 0.0f - maxHeight;
#if 1
		// all brim vertices have a y-position of + tunnellength / 2
		for (size_t j = 0; j < this->vertexLevelAttachment[i].size(); j++) {
			if (this->brimFlags[i][j]) {
				this->vertices[i][3 * j + 1] = maxHeight;
			}
		}
#endif
		// for the remaining vertices we have to perform a height diffusion, using the last vertices not belonging to the brim as source
		// first step: identify these vertices
		std::set<uint> borderSet;
		std::set<uint> southBorder;
		// go through all forward edges, if one vertex is on the brim and one is not, take the second one
		for (auto e : this->edgesForward[i]) {
			if (this->brimFlags[i][e.first] && !this->brimFlags[i][e.second]) {
				borderSet.insert(e.second);
			} else if (!this->brimFlags[i][e.first] && this->brimFlags[i][e.second]) {
				borderSet.insert(e.first);
			}
			if (e.first == bsVertex) {
				southBorder.insert(e.second);
			} else if (e.second == bsVertex) {
				southBorder.insert(e.first);
			}
		}

		// get the number of non-brim vertices
		uint newVertNum = static_cast<uint>(std::count(this->brimFlags[i].begin(), this->brimFlags[i].end(), false));
		std::set<uint> newVertices;
		// mapping of the new vertices to the old ones
		std::vector<uint> vertMappingToOld(newVertNum);
		// mapping of the old vertices to the new ones
		std::vector<uint> vertMappingToNew(this->vertexLevelAttachment[i].size(), UINT_MAX);
		uint idx = 0;
		for (size_t j = 0; j < this->vertexLevelAttachment[i].size(); j++) {
			if (!this->brimFlags[i][j]) {
				auto cnt = newVertices.size();
				newVertices.insert(static_cast<uint>(j));
				// we have to do this to prevent multiple inserts
				if (newVertices.size() != cnt) {
					vertMappingToOld[idx] = static_cast<uint>(j);
					vertMappingToNew[j] = idx;
					idx++;
				}
			}
		}

		// build the necessary input fields
		std::vector<float> zValues(newVertNum, 0.0f);
		std::vector<bool> zValidity(newVertNum, true);
		std::vector<std::vector<CUDAKernels::Edge>> zEdgeOffset(newVertNum);
		std::vector<uint> zEdgeOffsetDepth(newVertNum);

		zValues[vertMappingToNew[bsVertex]] = minHeight; // TODO
		zValidity[vertMappingToNew[bsVertex]] = false;
		for (auto v : borderSet) { // north border
			zValues[vertMappingToNew[v]] = maxHeight;
			zValidity[vertMappingToNew[v]] = false;
		}
		for (auto v : southBorder) {
			zValues[vertMappingToNew[v]] = minHeight; // TODO
			zValidity[vertMappingToNew[v]] = false;
		}

		for (auto e : this->edgesForward[i]) {
			if (newVertices.count(e.first) > 0 && newVertices.count(e.second) > 0) {
				CUDAKernels::Edge newEdge;
				newEdge.vertex_id_0 = vertMappingToNew[e.first];
				newEdge.vertex_id_1 = vertMappingToNew[e.second];
				zEdgeOffset[newEdge.vertex_id_0].push_back(newEdge);
			}
		}
		for (auto e : this->edgesReverse[i]) {
			if (newVertices.count(e.first) > 0 && newVertices.count(e.second) > 0) {
				CUDAKernels::Edge newEdge;
				newEdge.vertex_id_0 = vertMappingToNew[e.first];
				newEdge.vertex_id_1 = vertMappingToNew[e.second];
				zEdgeOffset[newEdge.vertex_id_0].push_back(newEdge);
			}
		}
		for (uint j = 0; j < zEdgeOffsetDepth.size(); j++) {
			zEdgeOffsetDepth[j] = static_cast<uint>(zEdgeOffset[j].size());
		}
		uint sum = zEdgeOffsetDepth[0];
		zEdgeOffsetDepth[0] = 0;
		for (uint j = 1; j < zEdgeOffsetDepth.size(); j++) {
			uint oldVal = zEdgeOffsetDepth[j];
			zEdgeOffsetDepth[j] = sum;
			sum += oldVal;
		}

		bool kernelRes = this->cuda_kernels->CreateZValues(20000, zValues, zValidity, zEdgeOffset, zEdgeOffsetDepth);
		if (!kernelRes) {
			vislib::sys::Log::DefaultLog.WriteError("The z-values kernel of the height computation failed!");
			return false;
		}

		for (uint j = 0; j < zValues.size(); j++) {
			auto idx = vertMappingToOld[j];
			this->vertices[i][3 * idx + 1] = zValues[j];
		}

	}
	return true;
}

/**
 * SombreroWarper::computeXZCoordinatePerVertex
 */
bool SombreroWarper::computeXZCoordinatePerVertex(void) {

	for (uint i = 0; i < static_cast<uint>(this->meshVector.size()); i++) {
		float minRad = this->sombreroRadius[i];
		float maxRad = minRad + this->brimWidth[i];

		std::set<uint> innerBorderSet;
		// go through all forward edges, if one vertex is on the brim and one is not, take the second one
		for (auto e : this->edgesForward[i]) {
			if (this->brimFlags[i][e.first] && !this->brimFlags[i][e.second]) {
				innerBorderSet.insert(e.second);
			} else if (!this->brimFlags[i][e.first] && this->brimFlags[i][e.second]) {
				innerBorderSet.insert(e.first);
			}
		}
		std::set<uint> outerBorderSet = std::set<uint>(this->brimIndices[i].begin(), this->brimIndices[i].end());

		std::set<uint> completeSet;
		for (uint j = 0; j < this->vertexLevelAttachment[i].size(); j++) {
			if (this->brimFlags[i][j]) {
				completeSet.insert(j);
			}
		}
		completeSet.insert(innerBorderSet.begin(), innerBorderSet.end());
		uint newVertNum = static_cast<uint>(completeSet.size());

		std::vector<float> zValues(newVertNum, 0.0f);
		std::vector<bool> zValidity(newVertNum, true);
		std::vector<std::vector<CUDAKernels::Edge>> zEdgeOffset(newVertNum);
		std::vector<uint> zEdgeOffsetDepth(newVertNum);

		// mapping of the new vertices to the old ones
		std::vector<uint> vertMappingToOld(newVertNum);
		// mapping of the old vertices to the new ones
		std::vector<uint> vertMappingToNew(this->vertexLevelAttachment[i].size(), UINT_MAX);
		for (uint j = 0; j < newVertNum; j++) {
			uint idx = *std::next(completeSet.begin(), j);
			vertMappingToOld[j] = idx;
			vertMappingToNew[idx] = j;
		}
		// compute validity flags and init values
		for (auto v : completeSet) {
			if (innerBorderSet.count(v) > 0) {
				zValues[vertMappingToNew[v]] = minRad;
				zValidity[vertMappingToNew[v]] = false;
			}
			if (outerBorderSet.count(v) > 0) {
				zValues[vertMappingToNew[v]] = maxRad;
				zValidity[vertMappingToNew[v]] = false;
			}
		}
		
		// add edges
		for (auto e : this->edgesForward[i]) {
			if (completeSet.count(e.first) > 0 && completeSet.count(e.second) > 0) {
				CUDAKernels::Edge newEdge;
				newEdge.vertex_id_0 = vertMappingToNew[e.first];
				newEdge.vertex_id_1 = vertMappingToNew[e.second];
				zEdgeOffset[newEdge.vertex_id_0].push_back(newEdge);
			}
		}
		for (auto e : this->edgesReverse[i]) {
			if (completeSet.count(e.first) > 0 && completeSet.count(e.second) > 0) {
				CUDAKernels::Edge newEdge;
				newEdge.vertex_id_0 = vertMappingToNew[e.first];
				newEdge.vertex_id_1 = vertMappingToNew[e.second];
				zEdgeOffset[newEdge.vertex_id_0].push_back(newEdge);
			}
		}
		// compute edge offset
		for (uint j = 0; j < zEdgeOffsetDepth.size(); j++) {
			zEdgeOffsetDepth[j] = static_cast<uint>(zEdgeOffset[j].size());
		}
		uint sum = zEdgeOffsetDepth[0];
		zEdgeOffsetDepth[0] = 0;
		for (uint j = 1; j < zEdgeOffsetDepth.size(); j++) {
			uint oldVal = zEdgeOffsetDepth[j];
			zEdgeOffsetDepth[j] = sum;
			sum += oldVal;
		}

		bool kernelRes = this->cuda_kernels->CreateZValues(20000, zValues, zValidity, zEdgeOffset, zEdgeOffsetDepth);
		if (!kernelRes) {
			vislib::sys::Log::DefaultLog.WriteError("The z-values kernel of the radius computation failed!");
			return false;
		}

		// radius computation finished

		float maxHeight = this->sombreroLength[i] / 2.0f;
		float minHeight = 0.0f - maxHeight;

		// compute the vertex position
		for (uint v = 0; v < static_cast<uint>(this->vertexLevelAttachment[i].size()); v++) {

			if (completeSet.count(v) > 0) {
				// special case for the brim vertices
				float radius = zValues[vertMappingToNew[v]];
				float xCoord = radius * std::cosf(this->rahiAngles[i][v]);
				float zCoord = radius * std::sinf(this->rahiAngles[i][v]);
				this->vertices[i][3 * v + 0] = xCoord;
				this->vertices[i][3 * v + 2] = zCoord;
			} else {
				float l = this->sombreroLength[i];
				float t = std::asinf((this->vertices[i][3 * v + 1] - maxHeight) / l);
				float cost = std::cosf(t);
				float xCoord = minRad * cost * std::cosf(this->rahiAngles[i][v]);
				float zCoord = minRad * cost * std::sinf(this->rahiAngles[i][v]);
				this->vertices[i][3 * v + 0] = xCoord;
				this->vertices[i][3 * v + 2] = zCoord;
			}
		}
	}

	return true;
}