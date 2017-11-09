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
	this->cutVertices.clear();
	this->cutVertices.resize(this->meshVector.size());
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
		for (size_t j = 0; j < vCnt; j++) {
			if (this->borderVertices[i].count(static_cast<uint>(j)) > 0) {
				this->brimFlags[i][j] = true;
				vertexLevels[j] = 0;
			}
		}

		// perform a region growing starting from the found border
		std::set<uint> brimCandidates = this->borderVertices[i];
		while (!brimCandidates.empty()) {
			uint current = static_cast<uint>(*brimCandidates.begin());
			brimCandidates.erase(current);

			// search for the start indices in both edge lists
			auto forward = std::lower_bound(edgesForward[i].begin(), edgesForward[i].end(), current, [](std::pair<unsigned int, unsigned int> &x, unsigned int val) {
				return x.first < val;
			});
			auto reverse = std::lower_bound(edgesReverse[i].begin(), edgesReverse[i].end(), current, [](std::pair<unsigned int, unsigned int> &x, unsigned int val) {
				return x.first < val;
			});

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

		this->cutVertices[i].resize(localBorder.size() - 1);

		uint myIndex = 0;
		// identify all real cut vertices
		for (size_t j = 0; j < localBorder.size(); j++) {
			if (j != maxIndex) { // we do this only when it is not the brim
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
						this->cutVertices[i][myIndex].insert(vIdx);
					}
				}
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
	}

	return true;
}

/*
 * SombreroWarper::warpMesh
 */
bool SombreroWarper::warpMesh(TunnelResidueDataCall& tunnelCall) {
	
	this->newBsDistances = this->bsDistanceAttachment;

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

		/*
		 * step 1: vertex level computation
		 */

		// TODO
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
				auto forward = std::lower_bound(edgesForward[i].begin(), edgesForward[i].end(), current, [](std::pair<unsigned int, unsigned int> &x, unsigned int val) {
					return x.first < val;
				});
				auto reverse = std::lower_bound(edgesReverse[i].begin(), edgesReverse[i].end(), current, [](std::pair<unsigned int, unsigned int> &x, unsigned int val) {
					return x.first < val;
				});
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
		
		
	}

	return true;
}