/*
 * TunnelCutter.h
 * Copyright (C) 2006-2017 by MegaMol Team
 * Alle Rechte vorbehalten.
 */
#include "stdafx.h"
#include "TunnelCutter.h"

#include "mmcore/param/IntParam.h"
#include "mmcore/param/BoolParam.h"

#include "protein_calls/BindingSiteCall.h"
#include "protein_calls/MolecularDataCall.h"
#include "mmstd_trisoup/CallTriMeshData.h"
#include "TunnelResidueDataCall.h"
#include <set>
#include <climits>
#include <cfloat>
#include <iostream>

using namespace megamol;
using namespace megamol::core;
using namespace megamol::trisoup;
using namespace megamol::sombreros;
using namespace megamol::protein_calls;

/*
 * TunnelCutter::TunnelCutter
 */
TunnelCutter::TunnelCutter(void) : Module(),
		meshInSlot("dataIn", "Receives the input mesh"),
		cutMeshOutSlot("getData", "Returns the mesh data of the wanted area"),
		tunnelInSlot("tunnelIn", "Receives the input tunnel data"),
		moleculeInSlot("molIn", "Receives the input molecular data"),
		bindingSiteInSlot("bsIn", "Receives the input binding site data"),
		growSizeParam("growSize", "The number of steps for the region growing"),
		isActiveParam("isActive", "Activates and deactivates the cutting performed by this Module. CURRENTLY NOT IN USE"){

	// Callee slot
	this->cutMeshOutSlot.SetCallback(CallTriMeshData::ClassName(), CallTriMeshData::FunctionName(0), &TunnelCutter::getData);
	this->cutMeshOutSlot.SetCallback(CallTriMeshData::ClassName(), CallTriMeshData::FunctionName(1), &TunnelCutter::getExtent);
	this->MakeSlotAvailable(&this->cutMeshOutSlot);

	// Caller slots
	this->meshInSlot.SetCompatibleCall<CallTriMeshDataDescription>();
	this->MakeSlotAvailable(&this->meshInSlot);

	this->tunnelInSlot.SetCompatibleCall<TunnelResidueDataCallDescription>();
	this->MakeSlotAvailable(&this->tunnelInSlot);

	this->moleculeInSlot.SetCompatibleCall<MolecularDataCallDescription>();
	this->MakeSlotAvailable(&this->moleculeInSlot);

	this->bindingSiteInSlot.SetCompatibleCall<BindingSiteCallDescription>();
	this->MakeSlotAvailable(&this->bindingSiteInSlot);

	// parameters
	this->growSizeParam.SetParameter(new param::IntParam(0, 0, 10));
	this->MakeSlotAvailable(&this->growSizeParam);

	this->isActiveParam.SetParameter(new param::BoolParam(true));
	this->MakeSlotAvailable(&this->isActiveParam);

	// other variables
	this->lastDataHash = 0;
	this->hashOffset = 0;
	this->dirt = false;
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

	MolecularDataCall * mdc = this->moleculeInSlot.CallAs<MolecularDataCall>();
	if (mdc == nullptr) return false;

	BindingSiteCall * bsc = this->bindingSiteInSlot.CallAs<BindingSiteCall>();
	if (bsc == nullptr) return false;

	inCall->SetFrameID(outCall->FrameID());
	tc->SetFrameID(outCall->FrameID());
	mdc->SetFrameID(outCall->FrameID());

	if (!(*inCall)(0)) return false;
	if (!(*tc)(0)) return false;
	if (!(*mdc)(0)) return false;
	if (!(*bsc)(0)) return false;

	if (this->dirt) {
		if (this->isActiveParam.Param<param::BoolParam>()->Value()) {
			cutMesh(inCall, tc, mdc, bsc);
		} else {
			this->meshVector.clear();
			this->meshVector.resize(inCall->Count());
			for (unsigned int i = 0; i < inCall->Count(); i++) {
				this->meshVector[i] = inCall->Objects()[i];
			}
		}
		this->dirt = false;
	}

	outCall->SetObjects(static_cast<unsigned int>(this->meshVector.size()), this->meshVector.data());

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

	MolecularDataCall * mdc = this->moleculeInSlot.CallAs<MolecularDataCall>();
	if (mdc == nullptr) return false;

	inCall->SetFrameID(outCall->FrameID());
	tc->SetFrameID(outCall->FrameID());
	mdc->SetFrameID(outCall->FrameID());

	if (!(*inCall)(1)) return false;
	if (!(*tc)(1)) return false;
	if (!(*mdc)(1)) return false;

	if (this->growSizeParam.IsDirty() || this->isActiveParam.IsDirty()) {
		this->hashOffset++;
		this->growSizeParam.ResetDirty();
		this->isActiveParam.ResetDirty();
		this->dirt = true;
	}

	if (inCall->DataHash() != this->lastDataHash) {
		this->dirt = true;
		this->lastDataHash = inCall->DataHash();
	}

	outCall->SetDataHash(inCall->DataHash() + this->hashOffset);
	outCall->SetFrameCount(inCall->FrameCount());
	outCall->SetExtent(inCall->FrameCount(), inCall->AccessBoundingBoxes());

	return true;
}

/*
 * TunnelCutter::cutMesh
 */
bool TunnelCutter::cutMesh(trisoup::CallTriMeshData * meshCall, TunnelResidueDataCall * tunnelCall, MolecularDataCall * molCall, BindingSiteCall * bsCall) {

	// generate set of allowed residue indices
	std::set<int> allowedSet;
	for (int i = 0; i < tunnelCall->getTunnelNumber(); i++) {
		for (int j = 0; j < tunnelCall->getTunnelDescriptions()[i].atomIdentifiers.size(); j++) {
			allowedSet.insert(tunnelCall->getTunnelDescriptions()[i].atomIdentifiers[j].first);
		}
	}
	std::set<unsigned int> allowedVerticesSet;
	
	this->meshVector.clear();
	this->meshVector.resize(meshCall->Count());
	this->vertexKeepFlags.clear();
	this->vertexKeepFlags.resize(meshCall->Count());

	this->vertices.clear();
	this->normals.clear();
	this->colors.clear();
	this->attributes.clear();
	this->levelAttributes.clear();
	this->faces.clear();
	this->vertices.resize(meshCall->Count());
	this->normals.resize(meshCall->Count());
	this->colors.resize(meshCall->Count());
	this->attributes.resize(meshCall->Count());
	this->levelAttributes.resize(meshCall->Count());
	this->faces.resize(meshCall->Count());

	for (int i = 0; i < static_cast<int>(meshCall->Count()); i++) {
		unsigned int vertCount = meshCall->Objects()[i].GetVertexCount();
		unsigned int triCount = meshCall->Objects()[i].GetTriCount();

		// check for the index of the atomIdx attribute
		auto atCnt = meshCall->Objects()[i].GetVertexAttribCount();
		unsigned int attIdx;
		bool found = false;
		if (atCnt != 0) {
			for (attIdx = 0; attIdx < atCnt; attIdx++) {
				if (meshCall->Objects()[i].GetVertexAttribDataType(attIdx) == CallTriMeshData::Mesh::DataType::DT_UINT32) {
					found = true;
					break;
				}
			}
		}

		if (!found) {
			vislib::sys::Log::DefaultLog.WriteError("The %i th object had no atom index attribute and can therefore not be processed", i);
			return false;
		}

		/*
		 * first step: compute which vertices to keep
		 */
		this->vertexKeepFlags[i].resize(vertCount);
		auto atomIndices = meshCall->Objects()[i].GetVertexAttribPointerUInt32(attIdx);
		
		int keptVertices = 0;
		for (int j = 0; j < static_cast<int>(vertCount); j++) {
			// the + 1 for the atom index reverses a -1 done in the MSMSMeshloader
			if (allowedSet.count(static_cast<int>(atomIndices[j]) + 1) > 0) {
				this->vertexKeepFlags[i][j] = true;
				allowedVerticesSet.insert(j);
				keptVertices++;
			} else {
				this->vertexKeepFlags[i][j] = false;
			}
		}

		/*
		 * second step: region growing
		 */

		// init distance vector
		std::vector<unsigned int> vertexDistances(vertCount, UINT_MAX);
		for (int j = 0; j < static_cast<int>(vertCount); j++) {
			if (this->vertexKeepFlags[i][j]) {
				vertexDistances[j] = 0;
			}
		}

		// build search datastructures
		std::vector<std::pair<unsigned int, unsigned int>> edgesForward;
		std::vector<std::pair<unsigned int, unsigned int>> edgesReverse;
		for (int j = 0; j < static_cast<int>(triCount); j++) {
			unsigned int vert1 = meshCall->Objects()[i].GetTriIndexPointerUInt32()[j * 3 + 0];
			unsigned int vert2 = meshCall->Objects()[i].GetTriIndexPointerUInt32()[j * 3 + 1];
			unsigned int vert3 = meshCall->Objects()[i].GetTriIndexPointerUInt32()[j * 3 + 2];

			edgesForward.push_back(std::pair<unsigned int, unsigned int>(vert1, vert2));
			edgesForward.push_back(std::pair<unsigned int, unsigned int>(vert2, vert3));
			edgesForward.push_back(std::pair<unsigned int, unsigned int>(vert1, vert3));
			edgesReverse.push_back(std::pair<unsigned int, unsigned int>(vert2, vert1));
			edgesReverse.push_back(std::pair<unsigned int, unsigned int>(vert3, vert2));
			edgesReverse.push_back(std::pair<unsigned int, unsigned int>(vert3, vert1));
		}
		// sort the search structures
		std::sort(edgesForward.begin(), edgesForward.end(), [](const std::pair<unsigned int, unsigned int> &left, const std::pair<unsigned int, unsigned int> &right) {
			return left.first < right.first;
		});
		std::sort(edgesReverse.begin(), edgesReverse.end(), [](const std::pair<unsigned int, unsigned int> &left, const std::pair<unsigned int, unsigned int> &right) {
			return left.first < right.first;
		});

		// we reuse the allowedset and switch it with the newset each step
		std::set<unsigned int> newset;
		for (int s = 0; s < this->growSizeParam.Param<param::IntParam>()->Value(); s++) {
			newset.clear();
			// for each currently allowed vertex
			for (auto element : allowedVerticesSet) {
				// search for the start indices in both edge lists
				auto forward = std::lower_bound(edgesForward.begin(), edgesForward.end(), element, [](std::pair<unsigned int, unsigned int> &x, unsigned int val) {
					return x.first < val;
				});
				auto reverse = std::lower_bound(edgesReverse.begin(), edgesReverse.end(), element, [](std::pair<unsigned int, unsigned int> &x, unsigned int val) {
					return x.first < val;
				});

				// go through all forward edges starting with the vertex
				while (forward != edgesForward.end() && (*forward).first == element) {
					auto val = (*forward).second;
					// check whether the endpoint val is not yet in our sets
					if (vertexDistances[val] > static_cast<unsigned int>(s + 1)) {
						if (!this->vertexKeepFlags[i][val]) {
							// if it is not, assign the distance
							this->vertexKeepFlags[i][val] = true;
							vertexDistances[val] = s + 1;
							newset.insert(val);
							keptVertices++;
						}
					}
					forward++;
				}

				// do the same thing for all reverse edges
				while (reverse != edgesReverse.end() && (*reverse).first == element) {
					auto val = (*reverse).second;
					// check whether the endpoint val is not yet in our sets
					if (vertexDistances[val] > static_cast<unsigned int>(s + 1)) {
						if (!this->vertexKeepFlags[i][val]) {
							// if it is not, assign the distance
							this->vertexKeepFlags[i][val] = true;
							vertexDistances[val] = s + 1;
							newset.insert(val);
							keptVertices++;
						}
					}
					reverse++;
				}
			}
			allowedVerticesSet = newset;
		}

		/*
		 * third step: find start vertex for the connected component
		 */
		if (bsCall->GetBindingSiteCount() < 1) {
			vislib::sys::Log::DefaultLog.WriteError("There are not binding sites provided. No further computation is possible!");
			return false;
		}

		// get the atom indices for the binding site
		unsigned int firstMol;
		unsigned int firstRes;
		unsigned int firstAtom;
		std::vector<vislib::math::Vector<float, 3>> atomPositions; // the atom positions
		std::set<int> atx; // the atom indices
		for (unsigned int cCnt = 0; cCnt < molCall->ChainCount(); cCnt++) {
			firstMol = molCall->Chains()[cCnt].FirstMoleculeIndex();
			for (unsigned int mCnt = firstMol; mCnt < firstMol + molCall->Chains()[cCnt].MoleculeCount(); mCnt++) {
				firstRes = molCall->Molecules()[mCnt].FirstResidueIndex();
				for (unsigned int rCnt = 0; rCnt < molCall->Molecules()[mCnt].ResidueCount(); rCnt++) {
					// only check the first binding site
					// we take the average location of the first described amino acid as starting point
					if (bsCall->GetBindingSite(0)->Count() < 1) {
						vislib::sys::Log::DefaultLog.WriteError("The provided binding site was empty. No further computation possible!");
						return false;
					}
					vislib::Pair<char, unsigned int> bsRes = bsCall->GetBindingSite(0)->operator[](0);
					if (molCall->Chains()[cCnt].Name() == bsRes.First() &&
						molCall->Residues()[firstRes + rCnt]->OriginalResIndex() == bsRes.Second() &&
						molCall->ResidueTypeNames()[molCall->Residues()[firstRes + rCnt]->Type()] == bsCall->GetBindingSiteResNames(0)->operator[](0)) {

						firstAtom = molCall->Residues()[firstRes + rCnt]->FirstAtomIndex();
						for (unsigned int aCnt = 0; aCnt < molCall->Residues()[firstRes + rCnt]->AtomCount(); aCnt++) {
							unsigned int aIdx = firstAtom + aCnt;
							float xcoord = molCall->AtomPositions()[3 * aIdx + 0];
							float ycoord = molCall->AtomPositions()[3 * aIdx + 1];
							float zcoord = molCall->AtomPositions()[3 * aIdx + 2];
							atomPositions.push_back(vislib::math::Vector<float, 3>(xcoord, ycoord, zcoord));
							atx.insert(static_cast<int>(aIdx));
						}
					}
				}
			}
		}

		// compute the average position of close atoms
		vislib::math::Vector<float, 3> avgPos(0.0f, 0.0f, 0.0f);
		for (auto s : atx) {
			avgPos += vislib::math::Vector<float, 3>(&molCall->AtomPositions()[3 * s]);
		}
		avgPos /= static_cast<float>(atx.size());

		// search for the vertex closest to the given position
		float minDist = FLT_MAX;
		unsigned int minIndex = UINT_MAX;
		for (unsigned int j = 0; j < vertCount; j++) {
			if (this->vertexKeepFlags[i][j]) {
				vislib::math::Vector<float, 3> vertPos = vislib::math::Vector<float, 3>(&meshCall->Objects()[i].GetVertexPointerFloat()[j * 3]);
				vislib::math::Vector<float, 3> distVec = avgPos - vertPos;
				if (distVec.Length() < minDist) {
					minDist = distVec.Length();
					minIndex = j;
				}
			}
		}

		// perform a region growing starting from minIndex
		// reusing the previously performed data structures.
		std::vector<bool> keepVector(this->vertexKeepFlags[i].size(), false);
		keepVector[minIndex] = true;
		allowedVerticesSet.clear();
		allowedVerticesSet.insert(minIndex);
		keptVertices = 1; // reset the value if we want to have just 1 connected component
		
		while (!allowedVerticesSet.empty()) {
			newset.clear();
			// for each currently allowed vertex
			for (auto element : allowedVerticesSet) {
				// search for the start indices in both edge lists
				auto forward = std::lower_bound(edgesForward.begin(), edgesForward.end(), element, [](std::pair<unsigned int, unsigned int> &x, unsigned int val) {
					return x.first < val;
				});
				auto reverse = std::lower_bound(edgesReverse.begin(), edgesReverse.end(), element, [](std::pair<unsigned int, unsigned int> &x, unsigned int val) {
					return x.first < val;
				});

				// go through all forward edges starting with the vertex
				while ((*forward).first == element && forward != edgesForward.end()) {
					auto val = (*forward).second;
					// check whether the endpoint val is not yet in our sets
					if (!keepVector[val] && this->vertexKeepFlags[i][val]) {
						// if it is not, assign the distance
						keepVector[val] = true;
						newset.insert(val);
						keptVertices++;
					}
					forward++;
				}

				// do the same thing for all reverse edges
				while ((*reverse).first == element && reverse != edgesReverse.end()) {
					auto val = (*reverse).second;
					// check whether the endpoint val is not yet in our sets
					if (!keepVector[val] && this->vertexKeepFlags[i][val]) {
						// if it is not, assign the distance
						keepVector[val] = true;
						newset.insert(val);
						keptVertices++;
					}
					reverse++;
				}
			}
			allowedVerticesSet = newset;
		}

		this->vertexKeepFlags[i] = keepVector; // overwrite the vector

		/*
		 * fourth step: mesh cutting
		 */
		this->vertices[i].resize(keptVertices * 3);
		this->normals[i].resize(keptVertices * 3);
		this->colors[i].resize(keptVertices * 3); // TODO accept other color configurations
		this->attributes[i].resize(keptVertices);
		this->levelAttributes[i].resize(keptVertices);

		// compute vertex mapping and fill vertex vectors
		std::vector<unsigned int> vertIndexMap(vertCount, UINT_MAX); // mapping of old vertex indices to new ones
		int help = 0;
		for (int j = 0; j < static_cast<int>(vertCount); j++) {
			if (this->vertexKeepFlags[i][j]) {
				// mapping
				vertIndexMap[j] = help;

				// vector fill
				this->vertices[i][help * 3 + 0] = meshCall->Objects()[i].GetVertexPointerFloat()[j * 3 + 0];
				this->vertices[i][help * 3 + 1] = meshCall->Objects()[i].GetVertexPointerFloat()[j * 3 + 1];
				this->vertices[i][help * 3 + 2] = meshCall->Objects()[i].GetVertexPointerFloat()[j * 3 + 2];

				this->normals[i][help * 3 + 0] = meshCall->Objects()[i].GetNormalPointerFloat()[j * 3 + 0];
				this->normals[i][help * 3 + 1] = meshCall->Objects()[i].GetNormalPointerFloat()[j * 3 + 1];
				this->normals[i][help * 3 + 2] = meshCall->Objects()[i].GetNormalPointerFloat()[j * 3 + 2];

				this->colors[i][help * 3 + 0] = meshCall->Objects()[i].GetColourPointerByte()[j * 3 + 0];
				this->colors[i][help * 3 + 1] = meshCall->Objects()[i].GetColourPointerByte()[j * 3 + 1];
				this->colors[i][help * 3 + 2] = meshCall->Objects()[i].GetColourPointerByte()[j * 3 + 2];

				// DEBUG
				/*if (j == minIndex) {
					this->colors[i][help * 3 + 0] = 255;
					this->colors[i][help * 3 + 1] = 0;
					this->colors[i][help * 3 + 2] = 0;
				}*/

				this->attributes[i][help] = meshCall->Objects()[i].GetVertexAttribPointerUInt32(attIdx)[j];
				this->levelAttributes[i][help] = vertexDistances[j];

				help++;
			}
		}

		this->faces[i].clear();
		// compute the triangles
		for (int j = 0; j < static_cast<int>(triCount); j++) {
			unsigned int vert1 = meshCall->Objects()[i].GetTriIndexPointerUInt32()[j * 3 + 0];
			unsigned int vert2 = meshCall->Objects()[i].GetTriIndexPointerUInt32()[j * 3 + 1];
			unsigned int vert3 = meshCall->Objects()[i].GetTriIndexPointerUInt32()[j * 3 + 2];

			// to keep a triangle, all vertices have to be kept, since we have no correct mapping for the other vertices
			if (vertexKeepFlags[i][vert1] && vertexKeepFlags[i][vert2] && vertexKeepFlags[i][vert3]) { 
				this->faces[i].push_back(vertIndexMap[vert1]);
				this->faces[i].push_back(vertIndexMap[vert2]);
				this->faces[i].push_back(vertIndexMap[vert3]);
			}
		}	


		/*
		 * nth step: fill the data into the structure
		 */
		this->meshVector[i].SetVertexData(static_cast<unsigned int>(this->vertices[i].size() / 3), this->vertices[i].data(), this->normals[i].data(), this->colors[i].data(), NULL, false);
		this->meshVector[i].SetTriangleData(static_cast<unsigned int>(this->faces[i].size() / 3), this->faces[i].data(), false);
		this->meshVector[i].SetMaterial(nullptr);
		this->meshVector[i].AddVertexAttribPointer(this->attributes[i].data());
		this->meshVector[i].AddVertexAttribPointer(this->levelAttributes[i].data());
	}

	return true;
}
