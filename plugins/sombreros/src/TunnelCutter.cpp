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

	inCall->SetFrameID(outCall->FrameID());
	tc->SetFrameID(outCall->FrameID());

	if (!(*inCall)(0)) return false;
	if (!(*tc)(0)) return false;

	if (this->dirt) {
		if (this->isActiveParam.Param<param::BoolParam>()->Value()) {
			cutMesh(inCall, tc);
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

	inCall->SetFrameID(outCall->FrameID());
	tc->SetFrameID(outCall->FrameID());

	if (!(*inCall)(1)) return false;
	if (!(*tc)(1)) return false;

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
void TunnelCutter::cutMesh(trisoup::CallTriMeshData * meshCall, TunnelResidueDataCall * tunnelCall) {

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
	this->faces.clear();
	this->vertices.resize(meshCall->Count());
	this->normals.resize(meshCall->Count());
	this->colors.resize(meshCall->Count());
	this->attributes.resize(meshCall->Count());
	this->faces.resize(meshCall->Count());

	for (int i = 0; i < static_cast<int>(meshCall->Count()); i++) {
		unsigned int vertCount = meshCall->Objects()[i].GetVertexCount();
		unsigned int triCount = meshCall->Objects()[i].GetTriCount();

		/*
		 * first step: compute which vertices to keep
		 */
		this->vertexKeepFlags[i].resize(vertCount);
		auto atomIndices = meshCall->Objects()[i].GetVertexAttribPointerUInt32();
		
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

		if (this->growSizeParam.Param<param::IntParam>() > 0) {
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
					while ((*forward).first == element && forward != edgesForward.end()) {
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
					while ((*reverse).first == element && reverse != edgesReverse.end()) {
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
		} /* end if(growsizeparam > 0) */


		/*
		 * third step: mesh cutting
		 */
		this->vertices[i].resize(keptVertices * 3);
		this->normals[i].resize(keptVertices * 3);
		this->colors[i].resize(keptVertices * 3); // TODO accept other color configurations
		this->attributes[i].resize(keptVertices);

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

				this->attributes[i][help] = meshCall->Objects()[i].GetVertexAttribPointerUInt32()[j];

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

		// TODO find start vertex for the connection component
		// needed: molecularDataCall, BindingSiteDataCall

		/*
		 * fourth step: fill the data into the structure
		 */
		this->meshVector[i].SetVertexData(static_cast<unsigned int>(this->vertices[i].size() / 3), this->vertices[i].data(), this->normals[i].data(), this->colors[i].data(), NULL, false);
		this->meshVector[i].SetTriangleData(static_cast<unsigned int>(this->faces[i].size() / 3), this->faces[i].data(), false);
		this->meshVector[i].SetMaterial(nullptr);
	}
}
