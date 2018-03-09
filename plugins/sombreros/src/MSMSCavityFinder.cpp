/*
 * MSMSCavityFinder.h
 * Copyright (C) 2006-2018 by MegaMol Team
 * Alle Rechte vorbehalten.
 */
#include "stdafx.h"
#include "MSMSCavityFinder.h"

#include "mmcore/param/IntParam.h"
#include "mmcore/param/BoolParam.h"
#include "mmcore/param/FloatParam.h"

#include "protein_calls/BindingSiteCall.h"
#include "protein_calls/MolecularDataCall.h"
#include "geometry_calls/CallTriMeshData.h"
#include "TunnelResidueDataCall.h"
#include <set>
#include <climits>
#include <cfloat>
#include <iostream>

#pragma warning(push)
#pragma warning(disable : 4996)
#pragma warning(disable : 4267)
#include "nanoflann.hpp"
#include "utils.h"
#pragma warning(pop)

using namespace megamol;
using namespace megamol::core;
using namespace megamol::geocalls;
using namespace megamol::sombreros;
using namespace megamol::protein_calls;

/*
 * MSMSCavityFinder::MSMSCavityFinder
 */
MSMSCavityFinder::MSMSCavityFinder(void) : Module(),
        innerMeshInSlot("innerDataIn", "Receives the inner input mesh"),
        outerMeshInSlot("outerDataIn", "Receives the outer input mesh"),
        cutMeshOutSlot("getData", "Returns the mesh data of the wanted area"),
        distanceParam("distance", "Mesh-mesh distance threshold for cavity detection"),
        dataHash(0), lastFrame(-1), lastHashInner(0), lastHashOuter(0) {

    std::cout << std::endl << std::endl << "HERASRAEWRARAWSDJSÖALKDJÖLK" << std::endl << std::endl << std::endl;

    // Callee slot
    this->cutMeshOutSlot.SetCallback(CallTriMeshData::ClassName(), CallTriMeshData::FunctionName(0), &MSMSCavityFinder::getData);
    this->cutMeshOutSlot.SetCallback(CallTriMeshData::ClassName(), CallTriMeshData::FunctionName(1), &MSMSCavityFinder::getExtent);
    this->MakeSlotAvailable(&this->cutMeshOutSlot);

    // Caller slots
    this->innerMeshInSlot.SetCompatibleCall<CallTriMeshDataDescription>();
    this->MakeSlotAvailable(&this->innerMeshInSlot);

    this->outerMeshInSlot.SetCompatibleCall<CallTriMeshDataDescription>();
    this->MakeSlotAvailable(&this->outerMeshInSlot);

    // Parameter slots
    this->distanceParam.SetParameter(new param::FloatParam(4.0f, 0.0f));
    this->MakeSlotAvailable(&this->distanceParam);
}

/*
 * MSMSCavityFinder::~MSMSCavityFinder
 */
MSMSCavityFinder::~MSMSCavityFinder(void) {
    this->Release();
}

/*
 * MSMSCavityFinder::create
 */
bool MSMSCavityFinder::create(void) {
    return true;
}

/*
 * MSMSCavityFinder::release
 */
void MSMSCavityFinder::release(void) {
}

/*
 * MSMSCavityFinder::getData
 */
bool MSMSCavityFinder::getData(Call& call) {
    CallTriMeshData * outCall = dynamic_cast<CallTriMeshData*>(&call);
    if (outCall == nullptr) return false;

    CallTriMeshData * inInnerCall = this->innerMeshInSlot.CallAs<CallTriMeshData>();
    if (inInnerCall == nullptr) return false;

    CallTriMeshData * inOuterCall = this->outerMeshInSlot.CallAs<CallTriMeshData>();
    if (inOuterCall == nullptr) return false;

    inInnerCall->SetFrameID(outCall->FrameID());
    if (!(*inInnerCall)(0)) return false;

    inOuterCall->SetFrameID(outCall->FrameID());
    if (!(*inOuterCall)(0)) return false;

    // squared distance threshold to outer mesh (for cavity detection)
    float sqDist = distanceParam.Param<param::FloatParam>()->Value();
    sqDist = sqDist * sqDist;

    // make sure that there is at least one mesh available and vertice are stored as float
    // CAUTION: THIS CODE ONLY USES THE FIRST MESH OF THE CALL!
    if (inInnerCall->Count() == 0)
        return false;
    auto innerObj = &inInnerCall->Objects()[0];
    // TODO implement branches for other data types
    if (innerObj->GetVertexDataType() != CallTriMeshData::Mesh::DT_FLOAT)
        return false;

    // only recompute vertex distances if something has changed
    if (this->lastFrame != outCall->FrameID() ||
        this->lastHashInner != inInnerCall->DataHash() ||
        this->lastHashOuter != inOuterCall->DataHash() ||
        this->distanceParam.IsDirty() ||
        this->vertexIndex.size() != innerObj->GetVertexCount() ) {

        // store values of this request
        this->lastFrame = outCall->FrameID();
        this->lastHashInner = inInnerCall->DataHash();
        this->lastHashOuter = inOuterCall->DataHash();
        this->distanceParam.ResetDirty();

        // insert all vertices of the outer mesh into the point cloud
        // CAUTION: THIS CODE ONLY USES THE FIRST MESH OF THE CALL!
        if (inOuterCall->Count() == 0)
            return false;
        auto outerObj = &inOuterCall->Objects()[0];
        // TODO implement branches for other data types
        if (outerObj->GetVertexDataType() != CallTriMeshData::Mesh::DT_FLOAT)
            return false;
        PointCloud<float> pointCloud;
        pointCloud.pts.resize(outerObj->GetVertexCount());
        for (unsigned int i = 0; i < outerObj->GetVertexCount(); i++) {
            pointCloud.pts[i].x = outerObj->GetVertexPointerFloat()[i * 3];
            pointCloud.pts[i].y = outerObj->GetVertexPointerFloat()[i * 3 + 1];
            pointCloud.pts[i].z = outerObj->GetVertexPointerFloat()[i * 3 + 2];
        }

        typedef nanoflann::KDTreeSingleIndexAdaptor<
            nanoflann::L2_Simple_Adaptor<float, PointCloud<float>>,
            PointCloud<float>,
            3> my_kd_tree_t;

        my_kd_tree_t searchIndex(3, pointCloud, nanoflann::KDTreeSingleIndexAdaptorParams(10));
        searchIndex.buildIndex();

        nanoflann::SearchParams searchParams;
    
        // search for closest point
        // CAUTION: THIS CODE ONLY USES THE FIRST MESH OF THE CALL!
        innerObj = &inInnerCall->Objects()[0];
        // TODO implement branches for other data types
        if (innerObj->GetVertexDataType() != CallTriMeshData::Mesh::DT_FLOAT)
            return false;
        this->vertexIndex.clear();
        this->vertexIndex.resize(innerObj->GetVertexCount());
        this->distanceToMesh.clear();
        this->distanceToMesh.resize(innerObj->GetVertexCount());
        // DEBUG HACK!!
        //auto color = const_cast<unsigned char*>(innerObj->GetColourPointerByte());
        for (unsigned int i = 0; i < innerObj->GetVertexCount(); i++) {
            auto nMatches = searchIndex.knnSearch(&innerObj->GetVertexPointerFloat()[i * 3], 1, &this->vertexIndex[i], &this->distanceToMesh[i]);
            // DEBUG HACK!!
            //if (this->distanceToMesh[i] > sqDist) {
            //    color[i * 3 + 0] = 250;
            //    color[i * 3 + 1] = 250;
            //    color[i * 3 + 2] = 250;
            //    //std::cout << "nMatches " << nMatches << " index: " << ret_index[0] << "; distance: " << sqrt(out_dist_sqr[0]) << std::endl;
            //}
        }

        // make new mesh containing only cavities
        this->cavityMesh = *innerObj;
        unsigned int triaCnt = this->cavityMesh.GetTriCount();
        // TODO implement branches for other data types
        if (this->cavityMesh.GetTriDataType() != CallTriMeshData::Mesh::DT_UINT32)
            return false;
        triaIndices.SetCount(0);
        triaIndices.AssertCapacity(triaCnt * 3);
        // check all triangle edges for distance to the outer mesh
        unsigned int idx0;
        unsigned int idx1;
        unsigned int idx2;
        for (unsigned int triaIdx = 0; triaIdx < triaCnt; triaIdx++) {
            idx0 = this->cavityMesh.GetTriIndexPointerUInt32()[triaIdx * 3 + 0];
            idx1 = this->cavityMesh.GetTriIndexPointerUInt32()[triaIdx * 3 + 1];
            idx2 = this->cavityMesh.GetTriIndexPointerUInt32()[triaIdx * 3 + 2];
            // add triangle, if all triangles are far from the outer mesh
            if (this->distanceToMesh[idx0] > sqDist && this->distanceToMesh[idx1] > sqDist && this->distanceToMesh[idx2] > sqDist) {
                triaIndices.Add(idx0);
                triaIndices.Add(idx1);
                triaIndices.Add(idx2);
            }
        }
        this->cavityMesh.SetTriangleData(static_cast<uint>(this->triaIndices.Count() / 3), &this->triaIndices[0], false);
        this->dataHash++;

        // --- find connected trianges forming independent submeshes (i.e., cavities) ---
        this->cavitySubmeshes.SetCount(0);
        // for each vertex, collect all connected faces
        vislib::Array<vislib::Array<unsigned int> > facesPerVertex;
        facesPerVertex.SetCount(this->cavityMesh.GetVertexCount());
        for (unsigned int triaIdx = 0; triaIdx < this->cavityMesh.GetTriCount(); triaIdx++) {
            facesPerVertex[this->cavityMesh.GetTriIndexPointerUInt32()[triaIdx * 3 + 0]].Add(triaIdx);
            facesPerVertex[this->cavityMesh.GetTriIndexPointerUInt32()[triaIdx * 3 + 1]].Add(triaIdx);
            facesPerVertex[this->cavityMesh.GetTriIndexPointerUInt32()[triaIdx * 3 + 2]].Add(triaIdx);
        }
        //reset the mesh ID for all vertices
        vislib::Array<int> vertexMeshId;
        vertexMeshId.SetCount(this->cavityMesh.GetVertexCount());
        int currentMeshId = -1;
        for (unsigned int vertIdx = 0; vertIdx < this->cavityMesh.GetVertexCount(); vertIdx++) {
            vertexMeshId[vertIdx] = -1;
        }
        // iterate over all vertices to propagate the mesh index
        vislib::Array<unsigned int> currentMeshVertices;
        for (unsigned int vertIdx = 0; vertIdx < this->cavityMesh.GetVertexCount(); vertIdx++) {
            currentMeshVertices.Clear();
            currentMeshVertices.AssertCapacity(this->cavityMesh.GetVertexCount());
            // does the current vertex have any faces connected?
            if (facesPerVertex[vertIdx].Count() < 1) {
                continue;
            }
            // does the current vertex already have a mesh id assigned?
            if (vertexMeshId[vertIdx] < 0) {
                // if not: proceed to next mesh id
                currentMeshId++;
            } else {
                // otherwise: proceed
                continue;
            }
            // add the current vertex to the list of vertices that belong to the same mesh
            currentMeshVertices.Add(vertIdx);
            // assign the new mesh id to the current vetex
            vertexMeshId[vertIdx] = currentMeshId;
            unsigned int vi = 0;
            // add all new vertices connected to the current one via a triangle to the mesh
            while (vi < currentMeshVertices.Count()) {
                // iterate over all triangles connected to the current face
                for (unsigned int ti = 0; ti < facesPerVertex[currentMeshVertices[vi]].Count(); ti++) {
                    unsigned int triaIdx = facesPerVertex[currentMeshVertices[vi]][ti];
                    // get vertex indices of the current triangle
                    unsigned int idx0 = this->cavityMesh.GetTriIndexPointerUInt32()[triaIdx * 3 + 0];
                    unsigned int idx1 = this->cavityMesh.GetTriIndexPointerUInt32()[triaIdx * 3 + 1];
                    unsigned int idx2 = this->cavityMesh.GetTriIndexPointerUInt32()[triaIdx * 3 + 2];
                    // add all vertices that have not yet been processed to the list of connected triangles
                    if (vertexMeshId[idx0] < 0) {
                        currentMeshVertices.Add(idx0);
                        vertexMeshId[idx0] = currentMeshId;
                    }
                    if (vertexMeshId[idx1] < 0) {
                        currentMeshVertices.Add(idx1);
                        vertexMeshId[idx1] = currentMeshId;
                    }
                    if (vertexMeshId[idx2] < 0) {
                        currentMeshVertices.Add(idx2);
                        vertexMeshId[idx2] = currentMeshId;
                    }
                }

                // go to the next vertex in the list of vertices belonging to the current mesh
                vi++;

            }

            if (currentMeshVertices.Count() > 0) {
                // temporary list of tringle indices
                std::vector<unsigned int> tmpTriaIndices;
                // copy all triangle indices of all vertices to the temporary list
                for (unsigned int i = 0; i < currentMeshVertices.Count(); i++) {
                    // iterate over all triangles connected to the current face
                    for (unsigned int ti = 0; ti < facesPerVertex[currentMeshVertices[i]].Count(); ti++) {
                        tmpTriaIndices.push_back(facesPerVertex[currentMeshVertices[i]][ti]);
                    }
                }
                // sort triangle indices
                std::sort(tmpTriaIndices.begin(), tmpTriaIndices.end());
                // remove duplicate triangle indices
                auto end = std::unique(tmpTriaIndices.begin(), tmpTriaIndices.end());
                // copy triangles (vertex indices) to new list
                auto triaCnt = end - tmpTriaIndices.begin();
                unsigned int *tmpTrias = new unsigned int[triaCnt * 3];
                for (unsigned int i = 0; i < triaCnt; i++) {
                    unsigned int triaIdx = tmpTriaIndices[i];
                    tmpTrias[i * 3 + 0] = this->cavityMesh.GetTriIndexPointerUInt32()[triaIdx * 3 + 0];
                    tmpTrias[i * 3 + 1] = this->cavityMesh.GetTriIndexPointerUInt32()[triaIdx * 3 + 1];
                    tmpTrias[i * 3 + 2] = this->cavityMesh.GetTriIndexPointerUInt32()[triaIdx * 3 + 2];
                }
                this->cavitySubmeshes.Add(this->cavityMesh);
                this->cavitySubmeshes.Last().SetTriangleData(static_cast<uint>(triaCnt), tmpTrias, true);
            }

            // DEBUG COLOR HACK!!
            //auto color = const_cast<unsigned char*>(this->cavityMesh.GetColourPointerByte());
            //float colTab[18] = { 255, 0, 0,   0, 255, 0,   0, 0, 255,    255, 255, 0,    0, 255, 255,    255, 255, 255 };
            //for (unsigned int i = 0; i < currentMeshVertices.Count(); i++) {
            //    int colInd = vertexMeshId[currentMeshVertices[i]] % 6;
            //    color[currentMeshVertices[i] * 3 + 0] = colTab[colInd * 3 + 0];
            //    color[currentMeshVertices[i] * 3 + 1] = colTab[colInd * 3 + 1];
            //    color[currentMeshVertices[i] * 3 + 2] = colTab[colInd * 3 + 2];
            //}
            // END DEBUG COLOR HACK!!
        }
        // --- END find connected trianges forming independent submeshes (i.e., cavities) ---

    } // END only recompute vertex distances if something has changed

    // assign cavity meshes to outgoing call
    //outCall->SetObjects( inInnerCall->Count(), inInnerCall->Objects());
    //outCall->SetObjects(1, &this->cavityMesh);
    outCall->SetObjects(this->cavitySubmeshes.Count(), this->cavitySubmeshes.PeekElements());
    outCall->SetDataHash(this->dataHash);

    return true;
}

/*
 * MSMSCavityFinder::getExtent
 */
bool MSMSCavityFinder::getExtent(Call& call) {
    CallTriMeshData * outCall = dynamic_cast<CallTriMeshData*>(&call);
    if (outCall == nullptr) return false;

    CallTriMeshData * inInnerCall = this->innerMeshInSlot.CallAs<CallTriMeshData>();
    if (inInnerCall == nullptr) return false;

    CallTriMeshData * inOuterCall = this->outerMeshInSlot.CallAs<CallTriMeshData>();
    if (inOuterCall == nullptr) return false;

    inInnerCall->SetFrameID(outCall->FrameID());
    if (!(*inInnerCall)(1)) return false;

    inOuterCall->SetFrameID(outCall->FrameID());
    if (!(*inOuterCall)(1)) return false;

    outCall->SetDataHash(inInnerCall->DataHash()); // +this->hashOffset);
    outCall->SetFrameCount(inInnerCall->FrameCount());
    outCall->SetExtent(inInnerCall->FrameCount(), inInnerCall->AccessBoundingBoxes());

    return true;
}
