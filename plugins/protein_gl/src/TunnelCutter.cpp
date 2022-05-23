/*
 * TunnelCutter.h
 * Copyright (C) 2006-2017 by MegaMol Team
 * Alle Rechte vorbehalten.
 */
#include "TunnelCutter.h"

#include "mmcore/param/BoolParam.h"
#include "mmcore/param/IntParam.h"

#include "geometry_calls_gl/CallTriMeshDataGL.h"
#include "protein_calls/BindingSiteCall.h"
#include "protein_calls/MolecularDataCall.h"
#include "protein_calls/TunnelResidueDataCall.h"
#include <cfloat>
#include <chrono>
#include <climits>
#include <iostream>
#include <iterator>
#include <set>

using namespace megamol;
using namespace megamol::core;
using namespace megamol::geocalls_gl;
using namespace megamol::protein_gl;
using namespace megamol::protein_calls;

/*
 * TunnelCutter::TunnelCutter
 */
TunnelCutter::TunnelCutter(void)
        : Module()
        , meshInSlot("dataIn", "Receives the input mesh")
        , cavityMeshInSlot("cavityMeshIn", "Receives teh input cavity mesh")
        , cutMeshOutSlot("getData", "Returns the mesh data of the wanted area")
        , tunnelInSlot("tunnelIn", "Receives the input tunnel data")
        , moleculeInSlot("molIn", "Receives the input molecular data")
        , bindingSiteInSlot("bsIn", "Receives the input binding site data")
        , growSizeParam("growSize", "The number of steps for the region growing")
        , isActiveParam(
              "isActive", "Activates and deactivates the cutting performed by this Module. CURRENTLY NOT IN USE")
        , tunnelIdParam(
              "tunnelID", "The id of the used tunnel. If no such tunnel is present, the first available one is used") {

    // Callee slot
    this->cutMeshOutSlot.SetCallback(
        CallTriMeshDataGL::ClassName(), CallTriMeshDataGL::FunctionName(0), &TunnelCutter::getData);
    this->cutMeshOutSlot.SetCallback(
        CallTriMeshDataGL::ClassName(), CallTriMeshDataGL::FunctionName(1), &TunnelCutter::getExtent);
    this->MakeSlotAvailable(&this->cutMeshOutSlot);

    // Caller slots
    this->meshInSlot.SetCompatibleCall<CallTriMeshDataGLDescription>();
    this->MakeSlotAvailable(&this->meshInSlot);

    this->cavityMeshInSlot.SetCompatibleCall<CallTriMeshDataGLDescription>();
    this->MakeSlotAvailable(&this->cavityMeshInSlot);

    this->tunnelInSlot.SetCompatibleCall<TunnelResidueDataCallDescription>();
    this->MakeSlotAvailable(&this->tunnelInSlot);

    this->moleculeInSlot.SetCompatibleCall<MolecularDataCallDescription>();
    this->MakeSlotAvailable(&this->moleculeInSlot);

    this->bindingSiteInSlot.SetCompatibleCall<BindingSiteCallDescription>();
    this->MakeSlotAvailable(&this->bindingSiteInSlot);

    // parameters
    this->growSizeParam.SetParameter(new param::IntParam(0, 0, 100));
    this->MakeSlotAvailable(&this->growSizeParam);

    this->isActiveParam.SetParameter(new param::BoolParam(true));
    this->MakeSlotAvailable(&this->isActiveParam);

    this->tunnelIdParam.SetParameter(new param::IntParam(0, 0, 1000));
    this->MakeSlotAvailable(&this->tunnelIdParam);

    // other variables
    this->lastDataHash = 0;
    this->lastCavityDataHash = 0;
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
void TunnelCutter::release(void) {}

/*
 * TunnelCutter::getData
 */
bool TunnelCutter::getData(Call& call) {
    CallTriMeshDataGL* outCall = dynamic_cast<CallTriMeshDataGL*>(&call);
    if (outCall == nullptr)
        return false;

    CallTriMeshDataGL* inCall = this->meshInSlot.CallAs<CallTriMeshDataGL>();
    if (inCall == nullptr)
        return false;

    CallTriMeshDataGL* inCav = this->cavityMeshInSlot.CallAs<CallTriMeshDataGL>();
    if (inCav == nullptr)
        return false;

    TunnelResidueDataCall* tc = this->tunnelInSlot.CallAs<TunnelResidueDataCall>();
    if (tc == nullptr)
        return false;

    MolecularDataCall* mdc = this->moleculeInSlot.CallAs<MolecularDataCall>();
    if (mdc == nullptr)
        return false;

    BindingSiteCall* bsc = this->bindingSiteInSlot.CallAs<BindingSiteCall>();
    if (bsc == nullptr)
        return false;

    inCall->SetFrameID(outCall->FrameID());
    inCav->SetFrameID(outCall->FrameID());
    tc->SetFrameID(outCall->FrameID());
    mdc->SetFrameID(outCall->FrameID());

    if (!(*inCall)(0))
        return false;
    if (!(*inCav)(0))
        return false;
    if (!(*tc)(0))
        return false;
    if (!(*mdc)(0))
        return false;
    if (!(*bsc)(0))
        return false;

    if (this->dirt) {
        if (this->isActiveParam.Param<param::BoolParam>()->Value()) {
#ifdef SOMBRERO_TIMING
            auto timebegin = std::chrono::steady_clock::now();
#endif
            cutMeshEqually(inCall, inCav, tc, mdc, bsc);
#ifdef SOMBRERO_TIMING
            auto timeend = std::chrono::steady_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(timeend - timebegin);
            std::cout << "***********Tunnel Cutting took " << elapsed.count() << " ms" << std::endl;
#endif
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
    CallTriMeshDataGL* outCall = dynamic_cast<CallTriMeshDataGL*>(&call);
    if (outCall == nullptr)
        return false;

    CallTriMeshDataGL* inCall = this->meshInSlot.CallAs<CallTriMeshDataGL>();
    if (inCall == nullptr)
        return false;

    CallTriMeshDataGL* inCav = this->cavityMeshInSlot.CallAs<CallTriMeshDataGL>();
    if (inCav == nullptr)
        return false;

    TunnelResidueDataCall* tc = this->tunnelInSlot.CallAs<TunnelResidueDataCall>();
    if (tc == nullptr)
        return false;

    MolecularDataCall* mdc = this->moleculeInSlot.CallAs<MolecularDataCall>();
    if (mdc == nullptr)
        return false;

    inCall->SetFrameID(outCall->FrameID());
    inCav->SetFrameID(outCall->FrameID());
    tc->SetFrameID(outCall->FrameID());
    mdc->SetFrameID(outCall->FrameID());

    if (!(*inCall)(1))
        return false;
    if (!(*inCav)(1))
        return false;
    if (!(*tc)(1))
        return false;
    if (!(*mdc)(1))
        return false;

    if (this->growSizeParam.IsDirty() || this->isActiveParam.IsDirty() || this->tunnelIdParam.IsDirty()) {
        this->hashOffset++;
        this->growSizeParam.ResetDirty();
        this->isActiveParam.ResetDirty();
        this->tunnelIdParam.ResetDirty();
        this->dirt = true;
    }

    if (inCall->DataHash() != this->lastDataHash) {
        this->dirt = true;
        this->lastDataHash = inCall->DataHash();
    }

    if (inCav->DataHash() != this->lastCavityDataHash) {
        this->dirt = true;
        this->lastCavityDataHash = inCav->DataHash();
    }

    outCall->SetDataHash(inCall->DataHash() + inCav->DataHash() + this->hashOffset);
    outCall->SetFrameCount(inCall->FrameCount());
    outCall->SetExtent(inCall->FrameCount(), inCall->AccessBoundingBoxes());

    return true;
}

/*
 * TunnelCutter::cutMeshEqually
 */
bool TunnelCutter::cutMeshEqually(CallTriMeshDataGL* meshCall, CallTriMeshDataGL* cavityMeshCall,
    TunnelResidueDataCall* tunnelCall, MolecularDataCall* molCall, BindingSiteCall* bsCall) {

    if (meshCall->Count() > 0 && cavityMeshCall->Count() > 0) {
        if (meshCall->Objects()[0].GetVertexCount() != cavityMeshCall->Objects()[0].GetVertexCount()) {
            megamol::core::utility::log::Log::DefaultLog.WriteError(
                "The vertex counts of the protein mesh and the cavity mesh do not match");
            return false;
        }
    }

    if (meshCall->Count() == 0 || cavityMeshCall->Count() == 0) {
        return true;
    }

    // generate set of allowed residue indices
    std::set<int> allowedSet;
    auto tid = static_cast<uint32_t>(this->tunnelIdParam.Param<param::IntParam>()->Value());
    if (tunnelCall->getTunnelNumber() == 0) {
        megamol::core::utility::log::Log::DefaultLog.WriteError("No tunnel descriptions found");
        return false;
    }
    if (static_cast<int>(tid) >= tunnelCall->getTunnelNumber()) {
        tid = 0;
        megamol::core::utility::log::Log::DefaultLog.WriteWarn("The given tunnel id was too large, using id 0 instead");
    }

    for (int j = 0; j < tunnelCall->getTunnelDescriptions()[tid].atomIdentifiers.size(); j++) {
        allowedSet.insert(tunnelCall->getTunnelDescriptions()[tid].atomIdentifiers[j].first);
    }
    std::set<uint32_t> allowedVerticesSet;

    int i = 0;

    uint32_t vertCount = meshCall->Objects()[i].GetVertexCount();
    uint32_t triCount = meshCall->Objects()[i].GetTriCount();

    this->meshVector.clear();
    this->meshVector.resize(1);
    this->vertexKeepFlags.clear();
    this->vertexKeepFlags.resize(vertCount);

    this->vertices.clear();
    this->normals.clear();
    this->colors.clear();
    this->attributes.clear();
    this->levelAttributes.clear();
    this->bindingDistanceAttributes.clear();
    this->faces.clear();
    this->vertices.resize(vertCount * 3);
    this->normals.resize(vertCount * 3);
    this->colors.resize(vertCount * 3);
    this->attributes.resize(vertCount);
    this->levelAttributes.resize(vertCount);
    this->bindingDistanceAttributes.resize(vertCount);
    this->faces.resize(triCount * 3);

    // check for the index of the atomIdx attribute
    auto atCnt = meshCall->Objects()[0].GetVertexAttribCount();
    unsigned int attIdx;
    bool found = false;
    if (atCnt != 0) {
        for (attIdx = 0; attIdx < atCnt; attIdx++) {
            if (meshCall->Objects()[0].GetVertexAttribDataType(attIdx) ==
                CallTriMeshDataGL::Mesh::DataType::DT_UINT32) {
                found = true;
                break;
            }
        }
    }

    if (!found) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "The %i th object had no atom index attribute and can therefore not be processed", 0);
        return false;
    }

    /*
     * first step: compute which vertices to keep
     */
    this->vertexKeepFlags.resize(vertCount);
    auto atomIndices = meshCall->Objects()[i].GetVertexAttribPointerUInt32(attIdx);

    int keptVertices = 0;
    for (int j = 0; j < static_cast<int>(vertCount); j++) {
        // the + 1 for the atom index reverses a -1 done in the MSMSMeshloader
        if (allowedSet.count(static_cast<int>(atomIndices[j]) + 1) > 0) {
            this->vertexKeepFlags[j] = true;
            allowedVerticesSet.insert(j);
            keptVertices++;
        } else {
            this->vertexKeepFlags[j] = false;
        }
    }

    /*
     * step 1 a: calculate the closest mesh
     */
    uint32_t maxCount = 0;
    uint32_t maxIndex = UINT_MAX;
    std::vector<uint32_t> localvector;
    std::set<uint32_t> localindices;
    std::set<uint32_t> foundindices;
    for (uint32_t j = 0; j < cavityMeshCall->Count(); j++) {
        const auto& mesh = cavityMeshCall->Objects()[j];
        if (mesh.GetVertexCount() != vertCount) {
            megamol::core::utility::log::Log::DefaultLog.WriteError("Vertex count mismatch");
            return false;
        }
        if (mesh.GetTriDataType() != mesh.DT_UINT32) {
            megamol::core::utility::log::Log::DefaultLog.WriteError("Triangle data type mismatch");
            return false;
        }

        localvector.clear();
        localindices.clear();
        for (uint32_t k = 0; k < mesh.GetTriCount() * 3; k++) {
            localindices.insert(mesh.GetTriIndexPointerUInt32()[k]);
        }
        std::set_intersection(localindices.begin(), localindices.end(), allowedVerticesSet.begin(),
            allowedVerticesSet.end(), std::back_inserter(localvector));
        if (localvector.size() > static_cast<size_t>(maxCount)) {
            maxCount = static_cast<uint32_t>(localvector.size());
            maxIndex = j;
            foundindices = localindices;
        }
    }

    if (maxCount == 0) {
        megamol::core::utility::log::Log::DefaultLog.WriteError("No matching mesh found");
        return false;
    }

    for (auto v : foundindices) {
        if (!this->vertexKeepFlags[v]) {
            this->vertexKeepFlags[v] = true;
            allowedVerticesSet.insert(v);
            keptVertices++;
        }
    }

    /*
     * second step: region growing
     */

    // init distance vector
    std::vector<unsigned int> vertexDistances(vertCount, UINT_MAX);
    for (int j = 0; j < static_cast<int>(vertCount); j++) {
        if (foundindices.count(j) > 0) {
            vertexDistances[j] = 0;
        }
    }

    // build search datastructures
    std::vector<std::pair<unsigned int, unsigned int>> edgesForward;
    std::vector<std::pair<unsigned int, unsigned int>> edgesReverse;
    for (int j = 0; j < static_cast<int>(triCount); j++) {
        unsigned int vert1 = meshCall->Objects()[0].GetTriIndexPointerUInt32()[j * 3 + 0];
        unsigned int vert2 = meshCall->Objects()[0].GetTriIndexPointerUInt32()[j * 3 + 1];
        unsigned int vert3 = meshCall->Objects()[0].GetTriIndexPointerUInt32()[j * 3 + 2];

        edgesForward.push_back(std::pair<unsigned int, unsigned int>(vert1, vert2));
        edgesForward.push_back(std::pair<unsigned int, unsigned int>(vert2, vert3));
        edgesForward.push_back(std::pair<unsigned int, unsigned int>(vert1, vert3));
        edgesReverse.push_back(std::pair<unsigned int, unsigned int>(vert2, vert1));
        edgesReverse.push_back(std::pair<unsigned int, unsigned int>(vert3, vert2));
        edgesReverse.push_back(std::pair<unsigned int, unsigned int>(vert3, vert1));
    }
    // sort the search structures
    std::sort(edgesForward.begin(), edgesForward.end(),
        [](const std::pair<unsigned int, unsigned int>& left, const std::pair<unsigned int, unsigned int>& right) {
            return left.first < right.first;
        });
    std::sort(edgesReverse.begin(), edgesReverse.end(),
        [](const std::pair<unsigned int, unsigned int>& left, const std::pair<unsigned int, unsigned int>& right) {
            return left.first < right.first;
        });

    // we reuse the allowedset and switch it with the newset each step
    std::set<unsigned int> newset;
    for (int s = 0; s < this->growSizeParam.Param<param::IntParam>()->Value(); s++) {
        newset.clear();
        // for each currently allowed vertex
        for (auto element : allowedVerticesSet) {
            // search for the start indices in both edge lists
            auto forward = std::lower_bound(edgesForward.begin(), edgesForward.end(), element,
                [](std::pair<unsigned int, unsigned int>& x, unsigned int val) { return x.first < val; });
            auto reverse = std::lower_bound(edgesReverse.begin(), edgesReverse.end(), element,
                [](std::pair<unsigned int, unsigned int>& x, unsigned int val) { return x.first < val; });

            // go through all forward edges starting with the vertex
            while (forward != edgesForward.end() && (*forward).first == element) {
                auto val = (*forward).second;
                // check whether the endpoint val is not yet in our sets
                if (vertexDistances[val] > static_cast<unsigned int>(s + 1)) {
                    if (!this->vertexKeepFlags[val]) {
                        // if it is not, assign the distance
                        this->vertexKeepFlags[val] = true;
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
                    if (!this->vertexKeepFlags[val]) {
                        // if it is not, assign the distance
                        this->vertexKeepFlags[val] = true;
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
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "There are not binding sites provided. No further computation is possible!");
        return false;
    }

    auto bspos = protein_calls::EstimateBindingSitePosition(molCall, bsCall);

    // search for the vertex closest to the given position
    float minDist = FLT_MAX;
    unsigned int minIndex = UINT_MAX;
    for (unsigned int j = 0; j < vertCount; j++) {
        if (this->vertexKeepFlags[j]) {
            vislib::math::Vector<float, 3> vertPos =
                vislib::math::Vector<float, 3>(&meshCall->Objects()[0].GetVertexPointerFloat()[j * 3]);
            vislib::math::Vector<float, 3> distVec = bspos - vertPos;
            if (distVec.Length() < minDist) {
                minDist = distVec.Length();
                minIndex = j;
            }
        }
    }

    // perform a region growing starting from minIndex
    // reusing the previously performed data structures.
    std::vector<bool> keepVector(this->vertexKeepFlags.size(), false);
    keepVector[minIndex] = true;
    allowedVerticesSet.clear();
    allowedVerticesSet.insert(minIndex);
    keptVertices = 1; // reset the value if we want to have just 1 connected component

    while (!allowedVerticesSet.empty()) {
        newset.clear();
        // for each currently allowed vertex
        for (auto element : allowedVerticesSet) {
            // search for the start indices in both edge lists
            auto forward = std::lower_bound(edgesForward.begin(), edgesForward.end(), element,
                [](std::pair<unsigned int, unsigned int>& x, unsigned int val) { return x.first < val; });
            auto reverse = std::lower_bound(edgesReverse.begin(), edgesReverse.end(), element,
                [](std::pair<unsigned int, unsigned int>& x, unsigned int val) { return x.first < val; });

            // go through all forward edges starting with the vertex
            while (forward != edgesForward.end() && (*forward).first == element) {
                auto val = (*forward).second;
                // check whether the endpoint val is not yet in our sets
                if (!keepVector[val] && this->vertexKeepFlags[val]) {
                    // if it is not, assign the distance
                    keepVector[val] = true;
                    newset.insert(val);
                    keptVertices++;
                }
                forward++;
            }

            // do the same thing for all reverse edges
            while (reverse != edgesReverse.end() && (*reverse).first == element) {
                auto val = (*reverse).second;
                // check whether the endpoint val is not yet in our sets
                if (!keepVector[val] && this->vertexKeepFlags[val]) {
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

    this->vertexKeepFlags = keepVector; // overwrite the vector

#if 1
    /*
     * intermediate step: recalculate vertex distances
     *
     * This step is necessary becaues the region growing may close holes which then need a distance of 0 assigned
     */
    std::vector<bool> isCandidate(vertexDistances.size(), false);

    bool changed;
    do {
        changed = false;
        std::fill(isCandidate.begin(), isCandidate.end(), false);

        // set the candidate field
        for (size_t i = 0; i < this->vertexKeepFlags.size(); i++) {
            if (!this->vertexKeepFlags[i])
                continue; // skip all non-kept vertices
            if (vertexDistances[i] == 0)
                continue; // skip all vertices with a level that is already 0

            auto ownlvl = vertexDistances[i];

            // check the levels of all neighbors
            auto forward = std::lower_bound(edgesForward.begin(), edgesForward.end(), static_cast<uint32_t>(i),
                [](std::pair<unsigned int, unsigned int>& x, unsigned int val) { return x.first < val; });
            auto reverse = std::lower_bound(edgesReverse.begin(), edgesReverse.end(), static_cast<uint32_t>(i),
                [](std::pair<unsigned int, unsigned int>& x, unsigned int val) { return x.first < val; });

            bool hasLargerNeighbor = false;

            // go through all forward edges starting with the vertex
            while (forward != edgesForward.end() && (*forward).first == i && !hasLargerNeighbor) {
                auto tgt = (*forward).second;
                auto tlevel = vertexDistances[tgt];
                if ((tlevel > ownlvl) || !this->vertexKeepFlags[tgt]) {
                    hasLargerNeighbor = true;
                }
                forward++;
            }

            // do the same thing for all reverse edges
            while (reverse != edgesReverse.end() && (*reverse).first == i && !hasLargerNeighbor) {
                auto tgt = (*reverse).second;
                auto tlevel = vertexDistances[tgt];
                if ((tlevel > ownlvl) || !this->vertexKeepFlags[tgt]) {
                    hasLargerNeighbor = true;
                }
                reverse++;
            }

            isCandidate[i] = !hasLargerNeighbor;
        }

        for (size_t i = 0; i < this->vertexKeepFlags.size(); i++) {
            if (!this->vertexKeepFlags[i])
                continue; // skip all non-kept vertices
            if (vertexDistances[i] == 0)
                continue; // skip all vertices with a level that is already 0

            auto ownlvl = vertexDistances[i];

            // check the levels of all neighbors
            auto forward = std::lower_bound(edgesForward.begin(), edgesForward.end(), static_cast<uint32_t>(i),
                [](std::pair<unsigned int, unsigned int>& x, unsigned int val) { return x.first < val; });
            auto reverse = std::lower_bound(edgesReverse.begin(), edgesReverse.end(), static_cast<uint32_t>(i),
                [](std::pair<unsigned int, unsigned int>& x, unsigned int val) { return x.first < val; });

            bool setZero = true;

            // go through all forward edges starting with the vertex
            while (forward != edgesForward.end() && (*forward).first == i) {
                auto tgt = (*forward).second;
                auto tlevel = vertexDistances[tgt];
                if ((tlevel > ownlvl) || (tlevel == ownlvl && !isCandidate[tgt])) {
                    setZero = false;
                }
                forward++;
            }

            // do the same thing for all reverse edges
            while (reverse != edgesReverse.end() && (*reverse).first == i) {
                auto tgt = (*reverse).second;
                auto tlevel = vertexDistances[tgt];
                if ((tlevel > ownlvl) || (tlevel == ownlvl && !isCandidate[tgt])) {
                    setZero = false;
                }
                reverse++;
            }

            if (setZero) {
                vertexDistances[i] = 0;
                changed = true;
            }
        }

    } while (changed);
#endif

    /*
     * fourth step: mesh cutting
     */
    this->vertices.resize(keptVertices * 3);
    this->normals.resize(keptVertices * 3);
    this->colors.resize(keptVertices * 3); // TODO accept other color configurations
    this->attributes.resize(keptVertices);
    this->levelAttributes.resize(keptVertices);
    this->bindingDistanceAttributes.clear();
    this->bindingDistanceAttributes.resize(keptVertices, UINT_MAX);

    // compute vertex mapping and fill vertex vectors
    std::vector<unsigned int> vertIndexMap(vertCount, UINT_MAX); // mapping of old vertex indices to new ones
    int help = 0;
    for (int j = 0; j < static_cast<int>(vertCount); j++) {
        if (this->vertexKeepFlags[j]) {
            // mapping
            vertIndexMap[j] = help;

            // vector fill
            this->vertices[help * 3 + 0] = meshCall->Objects()[0].GetVertexPointerFloat()[j * 3 + 0];
            this->vertices[help * 3 + 1] = meshCall->Objects()[0].GetVertexPointerFloat()[j * 3 + 1];
            this->vertices[help * 3 + 2] = meshCall->Objects()[0].GetVertexPointerFloat()[j * 3 + 2];

            this->normals[help * 3 + 0] = meshCall->Objects()[0].GetNormalPointerFloat()[j * 3 + 0];
            this->normals[help * 3 + 1] = meshCall->Objects()[0].GetNormalPointerFloat()[j * 3 + 1];
            this->normals[help * 3 + 2] = meshCall->Objects()[0].GetNormalPointerFloat()[j * 3 + 2];

            this->colors[help * 3 + 0] = meshCall->Objects()[0].GetColourPointerByte()[j * 3 + 0];
            this->colors[help * 3 + 1] = meshCall->Objects()[0].GetColourPointerByte()[j * 3 + 1];
            this->colors[help * 3 + 2] = meshCall->Objects()[0].GetColourPointerByte()[j * 3 + 2];

#if 0 // DEBUG
            if (j == minIndex) {
                this->colors[help * 3 + 0] = 255;
                this->colors[help * 3 + 1] = 0;
                this->colors[help * 3 + 2] = 0;
            }
#endif
#if 0
            if (isCandidate[j]) {
                this->colors[help * 3 + 0] = 0;
                this->colors[help * 3 + 1] = 255;
                this->colors[help * 3 + 2] = 0;
            }
#endif

            this->attributes[help] = meshCall->Objects()[0].GetVertexAttribPointerUInt32(attIdx)[j];
            this->levelAttributes[help] = vertexDistances[j];

            help++;
        }
    }

    this->faces.clear();
    // compute the triangles
    for (int j = 0; j < static_cast<int>(triCount); j++) {
        unsigned int vert1 = meshCall->Objects()[0].GetTriIndexPointerUInt32()[j * 3 + 0];
        unsigned int vert2 = meshCall->Objects()[0].GetTriIndexPointerUInt32()[j * 3 + 1];
        unsigned int vert3 = meshCall->Objects()[0].GetTriIndexPointerUInt32()[j * 3 + 2];

        // to keep a triangle, all vertices have to be kept, since we have no correct mapping for the other vertices
        if (vertexKeepFlags[vert1] && vertexKeepFlags[vert2] && vertexKeepFlags[vert3]) {
            this->faces.push_back(vertIndexMap[vert1]);
            this->faces.push_back(vertIndexMap[vert2]);
            this->faces.push_back(vertIndexMap[vert3]);
        }
    }

    /*
     * fifth step: vertex level computation
     */
    allowedVerticesSet.clear();
    allowedVerticesSet.insert(minIndex);
    this->bindingDistanceAttributes[vertIndexMap[minIndex]] = 0;
    while (!allowedVerticesSet.empty()) {
        newset.clear();
        // for each currently allowed vertex
        for (auto element : allowedVerticesSet) {
            // search for the start indices in both edge lists
            auto forward = std::lower_bound(edgesForward.begin(), edgesForward.end(), element,
                [](std::pair<unsigned int, unsigned int>& x, unsigned int val) { return x.first < val; });
            auto reverse = std::lower_bound(edgesReverse.begin(), edgesReverse.end(), element,
                [](std::pair<unsigned int, unsigned int>& x, unsigned int val) { return x.first < val; });

            // go through all forward edges starting with the vertex
            while (forward != edgesForward.end() && (*forward).first == element) {
                auto val = (*forward).second;
                // check whether the endpoint is valid
                if (this->vertexKeepFlags[val]) {
                    if (this->bindingDistanceAttributes[vertIndexMap[val]] >
                        this->bindingDistanceAttributes[vertIndexMap[element]] + 1) {
                        this->bindingDistanceAttributes[vertIndexMap[val]] =
                            this->bindingDistanceAttributes[vertIndexMap[element]] + 1;
                        newset.insert(val);
                    }
                }
                forward++;
            }

            // do the same thing for all reverse edges
            while (reverse != edgesReverse.end() && (*reverse).first == element) {
                auto val = (*reverse).second;
                // check whether the endpoint is valid
                if (this->vertexKeepFlags[val]) {
                    if (this->bindingDistanceAttributes[vertIndexMap[val]] >
                        this->bindingDistanceAttributes[vertIndexMap[element]] + 1) {
                        this->bindingDistanceAttributes[vertIndexMap[val]] =
                            this->bindingDistanceAttributes[vertIndexMap[element]] + 1;
                        newset.insert(val);
                    }
                }
                reverse++;
            }
        }
        allowedVerticesSet = newset;
    }

#if 0
    uint minVal = UINT_MAX;
    uint maxVal = 0;

    std::vector<uint>& bla = this->levelAttributes;

    for (size_t i = 0; i < bla.size(); i++) {
        if (bla[i] > maxVal) maxVal = bla[i];
        if (bla[i] < minVal) minVal = bla[i];
    }
    for (size_t i = 0; i < bla.size(); i++) {
        this->colors[3 * i + 0] = static_cast<unsigned char>(((bla[i] - minVal) * 255.0f) / (maxVal - minVal));
        this->colors[3 * i + 1] = 0; // this->colors[3 * i + 0];
        this->colors[3 * i + 2] = 0; // this->colors[3 * i + 0];
    }
#endif

    /*
     * sixth step: fill the data into the structure
     */
    this->meshVector[0].SetVertexData(static_cast<unsigned int>(this->vertices.size() / 3), this->vertices.data(),
        this->normals.data(), this->colors.data(), NULL, false);
    this->meshVector[0].SetTriangleData(static_cast<unsigned int>(this->faces.size() / 3), this->faces.data(), false);
    this->meshVector[0].SetMaterial(nullptr);
    this->meshVector[0].AddVertexAttribPointer(this->attributes.data());
    this->meshVector[0].AddVertexAttribPointer(this->levelAttributes.data());
    this->meshVector[0].AddVertexAttribPointer(this->bindingDistanceAttributes.data());

    return true;
}
