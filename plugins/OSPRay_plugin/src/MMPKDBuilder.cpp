#include "stdafx.h"
#include "MMPKDBuilder.h"
#include <iostream>
#include <stdint.h>
#include <thread>

#define POS(idx, dim) pos(idx, dim)

using namespace megamol;



ospray::MMPKDBuilder::MMPKDBuilder()
    : megamol::stdplugin::datatools::AbstractParticleManipulator("outData", "inData")
    , inDataHash(0)
    , outDataHash(0)
    , numParticles(0)
    , numInnerNodes(0) {
    model = std::make_shared<MMPKDParticleModel>();
}

ospray::MMPKDBuilder::~MMPKDBuilder() { Release(); }

bool ospray::MMPKDBuilder::manipulateData(
    core::moldyn::MultiParticleDataCall& outData, core::moldyn::MultiParticleDataCall& inData) {

    if ((inData.DataHash() != inDataHash) || (inData.FrameID() != frameID) || (inData.DataHash() == 0)) {
        inDataHash = inData.DataHash();
        outDataHash++;
        frameID = inData.FrameID();
    }

    outData = inData;
    inData.SetUnlocker(nullptr, false);
    outData.SetFrameID(frameID);
    outData.SetDataHash(outDataHash);

    for (unsigned int i = 0; i < inData.GetParticleListCount(); i++) {
        auto& parts = inData.AccessParticles(i);
        auto& out = outData.AccessParticles(i);

        // empty the model
        if (this->model->IsDoublePrecision()) {
            this->model->positiond.clear();
        } else {
            this->model->positionf.clear();
        }

        // put data the data into the model
        // and build the pkd tree
        this->model->fill(parts);
        this->build();

        if (this->model->IsDoublePrecision()) {
            out.SetVertexData(
                megamol::core::moldyn::SimpleSphericalParticles::VERTDATA_DOUBLE_XYZ, &this->model->positiond[0].x, 32);
            out.SetColourData(megamol::core::moldyn::SimpleSphericalParticles::COLDATA_USHORT_RGBA, &this->model->positiond[0].w, 32);
        } else {
            out.SetVertexData(
                megamol::core::moldyn::SimpleSphericalParticles::VERTDATA_FLOAT_XYZ, &this->model->positionf[0].x, 16);
            out.SetColourData(megamol::core::moldyn::SimpleSphericalParticles::COLDATA_UINT8_RGBA, &this->model->positionf[0].w, 16);
        }
    }

    return true;
}


void ospray::MMPKDBuilder::setDim(size_t ID, int dim) const {
    if (this->model->IsDoublePrecision()) {
        ospcommon::vec4d& particle = (ospcommon::vec4d&)this->model->positiond[ID];
        int64_t& pxAsInt = (int64_t&)particle.x;
        pxAsInt = (pxAsInt & ~3) | dim;
    } else {
        ospcommon::vec4f& particle = (ospcommon::vec4f&)this->model->positionf[ID];
        int& pxAsInt = (int&)particle.x;
        pxAsInt = (pxAsInt & ~3) | dim;
    }
}

inline size_t ospray::MMPKDBuilder::maxDim(const ospcommon::vec3f& v) const {
    const float maxVal = ospcommon::reduce_max(v);
    if (maxVal == v.x) {
        return 0;
    } else if (maxVal == v.y) {
        return 1;
    } else if (maxVal == v.z) {
        return 2;
    } else {
        assert(false && "Invalid max val index for vec!?");
        return -1;
    }
}


inline void ospray::MMPKDBuilder::swap(const size_t a, const size_t b) const {
    if (this->model->IsDoublePrecision()) {
        std::swap(model->positiond[a], model->positiond[b]);
    } else {
        std::swap(model->positionf[a], model->positionf[b]);
    }
}


void ospray::MMPKDBuilder::build() {
    // PING;
    assert(this->model != NULL);

    numParticles = model->GetNumParticles();
    assert(numParticles <= (1ULL << 31));

    numInnerNodes = numInnerNodesOf(numParticles);

    // determine num levels
    numLevels = 0;
    size_t nodeID = 0;
    while (isValidNode(nodeID)) {
        ++numLevels;
        nodeID = leftChildOf(nodeID);
    }
    // PRINT(numLevels);

    const ospcommon::box3f& bounds = model->getBounds();
    std::cout << "#osp:pkd: bounds of model " << bounds << std::endl;
    std::cout << "#osp:pkd: number of input particles " << numParticles << std::endl;
    this->buildRec(0, bounds, 0);
}


void ospray::MMPKDBuilder::buildRec(const size_t nodeID, const ospcommon::box3f& bounds, const size_t depth) const {
    // if (depth < 4)
    // std::cout << "#osp:pkd: building subtree " << nodeID << std::endl;
    if (!hasLeftChild(nodeID))
        // has no children -> it's a valid kd-tree already :-)
        return;

    // we have at least one child.
    const size_t dim = this->maxDim(bounds.size());
    // if (depth < 4) { PRINT(bounds); printf("depth %ld-> dim %ld\n",depth,dim); }
    const size_t N = numParticles;

    if (!hasRightChild(nodeID)) {
        // no right child, but not a leaf emtpy. must have exactly one
        // child on the left. see if we have to swap, but otherwise
        // nothing to do.
        size_t lChild = leftChildOf(nodeID);
        if (POS(lChild, dim) > POS(nodeID, dim)) swap(nodeID, lChild);
        // and done
        setDim(nodeID, dim);
        return;
    }

    {

        // we have a left and a right subtree, each of at least 1 node.
        SubtreeIterator l0(leftChildOf(nodeID));
        SubtreeIterator r0(rightChildOf(nodeID));

        SubtreeIterator l((size_t)l0); //(leftChildOf(nodeID));
        SubtreeIterator r((size_t)r0); //(rightChildOf(nodeID));

        // size_t numSwaps = 0, numComps = 0;
        float rootPos = POS(nodeID, dim);
        while (1) {

            while (isValidNode(l, N) && (POS(l, dim) <= rootPos)) ++l;
            while (isValidNode(r, N) && (POS(r, dim) >= rootPos)) ++r;

            if (isValidNode(l, N)) {
                if (isValidNode(r, N)) {
                    // both mis-mathces valid, just swap them and go on
                    swap(l, r);
                    ++l;
                    ++r;
                    continue;
                } else {
                    // mis-match on left side, but nothing on right side to swap with: swap with root
                    // --> can't go on on right side any more, but can still compact matches on left
                    l0 = l;
                    ++l;
                    while (isValidNode(l, N)) {
                        if (POS(l, dim) <= rootPos) {
                            swap(l0, l);
                            ++l0;
                        }
                        ++l;
                    }
                    swap(nodeID, l0);
                    ++l0;
                    rootPos = POS(nodeID, dim);

                    l = l0;
                    r = r0;
                    continue;
                }
            } else {
                if (isValidNode(r, N)) {
                    // mis-match on left side, but nothing on right side to swap with: swap with root
                    // --> can't go on on right side any more, but can still compact matches on left
                    r0 = r;
                    ++r;
                    while (isValidNode(r, N)) {
                        if (POS(r, dim) >= rootPos) {
                            swap(r0, r);
                            ++r0;
                        }
                        ++r;
                    }
                    swap(nodeID, r0);
                    ++r0;
                    rootPos = POS(nodeID, dim);

                    l = l0;
                    r = r0;
                    continue;
                } else {
                    // no mis-match on either side ... done.
                    break;
                }
            }
        }
    }

    ospcommon::box3f lBounds = bounds;
    ospcommon::box3f rBounds = bounds;

    setDim(nodeID, dim);

    lBounds.upper[dim] = rBounds.lower[dim] = pos(nodeID, dim);

    if ((numLevels - depth) > 20) {
        std::thread lThread(&pkdBuildThread, new PKDBuildJob(this, leftChildOf(nodeID), lBounds, depth + 1));
        buildRec(rightChildOf(nodeID), rBounds, depth + 1);
        lThread.join();
    } else {
        buildRec(leftChildOf(nodeID), lBounds, depth + 1);
        buildRec(rightChildOf(nodeID), rBounds, depth + 1);
    }
}

