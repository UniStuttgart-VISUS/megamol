#include "stdafx.h"
#include "HMMPKDBuilder.h"
#include <iostream>
#include <stdint.h>
#include <thread>
#include "mmcore/param/BoolParam.h"
#include "mmcore/param/IntParam.h"

#define POS(idx, dim) pos(idx, dim)

using namespace megamol;


ospray::HMMPKDBuilder::HMMPKDBuilder()
    : megamol::stdplugin::datatools::AbstractParticleManipulator("outData", "inData")
    , inDataHash(0)
    , outDataHash(0)
    , numParticles(0)
    , numInnerNodes(0)
    , numGhostLayersParam("num_ghost_layers", "set the number of ghost layers in the HMMPKD tree") {
    model = std::make_shared<HMMPKDParticleModel>();

    numGhostLayersParam << new megamol::core::param::IntParam(0, 0);
    this->MakeSlotAvailable(&numGhostLayersParam);
}


ospray::HMMPKDBuilder::~HMMPKDBuilder() { Release(); }


bool ospray::HMMPKDBuilder::manipulateData(
    core::moldyn::MultiParticleDataCall& outData, core::moldyn::MultiParticleDataCall& inData) {

    if ((inData.DataHash() != inDataHash) || (inData.FrameID() != frameID) || (inData.DataHash() == 0)) {
        inDataHash = inData.DataHash();
        outDataHash++;
        frameID = inData.FrameID();
    }

    data.clear();

    outData = inData;
    inData.SetUnlocker(nullptr, false);
    outData.SetFrameID(frameID);
    outData.SetDataHash(outDataHash);

    data.resize(inData.GetParticleListCount());

    for (unsigned int i = 0; i < inData.GetParticleListCount(); ++i) {
        auto& parts = inData.AccessParticles(i);
        auto& out = outData.AccessParticles(i);

        //// empty the model
        // if (this->model->IsDoublePrecision()) {
        //    this->model->positiond.clear();
        //} else {
        //    this->model->positionf.clear();
        //}

        // put data the data into the model
        // and build the pkd tree
        this->model->fill(parts);
        this->build();


        data[i].resize(this->model->pkd_particle.size());
        std::copy(this->model->pkd_particle.begin(), this->model->pkd_particle.end(), data[i].begin());
        out.SetVertexData(
            megamol::core::moldyn::SimpleSphericalParticles::VERTDATA_NONE, &this->data[i][0].position, 4);
        out.SetColourData(megamol::core::moldyn::SimpleSphericalParticles::COLDATA_NONE, &this->data[i][0].color, 4);

        auto const lbbox = this->model->GetLocalBBox();
        out.SetBBox(vislib::math::Cuboid<float>(
            lbbox.lower.x, lbbox.lower.y, lbbox.lower.z, lbbox.upper.x, lbbox.upper.y, lbbox.upper.z));
    }

    return true;
}


void ospray::HMMPKDBuilder::setDim(size_t ID, int dim) const {
    auto& particle = this->model->pkd_particle[ID];
    auto& pos = particle.position;

    pos = (pos & ~(3 << 30)) | (dim << 30);
}


void ospray::HMMPKDBuilder::setGhost(size_t ID, bool ghost) const {
    auto& particle = this->model->pkd_particle[ID];
    auto& col = particle.color;

    if (ghost) {
        col = (col & ~(1 << 8)) & ~(1 << 8);
    } else {
        col = (col & ~(1 << 8)) | (1 << 8);
    }
}


inline size_t ospray::HMMPKDBuilder::maxDim(const ospcommon::vec3f& v) const {
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


inline void ospray::HMMPKDBuilder::swapAndSet(const size_t a, const size_t b, ospcommon::box3f const& bounds) const {
    auto& parA = model->pkd_particle[a];
    auto& parB = model->pkd_particle[b];

    setPar(parA, model->position[b], bounds);
    setPar(parB, model->position[a], bounds);

    std::swap(model->position[a], model->position[b]);
}


void ospray::HMMPKDBuilder::setPar(
    HMMPKDParticle_t& pkd_par, ospcommon::vec4d const& par, ospcommon::box3f const& bounds) const {
    auto const* col = reinterpret_cast<uint16_t const*>(&par.w);
    pkd_par.color = (pkd_par.color & ~(255 << 24)) | static_cast<unsigned char>(col[0] / 257);
    pkd_par.color = (pkd_par.color & ~(255 << 16)) | static_cast<unsigned char>(col[1] / 257);
    pkd_par.color = (pkd_par.color & ~(255 << 8)) | static_cast<unsigned char>(col[2] / 257);
    pkd_par.color = (pkd_par.color & ~255) | static_cast<unsigned char>(col[3] / 257);

    auto bSize = bounds.size();

    pkd_par.position =
        (pkd_par.position & ~(1023 << 20)) | static_cast<unsigned int>(((par.x - bounds.lower[0]) / bSize[0]) * 1023.0);
    pkd_par.position =
        (pkd_par.position & ~(1023 << 10)) | static_cast<unsigned int>(((par.y - bounds.lower[1]) / bSize[1]) * 1023.0);
    pkd_par.position =
        (pkd_par.position & ~1023) | static_cast<unsigned int>(((par.z - bounds.lower[2]) / bSize[2]) * 1023.0);
}


void ospray::HMMPKDBuilder::build() {
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

    const ospcommon::box3f& bounds = model->GetLocalBBox();
    std::cout << "#osp:pkd: bounds of model " << bounds << std::endl;
    std::cout << "#osp:pkd: number of input particles " << numParticles << std::endl;
    this->buildRec(0, bounds, 0);
}


void ospray::HMMPKDBuilder::buildRec(const size_t nodeID, const ospcommon::box3f& bounds, const size_t depth) const {
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
        if (POS(lChild, dim) > POS(nodeID, dim)) swapAndSet(nodeID, lChild, bounds);
        // and done
        setDim(nodeID, dim);
        return;
    }

    {

        // we have a left and a right subtree, each of at least 1 node.
        HMMPKDSubtreeIterator l0(leftChildOf(nodeID));
        HMMPKDSubtreeIterator r0(rightChildOf(nodeID));

        HMMPKDSubtreeIterator l((size_t)l0); //(leftChildOf(nodeID));
        HMMPKDSubtreeIterator r((size_t)r0); //(rightChildOf(nodeID));

        // size_t numSwaps = 0, numComps = 0;
        float rootPos = POS(nodeID, dim);
        while (1) {

            while (isValidNode(l, N) && (POS(l, dim) <= rootPos)) ++l;
            while (isValidNode(r, N) && (POS(r, dim) >= rootPos)) ++r;

            if (isValidNode(l, N)) {
                if (isValidNode(r, N)) {
                    // both mis-mathces valid, just swap them and go on
                    swapAndSet(l, r, bounds);
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
                            swapAndSet(l0, l, bounds);
                            ++l0;
                        }
                        ++l;
                    }
                    swapAndSet(nodeID, l0, bounds);
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
                            swapAndSet(r0, r, bounds);
                            ++r0;
                        }
                        ++r;
                    }
                    swapAndSet(nodeID, r0, bounds);
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
        std::thread lThread(&hmmpkdBuildThread, new HMMPKDBuildJob(this, leftChildOf(nodeID), lBounds, depth + 1));
        buildRec(rightChildOf(nodeID), rBounds, depth + 1);
        lThread.join();
    } else {
        buildRec(leftChildOf(nodeID), lBounds, depth + 1);
        buildRec(rightChildOf(nodeID), rBounds, depth + 1);
    }
}
