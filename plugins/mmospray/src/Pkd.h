/*
 * Modified by MegaMol Dev Team
 * Based on project "ospray-module-pkd" files "PartiKD.h" and "PartiKD.cpp" (Apache License 2.0)
 */

#pragma once

#include "datatools/AbstractParticleManipulator.h"
#include "geometry_calls/MultiParticleDataCall.h"
#include "mmcore/CallerSlot.h"
#include "rkcommon/math/box.h"
#include "rkcommon/math/vec.h"
#include <map>


#include "pkd/ParticleModel.h"

namespace megamol::ospray {

static __forceinline size_t leftChildOf(const size_t nodeID) {
    return 2 * nodeID + 1;
}
static __forceinline size_t rightChildOf(const size_t nodeID) {
    return 2 * nodeID + 2;
}
static __forceinline size_t parentOf(const size_t nodeID) {
    return (nodeID - 1) / 2;
}


class PkdBuilder : public megamol::datatools::AbstractParticleManipulator {
public:
    static const char* ClassName() {
        return "PkdBuilder";
    }
    static const char* Description() {
        return "Converts MMPLD files to Pkd sorted MMPLD files.";
    }
    static bool IsAvailable() {
        return true;
    }

    PkdBuilder();
    ~PkdBuilder() override;


protected:
    bool manipulateData(geocalls::MultiParticleDataCall& outData, geocalls::MultiParticleDataCall& inData) override;

private:
    size_t inDataHash;
    size_t outDataHash;
    unsigned int frameID;
    unsigned int vertexLength;

    //std::shared_ptr<ParticleModel> model;
    std::vector<ParticleModel> models;
};

struct Pkd {
    ParticleModel* model;

    size_t numParticles;
    size_t numInnerNodes;
    size_t numLevels;

    __forceinline size_t isInnerNode(const size_t nodeID) const {
        return nodeID < numInnerNodes;
    }
    __forceinline size_t isLeafNode(const size_t nodeID) const {
        return nodeID >= numInnerNodes;
    }
    __forceinline size_t isValidNode(const size_t nodeID) const {
        return nodeID < numParticles;
    }
    __forceinline size_t numInnerNodesOf(const size_t N) const {
        return N / 2;
    }
    __forceinline bool hasLeftChild(const size_t nodeID) const {
        return isValidNode(leftChildOf(nodeID));
    }
    __forceinline bool hasRightChild(const size_t nodeID) const {
        return isValidNode(rightChildOf(nodeID));
    }
    __forceinline static size_t isValidNode(const size_t nodeID, const size_t numParticles) {
        return nodeID < numParticles;
    }
    __forceinline float pos(const size_t nodeID, const size_t dim) const {
        return model->position[nodeID][dim];
    }

    //! helper function for building - swap two particles in the model
    inline void swap(const size_t a, const size_t b) const;

    // save the given particle's split dimension
    void setDim(size_t ID, int dim) const;
    inline size_t maxDim(const rkcommon::math::vec3f& v) const;

    //! build particle tree over given model. WILL REORDER THE MODEL'S ELEMENTS
    void build();

    void buildRec(const size_t nodeID, const rkcommon::math::box3f& bounds, const size_t depth) const;
};


struct SubtreeIterator {
    size_t curInLevel;
    size_t maxInLevel;
    size_t current;

    __forceinline SubtreeIterator(size_t root) : curInLevel(0), maxInLevel(1), current(root) {}

    __forceinline operator size_t() const {
        return current;
    }

    __forceinline void operator++() {
        ++current;
        ++curInLevel;
        if (curInLevel == maxInLevel) {
            current = leftChildOf(current - maxInLevel);
            maxInLevel += maxInLevel;
            curInLevel = 0;
        }
    }

    __forceinline SubtreeIterator& operator=(const SubtreeIterator& other) {
        curInLevel = other.curInLevel;
        maxInLevel = other.maxInLevel;
        current = other.current;
        return *this;
    }
};

struct PKDBuildJob {
    const Pkd* const pkd;
    const size_t nodeID;
    const rkcommon::math::box3f bounds;
    const size_t depth;
    __forceinline PKDBuildJob(const Pkd* pkd, size_t nodeID, rkcommon::math::box3f bounds, size_t depth)
            : pkd(pkd)
            , nodeID(nodeID)
            , bounds(bounds)
            , depth(depth){};
};

static void pkdBuildThread(void* arg) {
    PKDBuildJob* job = (PKDBuildJob*)arg;
    job->pkd->buildRec(job->nodeID, job->bounds, job->depth);
    delete job;
}

} // namespace megamol::ospray
