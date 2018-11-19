#pragma once

#include <map>
#include "mmcore/CallerSlot.h"
#include "mmcore/moldyn/MultiParticleDataCall.h"
#include "mmstd_datatools/AbstractParticleManipulator.h"
#include "ospray/ospcommon/box.h"
#include "ospray/ospcommon/vec.h"

#include "pkd/HMMPKDParticleModel.h"
#include "mmcore/param/ParamSlot.h"

namespace megamol {
namespace ospray {



class HMMPKDBuilder : public megamol::stdplugin::datatools::AbstractParticleManipulator {
public:

    static __forceinline size_t leftChildOf(const size_t nodeID) { return 2 * nodeID + 1; }
    static __forceinline size_t rightChildOf(const size_t nodeID) { return 2 * nodeID + 2; }
    static __forceinline size_t parentOf(const size_t nodeID) { return (nodeID - 1) / 2; }

    static const char* ClassName(void) { return "HMMPKDBuilder"; }
    static const char* Description(void) { return "Converts MMPLD files to MegaMol-specific PkD sorted MMPLD files."; }
    static bool IsAvailable(void) { return true; }

    HMMPKDBuilder();
    virtual ~HMMPKDBuilder();

    void buildRec(const size_t nodeID, const ospcommon::box3f& bounds, const size_t depth) const;


protected:
    virtual bool manipulateData(
        megamol::core::moldyn::MultiParticleDataCall& outData, megamol::core::moldyn::MultiParticleDataCall& inData);

private:
    size_t inDataHash;
    size_t outDataHash;
    unsigned int frameID;
    unsigned int vertexLength;

    std::shared_ptr<HMMPKDParticleModel> model;
    size_t numParticles;
    size_t numInnerNodes;
    size_t numLevels;

    std::vector<std::vector<HMMPKDParticle_t>> data;

    megamol::core::param::ParamSlot numGhostLayersParam;

    __forceinline size_t isInnerNode(const size_t nodeID) const { return nodeID < numInnerNodes; }
    __forceinline size_t isLeafNode(const size_t nodeID) const { return nodeID >= numInnerNodes; }
    __forceinline size_t isValidNode(const size_t nodeID) const { return nodeID < numParticles; }
    __forceinline size_t numInnerNodesOf(const size_t N) const { return N / 2; }
    __forceinline bool hasLeftChild(const size_t nodeID) const { return isValidNode(leftChildOf(nodeID)); }
    __forceinline bool hasRightChild(const size_t nodeID) const { return isValidNode(rightChildOf(nodeID)); }
    __forceinline static size_t isValidNode(const size_t nodeID, const size_t numParticles) {
        return nodeID < numParticles;
    }
    __forceinline float pos(const size_t nodeID, const size_t dim) const { return model->GetCoord(nodeID, dim); }

    //! helper function for building - swap two particles in the model
    inline void swapAndSet(const size_t a, const size_t b, ospcommon::box3f const& bounds) const;

    inline void setPar(HMMPKDParticle_t& pkd_par, ospcommon::vec4d const& par, ospcommon::box3f const& bounds) const;

    // save the given particle's split dimension
    void setDim(size_t ID, int dim) const;
    void setGhost(size_t ID, bool ghost) const;
    inline size_t maxDim(const ospcommon::vec3f& v) const;

    //! build particle tree over given model. WILL REORDER THE MODEL'S ELEMENTS
    void build();
};


struct HMMPKDSubtreeIterator {
    static __forceinline size_t leftChildOf(const size_t nodeID) { return 2 * nodeID + 1; }
    static __forceinline size_t rightChildOf(const size_t nodeID) { return 2 * nodeID + 2; }
    static __forceinline size_t parentOf(const size_t nodeID) { return (nodeID - 1) / 2; }

    size_t curInLevel;
    size_t maxInLevel;
    size_t current;

    __forceinline HMMPKDSubtreeIterator(size_t root) : curInLevel(0), maxInLevel(1), current(root) {}

    __forceinline operator size_t() const { return current; }

    __forceinline void operator++() {
        ++current;
        ++curInLevel;
        if (curInLevel == maxInLevel) {
            current = leftChildOf(current - maxInLevel);
            maxInLevel += maxInLevel;
            curInLevel = 0;
        }
    }

    __forceinline HMMPKDSubtreeIterator& operator=(const HMMPKDSubtreeIterator& other) {
        curInLevel = other.curInLevel;
        maxInLevel = other.maxInLevel;
        current = other.current;
        return *this;
    }
};

struct MMPKDBuildJob {
    const HMMPKDBuilder* const pkd;
    const size_t nodeID;
    const ospcommon::box3f bounds;
    const size_t depth;
    __forceinline MMPKDBuildJob(const HMMPKDBuilder* pkd, size_t nodeID, ospcommon::box3f bounds, size_t depth)
        : pkd(pkd), nodeID(nodeID), bounds(bounds), depth(depth){};
};

static void mmpkdBuildThread(void* arg) {
    MMPKDBuildJob* job = (MMPKDBuildJob*)arg;
    job->pkd->buildRec(job->nodeID, job->bounds, job->depth);
    delete job;
}

} // namespace ospray
} // namespace megamol
