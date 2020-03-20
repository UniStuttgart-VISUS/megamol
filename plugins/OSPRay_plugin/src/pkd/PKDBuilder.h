#pragma once

#include <memory>

#include "pkd/ParticleModel.h"

namespace megamol {
namespace ospray {
namespace pkd {

static inline size_t leftChildOf(const size_t nodeID) { return 2 * nodeID + 1; }
static inline size_t rightChildOf(const size_t nodeID) { return 2 * nodeID + 2; }
static inline size_t parentOf(const size_t nodeID) { return (nodeID - 1) / 2; }
static inline size_t numInnerNodesOf(const size_t N) { return N / 2; }

class PKDBuilder {
public:
private:
    std::shared_ptr<ParticleModel> model;
    size_t numParticles;
    size_t numInnerNodes;
    size_t numLevels;

    size_t isInnerNode(const size_t nodeID) const { return nodeID < numInnerNodes; }
    size_t isLeafNode(const size_t nodeID) const { return nodeID >= numInnerNodes; }
    size_t isValidNode(const size_t nodeID) const { return nodeID < numParticles; }
    bool hasLeftChild(const size_t nodeID) const { return isValidNode(leftChildOf(nodeID)); }
    bool hasRightChild(const size_t nodeID) const { return isValidNode(rightChildOf(nodeID)); }
    static size_t isValidNode(const size_t nodeID, const size_t numParticles) { return nodeID < numParticles; }
    float pos(const size_t nodeID, const size_t dim) const { return model->position[nodeID][dim]; }

    //! helper function for building - swap two particles in the model
    void swap(const size_t a, const size_t b) const;

    // save the given particle's split dimension
    void setDim(size_t ID, int dim) const;
    size_t maxDim(const ospcommon::vec3f& v) const;

    void buildRec(const size_t nodeID, const ospcommon::box3f& bounds, const size_t depth) const;
}; // class PKDBuilder

struct SubtreeIterator {
    size_t curInLevel;
    size_t maxInLevel;
    size_t current;

    SubtreeIterator(size_t root) : curInLevel(0), maxInLevel(1), current(root) {}

    operator size_t() const { return current; }

    void operator++() {
        ++current;
        ++curInLevel;
        if (curInLevel == maxInLevel) {
            current = leftChildOf(current - maxInLevel);
            maxInLevel += maxInLevel;
            curInLevel = 0;
        }
    }

    SubtreeIterator& operator=(const SubtreeIterator& other) {
        curInLevel = other.curInLevel;
        maxInLevel = other.maxInLevel;
        current = other.current;
        return *this;
    }
};

} // namespace pkd
} // namespace ospray
} // namespace megamol