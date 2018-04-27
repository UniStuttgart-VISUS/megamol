#pragma once

#include "mmstd_datatools/AbstractParticleManipulator.h"
#include "mmcore/moldyn/MultiParticleDataCall.h"
#include "mmcore/CallerSlot.h"
#include <map>
#include "ospray/ospcommon/box.h"
#include "ospray/ospcommon/vec.h"

namespace megamol {
namespace ospray {

static __forceinline size_t leftChildOf(const size_t nodeID) { return 2 * nodeID + 1; }
static __forceinline size_t rightChildOf(const size_t nodeID) { return 2 * nodeID + 2; }
static __forceinline size_t parentOf(const size_t nodeID) { return (nodeID - 1) / 2; }

/*! complete input data for a particle model */
struct ParticleModel {


    ParticleModel() : radius(0) {}

    //! a set of attributes, one float per particle (with min/max info and logical name)
    struct Attribute {
        Attribute(const std::string &name)
            : name(name),
            minValue(+std::numeric_limits<float>::infinity()),
            maxValue(-std::numeric_limits<float>::infinity()) {
        };

        std::string        name;
        float              minValue, maxValue;
        std::vector<float> value;
    };
    struct AtomType {
        std::string name;
        ospcommon::vec3f       color;

        AtomType(const std::string &name) : name(name), color(1, 0, 0) {}
    };

    //! list of all declared atom types
    std::vector<AtomType *>           atomType;
    //! mapper that maps an atom type name to the ID in the 'atomType' vector
    std::map<std::string, int32_t> atomTypeByName;

    uint32_t getAtomTypeID(const std::string &name);

    std::vector<ospcommon::vec3f> position;   //!< particle position
    std::vector<int>   type;       //!< 'type' of particle (e.g., the atom type for atomistic models)
    std::vector<Attribute *> attribute;

    void fill(megamol::core::moldyn::SimpleSphericalParticles parts);

    //! get attributeset of given name; create a new one if not yet exists */
    Attribute *getAttribute(const std::string &name);

    //! return if attribute of this name exists 
    bool hasAttribute(const std::string &name);

    //! add one attribute value to set of attributes of given name
    void addAttribute(const std::string &attribName, float attribute);

    //! helper function for parser error recovery: 'clamp' all attributes to largest non-empty attribute
    void cullPartialData();

    //! return world bounding box of all particle *positions* (i.e., particles *ex* radius)
    ospcommon::box3f getBounds() const;

    float radius;  //!< radius to use (0 if not specified)
};


class PkdBuilder: public megamol::stdplugin::datatools::AbstractParticleManipulator {
public:
    static const char *ClassName(void) { return "PkdBuilder"; }
    static const char *Description(void) { return "Converts MMPLD files to Pkd sorted MMPLD files."; }
    static bool IsAvailable(void) { return true; }

    PkdBuilder();
    virtual ~PkdBuilder();

    void buildRec(const size_t nodeID, const ospcommon::box3f &bounds, const size_t depth) const;


protected:

    virtual bool manipulateData(
        megamol::core::moldyn::MultiParticleDataCall& outData,
        megamol::core::moldyn::MultiParticleDataCall& inData);

private:

    size_t inDataHash;
    size_t outDataHash;
    unsigned int frameID;
    unsigned int vertexLength;

    std::shared_ptr<ParticleModel> model;
    size_t numParticles;
    size_t numInnerNodes;
    size_t numLevels;

    __forceinline size_t isInnerNode(const size_t nodeID)   const { return nodeID < numInnerNodes; }
    __forceinline size_t isLeafNode(const size_t nodeID)    const { return nodeID >= numInnerNodes; }
    __forceinline size_t isValidNode(const size_t nodeID)   const { return nodeID < numParticles; }
    __forceinline size_t numInnerNodesOf(const size_t N)    const { return N / 2; }
    __forceinline bool hasLeftChild(const size_t nodeID)    const { return isValidNode(leftChildOf(nodeID)); }
    __forceinline bool hasRightChild(const size_t nodeID)   const { return isValidNode(rightChildOf(nodeID)); }
    __forceinline static size_t isValidNode(const size_t nodeID, const size_t numParticles) { return nodeID < numParticles; }
    __forceinline float pos(const size_t nodeID, const size_t dim) const { return model->position[nodeID][dim]; }

    //! helper function for building - swap two particles in the model
    inline void swap(const size_t a, const size_t b) const;

    // save the given particle's split dimension
    void setDim(size_t ID, int dim) const;
    inline size_t maxDim(const ospcommon::vec3f &v) const;

    //! build particle tree over given model. WILL REORDER THE MODEL'S ELEMENTS
    void build();
};


struct SubtreeIterator {
    size_t curInLevel;
    size_t maxInLevel;
    size_t current;

    __forceinline SubtreeIterator(size_t root)
        : curInLevel(0), maxInLevel(1), current(root)             {}

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
    
     __forceinline SubtreeIterator &operator=(const SubtreeIterator &other) {
         curInLevel = other.curInLevel;
         maxInLevel = other.maxInLevel;
         current = other.current;
         return *this;
     }
};

struct PKDBuildJob {
    const PkdBuilder *const pkd;
    const size_t nodeID;
    const ospcommon::box3f  bounds;
    const size_t depth;
    __forceinline PKDBuildJob(const PkdBuilder *pkd, size_t nodeID, ospcommon::box3f bounds, size_t depth)
        : pkd(pkd), nodeID(nodeID), bounds(bounds), depth(depth)             {
};
            };

static void pkdBuildThread(void *arg) {
    PKDBuildJob *job = (PKDBuildJob *)arg;
    job->pkd->buildRec(job->nodeID, job->bounds, job->depth);
    delete job;
}

inline ospcommon::vec3f makeRandomColor(const int i) {
    const int mx = 13 * 17 * 43;
    const int my = 11 * 29;
    const int mz = 7 * 23 * 63;
    const uint32_t g = (i * (3 * 5 * 127) + 12312314);
    return ospcommon::vec3f((g % mx)*(1.f / (mx - 1)),
        (g % my)*(1.f / (my - 1)),
        (g % mz)*(1.f / (mz - 1)));
}

} // namespace ospray
} // namespace megamol
