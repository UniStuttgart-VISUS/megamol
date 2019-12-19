#pragma once

#include "mmcore/CalleeSlot.h"
#include "mmcore/Module.h"
#include "mmcore/param/ParamSlot.h"
#include "mmcore/moldyn/MultiParticleDataCall.h"
#include "geometry_calls/LinesDataCall.h"

#include "mmpld.h"
#include "vislib/math/mathtypes.h"
#include "vislib/math/Cuboid.h"

namespace megamol {
namespace stdplugin {
namespace datatools {

// TODO extract template?
class FSOctreeMMPLD: public core::Module {
public:
    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char* ClassName(void) { return "FSOctreeMMPLD"; }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static const char* Description(void) { return "generates octree file names for a chunked MMPLD and outputs bounding boxes"; }

    /**
     * Answers whether this module is available on the current system.
     *
     * @return 'true' if the module is available, 'false' otherwise.
     */
    static bool IsAvailable(void) { return true; }

    /** Ctor. */
    FSOctreeMMPLD(void);

    /** Dtor. */
    virtual ~FSOctreeMMPLD(void);

protected:
    bool create() override;

    void release() override;

    core::param::ParamSlot pathSlot;
    core::param::ParamSlot filenamePrefixSlot;
    core::param::ParamSlot extensionSlot;
    core::param::ParamSlot maxSearchDepthSlot;

private:

    class octree_node {
    public:
        std::string filename;
        std::array<std::shared_ptr<octree_node>, 8> children;

        static void insert_node(std::shared_ptr<octree_node> root, const std::vector<uint8_t>& address, const std::string& filename) {
            auto curr_node = root;
            for (auto a: address) {
                if (curr_node->children[a] == nullptr) {
                    curr_node->children[a] = std::make_shared<octree_node>();
                }
                curr_node = curr_node->children[a];
            }
            curr_node->filename = filename;
        }
    };

    std::shared_ptr<octree_node> tree = std::make_shared<octree_node>();


    bool assertData(megamol::geocalls::LinesDataCall& outCall);

    bool getDataCallback(core::Call& c);

    bool getExtentCallback(core::Call& c);

    core::CalleeSlot outBoxesSlot;

    //std::vector<mmpld::frame_t> output_frames;

    //vislib::math::Cuboid<float> gbbox;

    uint32_t version = 0;
}; // end class FSOctreeMMPLD

} // end namespace datatools
} // end namespace stdplugin
} // end namespace megamol