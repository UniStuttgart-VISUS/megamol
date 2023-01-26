#pragma once

#include "geometry_calls/LinesDataCall.h"
#include "geometry_calls/MultiParticleDataCall.h"
#include "mesh/MeshCalls.h"
#include "mmcore/Call.h"
#include "mmcore/CalleeSlot.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/Module.h"
#include "mmcore/param/ParamSlot.h"
#include <map>

namespace megamol {
namespace datatools {

class LocalBoundingBoxExtractor : public core::Module {
public:
    /** Return module class name */
    static const char* ClassName(void) {
        return "LocalBoundingBoxExtractor";
    }

    /** Return module class description */
    static const char* Description(void) {
        return "Module extracting the local bounding box of MultiParticleDataCalls";
    }

    /** Module is always available */
    static bool IsAvailable(void) {
        return true;
    }

    /** Ctor */
    LocalBoundingBoxExtractor(void);

    /** Dtor */
    ~LocalBoundingBoxExtractor(void) override;

protected:
    /** Lazy initialization of the module */
    bool create(void) override;

    /** Resource release */
    void release(void) override;

private:
    /**
     * Called when the data is requested by this module
     *
     * @param c The incoming call
     *
     * @return True on success
     */
    bool getDataCallback(megamol::core::Call& c);

    /**
     * Called when the extend information is requested by this module
     *
     * @param c The incoming call
     *
     * @return True on success
     */
    bool getExtentCallback(megamol::core::Call& c);
    void calcLocalBox(geocalls::MultiParticleDataCall::Particles& parts, vislib::math::Cuboid<float>& box);

    /** The slot providing access to the data */
    megamol::core::CalleeSlot outLinesSlot;

    /** The slot providing access to the data */
    megamol::core::CalleeSlot outMeshSlot;

    /** The slot accessing the original data */
    megamol::core::CallerSlot inDataSlot;

    core::param::ParamSlot colorSlot;

    std::vector<geocalls::LinesDataCall::Lines> lines;
    std::map<std::string, std::array<float, 6>> lineMap;


    std::shared_ptr<mesh::MeshDataAccessCollection> mesh;
    std::vector<float> allVerts;
    std::vector<float> allCols;
    std::vector<unsigned int> allIdx;
    uint32_t mesh_version = 0;
};

} // namespace datatools
} // namespace megamol
