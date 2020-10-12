#pragma once

#include "mmcore/Module.h"
#include "mmcore/Call.h"
#include "mmcore/CalleeSlot.h"
#include "mmcore/CallerSlot.h"
#include "geometry_calls/LinesDataCall.h"
#include "mmcore/param/ParamSlot.h"
#include "geometry_calls/CallTriMeshData.h"
#include "mmcore/moldyn/MultiParticleDataCall.h"
#include <map>

namespace megamol {
namespace stdplugin {
namespace datatools {

class LocalBoundingBoxExtractor: public core::Module {
public:
    /** Return module class name */
    static const char* ClassName(void) { return "LocalBoundingBoxExtractor"; }

    /** Return module class description */
    static const char* Description(void) { return "Module extracting the local bounding box of MultiParticleDataCalls"; }

    /** Module is always available */
    static bool IsAvailable(void) { return true; }

    /** Ctor */
    LocalBoundingBoxExtractor(void);

    /** Dtor */
    ~LocalBoundingBoxExtractor(void);

protected:

    /** Lazy initialization of the module */
    bool create(void);

    /** Resource release */
    void release(void);

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
    void calcLocalBox(core::moldyn::MultiParticleDataCall::Particles& parts, vislib::math::Cuboid<float>& box);

    /** The slot providing access to the data */
    megamol::core::CalleeSlot outLinesSlot;

    /** The slot providing access to the data */
    megamol::core::CalleeSlot outMeshSlot;

    /** The slot accessing the original data */
    megamol::core::CallerSlot inDataSlot;

    core::param::ParamSlot colorSlot;

    std::vector<geocalls::LinesDataCall::Lines> lines;
    std::map<std::string, std::array<float,6>> lineMap;
    
    
    geocalls::CallTriMeshData::Mesh mesh;
    std::vector<float> allVerts;
    std::vector<float> allCols;
    std::vector<unsigned int> allIdx;



    };



} // namespace datatools
} // namespace stdplugin
} // namespace megamol