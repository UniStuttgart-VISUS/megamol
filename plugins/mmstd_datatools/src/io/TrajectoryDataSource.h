#ifndef MEGAMOL_STDPLUGIN_DATATOOLS_TRAJECTORYDATASOURCE_H_INCLUDED
#define MEGAMOL_STDPLUGIN_DATATOOLS_TRAJECTORYDATASOURCE_H_INCLUDED

#include <unordered_map>

#include "mmcore/Module.h"
#include "mmcore/Call.h"
#include "mmcore/CalleeSlot.h"
#include "mmcore/param/ParamSlot.h"

#include "geometry_calls/LinesDataCall.h"

namespace megamol {
namespace stdplugin {
namespace datatools {
namespace io {

class TrajectoryDataSource : public megamol::core::Module {
public:
    /** Return module class name */
    static char const* ClassName(void) {
        return "TrajectoryDataSource";
    }

    /** Return module class description */
    static char const* Description(void) {
        return "Reads a trajectory file and exposes its contents as line data.";
    }

    /** Module is always available */
    static bool IsAvailable(void) {
        return true;
    }

    /** Ctor */
    TrajectoryDataSource(void);

    /** Dtor */
    virtual ~TrajectoryDataSource(void);
protected:
    /** Lazy initialization of the module */
    virtual bool create(void);

    /** Resource release */
    virtual void release(void);
private:
    bool getDataCallback(megamol::core::Call& c);

    bool getExtentCallback(megamol::core::Call& c);

    bool filenameChanged(megamol::core::param::ParamSlot& p);

    bool dataParamChanged(megamol::core::param::ParamSlot& p);

    bool assertData();

    megamol::core::CalleeSlot trajOutSlot;

    megamol::core::param::ParamSlot trajFilepath;

    megamol::core::param::ParamSlot minFrameSlot;

    megamol::core::param::ParamSlot maxFrameSlot;

    megamol::core::param::ParamSlot minIDSlot;

    megamol::core::param::ParamSlot maxIDSlot;

    megamol::core::param::ParamSlot transitionOnlySlot;

    megamol::core::param::ParamSlot transitionAxisSlot;

    megamol::core::param::ParamSlot minTransitionPlaneSlot;

    megamol::core::param::ParamSlot maxTransitionPlaneSlot;

    size_t datahash;

    std::string filepath_;

    std::pair<unsigned int, unsigned int> frame_begin_end_;

    std::pair<uint64_t, uint64_t> id_begin_end_;

    bool data_param_changed_;

    struct file_header_t {
        uint64_t particle_count;
        unsigned int frame_count;
        float* bbox;
    } file_header_;

    std::unordered_map<uint64_t /* id */, size_t /* offset */> particle_file_offsets_;

    std::unordered_map<uint64_t /* id */, std::pair<unsigned int, unsigned int>> particle_frame_begin_end_;

    std::vector<uint64_t> sorted_id_list;

    std::vector<std::vector<float>> data;

    std::vector<std::vector<uint8_t>> col_data;

    std::vector<std::vector<char>> is_fluid_data;

    std::vector<megamol::geocalls::LinesDataCall::Lines> lines_data;

    std::vector<unsigned int> index_dummy;
}; /* end class TrajectoryDataSource */

} /* end namespace io */
} /* end namespace datatools */
} /* end namespace stdplugin */
} /* end namespace megamol */

#endif /* end ifndef MEGAMOL_STDPLUGIN_DATATOOLS_TRAJECTORYDATASOURCE_H_INCLUDED */
