#ifndef MEGAMOL_DATATOOLS_TRAJECTORYFACTORY_H_INCLUDED
#define MEGAMOL_DATATOOLS_TRAJECTORYFACTORY_H_INCLUDED

#include <memory>
#include <string>
#include <vector>
#include <unordered_map>

#include "mmcore/AbstractDataWriter.h"
#include "mmcore/Call.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/param/ParamSlot.h"

namespace megamol {
namespace stdplugin {
namespace datatools {

/**
 * Module to resort particles form timestep-separated into a particle-id-separated order.
 */
class TrajectoryFactory : public megamol::core::AbstractDataWriter {
public:
    /** Return module class name */
    static char const* ClassName(void) {
        return "TrajectoryFactory";
    }

    /** Return module class description */
    static char const* Description(void) {
        return "Resorts particles from a timestep based ordering to a particle-id based ordering.";
    }

    /** Module is always available */
    static bool IsAvailable(void) {
        return true;
    }

    /**
     * Disallow usage in quickstarts
     *
     * @return false
     */
    static bool SupportQuickstart(void) {
        return false;
    }

    /** Ctor */
    TrajectoryFactory(void);

    /** Dtor */
    virtual ~TrajectoryFactory(void);
protected:
    /** Lazy initialization of the module */
    virtual bool create(void);

    /** Resource release */
    virtual void release(void);

    /**
     * The main function
     *
     * @return True on success
     */
    virtual bool run(void);

    /**
     * Function querying the writers capabilities
     *
     * @param call The call to receive the capabilities
     *
     * @return True on success
     */
    virtual bool getCapabilities(core::DataWriterCtrlCall& call);
private:
    template<class cT>
    struct particle_t {
        cT x, y, z;
    };

    template<class cT>
    using trajectory_t = std::vector<particle_t<cT>>;

    struct abstract_trajectory_storage {
        virtual ~abstract_trajectory_storage() = default;
    };

    template<class idT, class cT>
    struct trajectory_storage : public abstract_trajectory_storage {
        std::unordered_map<idT, trajectory_t<cT>> data;
    };

    bool assertData(megamol::core::Call& c);

    void write(std::string const& filepath, unsigned int const pli,
        std::unordered_map<uint64_t, trajectory_t<float>>& cur_data,
        std::unordered_map<uint64_t, uint64_t>& cur_file_id_offsets,
        std::unordered_map<uint64_t, std::pair<unsigned int, unsigned int>>& cur_frame_id_assoc,
        std::vector<size_t>& max_offset, size_t const max_line_size,
        std::vector<char> const& zero_out_buf) const;

    void writeParticle(FILE* file, size_t const base_offset,
        std::pair<unsigned int, unsigned int> const& frame_start_end, uint64_t const id,
        trajectory_t<float> const& toWrite, bool const new_par = false) const;

    megamol::core::CallerSlot inDataSlot;

    megamol::core::param::ParamSlot filepathSlot;

    megamol::core::param::ParamSlot maxFramesInMemSlot;

    size_t datahash;

    /** path to the file holding the trajectories */
    //std::string trajectory_file_path_;

    /** upper bound for particle cache size before disk I/O is enforced */
    //unsigned int max_frames_in_mem_; //< translates to how many timesteps can be recorded per particle

    /** pointer to polymorph storage holding the trajectory data */
    //std::unique_ptr<abstract_trajectory_storage> t_storage_ptr_;
}; /* end class TrajectoryFactory */

} /* end namespace datatools */
} /* end namespace stdplugin */
} /* end namespace megamol */

#endif /* end ifndef MEGAMOL_DATATOOLS_TRAJECTORYFACTORY_H_INCLUDED */
