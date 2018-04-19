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

#include "PointcloudHelpers.h"

#include "nanoflann.hpp"

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

    void write(std::string const& filepath, unsigned int const pli, unsigned int const frameCount,
        std::unordered_map<uint64_t, trajectory_t<float>>& cur_data,
        std::unordered_map<uint64_t, std::vector<char>>& cur_is_fluid_data,
        std::unordered_map<uint64_t, uint64_t>& cur_file_id_offsets,
        std::unordered_map<uint64_t, std::pair<unsigned int, unsigned int>>& cur_frame_id_assoc,
        std::vector<size_t>& max_offset, size_t const max_line_size,
        std::vector<char> const& zero_out_buf) const;

    void writeParticle(FILE* file, unsigned int const frameCount, size_t const base_offset,
        std::pair<unsigned int, unsigned int>& frame_start_end, uint64_t const id,
        trajectory_t<float> const& toWrite, std::vector<char> const& is_fluid,
        bool const new_par = false) const;

    megamol::core::CallerSlot inDataSlot;

    megamol::core::param::ParamSlot filepathSlot;

    megamol::core::param::ParamSlot maxFramesInMemSlot;

    megamol::core::param::ParamSlot searchRadiusSlot;

    megamol::core::param::ParamSlot minPtsSlot;

    size_t datahash;

    typedef nanoflann::KDTreeSingleIndexAdaptor<
        nanoflann::L2_Simple_Adaptor<float, simplePointcloud>,
        simplePointcloud,
        3 /* dim */
    > my_kd_tree_t;
    typedef nanoflann::KDTreeSingleIndexAdaptor<
        nanoflann::L2_Simple_Adaptor<float, directionalPointcloud>,
        directionalPointcloud,
        3 /* dim */
    > my_dir_kd_tree_t;

    std::shared_ptr<my_kd_tree_t> particleTree;
    std::shared_ptr<my_dir_kd_tree_t> dirParticleTree;
    std::shared_ptr<simplePointcloud> myPts;
    std::shared_ptr<directionalPointcloud> myDirPts;

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
