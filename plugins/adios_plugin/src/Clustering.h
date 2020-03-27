#pragma once

#include <memory>

#include "mmcore/Module.h"
#include "mmcore/Call.h"
#include "mmcore/CalleeSlot.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/param/ParamSlot.h"

#include "CallADIOSData.h"

namespace megamol {
namespace adios {

class Clustering : public core::Module {
public:
    enum Algorithm {
        DBSCAN
    };

    /** Return module class name */
    static const char* ClassName(void) { return "Clustering"; }

    /** Return module class description */
    static const char* Description(void) { return "Applies clustering on input"; }

    /** Module is always available */
    static bool IsAvailable(void) { return true; }

    /** Ctor */
    Clustering(void);

    /** Dtor */
    virtual ~Clustering(void);

protected:
    /** Lazy initialization of the module */
    bool create(void) override;

    /** Resource release */
    void release(void) override;

private:
    bool getDataCallback(core::Call& c);
    bool getHeaderCallback(core::Call& c);

    bool changeAlgCallback(core::param::ParamSlot& p);

    core::CalleeSlot data_out_slot_;
    core::CallerSlot data_in_slot_;

    core::param::ParamSlot alg_selector_slot_;

    core::param::ParamSlot min_pts_slot_;
    core::param::ParamSlot sigma_slot_;

    std::shared_ptr<adiosDataMap> data_map_;

    size_t data_hash_;
}; // class Clustering

} // namespace adios
} // namespace megamol