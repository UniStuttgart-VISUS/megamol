#pragma once

#include <memory>

#include "mmcore/Call.h"
#include "mmcore/CalleeSlot.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/Module.h"
#include "mmcore/param/ParamSlot.h"

#include "CallADIOSData.h"

namespace megamol {
namespace adios {

class SignalPeaks : public core::Module {
public:
    /** Return module class name */
    static const char* ClassName(void) { return "SignalPeaks"; }

    /** Return module class description */
    static const char* Description(void) { return "Extracts local peaks from a signal"; }

    /** Module is always available */
    static bool IsAvailable(void) { return true; }

    /** Ctor */
    SignalPeaks(void);

    /** Dtor */
    virtual ~SignalPeaks(void);

protected:
    /** Lazy initialization of the module */
    bool create(void) override;

    /** Resource release */
    void release(void) override;

private:
    bool getDataCallback(core::Call& c);
    bool getHeaderCallback(core::Call& c);

    core::CalleeSlot data_out_slot_;
    core::CallerSlot data_in_slot_;

    core::param::ParamSlot num_peaks_slot_;

    std::shared_ptr<adiosDataMap> data_map_;

    size_t data_hash_;
}; // class SignalPeaks

} // namespace adios
} // namespace megamol
