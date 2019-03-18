#pragma once

#include <vector>
#include <fstream>
#include <string>

#include "mmcore/Module.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/param/ParamSlot.h"

namespace megamol {
namespace stdplugin {
namespace datatools {

class DumpIColTrend : public core::Module {
public:
    static const char* ClassName(void) { return "DumpIColTrend"; }

    static const char* Description(void) { return "Module to dump the ICol trend along a specified axis"; }

    static bool IsAvailable(void) { return true; }

    DumpIColTrend(void);

    virtual ~DumpIColTrend(void);

protected:
    bool create(void) override;

    void release(void) override;

private:
    bool dump(core::param::ParamSlot& p);

    static void dumpTrend(std::string const& filename, std::vector<float> const& trend, std::vector<float> const& mids) {
        if (trend.size() != mids.size()) return;
        std::ofstream out_str(filename);
        for (size_t idx = 0; idx < trend.size(); ++idx) {
            out_str << mids[idx] << "\t" << trend[idx] << std::endl;
        }
    }

    core::CallerSlot dataInSlot_;

    core::param::ParamSlot dumpSlot_;

    core::param::ParamSlot axisSlot_;

    core::param::ParamSlot numBucketsSlot_;

    core::param::ParamSlot frameSlot_;

}; // end class DumpIColTrend

} // end namespace datatools
} // end namespace stdplugin
} // end namespace megamol
