#pragma once

#include <array>

#include "mmcore/CalleeSlot.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/Module.h"
#include "mmcore/param/ParamSlot.h"

#include "thermodyn/BoxDataCall.h"
#include "vislib/StringTokeniser.h"


namespace megamol {
namespace thermodyn {

class PhaseSeparator : public core::Module {
public:
    /** Return module class name */
    static const char* ClassName(void) { return "PhaseSeparator"; }

    /** Return module class description */
    static const char* Description(void) { return "Determines the phases in an evap simulation"; }

    /** Module is always available */
    static bool IsAvailable(void) { return true; }

    PhaseSeparator();

    virtual ~PhaseSeparator();

protected:
    bool create() override;

    void release() override;

private:
    static std::array<float, 4> getColor(vislib::TString const& str) {
        if (!str.Contains(',')) {
            return std::array<float, 4>();
        }

        auto tokens = vislib::TStringTokeniser::Split(str, ',', true);

        if (tokens.Count() < 4) {
            return std::array<float, 4>();
        }

        std::array<float, 4> vals;
        for (int i = 0; i < 4; ++i) {
            vals[i] = vislib::TCharTraits::ParseDouble(tokens[i]);
        }

        return vals;
    }

    bool getDataCallback(core::Call& c);

    bool getExtentCallback(core::Call& c);

    core::CallerSlot dataInSlot_;

    core::CalleeSlot dataOutSlot_;

    core::param::ParamSlot criticalTempSlot_;

    core::param::ParamSlot ensembleTempSlot_;

    core::param::ParamSlot fluidColorSlot_;

    core::param::ParamSlot interfaceColorSlot_;

    core::param::ParamSlot gasColorSlot_;

    core::param::ParamSlot axisSlot_;

    core::param::ParamSlot numSlicesSlot_;

    size_t inDataHash_ = std::numeric_limits<size_t>::max();

    unsigned int frameID_ = 0;

    std::vector<BoxDataCall::box_entry_t> boxes_;
}; // end class PhaseSeparator

} // end namespace thermodyn
} // end namespace megamol
