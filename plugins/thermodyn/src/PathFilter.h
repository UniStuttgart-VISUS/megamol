#pragma once

#include <array>
#include <unordered_map>

#include "mmcore/Module.h"
#include "mmcore/Call.h"
#include "mmcore/CalleeSlot.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/param/ParamSlot.h"
#include "vislib/math/Cuboid.h"
#include "vislib/StringTokeniser.h"
#include "thermodyn/PathLineDataCall.h"


namespace megamol {
namespace thermodyn {

class PathFilter : public core::Module {
public:
    enum class FilterType : uint8_t {
        MainDirection = 0,
        Interface,
        Plane,
        BoxFilter,
        Hotness
    };

    /** Return module class name */
    static const char* ClassName(void) { return "PathFilter"; }

    /** Return module class description */
    static const char* Description(void) { return "Filter a particle pathlines"; }

    /** Module is always available */
    static bool IsAvailable(void) { return true; }

    PathFilter();

    virtual ~PathFilter();

protected:
    bool create() override;

    void release() override;

private:
    bool getDataCallback(core::Call& c);

    bool getExtentCallback(core::Call& c);

    static vislib::math::Cuboid<float> getBoxFromString(vislib::TString const& str) {
        if (!str.Contains(',')) {
            return vislib::math::Cuboid<float>(0.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f);
        }

        auto tokens = vislib::TStringTokeniser::Split(str, ',', true);

        if (tokens.Count()<6) {
            return vislib::math::Cuboid<float>(0.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f);
        }

        std::array<float, 6> vals;
        for (int i = 0; i < 6; ++i) {
            vals[i] = vislib::TCharTraits::ParseDouble(tokens[i]);
        }

        return vislib::math::Cuboid<float>(vals[0], vals[1], vals[2], vals[3], vals[4], vals[5]);
    }

    /** input of particle pathlines */
    core::CallerSlot dataInSlot_;

    /** output of a subset of particle pathlines */
    core::CalleeSlot dataOutSlot_;

    core::param::ParamSlot filterTypeSlot_;

    core::param::ParamSlot filterAxisSlot_;

    core::param::ParamSlot filterThresholdSlot_;

    core::param::ParamSlot maxIntSlot_;

    core::param::ParamSlot minIntSlot_;

    core::param::ParamSlot timeCutSlot_;

    core::param::ParamSlot boxSlot_;

    core::param::ParamSlot percSlot_;

    size_t inDataHash_ = std::numeric_limits<size_t>::max();

    std::vector<int> entrySizes_;

    std::vector<bool> colsPresent_;

    std::vector<bool> dirsPresent_;

    std::vector<std::unordered_map<uint64_t, std::vector<float>>> pathStore_;

    PathLineDataCall::pathline_frame_store_set_t pathFrameStore_;
};

} // end namespace thermodyn
} // end namespace megamol
