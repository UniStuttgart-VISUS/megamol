#pragma once

#include <array>

#include "mmcore/Call.h"
#include "mmcore/CalleeSlot.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/Module.h"
#include "mmcore/moldyn/MultiParticleDataCall.h"
#include "mmcore/param/ParamSlot.h"
#include "thermodyn/PathLineDataCall.h"
#include "vislib/StringTokeniser.h"
#include "vislib/math/Cuboid.h"

#include "MinSphereWrapper.h"

namespace megamol {
namespace thermodyn {

class PathIColClustering : public core::Module {
public:
    /** Return module class name */
    static const char* ClassName(void) { return "PathIColClustering"; }

    /** Return module class description */
    static const char* Description(void) { return "Selects paths according to ICol clusters"; }

    /** Module is always available */
    static bool IsAvailable(void) { return true; }

    PathIColClustering();

    virtual ~PathIColClustering();

protected:
    bool create() override;

    void release() override;

private:
    bool getDataCallback(core::Call& c);

    bool getExtentCallback(core::Call& c);

    static std::vector<float> preparePoints(core::moldyn::SimpleSphericalParticles const& part) {
        std::vector<float> ret;
        ret.reserve(part.GetCount());

        auto const& store = part.GetParticleStore();
        auto const& xacc = store.GetXAcc();
        auto const& yacc = store.GetYAcc();
        auto const& zacc = store.GetZAcc();
        auto const& iacc = store.GetCRAcc();

        for (size_t idx = 0; idx < part.GetCount(); ++idx) {
            ret.push_back(xacc->Get_f(idx));
            ret.push_back(yacc->Get_f(idx));
            ret.push_back(zacc->Get_f(idx));
            ret.push_back(iacc->Get_f(idx));
        }

        return ret;
    }

    static std::vector<float> replaceTempInPoints(std::vector<float> const& part, float const rad) {
        std::vector<float> ret = part;

        for (size_t idx = 0; idx < part.size() / 4; ++idx) {
            ret[idx * 4 + 3] = rad;
        }

        return ret;
    }

    static float getTemperatureAvg(std::vector<float> const& part) {
        float temp = 0.0f;
        for (size_t idx = 0; idx < part.size()/4; ++idx) {
            temp += part[idx*4+3];
        }
        temp /= part.size()/4;
        return temp;
    }

    static vislib::math::Cuboid<float> getBoxFromString(vislib::TString const& str) {
        if (!str.Contains(',')) {
            return vislib::math::Cuboid<float>(0.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f);
        }

        auto tokens = vislib::TStringTokeniser::Split(str, ',', true);

        if (tokens.Count() < 6) {
            return vislib::math::Cuboid<float>(0.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f);
        }

        std::array<float, 6> vals{0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        for (int i = 0; i < 6; ++i) {
            vals[i] = vislib::TCharTraits::ParseDouble(tokens[i]);
        }

        return vislib::math::Cuboid<float>(vals[0], vals[1], vals[2], vals[3], vals[4], vals[5]);
    }

    bool isDirty() const {
        return minPtsSlot_.IsDirty() || sigmaSlot_.IsDirty();
    }

    void resetDirty() {
        minPtsSlot_.ResetDirty();
        sigmaSlot_.ResetDirty();
    }

    /** input of path data */
    core::CallerSlot pathsInSlot_;

    /** input of particle data */
    core::CallerSlot particleInSlot_;

    /** output of particle pathlines */
    core::CalleeSlot dataOutSlot_;

    core::param::ParamSlot minPtsSlot_;

    core::param::ParamSlot sigmaSlot_;

    core::param::ParamSlot thresholdSlot_;

    core::param::ParamSlot similaritySlot_;

    size_t inPathsHash_ = std::numeric_limits<size_t>::max();

    size_t inParHash_ = std::numeric_limits<size_t>::max();

    size_t outDataHash_ = 0;

    int frameID_ = 0;//std::numeric_limits<unsigned int>::max();

    PathLineDataCall::pathline_store_set_t pathStore_;

    PathLineDataCall::pathline_frame_store_set_t pathFrameStore_;

}; // end class PathIColClustering

} // end namespace thermodyn
} // end namespace megamol
