#pragma once

#include <vector>

#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

#include "astro/AstroDataCall.h"
#include "geometry_calls/VolumetricDataCall.h"
#include "mmcore/Call.h"
#include "mmcore/CalleeSlot.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/Module.h"
#include "mmcore/param/ParamSlot.h"
#include "mmcore/utility/log/Log.h"

namespace megamol::astro {

class SpectralIntensityVolume : public core::Module {
public:
    static const char* ClassName() {
        return "SpectralIntensityVolume";
    }

    static const char* Description() {
        return "Creates of volume capturing spectral intensity.";
    }

    static bool IsAvailable() {
        return true;
    }

    SpectralIntensityVolume();

    ~SpectralIntensityVolume() override;

protected:
    bool create() override;

    void release() override;

private:
    bool getExtentCallback(core::Call& c);

    bool getDataCallback(core::Call& c);

    bool getLSUDataCallback(core::Call& c);

    bool getAbsorptionDataCallback(core::Call& c);

    bool dummyCallback(megamol::core::Call& c) {
        return true;
    }

    bool createVolumeCPU(geocalls::VolumetricDataCall const& volumeIn, geocalls::VolumetricDataCall const& tempIn,
        geocalls::VolumetricDataCall const& massIn, geocalls::VolumetricDataCall const& mwIn, AstroDataCall& astroIn);

    bool createBremsstrahlungVolume(geocalls::VolumetricDataCall const& volumeIn,
        geocalls::VolumetricDataCall const& tempIn, geocalls::VolumetricDataCall const& massIn,
        geocalls::VolumetricDataCall const& mwIn, AstroDataCall& astroIn);

    bool createAbsorptionVolume(geocalls::VolumetricDataCall const& volumeIn,
        geocalls::VolumetricDataCall const& tempIn, geocalls::VolumetricDataCall const& massIn,
        geocalls::VolumetricDataCall const& mwIn, AstroDataCall& astroIn);

    bool anythingDirty() const {
        return this->xResSlot.IsDirty() || this->yResSlot.IsDirty() || this->zResSlot.IsDirty() ||
               this->cyclXSlot.IsDirty() || this->cyclYSlot.IsDirty() || this->cyclZSlot.IsDirty() ||
               this->normalizeSlot.IsDirty() /*|| wavelength_slot_.IsDirty()*/ || numSamplesSlot.IsDirty() ||
               absorptionBiasSlot.IsDirty();
    }

    void resetDirty() {
        this->xResSlot.ResetDirty();
        this->yResSlot.ResetDirty();
        this->zResSlot.ResetDirty();
        this->cyclXSlot.ResetDirty();
        this->cyclYSlot.ResetDirty();
        this->cyclZSlot.ResetDirty();
        this->normalizeSlot.ResetDirty();
        //wavelength_slot_.ResetDirty();
        numSamplesSlot.ResetDirty();
        absorptionBiasSlot.ResetDirty();
    }

    inline glm::quat quat_from_vectors(glm::vec3 base, glm::vec3 org_dir) {
        auto m = std::sqrt(2.0f + 2.0f * glm::dot(base, org_dir));
        auto w = (1.0f / m) * glm::cross(base, org_dir);
        return glm::quat(m / 2.0f, w.x, w.y, w.z); // yes, the glm quaternion order is stupid...
    }

    inline glm::vec3 quat_rotate(glm::vec3 base, glm::quat quat) {
        glm::quat a(0.0f, base.x, base.y, base.z);
        glm::quat c = glm::conjugate(quat);
        auto retval = quat * a * c;
        return glm::vec3(retval.x, retval.y, retval.z);
    }

    core::CallerSlot volume_in_slot_;

    core::CallerSlot temp_in_slot_;

    core::CallerSlot mass_in_slot_;

    core::CallerSlot mw_in_slot_;

    core::CallerSlot astro_in_slot_;

    core::CalleeSlot volume_out_slot_;

    core::CalleeSlot lsu_out_slot_;

    core::CalleeSlot absorption_out_slot_;

    core::param::ParamSlot xResSlot;
    core::param::ParamSlot yResSlot;
    core::param::ParamSlot zResSlot;

    core::param::ParamSlot cyclXSlot;
    core::param::ParamSlot cyclYSlot;
    core::param::ParamSlot cyclZSlot;

    core::param::ParamSlot normalizeSlot;

    core::param::ParamSlot numSamplesSlot;

    core::param::ParamSlot absorptionBiasSlot;

    core::param::ParamSlot coneSampleNumSlot;

    core::param::ParamSlot coneAngleSlot;

    // core::param::ParamSlot wavelength_slot_;

    std::vector<std::vector<float>> vol_;

    float max_dens_ = 0.0f;
    float min_dens_ = std::numeric_limits<float>::max();

    size_t in_datahash = std::numeric_limits<size_t>::max();
    size_t datahash = 0;
    unsigned int time = std::numeric_limits<unsigned int>::max();

    geocalls::VolumetricDataCall::Metadata metadata;

    int sx = 0;
    int sy = 0;
    int sz = 0;
}; // end class SpectralIntensityVolume

} // namespace megamol::astro
