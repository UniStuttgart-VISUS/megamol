/*
 * ParticlesToDensity.h
 *
 * Copyright (C) 2018 by MegaMol team
 * Alle Rechte vorbehalten.
 */

#pragma once

#include "geometry_calls/MultiParticleDataCall.h"
#include "geometry_calls/VolumetricDataCall.h"
#include "mmcore/CalleeSlot.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/Module.h"
#include "mmcore/param/ParamSlot.h"

#include "datatools/table/TableDataCall.h"

#include "vislib/math/Vector.h"

#include <array>
#include <limits>
#include <vector>

namespace megamol {
namespace datatools {

/**
 * Module computing a density volume from particles.
 * One day, hopefully some precise particle-cell intersectors
 * will provide a very accurate result.
 */
class ParticlesToDensity : public megamol::core::Module {
public:
    /** Return module class name */
    static const char* ClassName() {
        return "ParticlesToDensity";
    }

    /** Return module class description */
    static const char* Description() {
        return "Computes a density volume from particles";
    }

    /** Module is always available */
    static bool IsAvailable() {
        return true;
    }

    /** Ctor */
    ParticlesToDensity();

    /** Dtor */
    ~ParticlesToDensity() override;

protected:
    /** Lazy initialization of the module */
    bool create() override;

    /** Resource release */
    void release() override;

private:
    /**
     * Called when the data is requested by this module
     *
     * @param c The incoming call
     *
     * @return True on success
     */
    bool getDataCallback(megamol::core::Call& c);

    bool dummyCallback(megamol::core::Call& c);

    bool createVolumeCPU(geocalls::MultiParticleDataCall* c2);

    void modifyBBox(geocalls::MultiParticleDataCall* c2);

    /**
     * Called when the extend information is requested by this module
     *
     * @param c The incoming call
     *
     * @return True on success
     */
    bool getExtentCallback(megamol::core::Call& c);

    inline bool anythingDirty() const {
        return this->aggregatorSlot.IsDirty() || this->xResSlot.IsDirty() || this->yResSlot.IsDirty() ||
               this->zResSlot.IsDirty() || this->cyclXSlot.IsDirty() || this->cyclYSlot.IsDirty() ||
               this->cyclZSlot.IsDirty() || this->normalizeSlot.IsDirty() || this->sigmaSlot.IsDirty();
    }

    inline void resetDirty() {
        this->aggregatorSlot.ResetDirty();
        this->xResSlot.ResetDirty();
        this->yResSlot.ResetDirty();
        this->zResSlot.ResetDirty();
        this->cyclXSlot.ResetDirty();
        this->cyclYSlot.ResetDirty();
        this->cyclZSlot.ResetDirty();
        this->normalizeSlot.ResetDirty();
        this->sigmaSlot.ResetDirty();
    }

    core::param::ParamSlot aggregatorSlot;

    core::param::ParamSlot xResSlot;
    core::param::ParamSlot yResSlot;
    core::param::ParamSlot zResSlot;

    core::param::ParamSlot cyclXSlot;
    core::param::ParamSlot cyclYSlot;
    core::param::ParamSlot cyclZSlot;

    core::param::ParamSlot normalizeSlot;

    core::param::ParamSlot sigmaSlot;

    core::param::ParamSlot surfaceSlot;

    std::vector<std::vector<float>> vol;
    std::vector<float> directions, colors, densities;
    std::vector<float> grid;

    std::array<datatools::table::TableDataCall::ColumnInfo, 7> info;
    std::vector<float> infoData;

    size_t in_datahash = std::numeric_limits<size_t>::max();
    size_t datahash = 0;
    unsigned int time = 0;
    float maxDens = 0.0f;
    float minDens = std::numeric_limits<float>::max();

    bool has_data;

    /** The slot providing access to the manipulated data */
    megamol::core::CalleeSlot outDataSlot;
    megamol::core::CalleeSlot outParticlesSlot;
    megamol::core::CalleeSlot outInfoSlot;

    /** The slot accessing the original data */
    megamol::core::CallerSlot inDataSlot;

    geocalls::VolumetricDataCall::Metadata metadata;
};

} /* end namespace datatools */
} /* end namespace megamol */
