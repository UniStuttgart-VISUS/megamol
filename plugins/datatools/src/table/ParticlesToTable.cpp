#include "ParticlesToTable.h"
#include "stdafx.h"

#include "geometry_calls//EllipsoidalDataCall.h"
#include "mmcore/param/EnumParam.h"
#include "mmcore/param/FlexEnumParam.h"
#include "mmcore/param/FloatParam.h"
#include "mmcore/param/StringParam.h"
#include "mmcore/utility/ColourParser.h"

#define GLM_FORCE_SWIZZLE
#include "glm/glm.hpp"
#include "glm/gtc/matrix_transform.hpp"
#include "glm/gtc/quaternion.hpp"
#include "glm/gtc/type_ptr.hpp"
#include "mmcore/utility/log/Log.h"
#include "vislib/StringConverter.h"
#include "vislib/Trace.h"
#include "vislib/sys/PerformanceCounter.h"
#include <glm/gtx/string_cast.hpp>

using namespace megamol::datatools;
using namespace megamol;

/*
 * TableToParticles::ParticlesToTable
 */
ParticlesToTable::ParticlesToTable(void)
        : Module()
        , slotTableOut("floattable", "Provides the data as table.")
        , slotParticlesIn("particles", "Particle input call") {

    /* Register parameters. */

    /* Register calls. */
    this->slotTableOut.SetCallback(table::TableDataCall::ClassName(), "GetData", &ParticlesToTable::getTableData);
    this->slotTableOut.SetCallback(table::TableDataCall::ClassName(), "GetHash", &ParticlesToTable::getTableHash);

    this->MakeSlotAvailable(&this->slotTableOut);

    this->slotParticlesIn.SetCompatibleCall<geocalls::MultiParticleDataCallDescription>();
    this->slotParticlesIn.SetCompatibleCall<geocalls::EllipsoidalParticleDataCallDescription>();
    this->MakeSlotAvailable(&this->slotParticlesIn);
}


/*
 * TableToParticles::~TableToParticles
 */
ParticlesToTable::~ParticlesToTable(void) {
    this->Release();
}


/*
 * megamol::pcl::PclDataSource::create
 */
bool ParticlesToTable::create(void) {
    return true;
}

bool ParticlesToTable::assertMPDC(geocalls::MultiParticleDataCall* in, table::TableDataCall* tc) {
    in->SetFrameID(tc->GetFrameID(), true);
    do {
        if (!(*in)(1))
            return false;
        if (!(*in)(0))
            return false;
    } while (in->FrameID() != tc->GetFrameID());

    if (in->DataHash() != inHash || in->FrameID() != inFrameID) {
        auto lc = in->GetParticleListCount();
        if (lc > 0) {
            // we cannot leave anything out, or else the indices break.
            total_particles = 0;
            for (auto l = 0; l < in->GetParticleListCount(); ++l) {
                total_particles += in->AccessParticles(l).GetCount();
            }

            column_infos.clear();
            const std::array<std::string, 11> column_names = {
                "x", "y", "z", "rad", "r", "g", "b", "i", "vx", "vy", "vz"};
            std::array<float, 11> minimums{}, maximums{};
            minimums.fill(std::numeric_limits<float>::max());
            maximums.fill(std::numeric_limits<float>::lowest());

            for (auto& col : column_names) {
                auto ci = column_infos.emplace_back();
                column_infos.back().SetName(col);
                column_infos.back().SetType(table::TableDataCall::ColumnType::QUANTITATIVE);
            }

            auto store_and_compute_extents = [&](uint32_t idx, uint32_t col, float val) {
                if (val < minimums[col]) {
                    minimums[col] = val;
                }
                if (val > maximums[col]) {
                    maximums[col] = val;
                }
                everything[column_names.size() * idx + col] = val;
            };

            everything.resize(column_names.size() * total_particles);
            uint32_t particle_idx = 0;
            for (auto l = 0; l < in->GetParticleListCount(); ++l) {
                auto pl = in->AccessParticles(l);
                const auto& store = pl.GetParticleStore();
                for (auto idx = 0; idx < pl.GetCount(); ++idx) {
                    store_and_compute_extents(particle_idx, 0, store.GetXAcc()->Get_f(idx));
                    store_and_compute_extents(particle_idx, 1, store.GetYAcc()->Get_f(idx));
                    store_and_compute_extents(particle_idx, 2, store.GetZAcc()->Get_f(idx));
                    store_and_compute_extents(particle_idx, 3, store.GetRAcc()->Get_f(idx));
                    store_and_compute_extents(particle_idx, 4, store.GetCRAcc()->Get_f(idx));
                    store_and_compute_extents(particle_idx, 5, store.GetCGAcc()->Get_f(idx));
                    store_and_compute_extents(particle_idx, 6, store.GetCBAcc()->Get_f(idx));
                    store_and_compute_extents(particle_idx, 7, store.GetCRAcc()->Get_f(idx));
                    store_and_compute_extents(particle_idx, 8, store.GetDXAcc()->Get_f(idx));
                    store_and_compute_extents(particle_idx, 9, store.GetDYAcc()->Get_f(idx));
                    store_and_compute_extents(particle_idx, 10, store.GetDZAcc()->Get_f(idx));
                    particle_idx++;
                }
            }

            for (auto i = 0; i < column_infos.size(); ++i) {
                column_infos[i].SetMinimumValue(minimums[i]);
                column_infos[i].SetMaximumValue(maximums[i]);
            }
        }
        inHash = in->DataHash();
        inFrameID = in->FrameID();
    }
    return true;
}

bool ParticlesToTable::assertEPDC(geocalls::EllipsoidalParticleDataCall* c, table::TableDataCall* tc) {
    core::utility::log::Log::DefaultLog.WriteError("ParticlesToTable: ellipsoidal particles not implemented yet!");
    return false;
}

/*
 * megamol::pcl::PclDataSource::getMultiParticleData
 */
bool ParticlesToTable::getTableData(core::Call& call) {
    auto* c = this->slotParticlesIn.CallAs<geocalls::MultiParticleDataCall>();
    auto* e = this->slotParticlesIn.CallAs<geocalls::EllipsoidalParticleDataCall>();

    auto* ft = dynamic_cast<table::TableDataCall*>(&call);
    if (ft == nullptr)
        return false;

    if (c != nullptr) {
        assertMPDC(c, ft);
        ft->Set(column_infos.size(), total_particles, column_infos.data(), everything.data());
    } else if (e != nullptr) {
        return false;
    }
    return true;
}


bool ParticlesToTable::getTableHash(core::Call& call) {
    auto* c = this->slotParticlesIn.CallAs<geocalls::MultiParticleDataCall>();
    auto* e = this->slotParticlesIn.CallAs<geocalls::EllipsoidalParticleDataCall>();

    auto* ft = dynamic_cast<table::TableDataCall*>(&call);
    if (ft == nullptr)
        return false;

    if (c != nullptr) {
        if (!(*c)(1))
            return false;
        ft->SetDataHash(c->DataHash());
        ft->SetFrameCount(c->FrameCount());
        return true;
    } else if (e != nullptr) {
        if (!(*e)(1))
            return false;
        ft->SetDataHash(e->DataHash());
        ft->SetFrameCount(e->FrameCount());
        return true;
    }
    return false;
}


/*
 * megamol::pcl::PclDataSource::release
 */
void ParticlesToTable::release(void) {}
