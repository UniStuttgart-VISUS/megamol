#include "MortonCompRenderer.h"

#include <mutex>

#include "mmcore/param/BoolParam.h"
#include "mmcore/param/FilePathParam.h"
#include "mmcore/param/FloatParam.h"
#include "mmcore/param/IntParam.h"

#include <owl/common/math/box.h>
#include <owl/common/math/vec.h>

#include "MortonCompCreate.h"
#include "PKDCreate.h"
#include "TreeletsCreate.h"

#include "CompErrorAnalysis.h"

#include "framestate.h"
#include "mortoncomp.h"
#include "raygen.h"

#include <glad/gl.h>

#include <tbb/parallel_for.h>

#include <cuda_runtime.h>

namespace megamol::optix_owl {
extern "C" const unsigned char mortoncompPrograms_ptx[];

MortonCompRenderer::MortonCompRenderer() : threshold_slot_("threshold", "") {
    threshold_slot_ << new core::param::IntParam(2048, 16, 2048);
    MakeSlotAvailable(&threshold_slot_);
}

MortonCompRenderer::~MortonCompRenderer() {
    this->Release();
}

bool MortonCompRenderer::create() {
    auto const ret = BaseRenderer::create();
    pkd_module_ = owlModuleCreate(ctx_, reinterpret_cast<const char*>(mortoncompPrograms_ptx));

    OWLVarDecl treeletsVars[] = {{"treeletBuffer", OWL_BUFPTR, OWL_OFFSETOF(device::MortonCompGeomData, treeletBuffer)},
        {"particleBuffer", OWL_BUFPTR, OWL_OFFSETOF(device::MortonCompGeomData, particleBuffer)},
        {"particleRadius", OWL_FLOAT, OWL_OFFSETOF(device::MortonCompGeomData, particleRadius)},
        {"bounds.lower", OWL_FLOAT3, OWL_OFFSETOF(device::MortonCompGeomData, bounds.lower)},
        {"bounds.upper", OWL_FLOAT3, OWL_OFFSETOF(device::MortonCompGeomData, bounds.upper)},
        {/* sentinel to mark end of list */}};

    OWLGeomType treeletsType =
        owlGeomTypeCreate(ctx_, OWL_GEOMETRY_USER, sizeof(device::MortonCompGeomData), treeletsVars, -1);
    owlGeomTypeSetBoundsProg(treeletsType, pkd_module_, "mortoncomp_bounds");
    owlGeomTypeSetIntersectProg(treeletsType, 0, pkd_module_, "mortoncomp_intersect");
    owlGeomTypeSetClosestHit(treeletsType, 0, pkd_module_, "mortoncomp_ch");

    geom_ = owlGeomCreate(ctx_, treeletsType);

    return ret;
}

void MortonCompRenderer::release() {
    owlBufferDestroy(treeletBuffer_);
    owlModuleRelease(pkd_module_);
    owlGeomRelease(geom_);

    BaseRenderer::release();
}


bool MortonCompRenderer::assertData(geocalls::MultiParticleDataCall const& call) {
    auto const pl_count = call.GetParticleListCount();

    std::vector<device::Particle> particles_;
    owl::common::box3f total_bounds;
    auto const global_radius = radius_slot_.Param<core::param::FloatParam>()->Value();

    for (unsigned int pl_idx = 0; pl_idx < pl_count; ++pl_idx) {
        auto const& particles = call.AccessParticles(pl_idx);

        auto const p_count = particles.GetCount();
        if (p_count == 0)
            continue;
        /*if (particles.GetVertexDataType() == geocalls::SimpleSphericalParticles::VERTDATA_FLOAT_XYZR)
            continue;*/

        particles_.reserve(particles_.size() + p_count);

        std::vector<device::Particle> data(p_count);

        auto x_acc = particles.GetParticleStore().GetXAcc();
        auto y_acc = particles.GetParticleStore().GetYAcc();
        auto z_acc = particles.GetParticleStore().GetZAcc();

        owl::common::box3f bounds;

        for (std::size_t i = 0; i < p_count; ++i) {
            data[i].pos = owl::common::vec3f(x_acc->Get_f(i), y_acc->Get_f(i), z_acc->Get_f(i));
            bounds.extend(
                owl::common::box3f().including(data[i].pos - global_radius).including(data[i].pos + global_radius));
        }

        particles_.insert(particles_.end(), data.begin(), data.end());
        total_bounds.extend(bounds);
    }


    device::MortonConfig config;

    std::vector<std::pair<uint64_t, uint64_t>> cells;
    std::vector<uint64_t> sorted_codes;

    // 1 create morton grid
    {
        auto mc = create_morton_codes(particles_, total_bounds, config);
        sort_morton_codes(mc);
        std::tie(cells, sorted_codes, particles_) = mask_morton_codes(mc, particles_, config);
    }

    std::vector<device::PKDlet> treelets;
    for (auto const& c : cells) {
        auto const temp_treelets = prePartition_inPlace(
            particles_, c.first, c.second, threshold_slot_.Param<core::param::IntParam>()->Value(), global_radius);

        treelets.insert(treelets.end(), temp_treelets.begin(), temp_treelets.end());
    }

    std::vector<MortonCompPKDlet> ctreelets(treelets.size());
    std::vector<MortonCompParticle> cparticles(particles_.size());
    auto diffs = std::make_shared<std::vector<vec3f>>();
    auto orgpos = std::make_shared<std::vector<vec3f>>();
    auto spos = std::make_shared<std::vector<vec3f>>();
    if (dump_debug_info_slot_.Param<core::param::BoolParam>()->Value()) {
        diffs->reserve(particles_.size());
        if (debug_rdf_slot_.Param<core::param::BoolParam>()->Value()) {
            orgpos->reserve(particles_.size());
            spos->reserve(particles_.size());
        }
    }
    std ::mutex debug_data_mtx;
#pragma omp parallel for
    for (int64_t i = 0; i < treelets.size(); ++i) {
        auto const [temp_pos, temp_rec, temp_diffs] =
            convert_morton_treelet(treelets[i], particles_, ctreelets[i], cparticles, total_bounds, config);
        if (dump_debug_info_slot_.Param<core::param::BoolParam>()->Value()) {
            std::lock_guard<std::mutex> guard(debug_data_mtx);
            orgpos->insert(orgpos->end(), temp_pos.begin(), temp_pos.end());
            if (debug_rdf_slot_.Param<core::param::BoolParam>()->Value()) {
                spos->insert(spos->end(), temp_rec.begin(), temp_rec.end());
                diffs->insert(diffs->end(), temp_diffs.begin(), temp_diffs.end());
            }
        }
    }

    for (auto& el : ctreelets) {
        adapt_morton_bbox(cparticles, el, total_bounds, global_radius, config);
        makePKD(cparticles, el, total_bounds, config);
    }

    if (dump_debug_info_slot_.Param<core::param::BoolParam>()->Value()) {
        auto const output_path = debug_output_path_slot_.Param<core::param::FilePathParam>()->Value();

        dump_analysis_data(
            output_path, orgpos, spos, diffs, global_radius, debug_rdf_slot_.Param<core::param::BoolParam>()->Value());
    }

    core::utility::log::Log::DefaultLog.WriteInfo(
        "[MortonCompRenderer] %d treelets for %d particles", ctreelets.size(), cparticles.size());

    if (particleBuffer_)
        owlBufferDestroy(particleBuffer_);
    particleBuffer_ =
        owlDeviceBufferCreate(ctx_, OWL_USER_TYPE(device::MortonCompParticle), cparticles.size(), cparticles.data());

    if (treeletBuffer_)
        owlBufferDestroy(treeletBuffer_);
    treeletBuffer_ =
        owlDeviceBufferCreate(ctx_, OWL_USER_TYPE(device::MortonCompPKDlet), ctreelets.size(), ctreelets.data());

    owlGeomSetPrimCount(geom_, ctreelets.size());

    owlGeomSetBuffer(geom_, "particleBuffer", particleBuffer_);
    owlGeomSetBuffer(geom_, "treeletBuffer", treeletBuffer_);
    owlGeomSet1f(geom_, "particleRadius", global_radius);
    owlGeomSet3f(geom_, "bounds.lower", (const owl3f&) total_bounds.lower);
    owlGeomSet3f(geom_, "bounds.upper", (const owl3f&) total_bounds.upper);

    owlBuildPrograms(ctx_);

    OWLGroup ug = owlUserGeomGroupCreate(ctx_, 1, &geom_);
    owlGroupBuildAccel(ug);

    world_ = owlInstanceGroupCreate(ctx_, 1, &ug);

    owlGroupBuildAccel(world_);

    if (dump_debug_info_slot_.Param<core::param::BoolParam>()->Value()) {
        size_t memFinal = 0;
        size_t memPeak = 0;
        owlGroupGetAccelSize(ug, &memFinal, &memPeak);

        size_t comp_data_size = cparticles.size() * sizeof(device::MortonCompParticle) +
                                ctreelets.size() * sizeof(device::MortonCompPKDlet);

        auto const output_path = debug_output_path_slot_.Param<core::param::FilePathParam>()->Value();
        auto of = std::ofstream(output_path / "size.csv");
        of << "BVHFinalSize[B],BVHPeakSize[B],CompDataSize[B]\n";
        of << memFinal << ",";
        of << memPeak << ",";
        of << comp_data_size << std::endl;
        of.close();
    }

    return true;
}

bool MortonCompRenderer::data_param_is_dirty() {
    return threshold_slot_.IsDirty();
}

void MortonCompRenderer::data_param_reset_dirty() {
    threshold_slot_.ResetDirty();
}
} // namespace megamol::optix_owl
