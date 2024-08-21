#include "ProgQuantRenderer.h"

#include "mmcore/param/BoolParam.h"
#include "mmcore/param/FilePathParam.h"
#include "mmcore/param/FloatParam.h"
#include "mmcore/param/IntParam.h"

#include <owl/common/math/box.h>
#include <owl/common/math/vec.h>

#include "PKDCreate.h"
#include "ProgQuantCreate.h"
#include "TreeletsCreate.h"

#include "CompErrorAnalysis.h"

#include "framestate.h"
#include "progquant.h"
#include "raygen.h"

#include <glad/gl.h>

#include <tbb/parallel_for.h>

#include <cuda_runtime.h>

namespace megamol::optix_owl {
extern "C" const unsigned char progquantPrograms_ptx[];

ProgQuantRenderer::ProgQuantRenderer() : threshold_slot_("threshold", "") {
    threshold_slot_ << new core::param::IntParam(2048, 16, 2048);
    MakeSlotAvailable(&threshold_slot_);
}

ProgQuantRenderer::~ProgQuantRenderer() {
    this->Release();
}

bool ProgQuantRenderer::create() {
    auto const ret = BaseRenderer::create();
    pkd_module_ = owlModuleCreate(ctx_, reinterpret_cast<const char*>(progquantPrograms_ptx));

    OWLVarDecl treeletsVars[] = {{"treeletBuffer", OWL_BUFPTR, OWL_OFFSETOF(device::ProgQuantGeomData, treeletBuffer)},
        {"particleBuffer", OWL_BUFPTR, OWL_OFFSETOF(device::ProgQuantGeomData, particleBuffer)},
        {"particleRadius", OWL_FLOAT, OWL_OFFSETOF(device::ProgQuantGeomData, particleRadius)},
        {/* sentinel to mark end of list */}};

    OWLGeomType treeletsType =
        owlGeomTypeCreate(ctx_, OWL_GEOMETRY_USER, sizeof(device::ProgQuantGeomData), treeletsVars, -1);
    owlGeomTypeSetBoundsProg(treeletsType, pkd_module_, "progquant_bounds");
    owlGeomTypeSetIntersectProg(treeletsType, 0, pkd_module_, "progquant_intersect");
    owlGeomTypeSetClosestHit(treeletsType, 0, pkd_module_, "progquant_ch");

    geom_ = owlGeomCreate(ctx_, treeletsType);

    return ret;
}

void ProgQuantRenderer::release() {
    owlBufferDestroy(treeletBuffer_);
    owlModuleRelease(pkd_module_);
    owlGeomRelease(geom_);

    BaseRenderer::release();
}


bool ProgQuantRenderer::assertData(geocalls::MultiParticleDataCall const& call) {
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

    auto const add_cond = [](device::box3f const& bounds) -> bool {
        constexpr auto const spatial_threshold = 16.f;
        auto const span = bounds.span();
        return span.x >= spatial_threshold || span.y >= spatial_threshold || span.z >= spatial_threshold;
    };

    auto const treelets = prePartition_inPlace(
        particles_, threshold_slot_.Param<core::param::IntParam>()->Value(), global_radius, add_cond);

    tbb::parallel_for(std::size_t(0), treelets.size(), [&](std::size_t treeletID) {
        makePKD(particles_, treelets[treeletID].begin, treelets[treeletID].end, treelets[treeletID].bounds);
    });

    std::vector<device::ProgQuantParticle> comp_particles_(particles_.size());
    tbb::parallel_for((size_t) 0, treelets.size(), [&](size_t treeletID) {
        convert_blets(0, treelets[treeletID].end - treelets[treeletID].begin,
            particles_.data() + treelets[treeletID].begin, comp_particles_.data() + treelets[treeletID].begin,
            global_radius, treelets[treeletID].bounds);
    });

    if (dump_debug_info_slot_.Param<core::param::BoolParam>()->Value()) {
        auto const output_path = debug_output_path_slot_.Param<core::param::FilePathParam>()->Value();

        auto diffs = std::make_shared<std::vector<vec3f>>();
        auto orgpos = std::make_shared<std::vector<vec3f>>();
        auto newpos = std::make_shared<std::vector<vec3f>>();
        diffs->resize(particles_.size());
        if (debug_rdf_slot_.Param<core::param::BoolParam>()->Value()) {
            orgpos->resize(particles_.size());
            newpos->resize(particles_.size());
        }

        tbb::parallel_for((size_t) 0, treelets.size(), [&](size_t treeletID) {
            reconstruct_blets(0, treelets[treeletID].end - treelets[treeletID].begin,
                particles_.data() + treelets[treeletID].begin, comp_particles_.data() + treelets[treeletID].begin,
                global_radius, treelets[treeletID].bounds,
                orgpos->empty() ? nullptr : orgpos->data() + treelets[treeletID].begin,
                newpos->empty() ? nullptr : newpos->data() + treelets[treeletID].begin,
                diffs->data() + treelets[treeletID].begin);
        });

        dump_analysis_data(output_path, orgpos, newpos, diffs, global_radius,
            debug_rdf_slot_.Param<core::param::BoolParam>()->Value());
    }

    core::utility::log::Log::DefaultLog.WriteInfo(
        "[ProgQuantRenderer] %d treelets for %d particles", treelets.size(), particles_.size());

    if (particleBuffer_)
        owlBufferDestroy(particleBuffer_);
    particleBuffer_ = owlDeviceBufferCreate(
        ctx_, OWL_USER_TYPE(device::ProgQuantParticle), comp_particles_.size(), comp_particles_.data());

    if (treeletBuffer_)
        owlBufferDestroy(treeletBuffer_);
    treeletBuffer_ = owlDeviceBufferCreate(ctx_, OWL_USER_TYPE(device::PKDlet), treelets.size(), treelets.data());

    owlGeomSetPrimCount(geom_, treelets.size());

    owlGeomSetBuffer(geom_, "particleBuffer", particleBuffer_);
    owlGeomSetBuffer(geom_, "treeletBuffer", treeletBuffer_);
    owlGeomSet1f(geom_, "particleRadius", global_radius);

    owlBuildPrograms(ctx_);

    OWLGroup ug = owlUserGeomGroupCreate(ctx_, 1, &geom_);
    owlGroupBuildAccel(ug);

    world_ = owlInstanceGroupCreate(ctx_, 1, &ug);

    owlGroupBuildAccel(world_);

    if (dump_debug_info_slot_.Param<core::param::BoolParam>()->Value()) {
        size_t memFinal = 0;
        size_t memPeak = 0;
        owlGroupGetAccelSize(ug, &memFinal, &memPeak);

        size_t comp_data_size =
            comp_particles_.size() * sizeof(device::ProgQuantParticle) + treelets.size() * sizeof(device::PKDlet);

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

bool ProgQuantRenderer::data_param_is_dirty() {
    return threshold_slot_.IsDirty();
}

void ProgQuantRenderer::data_param_reset_dirty() {
    threshold_slot_.ResetDirty();
}
} // namespace megamol::optix_owl
