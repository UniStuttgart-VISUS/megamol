#include "TreeletsRenderer.h"

#include <fstream>

#include "mmcore/param/BoolParam.h"
#include "mmcore/param/FloatParam.h"
#include "mmcore/param/IntParam.h"
#include "mmcore/param/FilePathParam.h"

#include <owl/common/math/box.h>
#include <owl/common/math/vec.h>

#include "PKDCreate.h"
#include "TreeletsCreate.h"

#include "framestate.h"
#include "raygen.h"
#include "treelets.h"

#include <glad/gl.h>

#include <tbb/parallel_for.h>

#include <cuda_runtime.h>

namespace megamol::optix_owl {
extern "C" const unsigned char treeletsPrograms_ptx[];

TreeletsRenderer::TreeletsRenderer() : threshold_slot_("threshold", "") {
    threshold_slot_ << new core::param::IntParam(2048, 16, 2048);
    MakeSlotAvailable(&threshold_slot_);
}

TreeletsRenderer::~TreeletsRenderer() {
    this->Release();
}

bool TreeletsRenderer::create() {
    auto const ret = BaseRenderer::create();
    pkd_module_ = owlModuleCreate(ctx_, reinterpret_cast<const char*>(treeletsPrograms_ptx));

    OWLVarDecl treeletsVars[] = {{"treeletBuffer", OWL_BUFPTR, OWL_OFFSETOF(device::TreeletsGeomData, treeletBuffer)},
        {"particleBuffer", OWL_BUFPTR, OWL_OFFSETOF(device::TreeletsGeomData, particleBuffer)},
        {"particleRadius", OWL_FLOAT, OWL_OFFSETOF(device::TreeletsGeomData, particleRadius)},
        {/* sentinel to mark end of list */}};

    OWLGeomType treeletsType =
        owlGeomTypeCreate(ctx_, OWL_GEOMETRY_USER, sizeof(device::TreeletsGeomData), treeletsVars, -1);
    owlGeomTypeSetBoundsProg(treeletsType, pkd_module_, "treelets_bounds");
    owlGeomTypeSetIntersectProg(treeletsType, 0, pkd_module_, "treelets_intersect");
    owlGeomTypeSetClosestHit(treeletsType, 0, pkd_module_, "treelets_ch");

    geom_ = owlGeomCreate(ctx_, treeletsType);

    return ret;
}

void TreeletsRenderer::release() {
    owlBufferDestroy(treeletBuffer_);
    owlModuleRelease(pkd_module_);
    owlGeomRelease(geom_);

    BaseRenderer::release();
}

bool TreeletsRenderer::assertData(geocalls::MultiParticleDataCall const& call) {
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

    auto const treelets =
        prePartition_inPlace(particles_, threshold_slot_.Param<core::param::IntParam>()->Value(), global_radius);

    tbb::parallel_for(std::size_t(0), treelets.size(), [&](std::size_t treeletID) {
        makePKD(particles_, treelets[treeletID].begin, treelets[treeletID].end, treelets[treeletID].bounds);
    });

    core::utility::log::Log::DefaultLog.WriteInfo(
        "[TreeletsRenderer] %d treelets for %d particles", treelets.size(), particles_.size());

    if (particleBuffer_)
        owlBufferDestroy(particleBuffer_);
    particleBuffer_ = owlDeviceBufferCreate(ctx_, OWL_USER_TYPE(device::Particle), particles_.size(), particles_.data());

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

        size_t comp_data_size = particles_.size() * sizeof(device::Particle) + treelets.size() * sizeof(device::PKDlet);

        auto const output_path = debug_output_path_slot_.Param<core::param::FilePathParam>()->Value();
        auto of = std::ofstream(output_path / "size.csv");
        of << "BVHFinalSize[B],BVHPeakSize[B],CompDataSize[B]\n";
        of << memFinal << "," << memPeak << "," << comp_data_size << "\n";
        of.close();
    }

    return true;
}

bool TreeletsRenderer::data_param_is_dirty() {
    return threshold_slot_.IsDirty();
}

void TreeletsRenderer::data_param_reset_dirty() {
    threshold_slot_.ResetDirty();
}
} // namespace megamol::optix_owl
