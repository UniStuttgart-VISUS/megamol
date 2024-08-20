#include "BVHRenderer.h"

#include <fstream>

#include "mmcore/param/FloatParam.h"
#include "mmcore/param/IntParam.h"
#include "mmcore/param/BoolParam.h"
#include "mmcore/param/FilePathParam.h"

#include <owl/common/math/box.h>
#include <owl/common/math/vec.h>

#include "PKDCreate.h"

#include "framestate.h"
#include "bvh.h"
#include "raygen.h"

#include <glad/gl.h>

#include <cuda_runtime.h>

namespace megamol::optix_owl {
extern "C" const unsigned char bvhPrograms_ptx[];

BVHRenderer::BVHRenderer() {}

BVHRenderer::~BVHRenderer() {
    this->Release();
}

bool BVHRenderer::create() {
    auto const ret = BaseRenderer::create();

    pkd_module_ = owlModuleCreate(ctx_, reinterpret_cast<const char*>(bvhPrograms_ptx));

    OWLVarDecl allPKDVars[] = {{"particleBuffer", OWL_BUFPTR, OWL_OFFSETOF(device::BVHGeomData, particleBuffer)},
        {"particleRadius", OWL_FLOAT, OWL_OFFSETOF(device::BVHGeomData, particleRadius)},
        {/* sentinel to mark end of list */}};

    OWLGeomType allPKDType = owlGeomTypeCreate(ctx_, OWL_GEOMETRY_USER, sizeof(device::BVHGeomData), allPKDVars, -1);
    owlGeomTypeSetBoundsProg(allPKDType, pkd_module_, "bvh_bounds");
    owlGeomTypeSetIntersectProg(allPKDType, 0, pkd_module_, "bvh_intersect");
    owlGeomTypeSetClosestHit(allPKDType, 0, pkd_module_, "bvh_ch");

    geom_ = owlGeomCreate(ctx_, allPKDType);

    return ret;
}

void BVHRenderer::release() {
    owlModuleRelease(pkd_module_);
    owlGeomRelease(geom_);
    BaseRenderer::release();
}

bool BVHRenderer::assertData(geocalls::MultiParticleDataCall const& call) {
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

    if (particleBuffer_)
        owlBufferDestroy(particleBuffer_);
    particleBuffer_ =
        owlDeviceBufferCreate(ctx_, OWL_USER_TYPE(device::Particle), particles_.size(), particles_.data());

    core::utility::log::Log::DefaultLog.WriteInfo("[BVHRenderer] Rendering %d particles", particles_.size());

    owlGeomSetPrimCount(geom_, particles_.size());

    owlGeomSetBuffer(geom_, "particleBuffer", particleBuffer_);
    owlGeomSet1f(geom_, "particleRadius", global_radius);

    owlBuildPrograms(ctx_);

    OWLGroup ug = owlUserGeomGroupCreate(ctx_, 1, &geom_);
    owlGroupBuildAccel(ug);

    world_ = owlInstanceGroupCreate(ctx_, 1, &ug);

    owlGroupBuildAccel(world_);

    if (dump_debug_info_slot_.Param<core::param::BoolParam>()->Value()) {
        size_t memFinal = 0;
        size_t memPeak = 0;
        owlGroupGetAccelSize(world_, &memFinal, &memPeak);

        size_t comp_data_size = particles_.size() * sizeof(device::Particle);

        auto const output_path = debug_output_path_slot_.Param<core::param::FilePathParam>()->Value();
        auto of = std::ofstream(output_path / "size.csv");
        of << "BVHFinalSize[B],BVHPeakSize[B],CompDataSize[B]\n";
        of << memFinal << "," << memPeak << "," << comp_data_size << "\n";
        of.close();
    }

    return true;
}

bool BVHRenderer::data_param_is_dirty() {
    return false;
}

void BVHRenderer::data_param_reset_dirty() {}
} // namespace megamol::optix_owl
