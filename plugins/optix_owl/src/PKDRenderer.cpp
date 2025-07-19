#include "PKDRenderer.h"

#include "mmcore/param/FloatParam.h"
#include "mmcore/param/IntParam.h"

#include <owl/common/math/box.h>
#include <owl/common/math/vec.h>

#include "PKDCreate.h"

#include "framestate.h"
#include "pkd.h"
#include "raygen.h"

#include <glad/gl.h>

#include <cuda_runtime.h>

namespace megamol::optix_owl {
extern "C" const unsigned char pkdPrograms_ptx[];

PKDRenderer::PKDRenderer() {}

PKDRenderer::~PKDRenderer() {
    this->Release();
}

bool PKDRenderer::create() {
    auto const ret = BaseRenderer::create();

    pkd_module_ = owlModuleCreate(ctx_, reinterpret_cast<const char*>(pkdPrograms_ptx));

    OWLVarDecl allPKDVars[] = {{"particleBuffer", OWL_BUFPTR, OWL_OFFSETOF(device::PKDGeomData, particleBuffer)},
        {"particleRadius", OWL_FLOAT, OWL_OFFSETOF(device::PKDGeomData, particleRadius)},
        {"particleCount", OWL_INT, OWL_OFFSETOF(device::PKDGeomData, particleCount)},
        {"bounds.lower", OWL_FLOAT3, OWL_OFFSETOF(device::PKDGeomData, worldBounds.lower)},
        {"bounds.upper", OWL_FLOAT3, OWL_OFFSETOF(device::PKDGeomData, worldBounds.upper)},
        {/* sentinel to mark end of list */}};

    OWLGeomType allPKDType = owlGeomTypeCreate(ctx_, OWL_GEOMETRY_USER, sizeof(device::PKDGeomData), allPKDVars, -1);
    owlGeomTypeSetBoundsProg(allPKDType, pkd_module_, "pkd_bounds");
    owlGeomTypeSetIntersectProg(allPKDType, 0, pkd_module_, "pkd_intersect");
    owlGeomTypeSetClosestHit(allPKDType, 0, pkd_module_, "pkd_ch");

    geom_ = owlGeomCreate(ctx_, allPKDType);

    return ret;
}

void PKDRenderer::release() {
    owlModuleRelease(pkd_module_);
    owlGeomRelease(geom_);
    BaseRenderer::release();
}

bool PKDRenderer::assertData(geocalls::MultiParticleDataCall const& call) {
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

    makePKD(particles_, total_bounds);
    if (particleBuffer_)
        owlBufferDestroy(particleBuffer_);
    particleBuffer_ =
        owlDeviceBufferCreate(ctx_, OWL_USER_TYPE(device::Particle), particles_.size(), particles_.data());

    core::utility::log::Log::DefaultLog.WriteInfo("[PKDRenderer] Rendering %d particles", particles_.size());

    owlGeomSetPrimCount(geom_, 1);

    owlGeomSetBuffer(geom_, "particleBuffer", particleBuffer_);
    owlGeomSet1f(geom_, "particleRadius", global_radius);
    owlGeomSet1i(geom_, "particleCount", (int) particles_.size());
    owlGeomSet3f(geom_, "bounds.lower", (const owl3f&) total_bounds.lower);
    owlGeomSet3f(geom_, "bounds.upper", (const owl3f&) total_bounds.upper);

    owlBuildPrograms(ctx_);

    OWLGroup ug = owlUserGeomGroupCreate(ctx_, 1, &geom_);
    owlGroupBuildAccel(ug);

    world_ = owlInstanceGroupCreate(ctx_, 1, &ug);

    owlGroupBuildAccel(world_);

    return true;
}

bool PKDRenderer::data_param_is_dirty() {
    return false;
}

void PKDRenderer::data_param_reset_dirty() {}
} // namespace megamol::optix_owl
