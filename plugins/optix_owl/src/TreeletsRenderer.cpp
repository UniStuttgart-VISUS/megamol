#include "TreeletsRenderer.h"

#include "mmcore/param/BoolParam.h"
#include "mmcore/param/FloatParam.h"
#include "mmcore/param/IntParam.h"

#include <owl/common/math/box.h>
#include <owl/common/math/vec.h>

#include "PKDCreate.h"

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

std::size_t sort_partition(
    std::vector<device::Particle>& particles, std::size_t begin, std::size_t end, box3f bounds, int& splitDim) {
    // -------------------------------------------------------
    // determine split pos
    // -------------------------------------------------------
    splitDim = arg_max(bounds.span());
    float splitPos = bounds.center()[splitDim];

    // -------------------------------------------------------
    // now partition ...
    // -------------------------------------------------------
    std::size_t mid = begin;
    std::size_t l = begin, r = (end - 1);
    // quicksort partition:
    while (l <= r) {
        while (l < r && particles[l].pos[splitDim] < splitPos)
            ++l;
        while (l < r && particles[r].pos[splitDim] >= splitPos)
            --r;
        if (l == r) {
            mid = l;
            break;
        }

        std::swap(particles[l], particles[r]);
    }

    // catch-all for extreme cases where all particles are on the same
    // spot, and can't be split:
    if (mid == begin || mid == end)
        mid = (begin + end) / 2;

    return mid;
}


/*! todo: make this a cmd-line parameter, so we can run scripts to
  measure perf impact per size (already made it a static, so we
  can set it from main() before class is created */
//int TreeletParticles::maxTreeletSize = 1000;

template<typename MakeLeafLambda>
void partitionRecursively(
    std::vector<device::Particle>& particles, std::size_t begin, std::size_t end, const MakeLeafLambda& makeLeaf) {
    if (makeLeaf(begin, end, false))
        // could make into a leaf, done.
        return;

    // -------------------------------------------------------
    // parallel bounding box computation
    // -------------------------------------------------------
    box3f bounds;
    std::mutex boundsMutex;
    parallel_for_blocked(begin, end, 32 * 1024, [&](size_t blockBegin, size_t blockEnd) {
        box3f blockBounds;
        for (size_t i = blockBegin; i < blockEnd; i++)
            blockBounds.extend(particles[i].pos);
        std::lock_guard<std::mutex> lock(boundsMutex);
        bounds.extend(blockBounds);
    });

    int splitDim;
    auto mid = sort_partition(particles, begin, end, bounds, splitDim);

    // -------------------------------------------------------
    // and recurse ...
    // -------------------------------------------------------
    tbb::parallel_for(0, 2, [&](int side) {
        if (side)
            partitionRecursively(particles, begin, mid, makeLeaf);
        else
            partitionRecursively(particles, mid, end, makeLeaf);
    });
}

std::vector<PKDlet> prePartition_inPlace(std::vector<device::Particle>& particles, std::size_t maxSize, float radius) {
    std::mutex resultMutex;
    std::vector<PKDlet> result;

    partitionRecursively(particles, 0ULL, particles.size(), [&](std::size_t begin, std::size_t end, bool force) {
        /*bool makeLeaf() :*/
        const std::size_t size = end - begin;
        if (size > maxSize && !force)
            return false;

        PKDlet treelet;
        treelet.begin = begin;
        treelet.end = end;
        treelet.bounds = box3f();
        for (std::size_t i = begin; i < end; i++) {
            treelet.bounds.extend(particles[i].pos - radius);
            treelet.bounds.extend(particles[i].pos + radius);
        }

        std::lock_guard<std::mutex> lock(resultMutex);
        result.push_back(treelet);
        return true;
    });

    return std::move(result);
}


bool TreeletsRenderer::assertData(geocalls::MultiParticleDataCall const& call) {
    auto const pl_count = call.GetParticleListCount();

    particles_.clear();
    owl::common::box3f total_bounds;
    auto const global_radius = radius_slot_.Param<core::param::FloatParam>()->Value();

    for (unsigned int pl_idx = 0; pl_idx < pl_count; ++pl_idx) {
        auto const& particles = call.AccessParticles(pl_idx);

        auto const p_count = particles.GetCount();
        if (p_count == 0)
            continue;
        /*if (particles.GetVertexDataType() == geocalls::SimpleSphericalParticles::VERTDATA_FLOAT_XYZR)
            continue;*/

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
    particleBuffer_ = owlDeviceBufferCreate(ctx_, OWL_USER_TYPE(particles_[0]), particles_.size(), particles_.data());

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

    return true;
}

bool TreeletsRenderer::data_param_is_dirty() {
    return threshold_slot_.IsDirty();
}

void TreeletsRenderer::data_param_reset_dirty() {
    threshold_slot_.ResetDirty();
}
} // namespace megamol::optix_owl
