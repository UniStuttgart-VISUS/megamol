#include "GridCompRenderer.h"

#include "mmcore/param/BoolParam.h"
#include "mmcore/param/FloatParam.h"
#include "mmcore/param/IntParam.h"
#include "mmcore/param/FilePathParam.h"

#include <owl/common/math/box.h>
#include <owl/common/math/vec.h>

#include "PKDCreate.h"
#include "TreeletsCreate.h"
#include "GridCompCreate.h"

#include "CompErrorAnalysis.h"

#include "framestate.h"
#include "raygen.h"
#include "gridcomp.h"

#include <glad/gl.h>

#include <tbb/parallel_for.h>

#include <cuda_runtime.h>

namespace megamol::optix_owl {
extern "C" const unsigned char gridcompPrograms_ptx[];

GridCompRenderer::GridCompRenderer() : threshold_slot_("threshold", "") {
    threshold_slot_ << new core::param::IntParam(2048, 16, 2048);
    MakeSlotAvailable(&threshold_slot_);
}

GridCompRenderer::~GridCompRenderer() {
    this->Release();
}

bool GridCompRenderer::create() {
    auto const ret = BaseRenderer::create();
    pkd_module_ = owlModuleCreate(ctx_, reinterpret_cast<const char*>(gridcompPrograms_ptx));

    OWLVarDecl treeletsVars[] = {{"treeletBuffer", OWL_BUFPTR, OWL_OFFSETOF(device::GridCompGeomData, treeletBuffer)},
        {"particleBuffer", OWL_BUFPTR, OWL_OFFSETOF(device::GridCompGeomData, particleBuffer)},
        {"particleRadius", OWL_FLOAT, OWL_OFFSETOF(device::GridCompGeomData, particleRadius)},
        {/* sentinel to mark end of list */}};

    OWLGeomType treeletsType =
        owlGeomTypeCreate(ctx_, OWL_GEOMETRY_USER, sizeof(device::GridCompGeomData), treeletsVars, -1);
    owlGeomTypeSetBoundsProg(treeletsType, pkd_module_, "gridcomp_bounds");
    owlGeomTypeSetIntersectProg(treeletsType, 0, pkd_module_, "gridcomp_intersect");
    owlGeomTypeSetClosestHit(treeletsType, 0, pkd_module_, "gridcomp_ch");

    geom_ = owlGeomCreate(ctx_, treeletsType);

    return ret;
}

void GridCompRenderer::release() {
    owlBufferDestroy(treeletBuffer_);
    owlModuleRelease(pkd_module_);
    owlGeomRelease(geom_);

    BaseRenderer::release();
}


bool GridCompRenderer::assertData(geocalls::MultiParticleDataCall const& call) {
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

    auto const cells = gridify(particles_, total_bounds.lower, total_bounds.upper);

    std::vector<GridCompParticle> s_particles(particles_.size());
    std::vector<GridCompPKDlet> s_treelets;

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

    for (auto const& c : cells) {
        auto const box = extendBounds(particles_, c.first, c.second, global_radius);
        auto tmp_t = partition_data(particles_, c.first, c.second, box.lower,
            threshold_slot_.Param<core::param::IntParam>()->Value(), global_radius);
        for (auto& el : tmp_t) {
            el.lower = box.lower;
        }
        /*std::transform(data.begin() + c.first, data.begin() + c.second, qparticles.begin() + c.first,
            [&box](auto const& p) { return encode_coord(p.pos - box.lower, glm::vec3(), glm::vec3()); });*/
        //std::vector<device::SPKDParticle> tmp_p(c.second - c.first);
        tbb::parallel_for((size_t) 0, tmp_t.size(), [&](size_t tID) {
            auto const& el = tmp_t[tID];
            std::transform(particles_.begin() + el.begin, particles_.begin() + el.end, s_particles.begin() + el.begin,
                [&el, &box](auto const& p) {
                    auto const qp = encode_coord(p.pos - box.lower, vec3f(), vec3f());
                    device::GridCompParticle sp;
                    byte_cast bc;
                    bc.ui = 0;
                    bc.ui = qp.x;
                    sp.x = bc.parts.a;
                    auto fit_x = std::find(el.sx, el.sx + device::spkd_array_size, bc.parts.b);
                    if (fit_x == el.sx + device::spkd_array_size) {
                        throw std::runtime_error("did not find propper index");
                    }
                    sp.sx_idx = std::distance(el.sx, fit_x);
                    bc.ui = qp.y;
                    sp.y = bc.parts.a;
                    auto fit_y = std::find(el.sy, el.sy + device::spkd_array_size, bc.parts.b);
                    if (fit_y == el.sy + device::spkd_array_size) {
                        throw std::runtime_error("did not find propper index");
                    }
                    sp.sy_idx = std::distance(el.sy, fit_y);
                    bc.ui = qp.z;
                    sp.z = bc.parts.a;
                    auto fit_z = std::find(el.sz, el.sz + device::spkd_array_size, bc.parts.b);
                    if (fit_z == el.sz + device::spkd_array_size) {
                        throw std::runtime_error("did not find propper index");
                    }
                    sp.sz_idx = std::distance(el.sz, fit_z);
                    return sp;
                });
        });
        /*for (auto const& el : tmp_t) {
            std::transform(particles_.begin() + el.begin, particles_.begin() + el.end, s_particles.begin() + el.begin,
                [&el, &box](auto const& p) {
                    auto const qp = encode_coord(p.pos - box.lower, vec3f(), vec3f());
                    device::GridCompParticle sp;
                    byte_cast bc;
                    bc.ui = 0;
                    bc.ui = qp.x;
                    sp.x = bc.parts.a;
                    auto fit_x = std::find(el.sx, el.sx + device::spkd_array_size, bc.parts.b);
                    if (fit_x == el.sx + device::spkd_array_size) {
                        throw std::runtime_error("did not find propper index");
                    }
                    sp.sx_idx = std::distance(el.sx, fit_x);
                    bc.ui = qp.y;
                    sp.y = bc.parts.a;
                    auto fit_y = std::find(el.sy, el.sy + device::spkd_array_size, bc.parts.b);
                    if (fit_y == el.sy + device::spkd_array_size) {
                        throw std::runtime_error("did not find propper index");
                    }
                    sp.sy_idx = std::distance(el.sy, fit_y);
                    bc.ui = qp.z;
                    sp.z = bc.parts.a;
                    auto fit_z = std::find(el.sz, el.sz + device::spkd_array_size, bc.parts.b);
                    if (fit_z == el.sz + device::spkd_array_size) {
                        throw std::runtime_error("did not find propper index");
                    }
                    sp.sz_idx = std::distance(el.sz, fit_z);
                    return sp;
                });*/
        //el.bounds = extendBounds(data, el.begin, el.end, particles.GetGlobalRadius());

        if (dump_debug_info_slot_.Param<core::param::BoolParam>()->Value()) {
            auto const [tmp_d, tmp_op, tmp_s] = compute_diffs(tmp_t, s_particles, particles_, c.first, c.second);
            diffs->insert(diffs->end(), tmp_d.begin(), tmp_d.end());
            if (debug_rdf_slot_.Param<core::param::BoolParam>()->Value()) {
                orgpos->insert(orgpos->end(), tmp_op.begin(), tmp_op.end());
                spos->insert(spos->end(), tmp_s.begin(), tmp_s.end());
            }
        }

        // make PKD
        tbb::parallel_for(
            (size_t) 0, tmp_t.size(), [&](size_t treeletID) { makePKD(s_particles, tmp_t[treeletID], 0); });
        s_treelets.insert(s_treelets.end(), tmp_t.begin(), tmp_t.end());
        //s_particles.insert(s_particles.end(), tmp_p.begin(), tmp_p.end());
    }

    if (dump_debug_info_slot_.Param<core::param::BoolParam>()->Value()) {
        auto const output_path = debug_output_path_slot_.Param<core::param::FilePathParam>()->Value();
        dump_analysis_data(
            output_path, orgpos, spos, diffs, global_radius, debug_rdf_slot_.Param<core::param::BoolParam>()->Value());
    }

    core::utility::log::Log::DefaultLog.WriteInfo(
        "[GridCompRenderer] %d treelets for %d particles", s_treelets.size(), s_particles.size());

    if (particleBuffer_)
        owlBufferDestroy(particleBuffer_);
    particleBuffer_ = owlDeviceBufferCreate(ctx_, OWL_USER_TYPE(device::GridCompParticle), s_particles.size(), s_particles.data());

    if (treeletBuffer_)
        owlBufferDestroy(treeletBuffer_);
    treeletBuffer_ = owlDeviceBufferCreate(ctx_, OWL_USER_TYPE(device::GridCompPKDlet), s_treelets.size(), s_treelets.data());

    owlGeomSetPrimCount(geom_, s_treelets.size());

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
        owlGroupGetAccelSize(world_, &memFinal, &memPeak);

        size_t comp_data_size = s_particles.size() * sizeof(device::GridCompParticle) + s_treelets.size() * sizeof(device::GridCompPKDlet);

        auto const output_path = debug_output_path_slot_.Param<core::param::FilePathParam>()->Value();
        auto of = std::ofstream(output_path / "size.csv");
        of << "BVHFinalSize[B],BVHPeakSize[B],CompDataSize[B]\n";
        of << memFinal << "," << memPeak << "," << comp_data_size << "\n";
        of.close();
    }

    return true;
}

bool GridCompRenderer::data_param_is_dirty() {
    return threshold_slot_.IsDirty() || dump_debug_info_slot_.IsDirty();
}

void GridCompRenderer::data_param_reset_dirty() {
    threshold_slot_.ResetDirty();
    dump_debug_info_slot_.ResetDirty();
}
} // namespace megamol::optix_owl
