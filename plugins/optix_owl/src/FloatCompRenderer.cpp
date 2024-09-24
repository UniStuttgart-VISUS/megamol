#include "FloatCompRenderer.h"

#include <memory>
#include <mutex>
#include <unordered_set>
#include <fstream>

#include "mmcore/param/BoolParam.h"
#include "mmcore/param/EnumParam.h"
#include "mmcore/param/FloatParam.h"
#include "mmcore/param/IntParam.h"
#include "mmcore/param/FilePathParam.h"

#include <owl/common/math/box.h>
#include <owl/common/math/vec.h>

#include "PKDCreate.h"
#include "TreeletsCreate.h"

#include "CompErrorAnalysis.h"

#include "FloatCompCreate.h"
#include "floatcomp.h"
#include "framestate.h"
#include "raygen.h"

#include <glad/gl.h>

#include <tbb/parallel_for.h>

#include <cuda_runtime.h>

namespace megamol::optix_owl {
extern "C" const unsigned char floatcompPrograms_ptx[];

FloatCompRenderer::FloatCompRenderer() : threshold_slot_("threshold", ""), type_slot_("type", "") {
    threshold_slot_ << new core::param::IntParam(2048, 16, 2048);
    MakeSlotAvailable(&threshold_slot_);

    auto ep = new core::param::EnumParam(static_cast<int>(device::FloatCompType::E5M15));
    ep->SetTypePair(static_cast<int>(device::FloatCompType::E5M15), "E5M15");
    ep->SetTypePair(static_cast<int>(device::FloatCompType::E5M15D), "E5M15D");
    type_slot_ << ep;
    MakeSlotAvailable(&type_slot_);
}

FloatCompRenderer::~FloatCompRenderer() {
    this->Release();
}

bool FloatCompRenderer::create() {
    auto const ret = BaseRenderer::create();
    pkd_module_ = owlModuleCreate(ctx_, reinterpret_cast<const char*>(floatcompPrograms_ptx));

    OWLVarDecl treeletsVars[] = {{"treeletBuffer", OWL_BUFPTR, OWL_OFFSETOF(device::FloatCompGeomData, treeletBuffer)},
        {"particleBuffer", OWL_BUFPTR, OWL_OFFSETOF(device::FloatCompGeomData, particleBuffer)},
        {"particleRadius", OWL_FLOAT, OWL_OFFSETOF(device::FloatCompGeomData, particleRadius)},
        {"expXBuffer", OWL_BUFPTR, OWL_OFFSETOF(device::FloatCompGeomData, expXBuffer)},
        {"expYBuffer", OWL_BUFPTR, OWL_OFFSETOF(device::FloatCompGeomData, expYBuffer)},
        {"expZBuffer", OWL_BUFPTR, OWL_OFFSETOF(device::FloatCompGeomData, expZBuffer)},
        {"use_localtables", OWL_CHAR, OWL_OFFSETOF(device::FloatCompGeomData, use_localtables)},
        {/* sentinel to mark end of list */}};

    OWLGeomType treeletsTypeE5M15 =
        owlGeomTypeCreate(ctx_, OWL_GEOMETRY_USER, sizeof(device::FloatCompGeomData), treeletsVars, -1);
    owlGeomTypeSetBoundsProg(treeletsTypeE5M15, pkd_module_, "floatcomp_bounds");
    owlGeomTypeSetIntersectProg(treeletsTypeE5M15, 0, pkd_module_, "floatcomp_intersect_e5m15");
    owlGeomTypeSetClosestHit(treeletsTypeE5M15, 0, pkd_module_, "floatcomp_ch");

    geom_e5m15_ = owlGeomCreate(ctx_, treeletsTypeE5M15);

    OWLGeomType treeletsTypeE5M15D =
        owlGeomTypeCreate(ctx_, OWL_GEOMETRY_USER, sizeof(device::FloatCompGeomData), treeletsVars, -1);
    owlGeomTypeSetBoundsProg(treeletsTypeE5M15D, pkd_module_, "floatcomp_bounds");
    owlGeomTypeSetIntersectProg(treeletsTypeE5M15D, 0, pkd_module_, "floatcomp_intersect_e5m15d");
    owlGeomTypeSetClosestHit(treeletsTypeE5M15D, 0, pkd_module_, "floatcomp_ch");

    geom_e5m15d_ = owlGeomCreate(ctx_, treeletsTypeE5M15D);

    return ret;
}

void FloatCompRenderer::release() {
    owlBufferDestroy(treeletBuffer_);
    owlBufferDestroy(exp_x_buffer_);
    owlBufferDestroy(exp_y_buffer_);
    owlBufferDestroy(exp_z_buffer_);
    owlModuleRelease(pkd_module_);
    owlGeomRelease(geom_e5m15_);
    owlGeomRelease(geom_e5m15d_);

    BaseRenderer::release();
}


bool FloatCompRenderer::assertData(geocalls::MultiParticleDataCall const& call) {
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

    // 1 partion particles
    auto const treelets =
        prePartition_inPlace(particles_, threshold_slot_.Param<core::param::IntParam>()->Value(), global_radius);

    auto const selected_type = static_cast<device::FloatCompType>(type_slot_.Param<core::param::EnumParam>()->Value());

    // 2 make PKDs and quantize
    std::vector<device::FloatCompPKDlet> qtreelets(treelets.size());
    std::vector<device::QTParticle> qtparticles(particles_.size());

    std::vector<unsigned int> bucket_count(treelets.size());
    std::vector<unsigned int> global_histo(256, 0);
    std::mutex histo_guard;

    tbb::parallel_for((size_t) 0, treelets.size(), [&](size_t treeletID) {
        makePKD(particles_, treelets[treeletID].begin, treelets[treeletID].end, treelets[treeletID].bounds);

        auto const histo_t = quantizeTree2(
            selected_type, particles_.data(), treelets[treeletID], qtparticles.data(), qtreelets[treeletID]);

        {
            auto num_el = std::count_if(
                std::get<0>(histo_t).begin(), std::get<0>(histo_t).end(), [](auto const& val) { return val != 0; });
            num_el = std::max(num_el, std::count_if(std::get<1>(histo_t).begin(), std::get<1>(histo_t).end(),
                                          [](auto const& val) { return val != 0; }));
            num_el = std::max(num_el, std::count_if(std::get<2>(histo_t).begin(), std::get<2>(histo_t).end(),
                                          [](auto const& val) { return val != 0; }));
            bucket_count[treeletID] = num_el;
            std::lock_guard<std::mutex> histo_lock(histo_guard);
            std::transform(std::get<0>(histo_t).begin(), std::get<0>(histo_t).end(), global_histo.begin(),
                global_histo.begin(), [](auto vala, auto valb) { return vala + valb; });
            std::transform(std::get<1>(histo_t).begin(), std::get<1>(histo_t).end(), global_histo.begin(),
                global_histo.begin(), [](auto vala, auto valb) { return vala + valb; });
            std::transform(std::get<2>(histo_t).begin(), std::get<2>(histo_t).end(), global_histo.begin(),
                global_histo.begin(), [](auto vala, auto valb) { return vala + valb; });
        }
    });

    // 3 create exponent maps
    std::unordered_set<char> exponents_x;
    std::unordered_set<char> exponents_y;
    std::unordered_set<char> exponents_z;

    unsigned num_idx = 0;
    switch (selected_type) {
    case device::FloatCompType::E5M15:
        num_idx = static_cast<unsigned int>(std::pow(2, device::QTParticle_e5m15::exp));
        break;
    /*case device::FloatCompType::E4M16:
        num_idx = static_cast<unsigned int>(std::pow(2, device::QTParticle_e4m16::exp));
        break;*/
    case device::FloatCompType::E5M15D:
        num_idx = static_cast<unsigned int>(std::pow(2, device::QTParticle_e5m15d::exp));
        break;
    /*case device::FloatCompType::E4M16D:
        num_idx = static_cast<unsigned int>(std::pow(2, device::QTParticle_e4m16d::exp));
        break;*/
    default:
        /*num_idx = static_cast<unsigned int>(std::pow(2, device::QTParticle_e4m16::exp));
        break;*/
        throw std::runtime_error("unexpected FloatCompType");
    }

    create_exp_maps(qtparticles, exponents_x, exponents_y, exponents_z, num_idx);

    auto exp_vec_x = std::vector<char>(exponents_x.begin(), exponents_x.end());
    auto exp_vec_y = std::vector<char>(exponents_y.begin(), exponents_y.end());
    auto exp_vec_z = std::vector<char>(exponents_z.begin(), exponents_z.end());

    bool using_localtables = false;
    unsigned int qt_exp_overflow = 0;

    if (exponents_x.size() > num_idx || exponents_y.size() > num_idx || exponents_z.size() > num_idx) {
        using_localtables = true;

        exp_vec_x.resize(treelets.size() * num_idx);
        exp_vec_y.resize(treelets.size() * num_idx);
        exp_vec_z.resize(treelets.size() * num_idx);

        //use_localtables_[pl_idx] = 1;

        std::vector<char> overflows(treelets.size(), 0);

        tbb::parallel_for((size_t) 0, treelets.size(), [&](size_t treeletID) {
            unsigned int offset = treeletID * num_idx;
            auto const overflow = create_exp_maps(qtreelets[treeletID], qtparticles, exp_vec_x.data() + offset,
                exp_vec_y.data() + offset, exp_vec_z.data() + offset, num_idx);
            if (overflow) {
                overflows[treeletID] = 1;
            }
        });

        qt_exp_overflow = std::count(overflows.begin(), overflows.end(), 1);
    }

    // 4 convert to qlets
    std::shared_ptr<QTPBufferBase> qtpbuffer;
    switch (selected_type) {
    case device::FloatCompType::E5M15:
        qtpbuffer = std::make_shared<QTPBuffer_e5m15>(particles_.size());
        break;
    /*case device::FloatCompType::E4M16:
        qtpbuffer = std::make_shared<QTPBuffer_e4m16>(particles_.size());
        break;*/
    case device::FloatCompType::E5M15D:
        qtpbuffer = std::make_shared<QTPBuffer_e5m15d>(particles_.size());
        break;
    /*case device::FloatCompType::E4M16D:
        qtpbuffer = std::make_shared<QTPBuffer_e4m16d>(particles_.size());
        break;*/
    default:
        //qtpbuffer = std::make_shared<QTPBuffer_e4m16>(particles_.size());
        throw std::runtime_error("unexpected FloatCompType");
    }

    switch (selected_type) {
    case device::FloatCompType::E5M15: {
        tbb::parallel_for((size_t) 0, treelets.size(), [&](size_t treeletID) {
            unsigned int offset = 0;
            if (using_localtables) {
                offset = treeletID * num_idx;
            }
            convert_qlet<device::QTParticle_e5m15>(qtreelets[treeletID], qtparticles,
                std::dynamic_pointer_cast<QTPBuffer_e5m15>(qtpbuffer)->buffer, exp_vec_x.data() + offset,
                exp_vec_y.data() + offset, exp_vec_z.data() + offset);
        });
    } break;
    /*case device::FloatCompType::E4M16: {
        tbb::parallel_for((size_t) 0, treelets.size(), [&](size_t treeletID) {
            unsigned int offset = 0;
            if (use_localtables_[pl_idx] > 0) {
                offset = treeletID * num_idx;
            }
            convert_qlet<device::QTParticle_e4m16>(qtreelets[treeletID], qtparticles,
                std::dynamic_pointer_cast<QTPBuffer_e4m16>(qtpbuffer)->buffer, exp_vec_x.data() + offset,
                exp_vec_y.data() + offset, exp_vec_z.data() + offset);
        });
    } break;*/
    case device::FloatCompType::E5M15D: {
        tbb::parallel_for((size_t) 0, treelets.size(), [&](size_t treeletID) {
            unsigned int offset = 0;
            if (using_localtables) {
                offset = treeletID * num_idx;
            }
            convert_qlet_dep<device::QTParticle_e5m15d>(qtreelets[treeletID], qtparticles,
                std::dynamic_pointer_cast<QTPBuffer_e5m15d>(qtpbuffer)->buffer, exp_vec_x.data() + offset,
                exp_vec_y.data() + offset, exp_vec_z.data() + offset);
        });
    } break;
    /*case device::FloatCompType::E4M16D: {
        tbb::parallel_for((size_t) 0, treelets.size(), [&](size_t treeletID) {
            unsigned int offset = 0;
            if (use_localtables_[pl_idx] > 0) {
                offset = treeletID * num_idx;
            }
            convert_qlet_dep<device::QTParticle_e4m16d>(qtreelets[treeletID], qtparticles,
                std::dynamic_pointer_cast<QTPBuffer_e4m16d>(qtpbuffer)->buffer, exp_vec_x.data() + offset,
                exp_vec_y.data() + offset, exp_vec_z.data() + offset);
        });
    } break;*/
    default:
        std::cout << "Should not happen" << std::endl;
    }

    if (dump_debug_info_slot_.Param<core::param::BoolParam>()->Value()) {
        auto const output_path = debug_output_path_slot_.Param<core::param::FilePathParam>()->Value();

        auto diffs = std::make_shared<std::vector<vec3f>>();
        auto orgpos = std::make_shared<std::vector<vec3f>>();
        auto newpos = std::make_shared<std::vector<vec3f>>();
        diffs->reserve(particles_.size());
        if (debug_rdf_slot_.Param<core::param::BoolParam>()->Value()) {
            orgpos->reserve(particles_.size());
            newpos->reserve(particles_.size());
        }

        for (size_t i = 0; i < treelets.size(); ++i) {
            unsigned int offset = 0;

            if (using_localtables) {
                offset = i * num_idx;
            }

            auto const [diffs_t, orgpos_t, newpos_t] =
                unified_sub_print(selected_type, 0, qtreelets[i].basePos, particles_.data(), qtpbuffer, qtreelets[i],
                    exp_vec_x.data() + offset, exp_vec_y.data() + offset, exp_vec_z.data() + offset);

            diffs->insert(diffs->end(), diffs_t.begin(), diffs_t.end());
            if (debug_rdf_slot_.Param<core::param::BoolParam>()->Value()) {
                orgpos->insert(orgpos->end(), orgpos_t.begin(), orgpos_t.end());
                newpos->insert(newpos->end(), newpos_t.begin(), newpos_t.end());
            }
        }

        dump_analysis_data(output_path, orgpos, newpos, diffs, global_radius,
            debug_rdf_slot_.Param<core::param::BoolParam>()->Value(), total_bounds);

        {
            if (using_localtables) {
                auto f = std::ofstream(output_path / "localtables.txt");
                f << "localtables\n";
                f.close();
            }
            if (qt_exp_overflow > 0) {
                auto f = std::ofstream(output_path / "overflow.txt");
                f << qt_exp_overflow << "\n";
                f.close();
            }
        }
    }

    core::utility::log::Log::DefaultLog.WriteInfo(
        "[FloatCompRenderer] %d treelets for %d particles", treelets.size(), particles_.size());

    if (particleBuffer_)
        owlBufferDestroy(particleBuffer_);
    switch (selected_type) {
    case device::FloatCompType::E5M15: {
        auto const buf = std::dynamic_pointer_cast<QTPBuffer_e5m15>(qtpbuffer);
        particleBuffer_ = owlDeviceBufferCreate(
            ctx_, OWL_USER_TYPE(device::QTParticle_e5m15), buf->buffer.size(), buf->buffer.data());
    } break;
    case device::FloatCompType::E5M15D: {
        auto const buf = std::dynamic_pointer_cast<QTPBuffer_e5m15d>(qtpbuffer);
        particleBuffer_ = owlDeviceBufferCreate(
            ctx_, OWL_USER_TYPE(device::QTParticle_e5m15d), buf->buffer.size(), buf->buffer.data());
    } break;
    default:
        throw std::runtime_error("unexpected FloatCompType");
    }

    if (exp_x_buffer_)
        owlBufferDestroy(exp_x_buffer_);
    if (exp_y_buffer_)
        owlBufferDestroy(exp_y_buffer_);
    if (exp_z_buffer_)
        owlBufferDestroy(exp_z_buffer_);

    exp_x_buffer_ = owlDeviceBufferCreate(ctx_, OWL_CHAR, exp_vec_x.size(), exp_vec_x.data());
    exp_y_buffer_ = owlDeviceBufferCreate(ctx_, OWL_CHAR, exp_vec_y.size(), exp_vec_y.data());
    exp_z_buffer_ = owlDeviceBufferCreate(ctx_, OWL_CHAR, exp_vec_z.size(), exp_vec_z.data());

    if (treeletBuffer_)
        owlBufferDestroy(treeletBuffer_);
    treeletBuffer_ =
        owlDeviceBufferCreate(ctx_, OWL_USER_TYPE(device::FloatCompPKDlet), qtreelets.size(), qtreelets.data());

    OWLGeom geom_;
    switch (selected_type) {
    case device::FloatCompType::E5M15: {
        geom_ = geom_e5m15_;
    } break;
    case device::FloatCompType::E5M15D: {
        geom_ = geom_e5m15d_;
    } break;
    default:
        throw std::runtime_error("unexpected FloatCompType");
    }

    owlGeomSetPrimCount(geom_, qtreelets.size());

    owlGeomSetBuffer(geom_, "particleBuffer", particleBuffer_);
    owlGeomSetBuffer(geom_, "treeletBuffer", treeletBuffer_);
    owlGeomSetBuffer(geom_, "expXBuffer", exp_x_buffer_);
    owlGeomSetBuffer(geom_, "expYBuffer", exp_y_buffer_);
    owlGeomSetBuffer(geom_, "expZBuffer", exp_z_buffer_);
    owlGeomSet1f(geom_, "particleRadius", global_radius);
    owlGeomSet1c(geom_, "use_localtables", using_localtables);

    owlBuildPrograms(ctx_);

    OWLGroup ug = owlUserGeomGroupCreate(ctx_, 1, &geom_);
    owlGroupBuildAccel(ug);

    world_ = owlInstanceGroupCreate(ctx_, 1, &ug);

    owlGroupBuildAccel(world_);

    if (dump_debug_info_slot_.Param<core::param::BoolParam>()->Value()) {
        size_t memFinal = 0;
        size_t memPeak = 0;
        owlGroupGetAccelSize(ug, &memFinal, &memPeak);

        size_t dtype_size = 0;
        switch (selected_type) {
        case device::FloatCompType::E5M15: {
            dtype_size = sizeof(device::QTParticle_e5m15);
        } break;
        case device::FloatCompType::E5M15D: {
            dtype_size = sizeof(device::QTParticle_e5m15);
        } break;
        default:
            throw std::runtime_error("unexpected FloatCompType");
        }

        size_t comp_data_size = particles_.size() * dtype_size + qtreelets.size() * sizeof(device::FloatCompPKDlet);

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

bool FloatCompRenderer::data_param_is_dirty() {
    return threshold_slot_.IsDirty() || type_slot_.IsDirty();
}

void FloatCompRenderer::data_param_reset_dirty() {
    threshold_slot_.ResetDirty();
    type_slot_.ResetDirty();
}
} // namespace megamol::optix_owl
