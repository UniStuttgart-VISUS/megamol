/*
 * ParticleSortFixHack.h
 *
 * Copyright (C) 2015 by S. Grottel
 * Alle Rechte vorbehalten.
 */
#include "ParticleSortFixHack.h"
#include "mmcore/param/BoolParam.h"
#include "mmcore/param/FloatParam.h"
#include "mmcore/utility/log/Log.h"
#include "mmcore/utility/sys/ConsoleProgressBar.h"
#include "vislib/sys/Thread.h"
#include <algorithm>
#include <cassert>
#include <cstdint>

using namespace megamol;


/*
 * datatools::ParticleSortFixHack::ParticleSortFixHack
 */
datatools::ParticleSortFixHack::ParticleSortFixHack(void)
        : AbstractParticleManipulator("outData", "indata")
        , data()
        , inDataTime(-1)
        , outDataTime(0)
        , ids()
        , inDataHash(-1)
        , outDataHash(0) {}


/*
 * datatools::ParticleSortFixHack::~ParticleSortFixHack
 */
datatools::ParticleSortFixHack::~ParticleSortFixHack(void) {
    this->Release();
}


/*
 * datatools::ParticleSortFixHack::manipulateData
 */
bool datatools::ParticleSortFixHack::manipulateData(
    geocalls::MultiParticleDataCall& outData, geocalls::MultiParticleDataCall& inData) {
    using geocalls::MultiParticleDataCall;

    // update id table if requried
    if ((inData.DataHash() == 0) || (inData.DataHash() != inDataHash)) {
        inDataHash = inData.DataHash();
        inDataTime = -1; // force update of frame data

        unsigned int fid = inData.FrameID();
        bool fid_f = inData.IsFrameForced();

        if (!updateIDdata(inData))
            return false;

        inData.SetFrameID(fid, fid_f);
        if (!inData(0))
            return false;
    }

    // update frame data if required
    if ((inData.FrameID() != inDataTime) || (data.size() != inData.GetParticleListCount())) {
        inDataTime = inData.FrameID();
        updateData(inData);
    }

    // output internal data copy
    outData.SetDataHash(outDataHash);
    outData.SetExtent(inData.FrameCount(), inData.AccessBoundingBoxes());
    outData.SetFrameID(outDataTime);
    outData.SetParticleListCount(static_cast<unsigned int>(data.size()));
    for (unsigned int i = 0; i < static_cast<unsigned int>(data.size()); i++) {
        outData.AccessParticles(i) = data[i].parts;
    }
    outData.SetUnlocker(nullptr); // since this is a hack module, I don't care for thread safety

    return true;
}

namespace {

class prev_part_info {
public:
    bool operator<(const prev_part_info& rhs) const {
        return dist < rhs.dist;
    }

    unsigned int idx;
    double dist;
};

class cur_part_info {
public:
    unsigned int idx;
    double dist;

    unsigned int prev_idx;
    std::vector<prev_part_info> prev;
};

bool inv_comp_cur_part_info_ptr(const cur_part_info* lhs, const cur_part_info* rhs) {
    return lhs->dist > rhs->dist;
}

} // namespace

bool datatools::ParticleSortFixHack::updateIDdata(geocalls::MultiParticleDataCall& inData) {
    outDataHash++;
    megamol::core::utility::log::Log::DefaultLog.WriteInfo("Updating Particle Sorting ID Data...");

    inData.SetFrameID(0, false);
    if (!inData(1))
        return false; // query extends first for number for frames

    // bbox to resolve cyclic boundary conditions
    vislib::math::Cuboid<float> bbox = inData.AccessBoundingBoxes().IsObjectSpaceBBoxValid()
                                           ? inData.AccessBoundingBoxes().ObjectSpaceBBox()
                                           : inData.AccessBoundingBoxes().WorldSpaceBBox();
    vislib::math::Dimension<float, 3> bboxsize = bbox.GetSize();
    megamol::core::utility::log::Log::DefaultLog.WriteInfo(
        "Bounding Box size (%f, %f, %f)", bboxsize[0], bboxsize[1], bboxsize[2]);

    unsigned int frame_cnt = inData.FrameCount();
    this->ids.resize(frame_cnt); // allocate data for all frames! We are memory-hungry! but I don't care

    vislib::sys::ConsoleProgressBar cpb;
    cpb.Start("Processing frame data", frame_cnt);

    std::vector<cur_part_info> dists_fix;
    std::vector<cur_part_info*> dists;

    std::vector<particle_data> pd[2];
    for (unsigned int frame_i = 0; frame_i < frame_cnt; ++frame_i) {
        std::vector<particle_data>& dat_cur = pd[frame_i % 2];
        std::vector<particle_data>& dat_prev = pd[(frame_i + 1) % 2];
        do { // ensure we get the right data
            inData.SetFrameID(frame_i, true);
            if (!inData(0))
                return false;
            if (inData.FrameID() != frame_i)
                vislib::sys::Thread::Sleep(1);
        } while (inData.FrameID() != frame_i);

        // ensure data structure for lists
        unsigned int list_cnt = inData.GetParticleListCount();
        if (frame_i > 0) {
            if (ids[frame_i - 1].size() != list_cnt) {
                megamol::core::utility::log::Log::DefaultLog.WriteError(
                    "Data sets changes lists over time. Unsupported!");
                return false;
            }
        }
        this->ids[frame_i].resize(list_cnt);
        dat_cur.resize(list_cnt);

        for (unsigned int list_i = 0; list_i < list_cnt; ++list_i) {
            // ensure data structure for particls
            auto& parts = inData.AccessParticles(list_i);
            unsigned int part_cnt = static_cast<unsigned int>(parts.GetCount());
            if (frame_i > 0) {
                if (ids[frame_i - 1][list_i].size() != part_cnt) {
                    megamol::core::utility::log::Log::DefaultLog.WriteError(
                        "Data sets changes particle numbers in list over time. Unsupported!");
                    return false;
                }
            }
            this->ids[frame_i][list_i].resize(part_cnt);

            if (frame_i > 1) {
// estimate new particle positions
//  dat_prev is data from frame_i - 1
//  dat_cur is data from frame_i - 2 (atm)
#pragma omp parallel for
                for (int part_i = 0; part_i < static_cast<int>(part_cnt); ++part_i) {
                    const float* part_k_pos = reinterpret_cast<const float*>(
                        static_cast<const unsigned char*>(dat_cur[list_i].parts.GetVertexData()) +
                        dat_cur[list_i].parts.GetVertexDataStride() * this->ids[frame_i - 1][list_i][part_i]);
                    float* part_j_pos = reinterpret_cast<float*>(const_cast<unsigned char*>(
                        static_cast<const unsigned char*>(dat_prev[list_i].parts.GetVertexData()) +
                        dat_prev[list_i].parts.GetVertexDataStride() * part_i));

                    for (int i = 0; i < 3; i++) {
                        part_j_pos[i] += 0.95f * (part_j_pos[i] - part_k_pos[i]);
                    }
                }
            }

            // copy data locally (for being data_prev in the next iteration)
            copyData(dat_cur[list_i], parts);

            // initialize with identity
            for (unsigned int part_i = 0; part_i < part_cnt; ++part_i) {
                this->ids[frame_i][list_i][part_i] = part_i;
            }
            if (frame_i == 0)
                continue; // identity is great for frame 0

            // for later frames find best matches for particles from current frame to previous frame trying to minimize the overall distance

            // 1. compute distance matrix
            // 2. select particle a with largest distance
            // 3. select pair-particle b' with which it would have the lowest distance
            // 4. test if a->a' b->b' would be switched to a->b' b->a' the overall distance would decrease
            // 5. if so, swap and continue with 2.
            // 6. if not, select b' with next lowest distance and continue with 4.
            // 7. if there is not suitable match with a, select a with next largest distance and continue with 3.
            // 8. if this was the last a, abort
            // x. abort after X iterations (?)

            // step 1
            //////////////////////////////////////////////////////////////////
            if (dists.size() < part_cnt) {
                dists_fix.resize(part_cnt);
                dists.resize(part_cnt);
                for (unsigned int i = 0; i < part_cnt; ++i) {
                    dists[i] = &dists_fix[i];
                    dists[i]->prev.resize(part_cnt);
                }
            }

#pragma omp parallel for
            for (int part_i = 0; part_i < static_cast<int>(part_cnt); ++part_i) {
                const float* part_i_pos = reinterpret_cast<const float*>(
                    static_cast<const unsigned char*>(dat_cur[list_i].parts.GetVertexData()) +
                    dat_cur[list_i].parts.GetVertexDataStride() * part_i);

                for (unsigned int part_j = 0; part_j < part_cnt; ++part_j) {
                    const float* part_j_pos = reinterpret_cast<const float*>(
                        static_cast<const unsigned char*>(dat_prev[list_i].parts.GetVertexData()) +
                        dat_prev[list_i].parts.GetVertexDataStride() * part_j);

                    double dist = std::sqrt(part_sqdist(part_i_pos, part_j_pos, bboxsize));

                    if (part_i == static_cast<int>(part_j)) {
                        dist *= 0.1;
                    } else {
                        dist += 0.001;
                    }

                    dists[part_i]->prev[part_j].dist = dist;
                    dists[part_i]->prev[part_j].idx = part_j;
                }

                dists[part_i]->idx = part_i;
                dists[part_i]->prev_idx = part_i; // id is a good starting point
                dists[part_i]->dist =
                    dists[part_i]->prev[part_i].dist; // this works here, because dists->prev is not sorted, yet

                //}
                //for (int part_i = 0; part_i < static_cast<int>(part_cnt); ++part_i) {
                // now sort, so we can iterate for the smallest distance (step 3.)
                std::sort(dists[part_i]->prev.begin(), dists[part_i]->prev.end());
            }

            // inverse sorting, so we find the pair with the largest distance (step 2.)
            std::sort(dists.begin(), dists.end(), inv_comp_cur_part_info_ptr);

            double whole_dist = 0.0;
            unsigned int update_print = 0;

            auto dists_it = dists.begin();
            auto dists_end = dists.end();
            for (; dists_it != dists_end; ++dists_it)
                whole_dist += (*dists_it)->dist;
            dists_it = dists.begin();

            // step 2
            //////////////////////////////////////////////////////////////////
            // for loop ensures the maximum iterations
            for (unsigned int max_iter = part_cnt * 100; (dists_it != dists_end) // condition step 8
                                                         && (max_iter > 0);      // condition step x: maximum iterations
                 --max_iter) {

                // dists_it is the pair with the largest distance a->a'
                cur_part_info* a = *dists_it;
                prev_part_info* a_to_a_tick = nullptr;
                for (prev_part_info& i_tick : a->prev) {
                    if (i_tick.idx == a->prev_idx) {
                        a_to_a_tick = &i_tick;
                        break;
                    }
                }
                assert(a_to_a_tick != nullptr);

                //bool swapped = false;

                // step 3: select b'
                //////////////////////////////////////////////////////////////
                unsigned int max_tries = 10;
                for (prev_part_info& a_to_b_tick : a->prev) {

                    // limit iterations in this loop
                    if (max_tries == 0)
                        break;
                    --max_tries;

                    // search for b
                    cur_part_info* b = nullptr;
                    for (cur_part_info* b_i : dists) {
                        if (b_i->prev_idx == a_to_b_tick.idx) {
                            b = b_i;
                            break;
                        }
                    }
                    assert(b != nullptr);
                    // search of alternative dists
                    prev_part_info* b_to_a_tick = nullptr;
                    for (prev_part_info& i_tick : b->prev) {
                        if (i_tick.idx == a_to_a_tick->idx) {
                            b_to_a_tick = &i_tick;
                            break;
                        }
                    }
                    assert(b_to_a_tick != nullptr);
                    prev_part_info* b_to_b_tick = nullptr;
                    for (prev_part_info& i_tick : b->prev) {
                        if (i_tick.idx == a_to_b_tick.idx) {
                            b_to_b_tick = &i_tick;
                            break;
                        }
                    }
                    assert(b_to_b_tick != nullptr);

                    // step 4: test distance difference
                    //////////////////////////////////////////////////////////
                    double distdiv = -a_to_a_tick->dist   // distance a->a'
                                     - b_to_b_tick->dist  // distance b->b'
                                     + a_to_b_tick.dist   // distance a->b'
                                     + b_to_a_tick->dist; // distance b->a'

                    //if ((a_to_b_tick.dist - a_to_a_tick->dist) > 0) continue;
                    //if ((b_to_a_tick->dist - b_to_b_tick->dist) > 0) continue;

                    if (distdiv >= -0.0001) {
                        // makes worse or no difference
                        // step 6: continue with next b'
                        //////////////////////////////////////////////////////

                        // should test, thou, if it makes sense continuing here, or if we should move on to the next particle
                        // For now, max_tries should help.

                        continue;
                    }

                    // step 5: swap and continue with next loop
                    //////////////////////////////////////////////////////////
                    a->prev_idx = a_to_b_tick.idx;
                    a->dist = a_to_b_tick.dist;
                    b->prev_idx = b_to_a_tick->idx;
                    b->dist = b_to_a_tick->dist;
                    whole_dist += distdiv;
                    //swapped = true;

                    if ((update_print % 10) == 0) {
                        megamol::core::utility::log::Log::DefaultLog.WriteInfo(
                            "[%u][%u] whole_dist = %g\n", frame_i, list_i, whole_dist);
                    }
                    ++update_print;

                    // resort: should be quick as only two elements changed their values
                    std::sort(dists.begin(), dists.end(), inv_comp_cur_part_info_ptr);

                    // just to be sure
                    dists_it = dists.begin();
                    dists_end = dists.end();
                    break;
                }

                //if (swapped) continue; // continue with step 2

                // step 7: no match with a: continue with next a at 3
                //////////////////////////////////////////////////////////////
            }

// transfer dists matrix
#pragma omp parallel for
            for (int i = 0; i < static_cast<int>(part_cnt); ++i) {
                this->ids[frame_i][list_i][i] = dists[i]->prev_idx;
            }

            // some testing output to see how good it works:
            unsigned int wcc = 0;
            for (unsigned int i = 0; i < part_cnt; ++i) {
                if (ids[frame_i][list_i][i] != i)
                    wcc++;
            }
            megamol::core::utility::log::Log::DefaultLog.WriteInfo(
                "[%u][%u] Particle list mixing: %u/%u\n", frame_i, list_i, wcc, part_cnt);
        }

        inData.Unlock();
        cpb.Set(frame_i);
    }

#if 0
    // first try was completely wrong!
    
    std::vector<particle_data> pd[2];

    typedef struct _sqdist_info_t {
        double d;
        unsigned int i, j;
    } sqdist_info;
    std::vector<sqdist_info> sqdists;
    std::vector<bool> part_i_available, part_j_available;

    for (unsigned int frame_i = 0; frame_i < frame_cnt; ++frame_i) {
        std::vector<particle_data> &dat_cur = pd[frame_i % 2];
        std::vector<particle_data> &dat_prev = pd[(frame_i + 1) % 2];

        do { // ensure we get the right data
            inData.SetFrameID(frame_i, true);
            if (!inData(0)) return false;
            if (inData.FrameID() != frame_i) vislib::sys::Thread::Sleep(1);
        } while (inData.FrameID() != frame_i);


        // ensure data structure for lists
        unsigned int list_cnt = inData.GetParticleListCount();
        if (frame_i > 0) {
            if (ids[frame_i - 1].size() != list_cnt) {
                megamol::core::utility::log::Log::DefaultLog.WriteError("Data sets changes lists over time. Unsupported!");
                return false;
            }
        }
        this->ids[frame_i].resize(list_cnt);
        dat_cur.resize(list_cnt);

        for (unsigned int list_i = 0; list_i < list_cnt; ++list_i) {
            // ensure data structure for particls
            auto& parts = inData.AccessParticles(list_i);
            unsigned int part_cnt = static_cast<unsigned int>(parts.GetCount());
            if (frame_i > 0) {
                if (ids[frame_i - 1][list_i].size() != part_cnt) {
                    megamol::core::utility::log::Log::DefaultLog.WriteError("Data sets changes particle numbers in list over time. Unsupported!");
                    return false;
                }
            }
            this->ids[frame_i][list_i].resize(part_cnt);


            if (frame_i > 1) {
                // estimate new particle positions
                //  dat_prev is data from frame_i - 1
                //  dat_cur is data from frame_i - 2 (atm)
#pragma omp parallel for
                for (int part_i = 0; part_i < static_cast<int>(part_cnt); ++part_i) {
                    const float * part_k_pos = reinterpret_cast<const float*>(static_cast<const unsigned char*>(dat_cur[list_i].parts.GetVertexData()) + dat_cur[list_i].parts.GetVertexDataStride() * 
                        this->ids[frame_i - 1][list_i][part_i]);
                    float * part_j_pos = reinterpret_cast<float*>(const_cast<unsigned char*>(static_cast<const unsigned char*>(dat_prev[list_i].parts.GetVertexData()) + dat_prev[list_i].parts.GetVertexDataStride() * part_i));

                    for (int i = 0; i < 3; i++) {
                        part_j_pos[i] += 0.5f * (part_j_pos[i] - part_k_pos[i]);
                    }
                }
            }

            // copy data locally (for being data_prev in the next iteration)
            copyData(dat_cur[list_i], parts);

            // Now compute ids for later sorting
            //  Here ids are the indices of the same particle in the last frame!
            if (frame_i == 0) {
                // initialize with identity for frame 0
                for (unsigned int part_i = 0; part_i < part_cnt; ++part_i) {
                    this->ids[frame_i][list_i][part_i] = part_i;
                }
            } else {
                // minimize sum of distances to particles from last frame
                if (sqdists.size() < part_cnt * part_cnt)
                    sqdists.resize(part_cnt * part_cnt);
                if (part_i_available.size() < part_cnt) part_i_available.resize(part_cnt);
                if (part_j_available.size() < part_cnt) part_j_available.resize(part_cnt);

                double diag_bonus_fac = (frame_i < 2) ? 0.1 : 0.666;
                double off_diag_malus = 0.00001;

#pragma omp parallel for
                for (int part_i = 0; part_i < static_cast<int>(part_cnt); ++part_i) {
                    const float * part_i_pos = reinterpret_cast<const float*>(
                        static_cast<const unsigned char*>(dat_cur[list_i].parts.GetVertexData()) 
                        + dat_cur[list_i].parts.GetVertexDataStride() * part_i);
                    part_i_available[part_i] = true;
                    part_j_available[part_i] = true;
                    for (unsigned int part_j = 0; part_j < part_cnt; ++part_j) {
                        const float * part_j_pos = reinterpret_cast<const float*>(
                            static_cast<const unsigned char*>(dat_prev[list_i].parts.GetVertexData()) 
                            + dat_prev[list_i].parts.GetVertexDataStride() * part_j);
                        // squared distance from particle i in this frame to particle j in the previous frame
                        double dist = part_sqdist(part_i_pos, part_j_pos, bboxsize);
                        if (part_i == part_j) dist *= diag_bonus_fac; else dist += off_diag_malus;
                        sqdists[part_i + part_j * part_cnt].d = dist;
                        sqdists[part_i + part_j * part_cnt].i = part_i;
                        sqdists[part_i + part_j * part_cnt].j = part_j;
                    }
                }
                std::sort(sqdists.begin(), sqdists.end(),
                    [](const sqdist_info& a, const sqdist_info& b) { return a.d < b.d; });

                unsigned int parts_to_go = part_cnt;
                for (auto& i : sqdists) {
                    if (!part_i_available[i.i]) continue;
                    if (!part_j_available[i.j]) continue;
                    ids[frame_i][list_i][i.i] = i.j;
                    part_i_available[i.i] = false;
                    part_j_available[i.j] = false;
                    parts_to_go--;
                    if (parts_to_go == 0) break;
                }
                assert(parts_to_go == 0);

            }

            unsigned int wcc = 0;
            for (unsigned int i = 0; i < part_cnt; ++i) {
                if (ids[frame_i][list_i][i] != i) wcc++;
            }
            megamol::core::utility::log::Log::DefaultLog.WriteInfo("[%u][%u] Particle list mixing: %u/%u\n", frame_i, list_i, wcc, part_cnt);
        }

        inData.Unlock();
        cpb.Set(frame_i);
    }
#endif

    cpb.Stop();

    // now we make the id list global!
    for (unsigned int frame_i = 1; frame_i < frame_cnt; ++frame_i) {
        unsigned int list_cnt = static_cast<unsigned int>(ids[frame_i].size());
        for (unsigned int list_i = 0; list_i < list_cnt; ++list_i) {
            int part_cnt = static_cast<int>(ids[frame_i][list_i].size());
#pragma omp parallel for
            for (int part_i = 0; part_i < part_cnt; ++part_i) {
                ids[frame_i][list_i][part_i] = ids[frame_i - 1][list_i][ids[frame_i][list_i][part_i]];
            }
        }
    }

    // preparation complete
    megamol::core::utility::log::Log::DefaultLog.WriteInfo("Particle Sorting ID Data updated.");
    return true;
}

void datatools::ParticleSortFixHack::updateData(geocalls::MultiParticleDataCall& inData) {
    data.resize(inData.GetParticleListCount());

    for (unsigned int i = 0; i < static_cast<unsigned int>(data.size()); i++) {
        copyData(data[i], inData.AccessParticles(i));
        vislib::RawStorage td(data[i].dat.GetSize());
        ::memcpy(td, data[i].dat, data[i].dat.GetSize());
        unsigned int ps = data[i].parts.GetVertexDataStride();
        int pc = static_cast<int>(data[i].parts.GetCount());
#pragma omp parallel for
        for (int pi = 0; pi < pc; ++pi) {
            ::memcpy(data[i].dat.At(pi * ps), td.At(ids[inData.FrameID()][i][pi] * ps), ps);
        }
    }

    outDataTime = inData.FrameID();
}

void datatools::ParticleSortFixHack::copyData(particle_data& tar, geocalls::SimpleSphericalParticles& src) {
    tar.parts = src;

    const unsigned char* colPtr = static_cast<const unsigned char*>(src.GetColourData());
    const unsigned char* vertPtr = static_cast<const unsigned char*>(src.GetVertexData());
    unsigned int colSize = 0;
    unsigned int colStride = src.GetColourDataStride();
    unsigned int vertSize = 0;
    unsigned int vertStride = src.GetVertexDataStride();
    bool vertRad = false;

    // colour
    switch (src.GetColourDataType()) {
    case geocalls::MultiParticleDataCall::Particles::COLDATA_NONE:
        break;
    case geocalls::MultiParticleDataCall::Particles::COLDATA_UINT8_RGB:
        colSize = 3;
        break;
    case geocalls::MultiParticleDataCall::Particles::COLDATA_UINT8_RGBA:
        colSize = 4;
        break;
    case geocalls::MultiParticleDataCall::Particles::COLDATA_FLOAT_RGB:
        colSize = 12;
        break;
    case geocalls::MultiParticleDataCall::Particles::COLDATA_FLOAT_RGBA:
        colSize = 16;
        break;
    case geocalls::MultiParticleDataCall::Particles::COLDATA_FLOAT_I:
        colSize = 4;
        break;
    default:
        break;
    }
    if (colStride < colSize)
        colStride = colSize;

    // radius and position
    switch (src.GetVertexDataType()) {
    case geocalls::MultiParticleDataCall::Particles::VERTDATA_NONE:
        break;
    case geocalls::MultiParticleDataCall::Particles::VERTDATA_FLOAT_XYZ:
        vertSize = 12;
        break;
    case geocalls::MultiParticleDataCall::Particles::VERTDATA_FLOAT_XYZR:
        vertSize = 16;
        break;
        // We do not support short-vertices ATM
    default:
        break;
    }
    if (vertStride < vertSize)
        vertStride = vertSize;

    tar.dat.EnforceSize(static_cast<size_t>((colSize + vertSize) * tar.parts.GetCount()));
    tar.parts.SetVertexData(src.GetVertexDataType(), tar.dat.At(0), colSize + vertSize);
    tar.parts.SetColourData(src.GetColourDataType(), tar.dat.At(vertSize), colSize + vertSize);

    for (UINT64 pi = 0; pi < tar.parts.GetCount(); ++pi) {
        ::memcpy(tar.dat.At(static_cast<size_t>(pi * (colSize + vertSize) + 0)), vertPtr, vertSize);
        vertPtr += vertStride;
        ::memcpy(tar.dat.At(static_cast<size_t>(pi * (colSize + vertSize) + vertSize)), colPtr, colSize);
        colPtr += colStride;
    }
}

double datatools::ParticleSortFixHack::part_sqdist(
    const float* p1, const float* p2, const vislib::math::Dimension<float, 3>& bboxsize) {
    float dx = p1[0] - p2[0];
    float dy = p1[1] - p2[1];
    float dz = p1[2] - p2[2];

    // direction does not matter
    if (dx < 0)
        dx = -dx;
    if (dy < 0)
        dy = -dy;
    if (dz < 0)
        dz = -dz;

    // now for the cyclic boundary condition
    assert(dx <= bboxsize[0]);
    assert(dy <= bboxsize[1]);
    assert(dz <= bboxsize[2]);

    if (dx > bboxsize[0] * 0.5f)
        dx = bboxsize[0] - dx;
    if (dy > bboxsize[1] * 0.5f)
        dy = bboxsize[1] - dy;
    if (dz > bboxsize[2] * 0.5f)
        dz = bboxsize[2] - dz;

    // squared distance
    return static_cast<double>(dx) * static_cast<double>(dx) + static_cast<double>(dy) * static_cast<double>(dy) +
           static_cast<double>(dz) * static_cast<double>(dz);
}
