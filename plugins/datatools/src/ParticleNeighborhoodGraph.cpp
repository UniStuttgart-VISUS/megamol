#include "ParticleNeighborhoodGraph.h"
#include "datatools/GraphDataCall.h"
#include "datatools/MultiParticleDataAdaptor.h"
#include "mmcore/param/BoolParam.h"
#include "mmcore/param/FloatParam.h"
#include "mmcore/param/IntParam.h"
#include "mmcore/utility/log/Log.h"
#include "vislib/math/ShallowPoint.h"
#include "vislib/math/ShallowVector.h"
#include <cfloat>
#include <chrono>
#include <omp.h>
#include <random>
#include <set>

using namespace megamol;
using namespace megamol::datatools;

ParticleNeighborhoodGraph::ParticleNeighborhoodGraph()
        : Module()
        , outGraphDataSlot("outGraphData", "Publishes graph edge data")
        , inParticleDataSlot("inParticle", "Fetches particle data")
        , radiusSlot("radius", "The neighborhood radius")
        , autoRadiusSlot("autoRadius::detect", "Flag to automatically assess the neighborhood radius")
        , autoRadiusSamplesSlot("autoRadius::samples", "Number of samples to determine the neighborhood radius")
        , autoRadiusSampleRndSeedSlot("autoRadius::randomSeed", "Seed for the random generator selecting the samples")
        , autoRadiusFactorSlot("autoRadius::resultsFactor", "Increase factor for the determined neighborhood radius")
        , forceConnectIsolatedSlot("forceConnectIsolated", "Forces to inter-connect isolated parts of the graph")
        , boundaryXCyclicSlot(
              "boundary::XCyclic", "Activates connection over cyclic boundary conditions in x direction")
        , boundaryYCyclicSlot(
              "boundary::YCyclic", "Activates connection over cyclic boundary conditions in y direction")
        , boundaryZCyclicSlot(
              "boundary::ZCyclic", "Activates connection over cyclic boundary conditions in z direction")
        , frameId(0)
        , inDataHash(0)
        , outDataHash(0)
        , edges() {

    static_assert(sizeof(index_t) * 2 == sizeof(GraphDataCall::edge), "Index type error.");

    outGraphDataSlot.SetCallback("GraphDataCall", "GetData", &ParticleNeighborhoodGraph::getData);
    outGraphDataSlot.SetCallback("GraphDataCall", "GetExtent", &ParticleNeighborhoodGraph::getExtent);
    MakeSlotAvailable(&outGraphDataSlot);

    inParticleDataSlot.SetCompatibleCall<geocalls::MultiParticleDataCallDescription>();
    MakeSlotAvailable(&inParticleDataSlot);

    autoRadiusSlot.SetParameter(new core::param::BoolParam(true));
    MakeSlotAvailable(&autoRadiusSlot);

    autoRadiusSamplesSlot.SetParameter(new core::param::IntParam(10, 1));
    MakeSlotAvailable(&autoRadiusSamplesSlot);

    autoRadiusSampleRndSeedSlot.SetParameter(new core::param::IntParam(1212));
    MakeSlotAvailable(&autoRadiusSampleRndSeedSlot);

    autoRadiusFactorSlot.SetParameter(new core::param::FloatParam(1.9f, 1.0f));
    MakeSlotAvailable(&autoRadiusFactorSlot);

    radiusSlot.SetParameter(new core::param::FloatParam(0.01f, 0.0f));
    MakeSlotAvailable(&radiusSlot);

    boundaryXCyclicSlot.SetParameter(new core::param::BoolParam(false));
    MakeSlotAvailable(&boundaryXCyclicSlot);

    boundaryYCyclicSlot.SetParameter(new core::param::BoolParam(false));
    MakeSlotAvailable(&boundaryYCyclicSlot);

    boundaryZCyclicSlot.SetParameter(new core::param::BoolParam(false));
    MakeSlotAvailable(&boundaryZCyclicSlot);

    forceConnectIsolatedSlot.SetParameter(new core::param::BoolParam(true));
    //MakeSlotAvailable(&forceConnectIsolatedSlot);
}

ParticleNeighborhoodGraph::~ParticleNeighborhoodGraph() {
    Release();
}

bool ParticleNeighborhoodGraph::create(void) {
    // intentionally empty
    return true;
}

void ParticleNeighborhoodGraph::release(void) {
    // intentionally empty
}

bool ParticleNeighborhoodGraph::getExtent(core::Call& c) {
    using datatools::GraphDataCall;
    GraphDataCall* gdc = dynamic_cast<GraphDataCall*>(&c);
    if (gdc == nullptr)
        return false;

    geocalls::MultiParticleDataCall* mpc = inParticleDataSlot.CallAs<geocalls::MultiParticleDataCall>();
    if (mpc == nullptr)
        return false;

    mpc->SetFrameID(gdc->FrameID(), true);
    if (!(*mpc)(1))
        return false;

    gdc->SetFrameCount(mpc->FrameCount());
    gdc->SetFrameID(mpc->FrameID());
    gdc->SetDataHash(mpc->DataHash());
    gdc->SetUnlocker(nullptr);

    return true;
}

bool ParticleNeighborhoodGraph::getData(core::Call& c) {
    using datatools::GraphDataCall;
    GraphDataCall* gdc = dynamic_cast<GraphDataCall*>(&c);
    if (gdc == nullptr)
        return false;

    geocalls::MultiParticleDataCall* mpc = inParticleDataSlot.CallAs<geocalls::MultiParticleDataCall>();
    if (mpc == nullptr)
        return false;

    mpc->SetFrameID(gdc->FrameID(), true);
    if (!(*mpc)(1))
        return false;
    core::BoundingBoxes bboxes = mpc->AccessBoundingBoxes();
    mpc->Unlock();

    mpc->SetFrameID(gdc->FrameID(), true);
    if (!(*mpc)(0))
        return false;
    mpc->AccessBoundingBoxes() = bboxes;

    if ((mpc->DataHash() != inDataHash) || (mpc->FrameID() != frameId) || (frameId != gdc->FrameID()) ||
        (inDataHash == 0) || autoRadiusSlot.IsDirty() || autoRadiusFactorSlot.IsDirty() ||
        autoRadiusSamplesSlot.IsDirty() || autoRadiusSampleRndSeedSlot.IsDirty() || radiusSlot.IsDirty() ||
        boundaryXCyclicSlot.IsDirty() || boundaryYCyclicSlot.IsDirty() || boundaryZCyclicSlot.IsDirty() ||
        forceConnectIsolatedSlot.IsDirty()) {
        // update data
        inDataHash = mpc->DataHash();
        frameId = mpc->FrameID();
        autoRadiusFactorSlot.ResetDirty();
        autoRadiusSamplesSlot.ResetDirty();
        autoRadiusSampleRndSeedSlot.ResetDirty();
        autoRadiusSlot.ResetDirty();
        radiusSlot.ResetDirty();
        boundaryXCyclicSlot.ResetDirty();
        boundaryYCyclicSlot.ResetDirty();
        boundaryZCyclicSlot.ResetDirty();
        forceConnectIsolatedSlot.ResetDirty();

        edges.clear();

        outDataHash++;

        this->calcData(mpc);
    }

    // set data
    gdc->SetDataHash(outDataHash);
    gdc->SetFrameID(frameId);
    gdc->Set(reinterpret_cast<const GraphDataCall::edge*>(edges.data()), edges.size() / 2, false);
    gdc->SetUnlocker(mpc->GetUnlocker());
    mpc->SetUnlocker(nullptr, false);

    mpc->Unlock();
    return true;
}

namespace {

struct graph_component {
    unsigned int id;
    size_t cnt;
    vislib::math::Vector<double, 3> pos;
};

} // namespace

void ParticleNeighborhoodGraph::calcData(geocalls::MultiParticleDataCall* data) {
    datatools::MultiParticleDataAdaptor d(*data);
    if (d.get_count() < 1)
        return;

    using std::chrono::high_resolution_clock;
    high_resolution_clock::time_point start = high_resolution_clock::now(), end;

    float neiRad = this->radiusSlot.Param<core::param::FloatParam>()->Value();
    if (this->autoRadiusSlot.Param<core::param::BoolParam>()->Value()) {
        // automatically select a neighborhood radius

        std::default_random_engine rnd_eng(autoRadiusSampleRndSeedSlot.Param<core::param::IntParam>()->Value());
        std::uniform_int_distribution<size_t> rnd_dist(0, d.get_count() - 1);

        int sample_cnt = autoRadiusSamplesSlot.Param<core::param::IntParam>()->Value();
        megamol::core::utility::log::Log::DefaultLog.WriteInfo("PNhG detecting radius from %d samples...", sample_cnt);

        float mean_dist = 0.0f;

        for (int sample = 0; sample < sample_cnt; ++sample) {
            size_t sample_idx = rnd_dist(rnd_eng);
            vislib::math::ShallowPoint<float, 3> sample_pt(const_cast<float*>(d.get_position(sample_idx)));

            float min_dist = FLT_MAX;

            for (size_t i = 0; i < d.get_count(); ++i) {
                if (i == sample_idx)
                    continue;
                vislib::math::ShallowPoint<float, 3> pt(const_cast<float*>(d.get_position(i)));
                float dist = (pt - sample_pt).SquareLength();
                if (dist < min_dist)
                    min_dist = dist;
            }

            if (min_dist == FLT_MAX)
                continue;

            mean_dist += std::sqrt(min_dist);
        }

        mean_dist /= static_cast<float>(sample_cnt);
        float dist_fac = autoRadiusFactorSlot.Param<core::param::FloatParam>()->Value();
        neiRad = dist_fac * mean_dist;
        end = high_resolution_clock::now();
        megamol::core::utility::log::Log::DefaultLog.WriteInfo("PNhG detecting radius (in %u ms) %f * %f => %f",
            std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count(), mean_dist, dist_fac, neiRad);
        start = end;

        this->radiusSlot.Param<core::param::FloatParam>()->SetValue(neiRad, false);
    }
    float neiRadSq = neiRad * neiRad;

    vislib::math::Cuboid<float> box(vislib::math::ShallowPoint<float, 3>(const_cast<float*>(d.get_position(0))),
        vislib::math::Dimension<float, 3>(0.0f, 0.0f, 0.0f));
    for (size_t i = 1; i < d.get_count(); ++i) {
        box.GrowToPoint(vislib::math::ShallowPoint<float, 3>(const_cast<float*>(d.get_position(i))));
    }

    unsigned int x_size = static_cast<unsigned int>(std::ceil(box.Width() / neiRad));
    unsigned int y_size = static_cast<unsigned int>(std::ceil(box.Height() / neiRad));
    unsigned int z_size = static_cast<unsigned int>(std::ceil(box.Depth() / neiRad));

    auto const& bboxR = data->AccessBoundingBoxes().ObjectSpaceBBox();
    auto const bboxCent = bboxR.CalcCenter();
    float bboxCentX = bboxCent.X();
    float bboxCentY = bboxCent.Y();
    float bboxCentZ = bboxCent.Z();

#define _COORD(x, y, z) ((x) + ((y) + ((z)*y_size)) * x_size)

    std::vector<size_t> grid(d.get_count());
    std::vector<size_t*> gridCell(x_size * y_size * z_size);

    megamol::core::utility::log::Log::DefaultLog.WriteInfo("PNhG search grid : (%u, %u, %u)", x_size, y_size, z_size);

    std::vector<vislib::math::Vector<unsigned int, 3>> cell(d.get_count());
    for (size_t i = 0; i < d.get_count(); ++i) {
        vislib::math::Vector<float, 3> c =
            vislib::math::ShallowPoint<float, 3>(const_cast<float*>(d.get_position(i))) - box.GetLeftBottomBack();
        cell[i] = c / neiRad;
        if (cell[i].X() >= x_size)
            cell[i].SetX(x_size - 1);
        if (cell[i].Y() >= y_size)
            cell[i].SetY(y_size - 1);
        if (cell[i].Z() >= z_size)
            cell[i].SetZ(z_size - 1);
        //assert(cell[i].X() < x_size);
        //assert(cell[i].Y() < y_size);
        //assert(cell[i].Z() < z_size);
    }

    std::vector<size_t> gridCellSize(x_size * y_size * z_size);
    std::fill(gridCellSize.begin(), gridCellSize.end(), 0);
    for (size_t i = 0; i < d.get_count(); ++i) {
        gridCellSize[_COORD(cell[i].X(), cell[i].Y(), cell[i].Z())]++;
    }

    size_t* p = grid.data();
    for (unsigned int gci = 0; gci < x_size * y_size * z_size; ++gci) {
        if (gridCellSize[gci] == 0) {
            gridCell[gci] = nullptr;
        } else {
            gridCell[gci] = p;
            p += gridCellSize[gci];
            gridCellSize[gci] = 0;
        }
    }
    assert((p - grid.data()) == d.get_count());

    for (size_t i = 0; i < d.get_count(); ++i) {
        size_t idx = _COORD(cell[i].X(), cell[i].Y(), cell[i].Z());
        *((gridCell[idx]) + gridCellSize[idx]) = i;
        gridCellSize[idx]++;
    }

    end = high_resolution_clock::now();
    megamol::core::utility::log::Log::DefaultLog.WriteInfo("PNhG search grid constructed in %u ms",
        std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count());
    start = end;

    edges.reserve(d.get_count() * 2 * 4); // something

    bool cycX = boundaryXCyclicSlot.Param<core::param::BoolParam>()->Value();
    bool cycY = boundaryYCyclicSlot.Param<core::param::BoolParam>()->Value();
    bool cycZ = boundaryZCyclicSlot.Param<core::param::BoolParam>()->Value();

    // iterate over cells
    int cellCnt = static_cast<int>(x_size * y_size * z_size);
    int maxThreads = omp_get_max_threads();
    std::vector<std::set<int>> neighborCellsMT(
        maxThreads); // all neighboring cells, from which particles need to be tested.
    std::vector<std::vector<vislib::math::Point<float, 3>>> testPossMT(
        maxThreads); // all positions of this one particle to be tested.

    float bboxSizeX = data->AccessBoundingBoxes().ObjectSpaceBBox().Width();
    float bboxSizeY = data->AccessBoundingBoxes().ObjectSpaceBBox().Height();
    float bboxSizeZ = data->AccessBoundingBoxes().ObjectSpaceBBox().Depth();

#pragma omp parallel for
    for (int cellIdx = 0; cellIdx < cellCnt; ++cellIdx) {
        int mt = omp_get_thread_num();
        std::set<int>& neighborCells = neighborCellsMT[mt];
        std::vector<vislib::math::Point<float, 3>>& testPoss = testPossMT[mt];
        neighborCells.clear();

        // cell coordinate from cell index
        int cellX = cellIdx % x_size;
        int cellZ = (cellIdx - cellX) / x_size;
        int cellY = cellZ % y_size;
        cellZ = (cellZ - cellY) / y_size;

        // collect neighboring cells
        for (int dx = -1; dx <= 1; ++dx) {
            int x = cellX + dx;
            if (x < 0) {
                if (cycX)
                    x = x_size - 1;
                else
                    continue; // skip this x
            }
            if (x >= static_cast<int>(x_size)) {
                if (cycX)
                    x = 0;
                else
                    continue; // skip this x
            }
            for (int dy = -1; dy <= 1; ++dy) {
                int y = cellY + dy;
                if (y < 0) {
                    if (cycY)
                        y = y_size - 1;
                    else
                        continue; // skip this y
                }
                if (y >= static_cast<int>(y_size)) {
                    if (cycY)
                        y = 0;
                    else
                        continue; // skip this y
                }
                for (int dz = -1; dz <= 1; ++dz) {
                    int z = cellZ + dz;
                    if (z < 0) {
                        if (cycZ)
                            z = z_size - 1;
                        else
                            continue; // skip this z
                    }
                    if (z >= static_cast<int>(z_size)) {
                        if (cycZ)
                            z = 0;
                        else
                            continue; // skip this z
                    }
                    neighborCells.insert(_COORD(x, y, z));
                }
            }
        }

        // for all points in the current cell
        for (size_t locPtIdx = 0; locPtIdx < gridCellSize[cellIdx]; ++locPtIdx) {
            // point position
            size_t ptIdx = gridCell[cellIdx][locPtIdx];
            vislib::math::ShallowPoint<float, 3> ptOrigPos(const_cast<float*>(d.get_position(ptIdx)));
            testPoss.clear();

            // multiply connections due to cyclic boundary tests
            float x = ptOrigPos.X(), y, z;
            for (int dx = 0; dx < (cycX ? 2 : 1); ++dx) {
                y = ptOrigPos.Y();
                for (int dy = 0; dy < (cycY ? 2 : 1); ++dy) {
                    z = ptOrigPos.Z();
                    for (int dz = 0; dz < (cycZ ? 2 : 1); ++dz) {
                        testPoss.push_back(vislib::math::Point<float, 3>(x, y, z));
                        if (dz == 0) {
                            if (z < bboxCentZ)
                                z += bboxSizeZ;
                            else
                                z -= bboxSizeZ;
                        }
                    }
                    if (dy == 0) {
                        if (y < bboxCentY)
                            y += bboxSizeY;
                        else
                            y -= bboxSizeY;
                    }
                }
                if (dx == 0) {
                    if (x < bboxCentX)
                        x += bboxSizeX;
                    else
                        x -= bboxSizeX;
                }
            }

            for (int neiCellIdx : neighborCells) {
                for (size_t locNPtIdx = 0; locNPtIdx < gridCellSize[neiCellIdx]; ++locNPtIdx) {
                    // point to test
                    size_t nPtIdx = gridCell[neiCellIdx][locNPtIdx];

                    if (ptIdx >= nPtIdx)
                        continue; // we only every construct edges from small to large indices

                    vislib::math::ShallowPoint<float, 3> nPtPos(const_cast<float*>(d.get_position(nPtIdx)));
                    for (vislib::math::Point<float, 3>& pt : testPoss) {
                        float sqDist = pt.SquareDistance(nPtPos);
                        if (sqDist < neiRadSq) {

#pragma omp critical
                            {
                                edges.push_back(static_cast<index_t>(ptIdx));
                                edges.push_back(static_cast<index_t>(nPtIdx));
                            }

                            break;
                        }
                    }
                }
            }
        }
    }

    end = high_resolution_clock::now();
    megamol::core::utility::log::Log::DefaultLog.WriteInfo(
        "PNhG edges computed in %u ms", std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count());
    start = end;

    //if (forceConnectIsolatedSlot.Param<core::param::BoolParam>()->Value()) {
    //    // 1. detect connected components

    //    // TODO: This is slow!

    //    std::vector<unsigned int> comp_id(d.get_count());
    //    std::fill(comp_id.begin(), comp_id.end(), 0);

    //    unsigned int next_comp = 1;
    //    for (size_t i = 0; i < d.get_count(); ++i) {
    //        if (comp_id[i] != 0) continue;
    //        comp_id[i] = next_comp;

    //        bool updated = true;
    //        while (updated) {
    //            updated = false;
    //            for (size_t e = 0; e < edges.size(); e += 2) {
    //                size_t e1 = edges[e];
    //                unsigned int &c1 = comp_id[e1];
    //                size_t e2 = edges[e + 1];
    //                unsigned int &c2 = comp_id[e2];
    //                if ((c1 == next_comp) && (c2 == next_comp)) continue;
    //                if (c1 == next_comp)  {
    //                    assert(c2 == 0);
    //                    c2 = next_comp;
    //                    updated = true;
    //                } else if (c2 == next_comp) {
    //                    assert(c1 == 0);
    //                    c1 = next_comp;
    //                    updated = true;
    //                }
    //            }
    //        }

    //        next_comp++;
    //    }

    //    end = high_resolution_clock::now();
    //    megamol::core::utility::log::Log::DefaultLog.WriteInfo("Neighborhood graph computed #2.1 in %u ms", std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count());
    //    start = end;

    //    if (next_comp > 2) {

    //        // TODO: This is slow!

    //        // compute centroids for connected components
    //        std::vector<std::shared_ptr<graph_component> > comps(next_comp - 1);
    //        for (unsigned int ci = 1; ci < next_comp; ci++) {
    //            comps[ci - 1] = std::make_shared<graph_component>();
    //            comps[ci - 1]->id = ci;
    //            comps[ci - 1]->cnt = 0;
    //            comps[ci - 1]->pos.SetNull();
    //        }
    //        for (size_t i = 0; i < d.get_count(); ++i) {
    //            comps[comp_id[i] - 1]->cnt++;
    //            comps[comp_id[i] - 1]->pos += vislib::math::ShallowVector<float, 3>(const_cast<float*>(d.get_position(i)));
    //        }
    //        size_t maxVal = 0;
    //        std::shared_ptr<graph_component> mainComp; // index of the biggest component to merge everything to
    //        for (std::shared_ptr<graph_component> c: comps) {
    //             c->pos /= static_cast<double>(c->cnt);
    //             if (c->cnt > maxVal) {
    //                 maxVal = c->cnt;
    //                 mainComp = c;
    //             }
    //        }
    //        assert(mainComp);

    //        // successively merge components into the biggest component
    //        while (comps.size() > 1) {
    //            // select the component closest to the biggest component
    //            double dist = DBL_MAX;
    //            std::shared_ptr<graph_component> selComp;
    //            for (std::shared_ptr<graph_component> c : comps) {
    //                if (c == mainComp) continue;
    //                double d = (c->pos - mainComp->pos).Length();
    //                if (d < dist) {
    //                    dist = d;
    //                    selComp = c;
    //                }
    //            }
    //            assert(selComp);

    //            // merge selComp into mainComp
    //            // add an edge between the two nodes closest to the other centroid
    //            size_t mainPId = static_cast<size_t>(-1);
    //            double mainPDist = DBL_MAX;
    //            for (size_t i = 0; i < d.get_count(); ++i) {
    //                vislib::math::ShallowVector<float, 3> p(const_cast<float*>(d.get_position(i)));
    //                if (comp_id[i] == mainComp->id) {
    //                    double d = (selComp->pos - p).Length();
    //                    if (d < mainPDist) {
    //                        mainPDist = d;
    //                        mainPId = i;
    //                    }
    //                }
    //            }
    //            vislib::math::ShallowVector<float, 3> maincomp_pos(const_cast<float*>(d.get_position(mainPId)));

    //            size_t selPId = static_cast<size_t>(-1);
    //            double selPDist = DBL_MAX;
    //            for (size_t i = 0; i < d.get_count(); ++i) {
    //                vislib::math::ShallowVector<float, 3> p(const_cast<float*>(d.get_position(i)));
    //                if (comp_id[i] == selComp->id) {
    //                    double d = (maincomp_pos - p).Length();
    //                    if (d < selPDist) {
    //                        selPDist = d;
    //                        selPId = i;
    //                    }
    //                }
    //            }
    //            assert(mainPId != static_cast<size_t>(-1));
    //            assert(selPId != static_cast<size_t>(-1));

    //            edges.push_back(selPId);
    //            edges.push_back(mainPId);

    //            // remove old ids
    //            for (size_t i = 0; i < d.get_count(); ++i) {
    //                if (comp_id[i] == selComp->id) {
    //                    comp_id[i] = mainComp->id;
    //                }
    //            }

    //            // merge centroids
    //            mainComp->pos *= static_cast<double>(mainComp->cnt);
    //            selComp->pos *= static_cast<double>(selComp->cnt);
    //            mainComp->pos += selComp->pos;
    //            mainComp->cnt += selComp->cnt;
    //            mainComp->pos /= static_cast<double>(mainComp->cnt);

    //            comps.erase(std::find(comps.begin(), comps.end(), selComp));
    //        }

    //    }
    //}

    edges.shrink_to_fit();

    end = high_resolution_clock::now();
    megamol::core::utility::log::Log::DefaultLog.WriteInfo(
        "PNhG completed in %u ms", std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count());
}
