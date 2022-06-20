/*
 * VoronoiChannelCalculator.cpp
 * Copyright (C) 2006-2016 by MegaMol Team
 * Alle Rechte vorbehalten.
 */

#include "VoronoiChannelCalculator.h"

using namespace megamol::core;
using namespace megamol::molecularmaps;
using namespace megamol::protein_calls;

/*
 * VoronoiChannelCalculator::~VoronoiChannelCalculator
 */
VoronoiChannelCalculator::~VoronoiChannelCalculator(void) {
    this->stopThreads();
    this->Release();
}

/*
 * VoronoiChannelCalculator::VoronoiChannelCalculator
 */
VoronoiChannelCalculator::VoronoiChannelCalculator(void) : AbstractLocalRenderer(), resultAvailable(false) {}

/*
 * VoronoiChannelCalculator::checkVertexValidity
 */
void VoronoiChannelCalculator::checkVertexAndGateValidity(MolecularDataCall* mdc) {
    // Initialise the valid vertices flag to be valid for all vertices.
    this->gateValidFlags.resize(this->voronoi_edges.size(), true);
    this->gateValidFlags.assign(gateValidFlags.size(), true);
    this->vertexValidFlags.resize(this->voronoi_vertices.size(), true);
    this->vertexValidFlags.assign(vertexValidFlags.size(), true);
    size_t filtered_inf_vertices = 0, filtered_radius_vertices = 0, filtered_convex_hull_vertices = 0, i = 0;
    std::vector<std::pair<size_t, vec4d>> valid_vertices;
    valid_vertices.reserve(this->voronoi_vertices.size());

    // Loop over all vertices.
    for (auto it = this->voronoi_vertices.begin(); it != this->voronoi_vertices.end(); it++) {
        // A vertex is valid if its radius is greater than the probe radius and if it is no infinity vertex.
        this->vertexValidFlags[i] =
            ((this->vertices[i].GetW() >= this->probeRadius) && !(it->second.infinity_count > 2));
        if (this->vertexValidFlags[i]) {
            valid_vertices.emplace_back(std::make_pair(i, this->vertices[i]));
        }

        // Check if the vertex radius is smaller than the probe radius.
        if (this->vertices[i].GetW() < this->probeRadius) {
            filtered_radius_vertices++;
        }

        // Check if the vertex is at infinity, i.e. from that vertex no other vertices could be reached.
        if (it->second.infinity_count > 2) {
            filtered_inf_vertices++;
        }
        i++;
    }
    // megamol::core::utility::log::Log::DefaultLog.WriteMsg(megamol::core::utility::log::Log::LEVEL_INFO, "Filtered %d
    // voronoi vertices, because their radius was lower than %f", filtered_radius_vertices, this->probeRadius);
    // megamol::core::utility::log::Log::DefaultLog.WriteMsg(megamol::core::utility::log::Log::LEVEL_INFO, "Filtered %d
    // voronoi infinity vertices", filtered_inf_vertices);

    // Initialise the threads to test if the vertices are inside or outside of the convex hull.
    size_t core_num = Concurrency::details::_CurrentScheduler::_GetNumberOfVirtualProcessors();
    size_t items_per_thread = std::max(static_cast<size_t>(1), valid_vertices.size() / core_num);
    this->stopThreads();

    // If the last thread has less than items_per_thread vertices to process, we need more than core_num threads in
    // total.
    if (valid_vertices.size() % core_num == 0) {
        this->voronoi_threads = std::vector<std::thread>(core_num);

    } else {
        this->voronoi_threads = std::vector<std::thread>(core_num + 1);
    }

    // Convert the atoms to a vec3f representation.
    std::vector<vec3f> atomData(mdc->AtomCount());
    auto ptr = mdc->AtomPositions();
    for (uint i = 0; i < mdc->AtomCount(); i++) {
        atomData[i].Set(ptr[i * 3 + 0], ptr[i * 3 + 1], ptr[i * 3 + 2]);
    }

#define CONVEX_HULL_FILTERING
#ifdef CONVEX_HULL_FILTERING
    // Check for all vertices that are still valid if they lie inside the convex hull if not, filter them.
    size_t begin_index;
    size_t end_index;
    for (size_t i = 0; i < this->voronoi_threads.size(); i++) {
        // Determine the begin and end index for the current thread.
        if (i < this->voronoi_threads.size() - 1) {
            begin_index = i * items_per_thread;
            end_index = (i + 1) * items_per_thread;

        } else {
            begin_index = i * items_per_thread;
            end_index = valid_vertices.size();
        }

        // Start the thread with the current begin and end index.
        this->voronoi_threads[i] = std::thread([&valid_vertices, this, &atomData, begin_index, end_index] {
            std::vector<vec3f> directions = std::vector<vec3f>(atomData.size());
            for (size_t a = begin_index; a < end_index; a++) {
                size_t idx = valid_vertices[a].first;
                this->vertexValidFlags[idx] =
                    Computations::LiesInsideConvexHull(atomData, valid_vertices[a].second, directions);
            }
        });
    }

    // Wait for the threads to finish.
    this->stopThreads();
#endif /* #ifdef CONVEX_HULL_FILTERING */

    std::vector<std::pair<size_t, vec4d>> valid_gates;
    valid_gates.reserve(this->voronoi_edges.size());

    // filter the gates also
    i = 0;
    for (auto it = this->voronoi_edges.begin(); it != this->voronoi_edges.end(); it++) {
        this->gateValidFlags[i] = (it->gate_sphere.GetW() >= this->probeRadius);
        if (this->gateValidFlags[i]) {
            valid_gates.emplace_back(std::make_pair(i, it->gate_sphere));
        }
        i++;
    }

    items_per_thread = std::max(static_cast<size_t>(1), valid_gates.size() / core_num);

    if (valid_gates.size() % core_num == 0) {
        this->voronoi_threads = std::vector<std::thread>(core_num);
    } else {
        this->voronoi_threads = std::vector<std::thread>(core_num + 1);
    }

#ifdef CONVEX_HULL_FILTERING
    for (size_t i = 0; i < this->voronoi_threads.size(); i++) {
        // Determine the begin and end index for the current thread.
        if (i < this->voronoi_threads.size() - 1) {
            begin_index = i * items_per_thread;
            end_index = (i + 1) * items_per_thread;

        } else {
            begin_index = i * items_per_thread;
            end_index = valid_gates.size();
        }

        // Start the thread with the current begin and end index.
        this->voronoi_threads[i] = std::thread([&valid_gates, this, &atomData, begin_index, end_index] {
            std::vector<vec3f> directions = std::vector<vec3f>(atomData.size());
            for (size_t a = begin_index; a < end_index; a++) {
                size_t idx = valid_gates[a].first;
                this->gateValidFlags[idx] =
                    Computations::LiesInsideConvexHull(atomData, valid_gates[a].second, directions);
            }
        });
    }
    this->stopThreads();
#endif /* #ifdef CONVEX_HULL_FILTERING */
}

/*
 * VoronoiChannelCalculator::constructVoronoiDiagram
 */
bool VoronoiChannelCalculator::constructVoronoiDiagram(MolecularDataCall* mdc) {
    // Delete the old computed Voronoi Diagram.
    this->edge_neighbours.clear();
    this->gates.clear();
    this->gateValidFlags.clear();
    this->vertexMap.clear();
    this->vertexValidFlags.clear();
    this->vertices.clear();
    this->voronoi_vertices.clear();
    this->voronoi_edges.clear();

    // Remove the remaining gates in the queue.
    this->gatesToTest_one.clear();
    this->gatesToTest_two.clear();

    // Get the true bounding box of the molecule. Then shrink the bounding box by 3 A
    // in each direction to fit tightly
    auto bb = mdc->AccessBoundingBoxes().ObjectSpaceBBox();
    bb.Grow(-3.0f);

    // Calculate the maximal and the middle atom radius.
    float rMax = FLT_MIN;
    for (unsigned int i = 0; i < mdc->AtomTypeCount(); i++) {
        if (mdc->AtomTypes()[i].Radius() > rMax) {
            rMax = mdc->AtomTypes()[i].Radius();
        }
    }
    float rMiddle = 0.0f;
    for (unsigned int i = 0; i < mdc->AtomCount(); i++) {
        rMiddle += mdc->AtomTypes()[mdc->AtomTypeIndices()[i]].Radius();
    }
    rMiddle /= static_cast<float>(mdc->AtomCount());

    // Compute the 4 start vertices.
    auto bbcenter = bb.CalcCenter();
    float movement = bb.GetLeft() - bbcenter.GetX();
    float constant = rMax * 2.5f;
    float plane = bbcenter.GetX() + movement - constant * rMiddle;

    vec4d start1(plane, bbcenter.GetY() + constant * rMiddle, bbcenter.GetZ(), rMiddle * 0.9);
    vec4d start2(plane, bbcenter.GetY() - constant * rMiddle, bbcenter.GetZ() + constant * rMiddle, rMiddle);
    vec4d start3(plane, bbcenter.GetY() - constant * rMiddle, bbcenter.GetZ() - constant * rMiddle, rMiddle * 1.1);
    vec4d start4(plane - 2.0 * constant * rMiddle, bbcenter.GetY(), bbcenter.GetZ(), rMiddle * 0.95);

    // Try to find the center sphere of the 4 input spheres.
    std::array<vec4d, 4> startVec{start1, start2, start3, start4};
    std::array<vec4d, 2> sphereVec{vec4d(), vec4d()};
    auto sphere_results = Computations::ComputeVoronoiSphereR(startVec, sphereVec);

    // Check if there is at least one sphere and if so take the one with the smallest radius,
    // i.e. the first sphere.
    if (sphere_results != 1) {
        megamol::core::utility::log::Log::DefaultLog.WriteMsg(megamol::core::utility::log::Log::LEVEL_ERROR,
            "The initial voronoi sphere could not be computed or is ambigous!"
            "\nPlease contact the developer to fix this.\n");
        return false;
    }
    auto centroid = sphereVec[0];

    // Copy the atom data.
    std::vector<vec4d> atomData(mdc->AtomCount());
    auto ptr = mdc->AtomPositions();
    for (uint i = 0; i < mdc->AtomCount(); i++) {
        atomData[i].Set(
            ptr[i * 3 + 0], ptr[i * 3 + 1], ptr[i * 3 + 2], mdc->AtomTypes()[mdc->AtomTypeIndices()[i]].Radius());
    }

    // Add the four start vertices to the atom list. They will be removed later on.
    atomData.push_back(start1);
    atomData.push_back(start2);
    atomData.push_back(start3);
    atomData.push_back(start4);

    // Create a search grid for the atoms.
    this->searchGrid.Init(atomData);

    // Add every possible gate of the start cell to a data structure.
    uint s1Idx = static_cast<uint>(this->searchGrid.GetAtoms().size() - 4);
    uint s2Idx = static_cast<uint>(this->searchGrid.GetAtoms().size() - 3);
    uint s3Idx = static_cast<uint>(this->searchGrid.GetAtoms().size() - 2);
    uint s4Idx = static_cast<uint>(this->searchGrid.GetAtoms().size() - 1);
    std::pair<vec4d, std::array<uint, 5>> g1(centroid, {s1Idx, s2Idx, s3Idx, s4Idx, 0});
    std::pair<vec4d, std::array<uint, 5>> g2(centroid, {s2Idx, s3Idx, s4Idx, s1Idx, 0});
    std::pair<vec4d, std::array<uint, 5>> g3(centroid, {s1Idx, s3Idx, s4Idx, s2Idx, 0});
    std::pair<vec4d, std::array<uint, 5>> g4(centroid, {s1Idx, s2Idx, s4Idx, s3Idx, 0});

    // Gate definition: 1 start voronoi sphere as vec4d + 3 gate sphere indices followed
    // by the index of the fourth vertex stored in an array. The fifth value is the
    // Voronoi vertex ID this can be ignored for now. Initialise the start gates.
    this->gatesToTest_one.reserve(2 * this->searchGrid.GetAtoms().size());
    this->gatesToTest_two.reserve(2 * this->searchGrid.GetAtoms().size());
    this->active_gates = 0;
    this->gatesToTest_one.push_back(g4);
    this->gatesToTest_one.push_back(g3);
    this->gatesToTest_one.push_back(g2);
    this->gatesToTest_one.push_back(g1);

    // Initialise the loop that looks for the initial Voronoi Vertex.
    this->initVertexFound = false;

    // This loop processes each element in the queue and tries to find a corresponding
    // end vertex for each of them. If the end vertex is only defined by "real" atoms,
    // we have found the initial voronoi vertex, if not we add three new gates to the
    // queue.
    size_t core_num = Concurrency::details::_CurrentScheduler::_GetNumberOfVirtualProcessors();
    this->stopThreads();
    this->voronoi_threads = std::vector<std::thread>(core_num);
    uint stop = 0;
    while (stop != 2) {
        // Compute the active gates and push the new gates to the inactive queue.
        stop = 0;
        for (size_t i = 0; i < this->voronoi_threads.size(); i++) {
            this->voronoi_threads[i] = std::thread(std::bind(&VoronoiChannelCalculator::nextVoronoiVertexInit, this));
        }

        // Wait for all threads to finish and swap the active an inactive queue.
        this->stopThreads();
        this->active_gates = (this->active_gates + 1) % 2;

        // Stop if either both gate queues are empty or if the initial vertex is found.
        if (this->gatesToTest_one.empty()) {
            stop++;
        }
        if (this->gatesToTest_two.empty()) {
            stop++;
        }
        if (this->initVertexFound) {
            stop = 2;
        }
    }

    if (!this->initVertexFound) {
        megamol::core::utility::log::Log::DefaultLog.WriteMsg(megamol::core::utility::log::Log::LEVEL_ERROR,
            "No initial Voronoi vertex could be found!"
            "\nPlease contact the developer to fix this.\n");
        return false;
    }

    // Remove the remaining gates in the queue.
    this->gatesToTest_one.clear();
    this->gatesToTest_two.clear();

    // Remove the start spheres from the list of atoms.
    this->searchGrid.RemoveStartSpheres();

    // Create the new gates for the start vertex.
    s1Idx = this->initVertexBorder[0];
    s2Idx = this->initVertexBorder[1];
    s3Idx = this->initVertexBorder[2];
    s4Idx = this->initVertexBorder[3];
    g1.first = this->initVertex;
    g1.second = {s1Idx, s2Idx, s3Idx, s4Idx, 0};
    g2.first = this->initVertex;
    g2.second = {s2Idx, s3Idx, s4Idx, s1Idx, 0};
    g3.first = this->initVertex;
    g3.second = {s1Idx, s3Idx, s4Idx, s2Idx, 0};
    g4.first = this->initVertex;
    g4.second = {s1Idx, s2Idx, s4Idx, s3Idx, 0};

    // Put the gate in the active queue.
    this->active_gates = 0;
    this->gatesToTest_one.push_back(g4);
    this->gatesToTest_one.push_back(g3);
    this->gatesToTest_one.push_back(g2);
    this->gatesToTest_one.push_back(g1);

    // Initialise the list of Voronoi vertices and edges.
    this->voronoi_edges.reserve(20 * this->searchGrid.GetAtoms().size());
    this->vertices.reserve(20 * this->searchGrid.GetAtoms().size());

    // Convert the intial Voronoi vertex to a Voronoi vertex and add it to the list.
    this->voronoi_id = 0;
    auto vertex = VoronoiVertex(initVertexBorder, this->voronoi_id++);
    this->voronoi_vertices.insert(std::pair<uint64_t, VoronoiVertex>(vertex.vertex_hash, vertex));
    this->vertices.push_back(this->initVertex);

    // Use the threadpool to compute all Voronoi vertices.
    this->stopThreads();
    this->voronoi_threads = std::vector<std::thread>(core_num);
    stop = 0;
    while (stop != 2) {
        // Compute the active gates and push the new gates to the inactive queue.
        stop = 0;
        for (size_t i = 0; i < this->voronoi_threads.size(); i++) {
            this->voronoi_threads[i] = std::thread(std::bind(&VoronoiChannelCalculator::nextVoronoiVertex, this));
        }

        // Wait for all threads to finish and swap the active an inactive queue.
        this->stopThreads();
        this->active_gates = (this->active_gates + 1) % 2;

        // If both queues are empty we found all vertices.
        if (this->gatesToTest_one.empty()) {
            stop++;
        }
        if (this->gatesToTest_two.empty()) {
            stop++;
        }
    }

    // Clear the search grid and free all used memory.
    std::thread cleanup([&]() { this->searchGrid.ClearSearchGrid(); });
    cleanup.detach();

    return true;
}

/*
 * VoronoiChannelCalculator::nextVoronoiVertex
 */
void VoronoiChannelCalculator::nextVoronoiVertex() {
    std::array<vec3d, 2> circles{vec3d(), vec3d()};
    std::pair<vec4d, std::array<uint, 5>> gate;
    std::array<vec4d, 2> gateCenter{vec4d(), vec4d()};
    std::array<vec4d, 2> incircle{vec4d(), vec4d()};
    std::array<vec4d, 2> sphereVec{vec4d(), vec4d()};
    bool stop = false;
    while (!stop) {
        // Get the gate and remove it from the queue.
        this->voronoi_mutex.lock();
        if (this->active_gates == 0) {
            // If the thread waited so long that the active queue is now empty we can stop it.
            if (this->gatesToTest_one.empty()) {
                this->voronoi_mutex.unlock();
                break;
            }

            // Get the last element from the queue and process it.
            gate = this->gatesToTest_one.back();
            this->gatesToTest_one.pop_back();

        } else {
            // If the thread waited so long that the active queue is now empty we can stop it.
            if (this->gatesToTest_two.empty()) {
                this->voronoi_mutex.unlock();
                break;
            }

            // Get the last element from the queue and process it.
            gate = this->gatesToTest_two.back();
            this->gatesToTest_two.pop_back();
        }
        this->voronoi_mutex.unlock();

        // Create the vector that contains all three gate spheres.
        std::array<vec4d, 4> gateVector{this->searchGrid.GetAtoms()[gate.second[0]],
            this->searchGrid.GetAtoms()[gate.second[1]], this->searchGrid.GetAtoms()[gate.second[2]], vec4d()};

        // Get the gate centers, we only need the first one, i.e. the one with the smaller radius.
        auto gateCenter_cnt = Computations::ComputeGateCenter(gateVector, gateCenter, incircle, circles);

        // Compute the pivot point of the current gate.
        vec3d pivot = Computations::ComputePivot(gateVector);

        // Compute the next voronoi vertex.
        EndVertexParams params = EndVertexParams(gate, gateCenter, gateVector, pivot);
        vec4d edgeEndResult;
        int minIdx = this->searchGrid.GetEndVertex(params, edgeEndResult);

        // Did we find a result for the currently processed gate?
        if (minIdx >= 0) {
            // Create the new Voronoi vertex and check if it already exists.
            this->voronoi_mutex.lock();
            VoronoiVertex possible_vertex =
                VoronoiVertex(vec4ui(gate.second[0], gate.second[1], gate.second[2], minIdx), 0);
            auto it = this->voronoi_vertices.find(possible_vertex.vertex_hash);
            if (it == this->voronoi_vertices.end()) {
                // The vertex is new so add it to the list and create the edge between the vertex we came from and the
                // new vertex.
                possible_vertex.id = this->voronoi_id++;
                this->voronoi_vertices.insert(
                    std::pair<uint64_t, VoronoiVertex>(possible_vertex.vertex_hash, possible_vertex));
                this->voronoi_edges.push_back(VoronoiEdge(possible_vertex.id, gate.first, gate.second[4]));
                this->vertices.push_back(edgeEndResult);
                this->voronoi_mutex.unlock();

                // Create the three new gates.
                std::pair<vec4d, std::array<uint, 5>> new1(edgeEndResult,
                    {static_cast<uint>(minIdx), gate.second[0], gate.second[1], gate.second[2], possible_vertex.id});
                std::pair<vec4d, std::array<uint, 5>> new2(edgeEndResult,
                    {static_cast<uint>(minIdx), gate.second[1], gate.second[2], gate.second[0], possible_vertex.id});
                std::pair<vec4d, std::array<uint, 5>> new3(edgeEndResult,
                    {static_cast<uint>(minIdx), gate.second[0], gate.second[2], gate.second[1], possible_vertex.id});

                // Add the new gates to the inactive queue.
                this->voronoi_mutex.lock();
                if (this->active_gates == 0) {
                    this->gatesToTest_two.push_back(new3);
                    this->gatesToTest_two.push_back(new2);
                    this->gatesToTest_two.push_back(new1);

                } else {
                    this->gatesToTest_one.push_back(new3);
                    this->gatesToTest_one.push_back(new2);
                    this->gatesToTest_one.push_back(new1);
                }
                this->voronoi_mutex.unlock();

            } else {
                // The vertex already exists so create the edge.
                this->voronoi_edges.push_back(VoronoiEdge(it->second.id, gate.first, gate.second[4]));
                this->voronoi_mutex.unlock();
            }

        } else {
            // Compute the hash value of the voronoi vertex and increase the infinity counter.
            this->voronoi_mutex.lock();
            auto vertex_hash =
                VoronoiVertex::ComputeHash(vec4ui(gate.second[0], gate.second[1], gate.second[2], gate.second[3]));
            auto it = this->voronoi_vertices.find(vertex_hash);
            if (it != this->voronoi_vertices.end()) {
                it->second.infinity_count++;
            }
            this->voronoi_mutex.unlock();
        }

        // Check if we need to stop the thread because all gates have been processed.
        if (this->active_gates == 0) {
            stop = this->gatesToTest_one.empty();

        } else {
            stop = this->gatesToTest_two.empty();
        }
    }
}

/*
 * VoronoiChannelCalculator::nextVoronoiVertexInit
 */
void VoronoiChannelCalculator::nextVoronoiVertexInit() {
    std::array<vec3d, 2> circles{vec3d(), vec3d()};
    std::pair<vec4d, std::array<uint, 5>> gate;
    std::array<vec4d, 2> gateCenter{vec4d(), vec4d()};
    std::array<vec4d, 2> incircle{vec4d(), vec4d()};
    std::array<vec4d, 2> sphereVec{vec4d(), vec4d()};
    bool stop = false;
    while (!stop && !this->initVertexFound) {
        // Get the gate and remove it from the queue.
        this->voronoi_mutex.lock();
        if (this->active_gates == 0) {
            // If the thread waited so long that the active queue is now empty we can stop it.
            if (this->gatesToTest_one.empty()) {
                this->voronoi_mutex.unlock();
                break;
            }

            // Get the last element from the queue and process it.
            gate = this->gatesToTest_one.back();
            this->gatesToTest_one.pop_back();

        } else {
            // If the thread waited so long that the active queue is now empty we can stop it.
            if (this->gatesToTest_two.empty()) {
                this->voronoi_mutex.unlock();
                break;
            }

            // Get the last element from the queue and process it.
            gate = this->gatesToTest_two.back();
            this->gatesToTest_two.pop_back();
        }
        this->voronoi_mutex.unlock();

        // Create the vector that contains all three gate spheres.
        std::array<vec4d, 4> gateVector{this->searchGrid.GetAtoms()[gate.second[0]],
            this->searchGrid.GetAtoms()[gate.second[1]], this->searchGrid.GetAtoms()[gate.second[2]], vec4d()};

        // Get the gate centers, we only need the first one, i.e. the one with the smaller radius.
        auto gateCenter_cnt = Computations::ComputeGateCenter(gateVector, gateCenter, incircle, circles);

        // Compute the pivot point of the current gate.
        vec3d pivot = Computations::ComputePivot(gateVector);

        // Compute the next voronoi vertex.
        EndVertexParams params = EndVertexParams(gate, gateCenter, gateVector, pivot);
        vec4d edgeEndResult;
        int minIdx = this->searchGrid.GetEndVertex(params, edgeEndResult);

        // Did we find a result for the currently processed gate?
        if (minIdx >= 0) {
            // Check if all 4 surrounding vertices of our voronoi vertex are "real" atoms. If all of them are real,
            // we have found our start vertex and can exit. If at least one is one of the four start vertices, add
            // the three new gates to the queue and process the next gate.
            this->voronoi_mutex.lock();
            uint thresh = static_cast<uint>(this->searchGrid.GetAtoms().size() - 4);
            if (gate.second[0] < thresh && gate.second[1] < thresh && gate.second[2] < thresh &&
                static_cast<uint>(minIdx) < thresh) {
                this->initVertexFound = true;
                this->initVertexBorder = vec4ui(gate.second[0], gate.second[1], gate.second[2], minIdx);
                this->initVertex = edgeEndResult;

            } else {
                std::pair<vec4d, std::array<uint, 5>> new1(
                    edgeEndResult, {static_cast<uint>(minIdx), gate.second[0], gate.second[1], gate.second[2], 0});
                std::pair<vec4d, std::array<uint, 5>> new2(
                    edgeEndResult, {static_cast<uint>(minIdx), gate.second[1], gate.second[2], gate.second[0], 0});
                std::pair<vec4d, std::array<uint, 5>> new3(
                    edgeEndResult, {static_cast<uint>(minIdx), gate.second[0], gate.second[2], gate.second[1], 0});
                if (this->active_gates == 0) {
                    this->gatesToTest_two.push_back(new3);
                    this->gatesToTest_two.push_back(new2);
                    this->gatesToTest_two.push_back(new1);

                } else {
                    this->gatesToTest_one.push_back(new3);
                    this->gatesToTest_one.push_back(new2);
                    this->gatesToTest_one.push_back(new1);
                }
            }
            this->voronoi_mutex.unlock();
        }
    }
}

/*
 * VoronoiChannelCalculator::stopThreads
 */
void VoronoiChannelCalculator::stopThreads() {
    for (size_t i = 0; i < this->voronoi_threads.size(); i++) {
        if (this->voronoi_threads[i].joinable()) {
            this->voronoi_threads[i].join();
        }
    }
}

/*
 * VoronoiChannelCalculator::create
 */
bool VoronoiChannelCalculator::create(void) {
    return true;
}

/*
 * VoronoiChannelCalculator::Render
 */
bool VoronoiChannelCalculator::Render(view::CallRender3DGL& call, bool lighting) {
    glLineWidth(1.0);
    glDisable(GL_LIGHTING);
    glBegin(GL_LINES);
    glColor3f(1.0, 1.0, 1.0);
    int i = 0;
    for (auto e : this->voronoi_edges) {
        if (this->vertexValidFlags[e.start_vertex] && this->vertexValidFlags[e.end_vertex] &&
            gateValidFlags[i]) { // the whole edge is valid
            glVertex3f(this->vertices[e.start_vertex].GetX(), this->vertices[e.start_vertex].GetY(),
                this->vertices[e.start_vertex].GetZ());
            glVertex3f(static_cast<float>(e.gate_sphere.GetX()), static_cast<float>(e.gate_sphere.GetY()),
                static_cast<float>(e.gate_sphere.GetZ()));
            glVertex3f(static_cast<float>(e.gate_sphere.GetX()), static_cast<float>(e.gate_sphere.GetY()),
                static_cast<float>(e.gate_sphere.GetZ()));
            glVertex3f(this->vertices[e.end_vertex].GetX(), this->vertices[e.end_vertex].GetY(),
                this->vertices[e.end_vertex].GetZ());
        } else if (this->vertexValidFlags[e.start_vertex] && gateValidFlags[i]) { // the first part of the edge is valid
            glVertex3f(this->vertices[e.start_vertex].GetX(), this->vertices[e.start_vertex].GetY(),
                this->vertices[e.start_vertex].GetZ());
            glVertex3f(static_cast<float>(e.gate_sphere.GetX()), static_cast<float>(e.gate_sphere.GetY()),
                static_cast<float>(e.gate_sphere.GetZ()));
        } else if (this->vertexValidFlags[e.end_vertex] && gateValidFlags[i]) { // the second part of the edge is valid
            glVertex3f(static_cast<float>(e.gate_sphere.GetX()), static_cast<float>(e.gate_sphere.GetY()),
                static_cast<float>(e.gate_sphere.GetZ()));
            glVertex3f(this->vertices[e.end_vertex].GetX(), this->vertices[e.end_vertex].GetY(),
                this->vertices[e.end_vertex].GetZ());
        }
        i++;
    }
    glEnd();
    glEnable(GL_LIGHTING);
    return true;
}

/*
 * VoronoiChannelCalculator::Update
 */
bool VoronoiChannelCalculator::Update(MolecularDataCall* mdc, std::vector<VoronoiVertex>& p_voronoi_vertices,
    std::vector<VoronoiEdge>& p_voronoi_edges, float probeRadius) {
    // Sanity check.
    if (mdc == nullptr) {
        this->resultAvailable = false;
        return false;
    }

    // If we have new data recompute the Voronoi diagram.
    bool newDiagram = false;
    if (mdc->DataHash() != this->lastDataHash) {
        // Set the new data hash and compute the voronoi diagram.
        this->lastDataHash = mdc->DataHash();
        newDiagram = this->constructVoronoiDiagram(mdc);
    }

    // Check if the diagram is valid or the probe radius has been changed and
    // filter the voronoi vertices based on the probe.
    if (std::abs(this->probeRadius - probeRadius) > FLT_EPSILON || newDiagram) {
        // Save the probe radius and filter the voronoi vertices and gates
        // based on the radius and the convex hull of the input data.
        this->probeRadius = probeRadius;
        this->checkVertexAndGateValidity(mdc);
        this->resultAvailable = true;
    }

    // Check if there are results available and return them.
    if (this->resultAvailable) {
        // Copy the valid vertices into the output vector. Also create the
        // offset vector for the vertex IDs.
        std::vector<int> vertex_offset = std::vector<int>(this->voronoi_vertices.size(), -1);
        p_voronoi_vertices.reserve(this->voronoi_vertices.size());
        int new_id = 0;
        size_t i = 0;
        for (auto it = this->voronoi_vertices.begin(); it != this->voronoi_vertices.end(); it++) {
            if (this->vertexValidFlags[i]) {
                p_voronoi_vertices.emplace_back((*it).second);
                p_voronoi_vertices.back().vertex = this->vertices[i];
                vertex_offset[i] = new_id++;
            }
            i++;
        }

        // Copy the valid edges to the output vector.For a valid edge both
        // vertices on the edge have to be valid.
        p_voronoi_edges.reserve(this->voronoi_edges.size());
        for (size_t i = 0; i < this->voronoi_edges.size(); i++) {
            if (this->gateValidFlags[i]) {
                if (vertex_offset[this->voronoi_edges[i].end_vertex] != -1 &&
                    vertex_offset[this->voronoi_edges[i].start_vertex] != -1) {
                    p_voronoi_edges.emplace_back(this->voronoi_edges[i]);
                    p_voronoi_edges.back().end_vertex = vertex_offset[p_voronoi_edges.back().end_vertex];
                    p_voronoi_edges.back().start_vertex = vertex_offset[p_voronoi_edges.back().start_vertex];
                }
            }
        }
        return true;
    }
    return false;
}

/*
 * VoronoiChannelCalculator::release
 */
void VoronoiChannelCalculator::release(void) {}
