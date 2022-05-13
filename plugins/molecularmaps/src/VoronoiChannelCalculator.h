/*
 * VoronoiChannelCalculator.h
 * Copyright (C) 2006-2017 by MegaMol Team
 * Alle Rechte vorbehalten.
 */
#ifndef MMMOLMAPPLG_VORONOICHANNELCALCULATOR_H_INCLUDED
#define MMMOLMAPPLG_VORONOICHANNELCALCULATOR_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "AbstractLocalRenderer.h"
#include "AtomGrid.h"
#include "Computations.h"

#include "protein_calls/MolecularDataCall.h"

#include "vislib/math/Plane.h"
#include "vislib/math/Vector.h"

#include <mutex>
#include <Eigen/Dense>

namespace megamol {
namespace molecularmaps {

class VoronoiChannelCalculator : public AbstractLocalRenderer {
public:
    /** Ctor. */
    VoronoiChannelCalculator(void);

    /** Dtor. */
    virtual ~VoronoiChannelCalculator(void);

    /**
     * Initializes the renderer.
     */
    virtual bool create(void);

    /**
     * Invokes the rendering calls.
     */
    virtual bool Render(core_gl::view::CallRender3DGL& call, bool lighting = true);

    /**
     * Update function for the local data to render.
     *
     * @param mdc The molecular data call containing the particle data
     */
    bool Update(protein_calls::MolecularDataCall* mdc, std::vector<VoronoiVertex>& p_voronoi_vertices,
        std::vector<VoronoiEdge>& p_voronoi_edges, float probeRadius = 1.5f);

protected:
    /**
     * Frees all needed resources used by this renderer
     */
    virtual void release(void);

private:
    /**
     * Checks the validity for each vertex
     */
    void checkVertexAndGateValidity(protein_calls::MolecularDataCall* mdc);

    /**
     * Constructs the voronoi diagram for a protein.
     *
     * @param mdc The molecular data call containing the protein data.
     *
     * @return true if the diagram was created, false otherwise
     */
    bool constructVoronoiDiagram(protein_calls::MolecularDataCall* mdc);

    /**
     *
     */
    void convexHullThread();

    /**
     * Computes the next Voronoi vertices until the thread is stopped.
     */
    void nextVoronoiVertex();

    /**
     * Computes the next Voronoi vertices until the thread is stopped.
     */
    void nextVoronoiVertexInit();

    /**
     * Stops all threads in the threadpool.
     */
    void stopThreads();

    /** Indicates the active queue for the Voronoi threads. */
    uint active_gates;

    /** List of neighbouring particle indices per edge */
    std::vector<vislib::math::Vector<int, 3>> edge_neighbours;

    /** List of all gate vertices, one for each edge */
    std::vector<vislib::math::Vector<float, 4>> gates;

    /** Queue of all gate vertices that need to be processed. */
    std::vector<std::pair<vec4d, std::array<uint, 5>>> gatesToTest_one;
    std::vector<std::pair<vec4d, std::array<uint, 5>>> gatesToTest_two;

    /** Validity flags for all gates. Non-valid gates are not initialized and do not belong to cavities. */
    std::vector<bool> gateValidFlags;

    /** The initial vertex. */
    vec4d initVertex;

    /** The gate spheres of the initial vertex. */
    vec4ui initVertexBorder;

    /** Flag that signals if the initial vertex is found. */
    bool initVertexFound;

    /** The probe radius the diagram is constructed for */
    float probeRadius;

    /** Flag showing whether a result is available. */
    bool resultAvailable;

    /** The search grid that contains all atoms of the protein. */
    AtomGrid searchGrid;

    /** map mapping vertex ids to identical other vertices */
    std::vector<int> vertexMap;

    /** validity flags of all vertices */
    std::vector<bool> vertexValidFlags;

    /** List of all voronoi vertex positions. */
    std::vector<vislib::math::Vector<float, 4>> vertices;

    /** List of all voronoi edges. */
    std::vector<VoronoiEdge> voronoi_edges;

    /** The ID of the next voronoi vertex. */
    uint voronoi_id;

    /** The mutex that locks access to the local variables for the threads. */
    std::mutex voronoi_mutex;

    /** Thread pool that computes the voronoi edges. */
    std::vector<std::thread> voronoi_threads;

    /** List of all voronoi vertices. */
    std::map<uint64_t, VoronoiVertex> voronoi_vertices;
};

} /* end namespace molecularmaps */
} /* end namespace megamol */

#endif /* MMMOLMAPPLG_VORONOICHANNELCALCULATOR_H_INCLUDED */
