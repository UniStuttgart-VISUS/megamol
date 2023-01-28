/*
 * AtomGrid.h
 * Copyright (C) 2006-2017 by MegaMol Team
 * Alle Rechte vorbehalten.
 */
#ifndef MMMOLMAPPLG_ATOMGRID_H_INCLUDED
#define MMMOLMAPPLG_ATOMGRID_H_INCLUDED
#pragma once

#include "vislib/math/Cuboid.h"
#include "vislib/math/Dimension.h"

#include "Computations.h"

#include <functional>
#include <queue>
#include <valarray>

namespace megamol {
namespace molecularmaps {

/**
 * A grid containing atoms for faster spatial search
 */
class AtomGrid {
public:
    /**
     * Destructor
     */
    virtual ~AtomGrid(void);

    /**
     * Constructor
     */
    AtomGrid(void);

    /**
     * Constructor for a atom grid with a given set of atoms
     *
     * @param atomVector Vector containing the atom data
     * (position & radius as vec4) that will be put into the grid.
     */
    AtomGrid(std::vector<vec4d>& atomVector);

    /**
     * Removes all date from the search grid and deletes the cell
     * rings and the closest atoms for each cell.
     */
    void ClearSearchGrid();

    /**
     * Return a reference to the atoms that are stored in the grid.
     *
     * @return the reference to the atoms
     */
    const std::vector<vec4d>& GetAtoms() const;

    /**
     * Get the number of cells.
     *
     * @return the number of cells.
     */
    size_t GetCellCnt();

    /**
     * Get the end vertex for the given start vertex by looking inside the grid
     * for the optimal end vertex.
     *
     * @param p_params contains every parameter for the function
     * @param edgeEndResult will contain the tangential sphere with the smallest
     * radius that is computed by the p_voronoi_sphere function
     *
     * @return the ID of the atom that is the end vertex or -1 if no good vertex
     * was found
     */
    int GetEndVertex(const EndVertexParams& p_params, vec4d& edgeEndResult);

    /**
     * Create a grid based on the given set of atoms. Does not use the Constructor
     * and also avoid the copy assignment operator. The constructor uses the copy
     * assignment operator and is therefore slower than calling this function.
     *
     * @param atomVector Vector containing the atom data
     * (position & radius as vec4) that will be put into the grid.
     */
    void Init(std::vector<vec4d>& atomVector);

    /**
     * Returns whether this grid is initialized or not
     *
     * @return True, if the grid is initialized, false otherwise
     */
    bool IsInitialized() const;

    /**
     * The copy assigment operator.
     */
    AtomGrid& operator=(const AtomGrid& rhs);

    /**
     * Removes the four start spheres from the list of atoms and the corresponding
     * cells.
     */
    void RemoveStartSpheres();

private:
    /**
     * Get all neighbouring cells and sort them into the rings around the cell.
     *
     * @param p_begin the first ID of the cells for which neighbours are created
     * @param p_end the last ID + 1 of the cells for which neighbours are created
     */
    void allCellNeighbours(const size_t p_begin, const size_t p_end);

    /**
     * Take the x, y and z index in the grid and return the ID of the corresponding
     * cell.
     *
     * @param p_position the x, y and z index in the grid
     *
     * @return the ID of the cell at the position
     */
    const int cellPositionToIndex(const vec3i& p_position);

    /**
     * Take the x, y and z index in the grid and return the ID of the corresponding
     * cell.
     *
     * @param p_x the x index of in the grid
     * @param p_y the y index of in the grid
     * @param p_z the z index of in the grid
     *
     * @return the ID of the cell at the position
     */
    const int cellPositionToIndex(const int p_x, const int p_y, const int p_z);

    /**
     * Checks if the potential end vertex intersects any of the atoms of the
     * molecule.
     *
     * @param p_end_vertex the potential end vertex
     * @param p_atom_id the ID of the current atom
     * @param p_gate_0 the ID of the first gate atom
     * @param p_gate_1 the ID of the second gate atom
     * @param p_gate_2 the ID of the third gate atom
     *
     * @return true if an intersection was found, false otherwise
     */
    bool checkForIntersections(
        const vec4d& p_end_vertex, const uint p_atom_id, const uint p_gate_0, const uint p_gate_1, const uint p_gate_2);

    /**
     * Returns the cell coordinate in which a certain position lies in.
     *
     * @param position The position we want to locate the cell for.
     *
     * @return The cell the position lies in, (-1, -1, -1) if the position is not
     * within the grid
     */
    const vec3i getCoordOf(const vec3d& position) const;

    /**
     * Returns the cell coordinate in which a certain sphere lies in.
     *
     * @param p_sphere The sphere we want to locate the cell for.
     *
     * @return The cell the position lies in, (-1, -1, -1) if the position is not
     * within the grid
     */
    const vec3i getCoordOf(const vec4d& p_sphere) const;

    /**
     * Returns the cell coordinate in which a certain position lies in.
     *
     * @param x The x-value of the position.
     * @param y The y-value of the position.
     * @param z The z-value of the position.
     *
     * @return The cell the position lies in, (-1, -1, -1) if the position is not
     * within the grid
     */
    const vec3i getCoordOf(const double x, const double y, const double z) const;

    /**
     * Initializes the sphere grid with a given set of spheres.
     *
     * @param sphereVec Vector containing the atoms (position & radius) that have to be put into the grid
     */
    void initialize(std::vector<vec4d>& atomVector);

    /**
     * Insert the atoms with the ID in the range [p_begin, p_end) into the cells.
     *
     * @param p_begin the first ID of the atoms to be inserted
     * @param p_end the last ID + 1 of the atoms to be inserted
     */
    void insertAtoms(const size_t p_begin, const size_t p_end);

    /**
     * Computes the distance of the current vertex to the nearest remaining potential atom
     * with the biggest radius. If the distance is greater the sum of their radii the
     * function returns true.
     *
     * @param p_center The center of the cell of the first loop
     * @param p_loop The number of the loop that was last performed
     * @param p_end_vertex The position and the radius of the current end vertex
     *
     * @return true if the distance is greater the sum of their radii, false otherwise
     */
    bool stopLoop(const vec3d& p_center, const size_t p_loop, const vec4d& p_end_vertex);

    /** The position and radius of all atoms. */
    std::vector<vec4d> atoms;

    /** The bounding box of the represented data set */
    vislib::math::Cuboid<double> boundingBox;

    /** The number of grid cells per dimension */
    vislib::math::Dimension<int, 3> cellNum;

    /** The size of a single grid cell */
    vislib::math::Dimension<double, 3> cellSize;

    /** The denumerator value for each cell size. */
    vislib::math::Dimension<double, 3> cellSizeDenom;

    /** The cells of the grid. */
    std::vector<megamol::molecularmaps::Cell> cells;

    /** The rings for every cell. */
    std::vector<std::vector<std::vector<uint16_t>>> cell_rings;

    /** Is this grid initialized? */
    bool isInitializedFlag;

    /** The maximum radius of the atoms. */
    double max_radius;

    /** The minimal cell dimension. */
    double min_cell_dim;

    /** The maximum number of cells within each ring. */
    std::vector<size_t> ring_sizes;
};

} /* end namespace molecularmaps */
} /* end namespace megamol */

#endif /* MMMOLMAPPLG_ATOMGRID_H_INCLUDED */
