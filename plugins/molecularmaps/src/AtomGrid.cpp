/*
 * AtomGrid.cpp
 * Copyright (C) 2006-2017 by MegaMol Team
 * Alle Rechte vorbehalten.
 */
#include "stdafx.h"
#include "AtomGrid.h"

using namespace megamol;
using namespace megamol::molecularmaps;

/*
 * AtomGrid::~AtomGrid
 */
AtomGrid::~AtomGrid(void) {
}

/*
 * AtomGrid::allCellNeighbours
 */
void AtomGrid::allCellNeighbours(const size_t p_begin, const size_t p_end) {
	// Initialise the neighbour computation.
	vec3i position;
	int w = this->cellNum.GetWidth();
	int h = this->cellNum.GetHeight();
	int w_h = w * h;

	// Create the neighbours for every cell within the given range.
	for (size_t cell_id = p_begin; cell_id < p_end; cell_id++) {
		// Compute the position of the cell, the major value is x the 
		// second value y and the last value is z.
		position.SetX(static_cast<int>(cell_id) % w);
		position.SetY(static_cast<int>(floor((cell_id % w_h) / w)));
		position.SetZ(static_cast<int>(floor(cell_id / w_h)));

		// Make shure the whole grid is looked at.
		int z_min = 0;
		int z_max = this->cellNum.GetDepth() - 1;
		int y_min = 0;
		int y_max = this->cellNum.GetHeight() - 1;
		int x_min = 0;
		int x_max = this->cellNum.GetWidth() - 1;

		// Get the number of rings.
		int max_x_dist = max(x_max - position.GetX(), position.GetX());
		int max_y_dist = max(y_max - position.GetY(), position.GetY());
		int max_z_dist = max(z_max - position.GetZ(), position.GetZ());
		int rings = max(max(max_x_dist, max_y_dist), max_z_dist);

		// Reserve enough space and store the IDs of neighbouring cells.
		this->cell_rings[cell_id].reserve(rings + 1);
		this->cell_rings[cell_id] = std::vector<std::vector<uint16_t>>(rings + 1);
		std::valarray<size_t> ring_indices = std::valarray<size_t>(rings + 1);
		for (size_t i = 0; i < rings + 1; i++) {
			size_t number = this->ring_sizes[std::min(i, this->ring_sizes.size() - 1)];
			this->cell_rings[cell_id][i].resize(static_cast<size_t>(number * 1.25));
		}

		// Get the ID out of the position of the cell and add it to the neighbours.
		for (int z = z_min; z <= z_max; z++) {
			for (int y = y_min; y <= y_max; y++) {
				for (int x = x_min; x <= x_max; x++) {
					auto id = this->cellPositionToIndex(x, y, z);

					// Get the ring the cell belongs to.
					auto x_r = abs(position.GetX() - x);
					auto y_r = abs(position.GetY() - y);
					auto z_r = abs(position.GetZ() - z);
					auto ring = max(max(x_r, y_r), z_r);

					// Add the cell to the ring.
					this->cell_rings[cell_id][ring][ring_indices[ring]++] = static_cast<uint16_t>(id);
				}
			}
		}

		// Set the correct size for every ring.
		for (size_t i = 0; i < rings + 1; i++) {
			this->cell_rings[cell_id][i].resize(ring_indices[i]);
			this->cell_rings[cell_id][i].shrink_to_fit();
		}
	}
}

/*
 * AtomGrid::AtomGrid
 */
AtomGrid::AtomGrid(void) : isInitializedFlag(false) {

}

/*
 * AtomGrid::AtomGrid
 */
AtomGrid::AtomGrid(std::vector<vec4d>& atomVector) {
	this->initialize(atomVector);
}

/*
 * AtomGrid::cellPositionToIndex
 */
const int AtomGrid::cellPositionToIndex(const vec3i& p_position) {
	return this->cellPositionToIndex(p_position.GetX(), p_position.GetY(), p_position.GetZ());
}

/*
 * AtomGrid::cellPositionToIndex
 */
const int AtomGrid::cellPositionToIndex(const int p_x, const int p_y, const int p_z) {
	return (p_x + this->cellNum.GetWidth() * (p_y + this->cellNum.GetHeight() * p_z));
}

/*
 * AtomGrid::checkForIntersections
 */
bool AtomGrid::checkForIntersections(const vec4d& p_end_vertex, const uint p_atom_id, 
		const uint p_gate_0, const uint p_gate_1, const uint p_gate_2) {
	// Get the coordinates of the cell that the start vertex belongs to.
	vec3i cell_coords = this->getCoordOf(p_end_vertex);

	// Get the ID of the cell and check if it is inside the grid.
	auto cell_id = this->cellPositionToIndex(cell_coords);
	if (cell_id >= static_cast<int>(this->cells.size()) || cell_id < 0) {
		return true;
	}

	// Get the atoms that possibly intersect the end vertex.
	double radius = p_end_vertex.GetW() + 6.0 * this->max_radius;
	radius *= radius;

	// Initialise the ring search.
	uint cur_cell;
	double min_dist = std::numeric_limits<double>::max();
	uint16_t* queue;
	size_t queue_idx;
	size_t ring = 0;

	// Get the current cell into the queue.
	queue = this->cell_rings[cell_id][ring].data();
	queue_idx = this->cell_rings[cell_id][ring++].size();
	while (true) {
		// Loop over the current ring of cells and check for intersections.
		while (queue_idx != 0) {
			// Get the ID of the cell and remove it from the queue.
			cur_cell = queue[--queue_idx];

			// Add the atoms of the cell to the list of closest atoms.
			for (auto atom_id : this->cells[cur_cell].atom_ids) {
				// Skipp the atoms the end vertex is tangent to.
				if (atom_id == p_atom_id || atom_id == p_gate_0 || atom_id == p_gate_1 || atom_id == p_gate_2) {
					continue;
				}

				// Compute the squared distance between the end vertex and the atom.
				double x = this->atoms[atom_id].GetX() - p_end_vertex.GetX();
				double y = this->atoms[atom_id].GetY() - p_end_vertex.GetY();
				double z = this->atoms[atom_id].GetZ() - p_end_vertex.GetZ();
				double r = this->atoms[atom_id].GetW() + p_end_vertex.GetW();
				double dist = x * x + y * y + z * z;

				// Compute the squared sum of the radii of the end vertex and the atom.
				double dist_rad = r * r;

				// Check if the squared distance is smaller than the squared sum of the radii.
				if (dist < dist_rad) {
					return true;
				}

				// Check if the distance is smaller than the minial distance so far.
				if (dist < min_dist) {
					min_dist = dist;
				}
			}
		}

		// Stop if all atoms of the current ring where outside of the search radius.
		if (min_dist > radius && min_dist != std::numeric_limits<double>::max()) {
			return false;
		}

		// Reset the minimum radius for the next ring.
		min_dist = std::numeric_limits<double>::max();

		// Get the next ring and add the neighbours of the current cell to the queue.
		if (ring < this->cell_rings[cell_id].size()) {
			queue = this->cell_rings[cell_id][ring].data();
			queue_idx = this->cell_rings[cell_id][ring++].size();

		} else {
			// No intersection was found so return false.
			return false;
		}
	}
}

/*
 * AtomGrid::ClearSearchGrid
 */
void AtomGrid::ClearSearchGrid() {
    this->atoms.clear();
    this->cells.clear();
	this->cell_rings.clear();
}

/*
 * AtomGrid::GetAtoms
 */
const std::vector<vec4d>& AtomGrid::GetAtoms() const {
	return this->atoms;
}

/*
 * AtomGrid::GetCellCnt
 */
size_t AtomGrid::GetCellCnt() {
	return this->cells.size();
}

/*
 * AtomGrid::getCoordOf
 */
const vec3i AtomGrid::getCoordOf(const vec3d& position) const {
	return this->getCoordOf(position.GetX(), position.GetY(), position.GetZ());
}

/*
 * AtomGrid::getCoordOf
 */
const vec3i AtomGrid::getCoordOf(const vec4d & p_sphere) const {
	return this->getCoordOf(p_sphere.GetX(), p_sphere.GetY(), p_sphere.GetZ());
}

/*
 * AtomGrid::getCoordOf
 */
const vec3i AtomGrid::getCoordOf(const double x, const double y, const double z) const {
	// Get the origin of the bounding box.
	auto bbOrigin = this->boundingBox.GetOrigin();
	
	// Check if one of the differences is negative, that means the point is not
	// inside of the grid.
	double x_diff = x - bbOrigin.GetX();
	double y_diff = y - bbOrigin.GetY();
	double z_diff = z - bbOrigin.GetZ();
	if (x_diff < 0.0 || y_diff < 0.0 || z_diff < 0.0) {
		return vec3i(-1, -1, -1);
	}

	// Compute the x, y and z index of the cell the point is in.
	int cell_x = static_cast<int>(x_diff * this->cellSizeDenom.GetWidth());
	int cell_y = static_cast<int>(y_diff * this->cellSizeDenom.GetHeight());
	int cell_z = static_cast<int>(z_diff * this->cellSizeDenom.GetDepth());

	// If one of the indices is bigger than the number of cells in that direction we
	// return a negative vector.
	if (cell_x >= this->cellNum.GetWidth() || cell_y >= this->cellNum.GetHeight() || 
			cell_z >= this->cellNum.GetDepth()) {
		return vec3i(-1, -1, -1);
	}

	// Return the cell otherwise.
	return vec3i(cell_x, cell_y, cell_z);
}

/*
 * AtomGrid::GetEndVertex
 */
int AtomGrid::GetEndVertex(const EndVertexParams& p_params, vec4d& edgeEndResult) {
	// Intialise computation of the new end vertex.
	vec3d center;
	uint cur_cell;
	vec3d start_pivot;
	int min_atom_id;
	double min_dist;
	uint16_t* queue;
	size_t queue_idx;
	size_t ring;
	std::array<vec4d, 2> spheres{ vec4d(), vec4d() };

	// Get the coordinates of the cell that the start vertex belongs to.
	vec3i cell_coords = this->getCoordOf(p_params.gate.first);

	// Get the ID of the cell and check if it is inside the grid.
	auto cell_id = this->cellPositionToIndex(cell_coords);
	if (cell_id >= static_cast<int>(this->cells.size()) || cell_id < 0) {
		return -1;
	}

	// Compute the center of the cell.
	auto bbox_origin = this->boundingBox.GetOrigin();
	center.SetX(static_cast<double>((bbox_origin.GetX() + cell_coords.GetX() * this->cellSize.GetWidth())
		+ this->cellSize.GetWidth() / 2.0f));
	center.SetY(static_cast<double>((bbox_origin.GetY() + cell_coords.GetY() * this->cellSize.GetHeight())
		+ this->cellSize.GetHeight() / 2.0f));
	center.SetZ(static_cast<double>((bbox_origin.GetZ() + cell_coords.GetZ() * this->cellSize.GetDepth())
		+ this->cellSize.GetDepth() / 2.0f));

	// Get the current cell into the queue.
	ring = 0;
	queue = this->cell_rings[cell_id][ring].data();
	queue_idx = this->cell_rings[cell_id][ring++].size();

	// Compute the vector from the pivot to the start vertex and normalise it.
	start_pivot = vec3d(p_params.gate.first.PeekComponents()) - p_params.pivot;
	Computations::NormaliseVector(start_pivot);

	// Loop over the rings to find the new end vertex.
	min_atom_id = -1;
	min_dist = 2.0 * vislib::math::PI_DOUBLE;
	while (true) {
		// Loop over the current ring of cells and compute end vertices.
		while (queue_idx != 0) {
			// Get the ID of the cell and remove it from the queue.
			cur_cell = queue[--queue_idx];

			// Get the end vertex in the cell by looking at all atoms in the cell.
			for (auto atom_id : this->cells[cur_cell].atom_ids) {
				if (atom_id != p_params.gate.second[0] &&
						atom_id != p_params.gate.second[1] &&
						atom_id != p_params.gate.second[2]) {
					// Get the atom based on the ID.
					p_params.gate_vector[3] = this->atoms[atom_id];

					// Compute the center sphere between the three gate atoms and the current atom.
					auto sphereResult = Computations::ComputeVoronoiSphereR(p_params.gate_vector, spheres);

					// We always take the result with the smaller radius, if available.
					if (sphereResult < 1) continue;
					auto edgeEnd = spheres[0];

					// Compute the angular distance to the found result sphere. If it is the currently smallest,
					// set the result values.
					if ((p_params.gate.first - edgeEnd).Length() > vislib::math::DOUBLE_EPSILON) {
						// Get the distance of the new end vertex.
						double angular_dist = Computations::AngularDistance(start_pivot, p_params.pivot, edgeEnd);
						if (angular_dist < min_dist) {
							// The new end vertex is closer to the start vertex, so check for intersections.
							bool intersection = this->checkForIntersections(edgeEnd, atom_id, p_params.gate.second[0],
								p_params.gate.second[1], p_params.gate.second[2]);

							// Check if there was an intersection, if not add the end vertex.
							if (!intersection) {
								min_dist = angular_dist;
								min_atom_id = static_cast<int>(atom_id);
								edgeEndResult = edgeEnd;
							}
						}
					}
				}
			}

			// Check if we need to stop the loop.
			if (this->stopLoop(center, ring, edgeEndResult)) {
				return min_atom_id;
			}
		}

		// Get the next ring and add the neighbours of the current cell to the queue.
		if (ring < this->cell_rings[cell_id].size()) {
			queue = this->cell_rings[cell_id][ring].data();
			queue_idx = this->cell_rings[cell_id][ring++].size();

		} else {
			// Return the current atom id because there are no further rings to look at.
			return min_atom_id;
		}
	}
}

/*
 * AtomGrid::Init
 */
void AtomGrid::Init(std::vector<vec4d>& atomVector) {
	this->initialize(atomVector);
}

/*
 * AtomGrid::initialize
 */
void AtomGrid::initialize(std::vector<vec4d>& atomVector) {
	// Store the vector locally.
	this->atoms = std::move(atomVector);
	this->atoms.shrink_to_fit();

	// Compute the bounding box of all atoms and the maximum radius.
	double xmin = DBL_MAX;
	double ymin = DBL_MAX;
	double zmin = DBL_MAX;
	double xmax = DBL_MIN;
	double ymax = DBL_MIN;
	double zmax = DBL_MIN;
	this->max_radius = 0.0;
	for (uint i = 0; i < this->atoms.size(); i++) {
		if (this->atoms[i].X() < xmin) xmin = this->atoms[i].X();
		if (this->atoms[i].Y() < ymin) ymin = this->atoms[i].Y();
		if (this->atoms[i].Z() < zmin) zmin = this->atoms[i].Z();
		if (this->atoms[i].X() > xmax) xmax = this->atoms[i].X();
		if (this->atoms[i].Y() > ymax) ymax = this->atoms[i].Y();
		if (this->atoms[i].Z() > zmax) zmax = this->atoms[i].Z();

		if (this->atoms[i].GetW() > this->max_radius) {
			this->max_radius = this->atoms[i].GetW();
		}
	}

	// Create the bounding box and enforce a positive size.
	this->boundingBox = vislib::math::Cuboid<double>(xmin, ymin, zmin, xmax, ymax, zmax);
	this->boundingBox.EnforcePositiveSize();

	// Compute the size of the grid.
	double smallestDimSize = this->boundingBox.Width();
	if (smallestDimSize > this->boundingBox.Height()) smallestDimSize = this->boundingBox.Height();
	if (smallestDimSize > this->boundingBox.Depth()) smallestDimSize = this->boundingBox.Depth();

	// The number of cell should be equal to the number of atoms but not exceed 65535, since the cells
	// are addressed with uint16 in the rings to save memory.
	double denom = static_cast<double>(Computations::Floor(pow(static_cast<double>(this->atoms.size()), 1.0 / 3.0)));
	double wantedCellSize = smallestDimSize / denom;
	this->cellNum.SetWidth(static_cast<int>(std::ceil(this->boundingBox.Width() / wantedCellSize)));
	this->cellNum.SetHeight(static_cast<int>(std::ceil(this->boundingBox.Height() / wantedCellSize)));
	this->cellNum.SetDepth(static_cast<int>(std::ceil(this->boundingBox.Depth() / wantedCellSize)));

	// Check the number of cells and correct it if necessary.
	int nbr_cells = this->cellNum.GetDepth() * this->cellNum.GetHeight() * this->cellNum.GetWidth();
	while (nbr_cells > 65535) {
		this->cellNum.SetWidth(this->cellNum.GetWidth() - 1);
		this->cellNum.SetHeight(this->cellNum.GetHeight() - 1);
		this->cellNum.SetDepth(this->cellNum.GetDepth() - 1);
		nbr_cells = this->cellNum.GetDepth() * this->cellNum.GetHeight() * this->cellNum.GetWidth();
	}

	// Set the dimensions of a cell and find the smallest dimension. Also compute the 1/cellSize values.
	this->cellSize.SetWidth(this->boundingBox.Width() / static_cast<double>(this->cellNum.Width()));
	this->cellSize.SetHeight(this->boundingBox.Height() / static_cast<double>(this->cellNum.Height()));
	this->cellSize.SetDepth(this->boundingBox.Depth() / static_cast<double>(this->cellNum.Depth()));
	this->min_cell_dim = std::min(std::min(this->cellSize.GetDepth(), this->cellSize.GetHeight()), this->cellSize.GetWidth());
	this->cellSizeDenom.SetWidth(1.0 / this->cellSize.GetWidth());
	this->cellSizeDenom.SetHeight(1.0 / this->cellSize.GetHeight());
	this->cellSizeDenom.SetDepth(1.0 / this->cellSize.GetDepth());

	// Initialise the cells, the neighbours of the cells and the closest atoms.
	this->cells = std::vector<Cell>(this->cellNum.GetDepth() * this->cellNum.GetHeight() * this->cellNum.GetWidth());
	this->cells.shrink_to_fit();

	// Determine the sizes of the rings and the maximum size.
	size_t ring_size = static_cast<size_t>(min(min(this->cellNum.GetDepth(), this->cellNum.GetHeight()), this->cellNum.GetWidth()));
	ring_size = static_cast<size_t>(ceil(ring_size / 2));
	this->ring_sizes = std::vector<size_t>(ring_size + 1);
	this->ring_sizes.shrink_to_fit();
	this->ring_sizes[0] = 1;
	for (size_t i = 1; i < this->ring_sizes.size(); i++) {
		size_t x = 1 + 2 * i;
		size_t y = 1 + 2 * (i - 1);
		this->ring_sizes[i] = 2 * (x * x) + y * ((x * x) - (y * y));
	}

	// Initialise the threads that are used for the computation of the neighbours and the
	// closest atoms.
	size_t core_num = Concurrency::details::_CurrentScheduler::_GetNumberOfVirtualProcessors();
	std::vector<std::thread> cell_threads = std::vector<std::thread>(core_num);
	size_t part_size = static_cast<size_t>(this->cells.size() / cell_threads.size());

	// Create neighbours of cells.	
	this->cell_rings = std::vector<std::vector<std::vector<uint16_t>>>(this->cells.size());
	this->cell_rings.shrink_to_fit();
	for (size_t i = 0; i < cell_threads.size(); i++) {
		if (i < cell_threads.size() - 1) {
			cell_threads[i] = std::thread(std::bind(
				&AtomGrid::allCellNeighbours, this, i * part_size, (i + 1) * part_size));

		} else {
			cell_threads[i] = std::thread(std::bind(
				&AtomGrid::allCellNeighbours, this, i * part_size, this->cells.size()));
		}
	}

	// Wait for the create cell threads to finish.
	for (size_t i = 0; i < cell_threads.size(); i++) {
		if (cell_threads[i].joinable()) {
			cell_threads[i].join();
		}
	}

	// Insert the atoms in the grid.
	part_size = static_cast<size_t>(this->atoms.size() / cell_threads.size());
	for (size_t i = 0; i < cell_threads.size(); i++) {
		if (i < cell_threads.size() - 1) {
			cell_threads[i] = std::thread(std::bind(
				&AtomGrid::insertAtoms, this, i * part_size, (i + 1) * part_size));

		} else {
			cell_threads[i] = std::thread(std::bind(
				&AtomGrid::insertAtoms, this, i * part_size, this->atoms.size()));
		}
	}

	// Wait for the create cell threads to finish.
	for (size_t i = 0; i < cell_threads.size(); i++) {
		if (cell_threads[i].joinable()) {
			cell_threads[i].join();
		}
	}

	// Set the initalised flag.
	this->isInitializedFlag = true;
}

/*
 * AtomGrid::insertAtoms
 */
void AtomGrid::insertAtoms(const size_t p_begin, const size_t p_end) {
	int id;
	vec3i cell_coords;
	auto bbOrigin = this->boundingBox.GetOrigin();
	for (size_t i = p_begin; i < p_end; i++) {
		// Get the differences along every axis to the origin of the bounding box.
		double x_diff = this->atoms[i].X() - bbOrigin.GetX();
		double y_diff = this->atoms[i].Y() - bbOrigin.GetY();
		double z_diff = this->atoms[i].Z() - bbOrigin.GetZ();

		// Compute the x, y and z index of the cell the point is in.
		int cell_x = static_cast<int>(x_diff * this->cellSizeDenom.GetWidth());
		int cell_y = static_cast<int>(y_diff * this->cellSizeDenom.GetHeight());
		int cell_z = static_cast<int>(z_diff * this->cellSizeDenom.GetDepth());

		// The three atoms that define the maximum x, y and z value need to be shifted
		// by one along their respective axis.
		if (cell_x == this->cellNum.GetWidth()) {
			cell_x -= 1;
		}
		if (cell_y == this->cellNum.GetHeight()) {
			cell_y -= 1;
		}
		if (cell_z == this->cellNum.GetDepth()) {
			cell_z -= 1;
		}

		// Set the cell coordinates.
		cell_coords.Set(cell_x, cell_y, cell_z);

		// Get the cell ID from the coordiantes and add the atom to the cell.
		id = this->cellPositionToIndex(cell_coords);
		this->cells[id].atom_ids.push_back(static_cast<uint>(i));
	}
}

/*
 * AtomGrid::IsInitialized
 */
bool AtomGrid::IsInitialized(void) const {
	return this->isInitializedFlag;
}

/*
 * AtomGrid::operator=
 */
AtomGrid& AtomGrid::operator=(const AtomGrid& rhs) {
	this->atoms = rhs.atoms;
	this->boundingBox = rhs.boundingBox;
	this->cellNum = rhs.cellNum;
	this->cells = rhs.cells;
	this->cellSize = rhs.cellSize;
	this->isInitializedFlag = rhs.isInitializedFlag;
	this->max_radius = rhs.max_radius;
	this->min_cell_dim = rhs.min_cell_dim;

	return *this;
}

/*
 * AtomGrid::RemoveStartSpheres
 */
void AtomGrid::RemoveStartSpheres() {
	// Get the IDs of the start spheres. They where added at the end of the list.
	vec4ui sphere_ids;
	sphere_ids.Set(
		static_cast<uint>(this->atoms.size() - 1),
		static_cast<uint>(this->atoms.size() - 2),
		static_cast<uint>(this->atoms.size() - 3),
		static_cast<uint>(this->atoms.size() - 4));

	// Loop over all cells and remove the start spheres also recompute the maximal
	// radius of all atoms except for the start spheres.
	this->max_radius = 0.0;
	for (auto& cell : this->cells) {
		for (int i = 0; i < 4; i++) {
			auto it = std::remove_if(cell.atom_ids.begin(), cell.atom_ids.end(),
				[&sphere_ids, &i](const uint& id) { return id == sphere_ids[i]; });
			cell.atom_ids.erase(it, cell.atom_ids.end());
		}

		for (const auto atom_id : cell.atom_ids) {
			if (this->atoms[atom_id].GetW() > this->max_radius) {
				this->max_radius = this->atoms[atom_id].GetW();
			}
		}
	}

	// Remove the start spheres from the atoms that the grid stores.
	for (size_t i = 0; i < 4; i++) {
		this->atoms.pop_back();
	}
}

/*
 * AtomGrid::stopLoop
 */
bool AtomGrid::stopLoop(const vec3d& p_center, const size_t p_loop, const vec4d& p_end_vertex) {
	// Compute c_r from the Lindow paper and get the position of the end vertex.
	double cr = (static_cast<double>(p_loop) - 0.5) * this->min_cell_dim;
	vec3d end_vertex = vec3d(p_end_vertex.PeekComponents());

	// Compute the distance approximation.
	double dist = cr - (p_center - end_vertex).Length() - p_end_vertex.GetW();

	// Return the condition.
	return dist > this->max_radius;
}
