#include "mesh/TriangleMeshCall.h"

#include "vislib/math/Cuboid.h"
#include "vislib/math/Rectangle.h"

#include <memory>
#include <vector>

namespace megamol {
namespace mesh {
TriangleMeshCall::TriangleMeshCall() : dimension(dimension_t::INVALID) {}

TriangleMeshCall::dimension_t TriangleMeshCall::get_dimension() const {
    return this->dimension;
}

void TriangleMeshCall::set_dimension(dimension_t dimension) {
    this->dimension = dimension;
}

const vislib::math::Rectangle<float>& TriangleMeshCall::get_bounding_rectangle() const {
    return this->bounding_rectangle;
}

void TriangleMeshCall::set_bounding_rectangle(const vislib::math::Rectangle<float>& bounding_rectangle) {
    this->bounding_rectangle = bounding_rectangle;
}

const vislib::math::Cuboid<float>& TriangleMeshCall::get_bounding_box() const {
    return this->bounding_box;
}

void TriangleMeshCall::set_bounding_box(const vislib::math::Cuboid<float>& bounding_box) {
    this->bounding_box = bounding_box;
}

std::shared_ptr<std::vector<float>> TriangleMeshCall::get_vertices() const {
    return this->vertices;
}

void TriangleMeshCall::set_vertices(std::shared_ptr<std::vector<float>> vertices) {
    this->vertices = vertices;
}

std::shared_ptr<std::vector<float>> TriangleMeshCall::get_normals() const {
    return this->normals;
}

void TriangleMeshCall::set_normals(std::shared_ptr<std::vector<float>> normals) {
    this->normals = normals;
}

std::shared_ptr<std::vector<unsigned int>> TriangleMeshCall::get_indices() const {
    return this->indices;
}

void TriangleMeshCall::set_indices(std::shared_ptr<std::vector<unsigned int>> indices) {
    this->indices = indices;
}
} // namespace mesh
} // namespace megamol
