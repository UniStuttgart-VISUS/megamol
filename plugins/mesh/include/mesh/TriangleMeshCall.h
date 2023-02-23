/*
 * TriangleMeshCall.h
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */
#pragma once

#include "mmcore/factories/CallAutoDescription.h"
#include "mmstd/data/AbstractGetDataCall.h"

#include "vislib/math/Cuboid.h"
#include "vislib/math/Rectangle.h"

#include <memory>
#include <vector>

namespace megamol::mesh {
/**
 * Call for transporting a triangle mesh in an ready-to-use fashion (for OpenGL).
 *
 * @author Alexander Straub
 */
class TriangleMeshCall : public core::AbstractGetDataCall {
public:
    typedef core::factories::CallAutoDescription<TriangleMeshCall> triangle_mesh_description;

    /**
     * Human-readable class name
     */
    static const char* ClassName() {
        return "TriangleMeshCall";
    }

    /**
     * Human-readable class description
     */
    static const char* Description() {
        return "Call transporting triangle data";
    }

    /**
     * Number of available functions
     */
    static unsigned int FunctionCount() {
        return 2;
    }

    /**
     * Names of available functions
     */
    static const char* FunctionName(unsigned int idx) {
        switch (idx) {
        case 0:
            return "get_data";
        case 1:
            return "get_extent";
        }

        return nullptr;
    }

    /** Dimension */
    enum class dimension_t { INVALID, TWO, THREE };

    /**
     * Constructor
     */
    TriangleMeshCall();

    /**
     * Getter for the dimension
     */
    dimension_t get_dimension() const;

    /**
     * Setter for the dimension
     */
    void set_dimension(dimension_t dimension);

    /**
     * Getter for the bounding rectangle for 2D data
     */
    const vislib::math::Rectangle<float>& get_bounding_rectangle() const;

    /**
     * Setter for the bounding rectangle for 2D data
     */
    void set_bounding_rectangle(const vislib::math::Rectangle<float>& bounding_rectangle);

    /**
     * Getter for the bounding box for 3D data
     */
    const vislib::math::Cuboid<float>& get_bounding_box() const;

    /**
     * Setter for the bounding box for 3D data
     */
    void set_bounding_box(const vislib::math::Cuboid<float>& bounding_box);

    /**
     * Getter for the vertices defining the triangle mesh
     */
    std::shared_ptr<std::vector<float>> get_vertices() const;

    /**
     * Setter for the vertices defining the triangle mesh
     */
    void set_vertices(std::shared_ptr<std::vector<float>> vertices);

    /**
     * Getter for the normals
     */
    std::shared_ptr<std::vector<float>> get_normals() const;

    /**
     * Setter for the normals
     */
    void set_normals(std::shared_ptr<std::vector<float>> normals);

    /**
     * Getter for the indices defining the triangle mesh
     */
    std::shared_ptr<std::vector<unsigned int>> get_indices() const;

    /**
     * Setter for the indices defining the triangle mesh
     */
    void set_indices(std::shared_ptr<std::vector<unsigned int>> indices);

protected:
    /** Dimension */
    dimension_t dimension;

    /** Bounding rectangle */
    vislib::math::Rectangle<float> bounding_rectangle;

    /** Bounding box */
    vislib::math::Cuboid<float> bounding_box;

    /** Vertices, normals and indices defining the triangle mesh */
    std::shared_ptr<std::vector<float>> vertices;
    std::shared_ptr<std::vector<float>> normals;
    std::shared_ptr<std::vector<unsigned int>> indices;
};
} // namespace megamol::mesh
