/*
 * streamlines_2d.h
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */
#pragma once

#include "mmcore/Call.h"
#include "mmcore/CalleeSlot.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/Module.h"
#include "mmcore/param/ParamSlot.h"

#include "vislib/math/Rectangle.h"

#include "Eigen/Dense"

#include <array>
#include <memory>
#include <vector>

namespace megamol {
namespace flowvis {

/**
 * Module for computing and visualizing the the periodic orbits of a vector field.
 *
 * @author Alexander Straub
 */
class streamlines_2d : public core::Module {
public:
    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static inline const char* ClassName() { return "streamlines_2d"; }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static inline const char* Description() {
        return "Compute streamlines for 2D vector fields";
    }

    /**
     * Answers whether this module is available on the current system.
     *
     * @return 'true' if the module is available, 'false' otherwise.
     */
    static inline bool IsAvailable() { return true; }

    /**
     * Initialises a new instance.
     */
    streamlines_2d();

    /**
     * Finalises an instance.
     */
    virtual ~streamlines_2d();

protected:
    /**
     * Implementation of 'Create'.
     *
     * @return 'true' on success, 'false' otherwise.
     */
    virtual bool create() override;

    /**
     * Implementation of 'Release'.
     */
    virtual void release() override;

private:
    /** Get input data and extent from called modules */
    bool get_input_data();
    bool get_input_extent();

    /**
     * Compute streamlines in a 2D vector field
     */
    bool compute_streamlines();

    /** Callbacks for the computed streamlines */
    bool get_streamlines_data(core::Call& call);
    bool get_streamlines_extent(core::Call& call);

    /** Output slot for the streamlines */
    core::CalleeSlot streamlines_slot;

    /** Input slot for getting the vector field */
    core::CallerSlot vector_field_slot;

    /** Input slot for getting the seed points */
    core::CallerSlot seed_points_slot;

    /** Parameters for streamline computation */
    core::param::ParamSlot integration_method;
    core::param::ParamSlot num_integration_steps;
    core::param::ParamSlot integration_timestep;
    core::param::ParamSlot max_integration_error;
    core::param::ParamSlot direction;

    /** Bounding rectangle and box */
    vislib::math::Rectangle<float> bounding_rectangle;

    /** Input vector field */
    SIZE_T vector_field_hash;
    bool vector_field_changed;

    std::array<unsigned int, 2> resolution;
    std::shared_ptr<std::vector<float>> grid_positions;
    std::shared_ptr<std::vector<float>> vectors;

    /** Input seed points */
    SIZE_T seed_points_hash;
    bool seed_points_changed;

    std::shared_ptr<std::vector<float>> seed_points;

    /** Output streamlines */
    SIZE_T streamlines_hash;

    std::vector<std::pair<float, std::vector<Eigen::Vector2f>>> streamlines;
};

} // namespace flowvis
} // namespace megamol