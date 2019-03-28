#include "stdafx.h"
#include "implicit_topology.h"

#include "implicit_topology_computation.h"
#include "mesh_data_call.h"
#include "triangle_mesh_call.h"
#include "triangulation.h"

#include "mmcore/Call.h"
#include "mmcore/param/BoolParam.h"
#include "mmcore/param/ButtonParam.h"
#include "mmcore/param/FloatParam.h"
#include "mmcore/param/FilePathParam.h"
#include "mmcore/param/IntParam.h"
#include "mmcore/param/LinearTransferFunctionParam.h"

#include "vislib/math/Rectangle.h"
#include "vislib/sys/Log.h"

#include "glad/glad.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <future>
#include <iostream>
#include <limits>
#include <memory>
#include <vector>

namespace megamol
{
    namespace flowvis
    {
        implicit_topology::implicit_topology() :
            triangle_mesh_slot("set_triangle_mesh", "Triangle mesh output"),
            mesh_data_slot("set_mesh_data", "Mesh data output"),
            start_computation("start_computation", "Start the computation"),
            stop_computation("stop_computation", "Stop the computation"),
            reset_computation("reset_computation", "Reset the computation"),
            vector_field_path("vector_field_path", "Path to the input vector field"),
            convergence_structures_path("convergence_structures_path", "Path to the input convergence structures"),
            label_transfer_function("label_transfer_function", "Transfer function for labels"),
            distance_transfer_function("distance_transfer_function", "Transfer function for distances"),
            termination_transfer_function("termination_transfer_function", "Transfer function for reasons of termination"),
            num_integration_steps("num_integration_steps", "Number of stream line integration steps"),
            integration_timestep("integration_timestep", "Initial time step for stream line integration"),
            max_integration_error("max_integration_error", "Maximum integration error for Runge-Kutta 4-5"),
            num_particles_per_batch("num_particles_per_batch", "Number of particles per batch (influences GPU utilization)"),
            num_integration_steps_per_batch("num_integration_steps_per_batch", "Number of integration steps per batch, after which a result can be visualized"),
            refinement_threshold("refinement_threshold", "Threshold for grid refinement, defined as minimum edge length"),
            refine_at_labels("refine_at_labels", "Should the grid be refined in regions of different labels?"),
            distance_difference_threshold("distance_difference_threshold", "Threshold for refining the grid when neighboring nodes exceed a distance difference"),
            computation_running(false), mesh_output_changed(false), data_output_changed(false), computation(nullptr)
        {
            // Connect output
            this->triangle_mesh_slot.SetCallback(triangle_mesh_call::ClassName(), triangle_mesh_call::FunctionName(0), &implicit_topology::get_triangle_data_callback);
            this->triangle_mesh_slot.SetCallback(triangle_mesh_call::ClassName(), triangle_mesh_call::FunctionName(1), &implicit_topology::get_triangle_extent_callback);
            this->MakeSlotAvailable(&this->triangle_mesh_slot);

            this->mesh_data_slot.SetCallback(mesh_data_call::ClassName(), mesh_data_call::FunctionName(0), &implicit_topology::get_data_data_callback);
            this->mesh_data_slot.SetCallback(mesh_data_call::ClassName(), mesh_data_call::FunctionName(1), &implicit_topology::get_data_extent_callback);
            this->MakeSlotAvailable(&this->mesh_data_slot);

            // Create path parameters
            this->vector_field_path << new core::param::FilePathParam("");
            this->MakeSlotAvailable(&this->vector_field_path);

            this->convergence_structures_path << new core::param::FilePathParam("");
            this->MakeSlotAvailable(&this->convergence_structures_path);

            // Create computation parameters
            this->num_integration_steps << new core::param::IntParam(0);
            this->MakeSlotAvailable(&this->num_integration_steps);

            this->integration_timestep << new core::param::FloatParam(0.1f);
            this->MakeSlotAvailable(&this->integration_timestep);

            this->max_integration_error << new core::param::FloatParam(1.0f);
            this->MakeSlotAvailable(&this->max_integration_error);

            this->num_particles_per_batch << new core::param::IntParam(10000);
            this->MakeSlotAvailable(&this->num_particles_per_batch);

            this->num_integration_steps_per_batch << new core::param::IntParam(1000);
            this->MakeSlotAvailable(&this->num_integration_steps_per_batch);

            this->refinement_threshold << new core::param::FloatParam(1.0f);
            this->MakeSlotAvailable(&this->refinement_threshold);

            this->refine_at_labels << new core::param::BoolParam(false);
            this->MakeSlotAvailable(&this->refine_at_labels);

            this->distance_difference_threshold << new core::param::FloatParam(0.0f);
            this->MakeSlotAvailable(&this->distance_difference_threshold);

            // Create computation buttons
            this->start_computation << new core::param::ButtonParam();
            this->start_computation.SetUpdateCallback(&implicit_topology::start_computation_callback);
            this->MakeSlotAvailable(&this->start_computation);

            this->stop_computation << new core::param::ButtonParam();
            this->stop_computation.SetUpdateCallback(&implicit_topology::stop_computation_callback);
            this->MakeSlotAvailable(&this->stop_computation);

            this->reset_computation << new core::param::ButtonParam();
            this->reset_computation.SetUpdateCallback(&implicit_topology::reset_computation_callback);
            this->MakeSlotAvailable(&this->reset_computation);

            // Create transfer function parameters
            this->label_transfer_function << new core::param::LinearTransferFunctionParam();
            this->MakeSlotAvailable(&this->label_transfer_function);

            this->distance_transfer_function << new core::param::LinearTransferFunctionParam();
            this->MakeSlotAvailable(&this->distance_transfer_function);

            this->termination_transfer_function << new core::param::LinearTransferFunctionParam();
            this->MakeSlotAvailable(&this->termination_transfer_function);
        }

        implicit_topology::~implicit_topology()
        {
            this->Release();
        }

        bool implicit_topology::create()
        {
            return true;
        }

        void implicit_topology::release()
        {
        }

        bool implicit_topology::initialize_computation()
        {
            // Try to load input vector field
            if (this->computation == nullptr)
            {
                std::ifstream vectors_ifs(this->vector_field_path.Param<core::param::FilePathParam>()->Value(), std::ios_base::in | std::ios_base::binary);
                std::ifstream structures_ifs(this->convergence_structures_path.Param<core::param::FilePathParam>()->Value(), std::ios_base::in | std::ios_base::binary);

                if (vectors_ifs.good() && structures_ifs.good())
                {
                    // Get dimension from file
                    unsigned int dimension, components;

                    vectors_ifs.read(reinterpret_cast<char*>(&dimension), sizeof(unsigned int));
                    vectors_ifs.read(reinterpret_cast<char*>(&components), sizeof(unsigned int));

                    if (dimension != 2)
                    {
                        vislib::sys::Log::DefaultLog.WriteError("Vector field file must have exactly two dimensions '%s'",
                            this->vector_field_path.Param<core::param::FilePathParam>()->Value());

                        return false;
                    }

                    if (components != 2)
                    {
                        vislib::sys::Log::DefaultLog.WriteError("Vectors must have exactly two components '%s'",
                            this->vector_field_path.Param<core::param::FilePathParam>()->Value());

                        return false;
                    }

                    // Read extents from file
                    float x_min, x_max, y_min, y_max;
                    unsigned int x_num, y_num, num;

                    vectors_ifs.read(reinterpret_cast<char*>(&x_num), sizeof(unsigned int));
                    vectors_ifs.read(reinterpret_cast<char*>(&x_min), sizeof(float));
                    vectors_ifs.read(reinterpret_cast<char*>(&x_max), sizeof(float));
                    vectors_ifs.read(reinterpret_cast<char*>(&y_num), sizeof(unsigned int));
                    vectors_ifs.read(reinterpret_cast<char*>(&y_min), sizeof(float));
                    vectors_ifs.read(reinterpret_cast<char*>(&y_max), sizeof(float));

                    num = x_num * y_num;

                    // Read file content
                    const float x_step = (x_max - x_min) / (x_num - 1);
                    const float y_step = (y_max - y_min) / (y_num - 1);

                    std::vector<GLfloat> positions(num * 2);
                    std::vector<GLfloat> vectors(num * 2);

                    for (unsigned int y = 0; y < y_num; ++y)
                    {
                        for (unsigned int x = 0; x < x_num; ++x)
                        {
                            const unsigned int xy = y * x_num + x;

                            // Calculate positions
                            const float x_pos = x_min + x * x_step;
                            const float y_pos = y_min + y * y_step;

                            positions[xy * 2 + 0] = x_pos;
                            positions[xy * 2 + 1] = y_pos;

                            // Read vectors
                            vectors_ifs.read(reinterpret_cast<char*>(&vectors[xy * 2 + 0]), sizeof(float));
                            vectors_ifs.read(reinterpret_cast<char*>(&vectors[xy * 2 + 1]), sizeof(float));
                        }
                    }

                    vectors_ifs.close();

                    // Load convergence structures
                    unsigned int num_convergence_structures;

                    structures_ifs.read(reinterpret_cast<char*>(&num_convergence_structures), sizeof(unsigned int));

                    std::vector<GLfloat> points, lines;
                    std::vector<int> point_ids, line_ids;

                    points.reserve(2 * num_convergence_structures);
                    lines.reserve(4 * num_convergence_structures);

                    point_ids.reserve(num_convergence_structures);
                    line_ids.reserve(num_convergence_structures);

                    for (unsigned int i = 0; i < num_convergence_structures; ++i)
                    {
                        unsigned int type;
                        structures_ifs.read(reinterpret_cast<char*>(&type), sizeof(unsigned int));

                        float value;

                        switch (type)
                        {
                        case 0: // Point
                            structures_ifs.read(reinterpret_cast<char*>(&value), sizeof(float));
                            points.push_back(value);
                            structures_ifs.read(reinterpret_cast<char*>(&value), sizeof(float));
                            points.push_back(value);

                            point_ids.push_back(i);

                            break;
                        case 1: // Line
                            structures_ifs.read(reinterpret_cast<char*>(&value), sizeof(float));
                            lines.push_back(value);
                            structures_ifs.read(reinterpret_cast<char*>(&value), sizeof(float));
                            lines.push_back(value);

                            structures_ifs.read(reinterpret_cast<char*>(&value), sizeof(float));
                            lines.push_back(value);
                            structures_ifs.read(reinterpret_cast<char*>(&value), sizeof(float));
                            lines.push_back(value);

                            line_ids.push_back(i);

                            break;
                        default:
                            vislib::sys::Log::DefaultLog.WriteError("Unknown convergence structure type in file '%s'!",
                                this->convergence_structures_path.Param<core::param::FilePathParam>()->Value());

                            return false;
                        }
                    }

                    // Create new computation object
                    this->computation = std::make_unique<implicit_topology_computation>(std::array<int, 2>{ static_cast<int>(x_num), static_cast<int>(y_num) },
                        std::array<float, 4>{ x_min, x_max, y_min, y_max },
                        std::move(positions), std::move(vectors), std::move(points), std::move(point_ids), std::move(lines), std::move(line_ids),
                        this->integration_timestep.Param<core::param::FloatParam>()->Value(),
                        this->max_integration_error.Param<core::param::FloatParam>()->Value());

                    set_readonly_fixed_parameters(true);
                }
                else if (!vectors_ifs.good())
                {
                    vislib::sys::Log::DefaultLog.WriteWarn("Unable to open input vector field file '%s'!",
                        this->vector_field_path.Param<core::param::FilePathParam>()->Value());

                    return false;
                }
                else if (!structures_ifs.good())
                {
                    vislib::sys::Log::DefaultLog.WriteWarn("Unable to open input convergence structures file '%s'!",
                        this->convergence_structures_path.Param<core::param::FilePathParam>()->Value());

                    return false;
                }
            }

            return true;
        }

        void implicit_topology::update_results()
        {
            // Try to get new results
            if (this->computation_running && !(this->mesh_output_changed || this->data_output_changed))
            {
                // Get new results
                if (this->last_result.wait_for(std::chrono::milliseconds(1)) != std::future_status::ready)
                {
                    return;
                }

                vislib::sys::Log::DefaultLog.WriteInfo("Computation of stream lines yielded new results.");

                // Store triangles
                auto result = this->last_result.get();

                this->vertices = result.vertices;
                this->indices = result.indices;

                this->labels_forward = result.labels_forward;
                this->distances_forward = result.distances_forward;
                this->terminations_forward = result.terminations_forward;

                this->labels_backward = result.labels_backward;
                this->distances_backward = result.distances_backward;
                this->terminations_backward = result.terminations_backward;

                this->computation_running = !result.finished;

                if (result.finished)
                {
                    vislib::sys::Log::DefaultLog.WriteInfo("Computation of stream lines ended.");

                    // Reset parameters to read-write
                    set_readonly_variable_parameters(false);
                }

                // Save new last result
                this->last_result = this->computation->get_results();

                this->mesh_output_changed = true;
                this->data_output_changed = true;
            }
        }

        void implicit_topology::set_readonly_fixed_parameters(const bool read_only)
        {
            this->vector_field_path.Parameter()->SetGUIReadOnly(read_only);
            this->convergence_structures_path.Parameter()->SetGUIReadOnly(read_only);

            this->integration_timestep.Parameter()->SetGUIReadOnly(read_only);
            this->max_integration_error.Parameter()->SetGUIReadOnly(read_only);
        }

        void implicit_topology::set_readonly_variable_parameters(const bool read_only)
        {
            this->num_integration_steps.Parameter()->SetGUIReadOnly(read_only);
            this->num_particles_per_batch.Parameter()->SetGUIReadOnly(read_only);
            this->num_integration_steps_per_batch.Parameter()->SetGUIReadOnly(read_only);

            this->refinement_threshold.Parameter()->SetGUIReadOnly(read_only);
            this->refine_at_labels.Parameter()->SetGUIReadOnly(read_only);
            this->distance_difference_threshold.Parameter()->SetGUIReadOnly(read_only);
        }

        bool implicit_topology::get_triangle_data_callback(core::Call& call)
        {
            auto* triangle_call = dynamic_cast<triangle_mesh_call*>(&call);
            if (triangle_call == nullptr) return false;

            // Update render output if there are new results
            update_results();

            if (this->mesh_output_changed)
            {
                triangle_call->set_vertices(this->vertices);
                triangle_call->set_indices(this->indices);

                triangle_call->SetDataHash(triangle_call->DataHash() + 1);

                this->mesh_output_changed = false;
            }
            
            return true;
        }

        bool implicit_topology::get_triangle_extent_callback(core::Call& call)
        {
            auto* triangle_call = dynamic_cast<triangle_mesh_call*>(&call);
            if (triangle_call == nullptr) return false;

            if (this->vector_field_path.IsDirty())
            {
                // Try to load input vector field
                std::ifstream vectors_ifs(this->vector_field_path.Param<core::param::FilePathParam>()->Value(), std::ios_base::in | std::ios_base::binary);

                if (vectors_ifs.good())
                {
                    // Read dimension from file
                    unsigned int dimension, components;

                    vectors_ifs.read(reinterpret_cast<char*>(&dimension), sizeof(unsigned int));
                    vectors_ifs.read(reinterpret_cast<char*>(&components), sizeof(unsigned int));

                    if (dimension != 2)
                    {
                        vislib::sys::Log::DefaultLog.WriteError("Vector field file must have exactly two dimensions '%s'",
                            this->vector_field_path.Param<core::param::FilePathParam>()->Value());

                        return false;
                    }

                    if (components != 2)
                    {
                        vislib::sys::Log::DefaultLog.WriteError("Vectors must have exactly two components '%s'",
                            this->vector_field_path.Param<core::param::FilePathParam>()->Value());

                        return false;
                    }

                    // Read extents from file
                    float x_min, x_max, y_min, y_max;

                    vectors_ifs.ignore(sizeof(unsigned int));
                    vectors_ifs.read(reinterpret_cast<char*>(&x_min), sizeof(float));
                    vectors_ifs.read(reinterpret_cast<char*>(&x_max), sizeof(float));
                    vectors_ifs.ignore(sizeof(unsigned int));
                    vectors_ifs.read(reinterpret_cast<char*>(&y_min), sizeof(float));
                    vectors_ifs.read(reinterpret_cast<char*>(&y_max), sizeof(float));

                    triangle_call->set_bounding_rectangle(vislib::math::Rectangle<float>(x_min, y_min, x_max, y_max));
                }
                else
                {
                    triangle_call->SetDataHash(0);

                    this->vector_field_path.ResetDirty();

                    return false;
                }
            }

            return true;
        }

        bool implicit_topology::get_data_data_callback(core::Call& call)
        {
            auto* data_call = dynamic_cast<mesh_data_call*>(&call);
            if (data_call == nullptr) return false;

            // Update render output if there are new results
            update_results();

            if (this->data_output_changed)
            {
                // Prepare labels
                {
                    auto label_data = std::make_shared<mesh_data_call::data_set>();
                    label_data->transfer_function = this->label_transfer_function.Param<core::param::LinearTransferFunctionParam>()->Value();

                    const auto min_max_value = std::minmax_element(this->labels_forward->begin(), this->labels_forward->end());
                    label_data->min_value = *min_max_value.first;
                    label_data->max_value = *min_max_value.second;

                    label_data->data = this->labels_forward;

                    data_call->set_data("labels (forward)", label_data);
                }
                {
                    auto label_data = std::make_shared<mesh_data_call::data_set>();
                    label_data->transfer_function = this->label_transfer_function.Param<core::param::LinearTransferFunctionParam>()->Value();

                    const auto min_max_value = std::minmax_element(this->labels_backward->begin(), this->labels_backward->end());
                    label_data->min_value = *min_max_value.first;
                    label_data->max_value = *min_max_value.second;

                    label_data->data = this->labels_backward;

                    data_call->set_data("labels (backward)", label_data);
                }
                
                // Prepare distances
                {
                    auto distance_data = std::make_shared<mesh_data_call::data_set>();
                    distance_data->transfer_function = this->distance_transfer_function.Param<core::param::LinearTransferFunctionParam>()->Value();

                    const auto min_max_value = std::minmax_element(this->distances_forward->begin(), this->distances_forward->end());
                    distance_data->min_value = *min_max_value.first;
                    distance_data->max_value = *min_max_value.second;

                    distance_data->data = this->distances_forward;

                    data_call->set_data("distances (forward)", distance_data);
                }
                {
                    auto distance_data = std::make_shared<mesh_data_call::data_set>();
                    distance_data->transfer_function = this->distance_transfer_function.Param<core::param::LinearTransferFunctionParam>()->Value();

                    const auto min_max_value = std::minmax_element(this->distances_backward->begin(), this->distances_backward->end());
                    distance_data->min_value = *min_max_value.first;
                    distance_data->max_value = *min_max_value.second;

                    distance_data->data = this->distances_backward;

                    data_call->set_data("distances (backward)", distance_data);
                }

                // Prepare reasons for termination
                {
                    auto termination_data = std::make_shared<mesh_data_call::data_set>();
                    termination_data->transfer_function = this->termination_transfer_function.Param<core::param::LinearTransferFunctionParam>()->Value();

                    const auto min_max_value = std::minmax_element(this->terminations_forward->begin(), this->terminations_forward->end());
                    termination_data->min_value = *min_max_value.first;
                    termination_data->max_value = *min_max_value.second;

                    termination_data->data = this->terminations_forward;

                    data_call->set_data("reasons for termination (forward)", termination_data);
                }
                {
                    auto termination_data = std::make_shared<mesh_data_call::data_set>();
                    termination_data->transfer_function = this->termination_transfer_function.Param<core::param::LinearTransferFunctionParam>()->Value();

                    const auto min_max_value = std::minmax_element(this->terminations_forward->begin(), this->terminations_forward->end());
                    termination_data->min_value = *min_max_value.first;
                    termination_data->max_value = *min_max_value.second;

                    termination_data->data = this->terminations_forward;

                    data_call->set_data("reasons for termination (backward)", termination_data);
                }

                // Set new data hash
                data_call->SetDataHash(data_call->DataHash() + 1);

                this->data_output_changed = false;
            }

            // Update transfer functions
            if (this->label_transfer_function.IsDirty())
            {
                {
                    auto label_data = data_call->get_data("labels (forward)");

                    if (label_data != nullptr)
                    {
                        label_data->transfer_function = this->label_transfer_function.Param<core::param::LinearTransferFunctionParam>()->Value();
                        label_data->transfer_function_dirty = true;
                    }
                }
                {
                    auto label_data = data_call->get_data("labels (backward)");

                    if (label_data != nullptr)
                    {
                        label_data->transfer_function = this->label_transfer_function.Param<core::param::LinearTransferFunctionParam>()->Value();
                        label_data->transfer_function_dirty = true;
                    }
                }

                this->label_transfer_function.ResetDirty();
            }

            if (this->distance_transfer_function.IsDirty())
            {
                {
                    auto distance_data = data_call->get_data("distances (forward)");

                    if (distance_data != nullptr)
                    {
                        distance_data->transfer_function = this->distance_transfer_function.Param<core::param::LinearTransferFunctionParam>()->Value();
                        distance_data->transfer_function_dirty = true;
                    }
                }
                {
                    auto distance_data = data_call->get_data("distances (backward)");

                    if (distance_data != nullptr)
                    {
                        distance_data->transfer_function = this->distance_transfer_function.Param<core::param::LinearTransferFunctionParam>()->Value();
                        distance_data->transfer_function_dirty = true;
                    }
                }

                this->distance_transfer_function.ResetDirty();
            }

            if (this->termination_transfer_function.IsDirty())
            {
                {
                    auto termination_data = data_call->get_data("reasons for termination (forward)");

                    if (termination_data != nullptr)
                    {
                        termination_data->transfer_function = this->termination_transfer_function.Param<core::param::LinearTransferFunctionParam>()->Value();
                        termination_data->transfer_function_dirty = true;
                    }
                }
                {
                    auto termination_data = data_call->get_data("reasons for termination (backward)");

                    if (termination_data != nullptr)
                    {
                        termination_data->transfer_function = this->termination_transfer_function.Param<core::param::LinearTransferFunctionParam>()->Value();
                        termination_data->transfer_function_dirty = true;
                    }
                }

                this->termination_transfer_function.ResetDirty();
            }

            return true;
        }

        bool implicit_topology::get_data_extent_callback(core::Call& call)
        {
            auto* data_call = dynamic_cast<mesh_data_call*>(&call);
            if (data_call == nullptr) return false;

            data_call->set_data("labels (forward)");
            data_call->set_data("labels (backward)");

            data_call->set_data("distances (forward)");
            data_call->set_data("distances (backward)");

            data_call->set_data("reasons for termination (forward)");
            data_call->set_data("reasons for termination (backward)");

            return true;
        }

        bool implicit_topology::start_computation_callback(core::param::ParamSlot&)
        {
            // Initialize computation object
            if (!initialize_computation())
            {
                return false;
            }

            // Start computation with current values
            this->computation->start(this->num_integration_steps.Param<core::param::IntParam>()->Value(),
                this->refinement_threshold.Param<core::param::FloatParam>()->Value(),
                this->refine_at_labels.Param<core::param::BoolParam>()->Value(),
                this->distance_difference_threshold.Param<core::param::FloatParam>()->Value(),
                this->num_particles_per_batch.Param<core::param::IntParam>()->Value(),
                this->num_integration_steps_per_batch.Param<core::param::IntParam>()->Value());

            this->last_result = this->computation->get_results();

            this->computation_running = true;

            vislib::sys::Log::DefaultLog.WriteInfo("Computation of stream lines started...");

            // Set parameters to read-only
            set_readonly_variable_parameters(true);

            return true;
        }

        bool implicit_topology::stop_computation_callback(core::param::ParamSlot&)
        {
            // Terminate computation
            if (this->computation != nullptr && this->computation_running)
            {
                this->computation->terminate();

                vislib::sys::Log::DefaultLog.WriteInfo("Computation of stream lines terminated!");
            }

            this->computation_running = false;

            // Reset parameters to read-write
            set_readonly_variable_parameters(false);

            return true;
        }

        bool implicit_topology::reset_computation_callback(core::param::ParamSlot&)
        {
            // Terminate earlier computation
            stop_computation_callback();
            this->computation = nullptr;

            // Reset parameters to read-write
            set_readonly_fixed_parameters(false);
            set_readonly_variable_parameters(false);

            return true;
        }
    }
}
