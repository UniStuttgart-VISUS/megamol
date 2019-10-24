#include "stdafx.h"
#include "periodic_orbits_theisel.h"

#include "glyph_data_call.h"
#include "vector_field_call.h"

#include "mmcore/Call.h"
#include "mmcore/DirectDataWriterCall.h"
#include "mmcore/param/EnumParam.h"
#include "mmcore/param/FloatParam.h"
#include "mmcore/param/IntParam.h"
#include "mmcore/param/TransferFunctionParam.h"

#include "mesh/CallGPUMeshData.h"

#include "vislib/math/Cuboid.h"
#include "vislib/math/Rectangle.h"
#include "vislib/sys/Log.h"

#include "Eigen/Dense"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <future>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <utility>
#include <vector>

namespace megamol
{
    namespace flowvis
    {
        periodic_orbits_theisel::periodic_orbits_theisel() :
            periodic_orbits_slot("periodic_orbits", "Computed periodic orbits as line glyphs"),
            stream_surface_slot("stream_surface", "Computed stream surfaces"),
            seed_line_slot("seed_line", "Computed seed lines"),
            result_writer_slot("result_writer_slot", "Results writer for storing periodic orbits"),
            vector_field_slot("vector_field_slot", "Vector field input"),
            critical_points_slot("critical_points", "Critical points input"),
            transfer_function("transfer_function", "Transfer function for coloring the stream surfaces"),
            integration_method("integration_method", "Method for stream line integration"),
            num_integration_steps("num_integration_steps", "Number of stream line integration steps"),
            integration_timestep("integration_timestep", "Initial time step for stream line integration"),
            max_integration_error("max_integration_error", "Maximum integration error for Runge-Kutta 4-5")
        {
            // Connect output
            this->periodic_orbits_slot.SetCallback(glyph_data_call::ClassName(), glyph_data_call::FunctionName(0), &periodic_orbits_theisel::get_periodic_orbits_data);
            this->periodic_orbits_slot.SetCallback(glyph_data_call::ClassName(), glyph_data_call::FunctionName(1), &periodic_orbits_theisel::get_periodic_orbits_extent);
            this->MakeSlotAvailable(&this->periodic_orbits_slot);

            this->stream_surface_slot.SetCallback(mesh::CallGPUMeshData::ClassName(), mesh::CallGPUMeshData::FunctionName(0), &periodic_orbits_theisel::get_stream_surfaces_data);
            this->stream_surface_slot.SetCallback(mesh::CallGPUMeshData::ClassName(), mesh::CallGPUMeshData::FunctionName(1), &periodic_orbits_theisel::get_stream_surfaces_extent);
            this->MakeSlotAvailable(&this->stream_surface_slot);

            this->seed_line_slot.SetCallback(glyph_data_call::ClassName(), glyph_data_call::FunctionName(0), &periodic_orbits_theisel::get_seed_lines_data);
            this->seed_line_slot.SetCallback(glyph_data_call::ClassName(), glyph_data_call::FunctionName(1), &periodic_orbits_theisel::get_seed_lines_extent);
            this->MakeSlotAvailable(&this->seed_line_slot);

            this->result_writer_slot.SetCallback(core::DirectDataWriterCall::ClassName(), core::DirectDataWriterCall::FunctionName(0), &periodic_orbits_theisel::get_writer_callback);
            this->MakeSlotAvailable(&this->result_writer_slot);
            this->get_writer = []() -> std::ostream& { static std::ostream dummy(nullptr); return dummy; };

            // Connect input
            this->vector_field_slot.SetCompatibleCall<vector_field_call::vector_field_description>();
            this->MakeSlotAvailable(&this->vector_field_slot);

            this->critical_points_slot.SetCompatibleCall<glyph_data_call::glyph_data_description>();
            this->MakeSlotAvailable(&this->critical_points_slot);

            // Create transfer function parameters
            this->transfer_function << new core::param::TransferFunctionParam(
                "{\"Interpolation\":\"LINEAR\",\"Nodes\":[[0.0,0.0,0.423499,1.0,0.0],[0.0,0.119346,0.529237,1.0,0.125],"
                "[0.0,0.238691,0.634976,1.0,0.1875],[0.0,0.346852,0.68788,1.0,0.25],[0.0,0.45022,0.718141,1.0,0.3125],"
                "[0.0,0.553554,0.664839,1.0,0.375],[0.0,0.651082,0.519303,1.0,0.4375],[0.115841,0.72479,0.352857,1.0,0."
                "5],"
                "[0.326771,0.781195,0.140187,1.0,0.5625],[0.522765,0.798524,0.0284624,1.0,0.625],[0.703162,0.788685,0."
                "00885756,1.0,0.6875],"
                "[0.845118,0.751133,0.0,1.0,0.75],[0.955734,0.690825,0.0,1.0,0.8125],[0.995402,0.567916,0.0618524,1.0,"
                "0.875],"
                "[0.987712,0.403398,0.164851,1.0,0.9375],[0.980407,0.247105,0.262699,1.0,1.0]],\"TextureSize\":128}");
            this->MakeSlotAvailable(&this->transfer_function);

            // Create computation parameters
            this->integration_method << new core::param::EnumParam(0);
            this->integration_method.Param<core::param::EnumParam>()->SetTypePair(0, "Runge-Kutta 4 (fixed)");
            this->integration_method.Param<core::param::EnumParam>()->SetTypePair(1, "Runge-Kutta 4-5 (dynamic)");
            this->MakeSlotAvailable(&this->integration_method);

            this->num_integration_steps << new core::param::IntParam(0);
            this->MakeSlotAvailable(&this->num_integration_steps);

            this->integration_timestep << new core::param::FloatParam(0.01f);
            this->MakeSlotAvailable(&this->integration_timestep);

            this->max_integration_error << new core::param::FloatParam(0.000001f);
            this->MakeSlotAvailable(&this->max_integration_error);
        }

        periodic_orbits_theisel::~periodic_orbits_theisel()
        {
            this->Release();
        }

        bool periodic_orbits_theisel::create()
        {
            return true;
        }

        void periodic_orbits_theisel::release()
        {
        }

        bool periodic_orbits_theisel::get_periodic_orbits_data(core::Call& call)
        {


            return true;
        }

        bool periodic_orbits_theisel::get_periodic_orbits_extent(core::Call& call)
        {


            return true;
        }

        bool periodic_orbits_theisel::get_stream_surfaces_data(core::Call& call)
        {


            return true;
        }

        bool periodic_orbits_theisel::get_stream_surfaces_extent(core::Call& call)
        {


            return true;
        }

        bool periodic_orbits_theisel::get_seed_lines_data(core::Call& call)
        {


            return true;
        }

        bool periodic_orbits_theisel::get_seed_lines_extent(core::Call& call)
        {


            return true;
        }

        bool periodic_orbits_theisel::get_writer_callback(core::Call& call)
        {


            return true;
        }
    }
}
