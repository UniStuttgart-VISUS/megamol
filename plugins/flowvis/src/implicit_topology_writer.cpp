#include "stdafx.h"

#include "implicit_topology_results.h"
#include "implicit_topology_writer.h"

#include "vislib/sys/Log.h"

#include <fstream>
#include <iostream>
#include <string>

namespace megamol
{
    namespace flowvis
    {
        implicit_topology_writer::implicit_topology_writer()
        {
        }

        implicit_topology_writer::~implicit_topology_writer()
        {
            this->Release();
        }

        bool implicit_topology_writer::create()
        {
            return true;
        }

        void implicit_topology_writer::release()
        {
        }

        bool implicit_topology_writer::write(const std::string& filename, const implicit_topology_results& content)
        {
            // Open output file
            try
            {
                std::ofstream ofs(filename, std::ios_base::out | std::ios_base::binary);

                if (!ofs.good())
                {
                    vislib::sys::Log::DefaultLog.WriteWarn("Unable to open implicit topology results file '%s'!", filename.c_str());
                    return false;
                }

                // Gather information for the file header and write it
                const unsigned int num_particles = content.vertices->size() / 2;
                const unsigned int num_indices = content.indices->size();

                ofs.write(reinterpret_cast<const char*>(&num_particles), sizeof(unsigned int));
                ofs.write(reinterpret_cast<const char*>(&num_indices), sizeof(unsigned int));
                ofs.write(reinterpret_cast<const char*>(&content.computation_state.num_integration_steps), sizeof(unsigned int));
                ofs.write(reinterpret_cast<const char*>(&content.computation_state.integration_timestep), sizeof(float));
                ofs.write(reinterpret_cast<const char*>(&content.computation_state.max_integration_error), sizeof(float));

                // Write vertices and indices
                ofs.write(reinterpret_cast<const char*>(content.vertices->data()), content.vertices->size() * sizeof(float));
                ofs.write(reinterpret_cast<const char*>(content.indices->data()), content.indices->size() * sizeof(unsigned int));

                // Write stream line end positions
                ofs.write(reinterpret_cast<const char*>(content.positions_forward->data()), content.positions_forward->size() * sizeof(float));
                ofs.write(reinterpret_cast<const char*>(content.positions_backward->data()), content.positions_backward->size() * sizeof(float));

                // Write labels
                ofs.write(reinterpret_cast<const char*>(content.labels_forward->data()), content.labels_forward->size() * sizeof(float));
                ofs.write(reinterpret_cast<const char*>(content.labels_backward->data()), content.labels_backward->size() * sizeof(float));

                // Write distances
                ofs.write(reinterpret_cast<const char*>(content.distances_forward->data()), content.distances_forward->size() * sizeof(float));
                ofs.write(reinterpret_cast<const char*>(content.distances_backward->data()), content.distances_backward->size() * sizeof(float));

                // Write reason of termination
                ofs.write(reinterpret_cast<const char*>(content.terminations_forward->data()), content.terminations_forward->size() * sizeof(float));
                ofs.write(reinterpret_cast<const char*>(content.terminations_backward->data()), content.terminations_backward->size() * sizeof(float));

                // Finish writing
                ofs.close();
            }
            catch (const std::exception& e)
            {
                vislib::sys::Log::DefaultLog.WriteError("Unable to write implicit topology results file '%s': %s", filename.c_str(), e.what());

                return false;
            }

            return true;
        }
    }
}