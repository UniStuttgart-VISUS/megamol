#include "stdafx.h"

#include "implicit_topology_results.h"
#include "implicit_topology_reader.h"

#include "vislib/sys/Log.h"

#include <fstream>
#include <iostream>
#include <string>

namespace megamol
{
    namespace flowvis
    {
        implicit_topology_reader::implicit_topology_reader()
        {
        }

        implicit_topology_reader::~implicit_topology_reader()
        {
            this->Release();
        }

        bool implicit_topology_reader::create()
        {
            return true;
        }

        void implicit_topology_reader::release()
        {
        }

        bool implicit_topology_reader::read(const std::string& filename, implicit_topology_results& content)
        {
            // Open output file
            try
            {
                std::ifstream ifs(filename, std::ios_base::in | std::ios_base::binary);

                if (!ifs.good())
                {
                    vislib::sys::Log::DefaultLog.WriteWarn("Unable to open implicit topology results file '%s'!", filename.c_str());
                    return false;
                }

                // Gather information for the file header and write it
                unsigned int num_particles, num_indices;

                ifs.read(reinterpret_cast<char*>(&num_particles), sizeof(unsigned int));
                ifs.read(reinterpret_cast<char*>(&num_indices), sizeof(unsigned int));
                ifs.read(reinterpret_cast<char*>(&content.computation_state.num_integration_steps), sizeof(unsigned int));
                ifs.read(reinterpret_cast<char*>(&content.computation_state.integration_timestep), sizeof(float));
                ifs.read(reinterpret_cast<char*>(&content.computation_state.max_integration_error), sizeof(float));

                // Read vertices and indices
                content.vertices = std::make_shared<std::vector<float>>(2 * num_particles);
                content.indices = std::make_shared<std::vector<unsigned int>>(num_indices);

                ifs.read(reinterpret_cast<char*>(content.vertices->data()), content.vertices->size() * sizeof(float));
                ifs.read(reinterpret_cast<char*>(content.indices->data()), content.indices->size() * sizeof(unsigned int));

                // Read stream line end positions
                content.positions_forward = std::make_shared<std::vector<float>>(2 * num_particles);
                content.positions_backward = std::make_shared<std::vector<float>>(2 * num_particles);

                ifs.read(reinterpret_cast<char*>(content.positions_forward->data()), content.positions_forward->size() * sizeof(float));
                ifs.read(reinterpret_cast<char*>(content.positions_backward->data()), content.positions_backward->size() * sizeof(float));

                // Read labels
                content.labels_forward = std::make_shared<std::vector<float>>(num_particles);
                content.labels_backward = std::make_shared<std::vector<float>>(num_particles);

                ifs.read(reinterpret_cast<char*>(content.labels_forward->data()), content.labels_forward->size() * sizeof(float));
                ifs.read(reinterpret_cast<char*>(content.labels_backward->data()), content.labels_backward->size() * sizeof(float));

                // Read distances
                content.distances_forward = std::make_shared<std::vector<float>>(num_particles);
                content.distances_backward = std::make_shared<std::vector<float>>(num_particles);

                ifs.read(reinterpret_cast<char*>(content.distances_forward->data()), content.distances_forward->size() * sizeof(float));
                ifs.read(reinterpret_cast<char*>(content.distances_backward->data()), content.distances_backward->size() * sizeof(float));

                // Read reason of termination
                content.terminations_forward = std::make_shared<std::vector<float>>(num_particles);
                content.terminations_backward = std::make_shared<std::vector<float>>(num_particles);

                ifs.read(reinterpret_cast<char*>(content.terminations_forward->data()), content.terminations_forward->size() * sizeof(float));
                ifs.read(reinterpret_cast<char*>(content.terminations_backward->data()), content.terminations_backward->size() * sizeof(float));

                // Finish reading
                ifs.close();
            }
            catch (const std::exception& e)
            {
                vislib::sys::Log::DefaultLog.WriteError("Unable to read implicit topology results file '%s': %s", filename.c_str(), e.what());

                return false;
            }

            return true;
        }
    }
}