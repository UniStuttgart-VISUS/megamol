#include "stdafx.h"

#include "implicit_topology_results.h"
#include "implicit_topology_writer.h"

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
        }

        bool implicit_topology_writer::create()
        {
            return true;
        }

        void implicit_topology_writer::release()
        {
        }

        void implicit_topology_writer::write(const std::string& filename, const implicit_topology_results& content)
        {
            // TODO
        }
    }
}