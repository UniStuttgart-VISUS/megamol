#include "stdafx.h"
#include "mouse_click_call.h"

namespace megamol
{
    namespace flowvis
    {
        void mouse_click_call::set_coordinates(std::pair<float, float> coordinates)
        {
            std::swap(this->coordinates, coordinates);
        }

        std::pair<float, float> mouse_click_call::get_coordinates() const
        {
            return this->coordinates;
        }
    }
}
