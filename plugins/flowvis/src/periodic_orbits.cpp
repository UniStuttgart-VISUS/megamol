#include "stdafx.h"
#include "periodic_orbits.h"

#include "glyph_data_call.h"
#include "vector_field_call.h"

#include "mmcore/Call.h"

#include "vislib/math/Rectangle.h"
#include "vislib/sys/Log.h"

#include <array>

namespace megamol
{
    namespace flowvis
    {
        periodic_orbits::periodic_orbits() :
            glyph_slot("set_glyphs", "Glyph output"),
            vector_field_slot("get_vector_field", "Vector field input"),
            vector_field_hash(-1)
        {
            // Connect output
            this->glyph_slot.SetCallback(glyph_data_call::ClassName(), glyph_data_call::FunctionName(0), &periodic_orbits::get_glyph_data_callback);
            this->glyph_slot.SetCallback(glyph_data_call::ClassName(), glyph_data_call::FunctionName(1), &periodic_orbits::get_glyph_extent_callback);
            this->MakeSlotAvailable(&this->glyph_slot);

            // Connect input
            this->vector_field_slot.SetCompatibleCall<vector_field_call::vector_field_description>();
            this->MakeSlotAvailable(&this->vector_field_slot);
        }

        periodic_orbits::~periodic_orbits()
        {
            this->Release();
        }

        bool periodic_orbits::create()
        {
            return true;
        }

        void periodic_orbits::release()
        {
        }

        bool periodic_orbits::get_glyph_data_callback(core::Call& call)
        {
            auto* glyph_call = dynamic_cast<glyph_data_call*>(&call);

            if (glyph_call != nullptr)
            {
                // Get vector field
                auto* get_vector_field = this->vector_field_slot.CallAs<vector_field_call>();

                if (get_vector_field != nullptr && (*get_vector_field)(0) && get_vector_field->DataHash() != this->vector_field_hash)
                {
                    // TODO
                }
            }

            return true;
        }

        bool periodic_orbits::get_glyph_extent_callback(core::Call& call)
        {
            auto* get_vector_field = this->vector_field_slot.CallAs<vector_field_call>();

            return get_vector_field != nullptr && (*get_vector_field)(1);
        }
    }
}
