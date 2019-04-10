/*
 * mouse_click_call.h
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */
#pragma once

#include "mmcore/Call.h"
#include "mmcore/factories/CallAutoDescription.h"

#include <array>
#include <memory>
#include <vector>

namespace megamol
{
    namespace flowvis
    {
        /**
        * Call for transporting 2D coordinates of a mouse click.
        *
        * @author Alexander Straub
        */
        class mouse_click_call : public core::Call
        {
        public:
            typedef core::factories::CallAutoDescription<mouse_click_call> mouse_click_description;

            /**
            * Human-readable class name
            */
            static const char* ClassName() { return "mouse_click_call"; }

            /**
            * Human-readable class description
            */
            static const char* Description() { return "Call transporting mouse coordinates"; }

            /**
            * Number of available functions
            */
            static unsigned int FunctionCount() { return 1; }

            /**
            * Names of available functions
            */
            static const char* FunctionName(unsigned int idx)
            {
                switch (idx)
                {
                case 0: return "get_coordinates";
                }

                return nullptr;
            }

            /**
            * Getter for the mouse coordinates
            */
            void set_coordinates(std::pair<float, float> coordinates);

            /**
            * Getter for the mouse coordinates
            */
            std::pair<float, float> get_coordinates() const;

        protected:
            /** Mouse coordinates */
            std::pair<float, float> coordinates;
        };
    }
}
