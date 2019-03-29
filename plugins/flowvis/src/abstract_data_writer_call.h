/*
 * abstract_data_writer_call.h
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */
#pragma once

#include "mmcore/Call.h"

namespace megamol
{
    namespace flowvis
    {
        /**
        * Call transporting a callback for writing data.
        *
        * @author Alexander Straub
        */
        template <typename function_t>
        class abstract_data_writer_call : public core::Call
        {
        public:
            /**
            * Set the callback
            *
            * @param callback New callback
            */
            void set_callback(function_t callback)
            {
                this->callback = callback;
            }

            /**
            * Get the stored callback
            *
            * @return Callback
            */
            function_t get_callback() const
            {
                return this->callback;
            }

        protected:
            /**
            * Constructor
            */
            abstract_data_writer_call() {}

        private:
            /** Store callback */
            function_t callback;
        };
    }
}