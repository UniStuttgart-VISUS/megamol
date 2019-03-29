/*
 * abstract_stream_provider.h
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */
#pragma once

#include "direct_data_writer_call.h"

#include "mmcore/CalleeSlot.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/Module.h"
#include "mmcore/job/TickCall.h"

#include <functional>
#include <iostream>

namespace megamol
{
    namespace flowvis
    {
        /**
        * Provides a stream.
        *
        * @author Alexander Straub
        */
        class abstract_stream_provider : public core::Module
        {
        public:
            /**
            * Constructor
            */
            abstract_stream_provider() :
                input_slot("input_slot", "Slot for providing a callback"),
                tick_slot("tick_slot", "Slot for receiving a tick")
            {
                this->input_slot.SetCompatibleCall<direct_data_writer_call::direct_data_writer_description>();
                this->MakeSlotAvailable(&this->input_slot);

                this->tick_slot.SetCallback(core::job::TickCall::ClassName(), core::job::TickCall::FunctionName(0), &abstract_stream_provider::run);
                this->MakeSlotAvailable(&this->tick_slot);
            }

            /**
            * Destructor
            */
            ~abstract_stream_provider()
            {
                this->Release();
            }

        protected:
            /**
            * Callback function providing the stream.
            *
            * @return Stream
            */
            virtual std::ostream& get_stream() = 0;

            /**
             * Implementation of 'Create'.
             *
             * @return 'true' on success, 'false' otherwise.
             */
            virtual bool create() override { return true; };

            /**
             * Implementation of 'Release'.
             */
            virtual void release() override { };

        private:
            /**
             * Starts the job.
             *
             * @return true if the job has been successfully started.
             */
            bool run(core::Call&)
            {
                auto* call = this->input_slot.CallAs<direct_data_writer_call>();

                if (call != nullptr)
                {
                    call->set_callback(std::bind(&abstract_stream_provider::get_stream, this));

                    return (*call)(0);
                }

                return true;
            };

            /** Input slot  */
            core::CallerSlot input_slot;

            /** Tick slot */
            core::CalleeSlot tick_slot;
        };
    }
}