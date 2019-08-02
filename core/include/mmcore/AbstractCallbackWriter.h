/*
 * AbstractCallbackWriter.h
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */
#pragma once

#include "mmcore/AbstractCallbackCall.h"
#include "mmcore/AbstractWriterParams.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/job/AbstractTickJob.h"

#include "vislib/sys/Log.h"

#include <functional>
#include <string>
#include <type_traits>

namespace megamol {
namespace core {

    /**
    * Abstract class for implementing a writer based on a callback.
    *
    * @author Alexander Straub
    */
    template <typename CallDescT, typename... ContentT>
    class AbstractCallbackWriter : public job::AbstractTickJob, protected AbstractWriterParams {

    public:
        using FunctionT = std::function<bool(ContentT...)>;

        static_assert(std::is_base_of<AbstractCallbackCall<FunctionT>, typename CallDescT::CallT>::value,
            "Call not derived from AbstractCallbackCall, or using wrong template parameter.");

        /**
        * Constructor
        */
        AbstractCallbackWriter() :
            AbstractWriterParams(std::bind(&AbstractCallbackWriter::MakeSlotAvailable, this, std::placeholders::_1)),
            inputSlot("input", "Slot for providing a callback") {
            
            this->inputSlot.SetCompatibleCall<CallDescT>();
            this->MakeSlotAvailable(&this->inputSlot);
        }

        /**
        * Destructor
        */
        virtual ~AbstractCallbackWriter() {
            this->Release();
        }

    protected:
        /**
         * Implementation of 'Create'.
         *
         * @return 'true' on success, 'false' otherwise.
         */
        virtual bool create() = 0;

        /**
         * Implementation of 'Release'.
         */
        virtual void release() = 0;

        /**
        * Callback function for writing data to file.
        *
        * @param path Output file path
        * @param content Content to write
        *
        * @return 'true' on success, 'false' otherwise.
        */
        virtual bool write(const std::string& path, ContentT... content) = 0;

        /**
         * Starts the job.
         *
         * @return true if the job has been successfully started.
         */
        virtual bool run() final {
            auto* call = this->inputSlot.CallAs<AbstractCallbackCall<FunctionT>>();

            if (call != nullptr)
            {
                auto proxy = [this](ContentT... content) -> bool { return Write(content...); };
                call->SetCallback(proxy);

                return (*call)(0);
            }

            return true;
        }

    private:
        /**
        * Callback function for writing data to file.
        *
        * @param content Content to write
        *
        * @return 'true' on success, 'false' otherwise.
        */
        bool Write(ContentT... content) {
            const auto filename = AbstractWriterParams::getNextFilename();

            if (filename.first) {
                return write(filename.second, content...);
            }

            return false;
        }

        /** Input slot */
        CallerSlot inputSlot;
    };

}
}
