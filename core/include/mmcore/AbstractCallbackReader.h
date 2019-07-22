/*
 * AbstractCallbackReader.h
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */
#pragma once

#include "mmcore/AbstractCallbackCall.h"
#include "mmcore/CalleeSlot.h"
#include "mmcore/Module.h"
#include "mmcore/param/FilePathParam.h"
#include "mmcore/param/ParamSlot.h"

#include <functional>
#include <iostream>
#include <string>
#include <type_traits>

namespace megamol {
namespace core {

    /**
    * Abstract class for implementing a reader based on a callback.
    *
    * @author Alexander Straub
    */
    template <typename CallT, typename... ContentT>
    class AbstractCallbackReader : public core::Module {

    public:
        using FunctionT = std::function<bool(ContentT...)>;

        static_assert(std::is_base_of<AbstractCallbackCall<FunctionT>, CallT>::value,
            "Call not derived from AbstractCallbackCall, or using wrong template parameter.");

        /**
        * Constructor
        */
        AbstractCallbackReader() :
            outputSlot("output", "Slot for providing a callback"),
            filePathSlot("inputFile", "Path to file which should be read from") {
            
            this->outputSlot.SetCallback(CallT::ClassName(), CallT::FunctionName(0), &AbstractCallbackReader::SetCallback);
            this->MakeSlotAvailable(&this->outputSlot);

            this->filePathSlot << new param::FilePathParam("");
            this->MakeSlotAvailable(&this->filePathSlot);
        }

        /**
        * Destructor
        */
        virtual ~AbstractCallbackReader() {
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
        virtual bool read(const std::string& path, ContentT... content) = 0;

        /**
         * Callback for handling the callback request.
         *
         * @return 'true' on success, 'false' otherwise.
         */
        bool SetCallback(Call& call) {
            auto* callbackCall = dynamic_cast<AbstractCallbackCall<FunctionT>*>(&call);

            if (callbackCall != nullptr)
            {
                auto proxy = [this](ContentT... content) -> bool { return Read(content...); };
                callbackCall->SetCallback(proxy);
            }

            return true;
        }

    private:
        /**
        * Callback function for reading data from file.
        *
        * @param content Content to fill
        *
        * @return 'true' on success, 'false' otherwise.
        */
        bool Read(ContentT... content) {
            return read(static_cast<std::string>(this->filePathSlot.template Param<param::FilePathParam>()->Value()), content...);
        }

        /** Output slot */
        CalleeSlot outputSlot;

        /** File path parameter */
        param::ParamSlot filePathSlot;
    };

}
}
