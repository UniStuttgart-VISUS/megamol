/*
 * AbstractCallbackWriter.h
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */
#pragma once

#include "mmcore/AbstractCallbackCall.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/job/AbstractTickJob.h"
#include "mmcore/param/FilePathParam.h"
#include "mmcore/param/ParamSlot.h"

#include <functional>
#include <iostream>
#include <string>
#include <type_traits>

#ifndef __VARIADIC_BIND__
#define __VARIADIC_BIND__
namespace {
    template <int>
    struct variadic_placeholder {};
}

namespace std {
    template <int N>
    struct is_placeholder<variadic_placeholder<N>> : integral_constant<int, N + 1>
    {
    };
}

namespace {
    template <typename Ret, typename Class, typename... Args, size_t... Is, typename... Args2>
    inline auto bind(std::index_sequence<Is...>, Ret(Class::*fptr)(Args...), Args2&&... args) {
        return std::bind(fptr, std::forward<Args2>(args)..., variadic_placeholder<Is>{}...);
    }

    template <typename Ret, typename Class, typename... Args, typename... Args2>
    inline auto bind(Ret(Class::*fptr)(Args...), Args2&&... args) {
        return bind(std::make_index_sequence<sizeof...(Args) - sizeof...(Args2) + 1>{}, fptr, std::forward<Args2>(args)...);
    }
}
#endif

namespace megamol {
namespace core {

    /**
    * Abstract class for implementing a writer based on a callback.
    *
    * @author Alexander Straub
    */
    template <typename CallDescT, typename... ContentT>
    class AbstractCallbackWriter : public job::AbstractTickJob {

    public:
        using FunctionT = std::function<bool(ContentT...)>;

        static_assert(std::is_base_of<AbstractCallbackCall<FunctionT>, typename CallDescT::CallT>::value,
            "Call not derived from AbstractCallbackCall, or using wrong template parameter.");

        /**
        * Constructor
        */
        AbstractCallbackWriter() :
            inputSlot("input", "Slot for providing a callback"),
            filePathSlot("outputFile", "Path to file which should be written into") {
            
            this->inputSlot.SetCompatibleCall<CallDescT>();
            this->MakeSlotAvailable(&this->inputSlot);

            this->filePathSlot << new param::FilePathParam("");
            this->MakeSlotAvailable(&this->filePathSlot);
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
                call->SetCallback(bind(&AbstractCallbackWriter::Write, this));

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
            return write(static_cast<std::string>(this->filePathSlot.template Param<param::FilePathParam>()->Value()), content...);
        }

        /** Input slot */
        CallerSlot inputSlot;

        /** File path parameter */
        param::ParamSlot filePathSlot;
    };

}
}
