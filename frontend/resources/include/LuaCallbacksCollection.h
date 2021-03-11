/*
 * LuaCallbacksCollection.h
 *
 * Copyright (C) 2021 by VISUS (Universitaet Stuttgart).
 * Alle Rechte vorbehalten.
 */

#pragma once

#include <functional>
#include <type_traits>
#include <string>
#include <tuple>
#include <list>

namespace megamol {
namespace frontend_resources {

struct LuaCallbacksCollection {

    struct LuaState {
        template <typename T>
        typename std::remove_reference<T>::type
        read(size_t index);

        template <typename T>
        void write(T item);

        void error(std::string reason);

        void* state_ptr = nullptr;
    };

    struct LuaError {
        std::string reason;
    };

    template <typename T>
    struct LuaResult {
        LuaResult(LuaError e)
        : exit_success{false} 
        , exit_reason{e.reason}
        {}

        LuaResult(T result)
        : exit_success{true}
        , exit_reason{}
        , result{result}
        {}

        bool exit_success = false;
        std::string exit_reason = "unknown reason";
        T result;
    };

    template <>
    struct LuaResult<void> {
        LuaResult(LuaError e)
        : exit_success{false} 
        , exit_reason{e.reason}
        {}
    
        LuaResult()
        : exit_success{true}
        , exit_reason{}
        {}
    
        bool exit_success = false;
        std::string exit_reason = "unknown reason";
    };

    using VoidResult = LuaResult<void>;
    using BoolResult = LuaResult<bool>;
    using LongResult = LuaResult<long>;
    using FloatResult = LuaResult<float>;
    using DoubleResult = LuaResult<double>;
    using StringResult = LuaResult<std::string>;

    template <typename ReturnType, typename FuncType, typename... FuncArgs, size_t... I>
    ReturnType unpack(LuaState state, FuncType func, std::tuple<FuncArgs...> tuple, std::index_sequence<I...>) {
        return func( state.read< typename std::tuple_element<I, std::tuple<FuncArgs...>>::type >(I+1)... );
    }

    template <typename FuncType, typename Result, typename... FuncArgs>
    std::function<int(LuaState)> resolve(std::string func_name, FuncType func) {
        return
            [=](LuaState state) -> int {
                //const bool stack_size_ok = (sizeof...(FuncArgs) == state.read<>(0));

                const Result result = unpack<Result>(state, func, std::tuple<FuncArgs...> {} , std::index_sequence_for<FuncArgs...> {});

                if(!result.exit_success) {
                    state.error(func_name + ": " + result.exit_reason);
                    return 0;
                }

                if constexpr (!std::is_same<Result, LuaResult<void>>::value) {
                    state.write(result.result);
                    return 1;
                }

                return 0;
            };
    }

    template <typename Result, typename... FuncArgs>
    void add(std::string func_name, std::string func_description, std::function<Result(FuncArgs...)> func) {
        callbacks.push_back({ func_name, func_description, 
            resolve<std::function<Result(FuncArgs...)>, Result, FuncArgs...>(func_name, func) }
        );
    }

    using LuaCallbackEntryType = std::function<int(LuaState)>;
    using LuaCallbacksListEntry = std::tuple<std::string, std::string, LuaCallbackEntryType>;

    std::list<LuaCallbacksListEntry> callbacks;
    bool is_registered = false;
    bool config_valid = false;
    bool render_valid = false;
};

// we implement reading/writing the lua stack in LuaAPI.cpp
// but we tell the world which types we support on the stack here
#define make_read_write(Type) \
    template <> \
    typename std::remove_reference<Type>::type \
    LuaCallbacksCollection::LuaState::read<typename std::remove_reference<Type>::type>(size_t index); \
    \
    template <> \
    void LuaCallbacksCollection::LuaState::write(Type item);

    make_read_write(bool);
    make_read_write(int);
    make_read_write(long);
    make_read_write(float);
    make_read_write(double);
    make_read_write(std::string);

} /* end namespace frontend_resources */
} /* end namespace megamol */
