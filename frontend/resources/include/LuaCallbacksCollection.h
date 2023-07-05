/*
 * LuaCallbacksCollection.h
 *
 * Copyright (C) 2021 by VISUS (Universitaet Stuttgart).
 * Alle Rechte vorbehalten.
 */

#pragma once

#include <functional>
#include <list>
#include <sstream>
#include <string>
#include <tuple>
#include <type_traits>

#ifdef MEGAMOL_USE_TRACY
#include <tracy/Tracy.hpp>
#endif

namespace megamol::frontend_resources {

namespace {
template<typename T>
std::string type_name();

#define make_name(Type)             \
    template<>                      \
    std::string type_name<Type>() { \
        return {#Type};             \
    }

make_name(bool);
make_name(int);
make_name(long);
make_name(float);
make_name(double);
template<>
std::string type_name<std::string>() {
    return "string";
}
#undef make_name
} // namespace

struct LuaCallbacksCollection {

    struct LuaState {
        template<typename T>
        typename std::remove_reference<T>::type read(size_t index);

        template<typename T>
        void write(T item);

        void error(std::string reason);
        int stack_size();

        void* state_ptr = nullptr;
    };

    struct LuaError {
        std::string reason;
    };

    template<typename T>
    struct LuaResult {
        LuaResult(LuaError e) : exit_success{false}, exit_reason{e.reason} {}

        LuaResult(T result) : exit_success{true}, exit_reason{}, result{result} {}

        bool exit_success = false;
        std::string exit_reason = "unknown reason";
        T result;
    };

    template<typename ReturnType, typename FuncType, typename... FuncArgs, size_t... I>
    ReturnType unpack(LuaState state, FuncType func, std::tuple<FuncArgs...> tuple, std::index_sequence<I...>) {
#ifdef MEGAMOL_USE_TRACY
        std::ostringstream stream;
        if constexpr (sizeof...(FuncArgs) > 0) {
            ((stream << state.read<typename std::tuple_element<I, std::tuple<FuncArgs...>>::type>(I + 1) << ", "), ...);
        }
        ZoneScopedC(0xA6963B);
        ZoneName(stream.str().c_str(), stream.str().size());
#endif
        return func(state.read<typename std::tuple_element<I, std::tuple<FuncArgs...>>::type>(I + 1)...);
    }

    template<typename FuncType, typename Result, typename... FuncArgs>
    std::function<int(LuaState)> resolve(std::string func_name, FuncType func) {
        return [=](LuaState state) -> int {
#ifdef MEGAMOL_USE_TRACY
            ZoneScopedC(0xA6963B);
            ZoneName(func_name.c_str(), func_name.size());
#endif
            if (sizeof...(FuncArgs) != state.stack_size()) {
                // if no function arguments given, cant expand FuncArgs during compile time - need to catch that case
                std::string args;
                if constexpr (sizeof...(FuncArgs) > 0) {
                    args = {((type_name<FuncArgs>() + ", ") + ...)};
                } else {
                    args = "";
                }
                state.error(func_name + ": " + " expects " + std::to_string(sizeof...(FuncArgs)) +
                            " arguments of type (" + args.substr(0, args.find_last_of(',')) + ")" + " but has " +
                            std::to_string(state.stack_size()));
                return 0;
            }

            const Result result =
                unpack<Result>(state, func, std::tuple<FuncArgs...>{}, std::index_sequence_for<FuncArgs...>{});

            if (!result.exit_success) {
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

    template<typename Result, typename... FuncArgs>
    void add(std::string func_name, std::string func_description, std::function<Result(FuncArgs...)> func) {
        callbacks.push_back({func_name, func_description,
            resolve<std::function<Result(FuncArgs...)>, Result, FuncArgs...>(func_name, func)});
    }

    using LuaCallbackEntryType = std::function<int(LuaState)>;
    using LuaCallbacksListEntry = std::tuple<std::string, std::string, LuaCallbackEntryType>;

    std::list<LuaCallbacksListEntry> callbacks;
    bool is_registered = false;
    bool config_valid = false;
    bool render_valid = false;

    using VoidResult = LuaCallbacksCollection::LuaResult<void>;
    using BoolResult = LuaCallbacksCollection::LuaResult<bool>;
    using LongResult = LuaCallbacksCollection::LuaResult<long>;
    using FloatResult = LuaCallbacksCollection::LuaResult<float>;
    using DoubleResult = LuaCallbacksCollection::LuaResult<double>;
    using StringResult = LuaCallbacksCollection::LuaResult<std::string>;
    using Error = LuaCallbacksCollection::LuaError;
    using LuaError = LuaCallbacksCollection::LuaError;
};

template<>
struct LuaCallbacksCollection::LuaResult<void> {
    LuaResult(LuaError e) : exit_success{false}, exit_reason{e.reason} {}

    LuaResult() : exit_success{true}, exit_reason{} {}

    bool exit_success = false;
    std::string exit_reason = "unknown reason";
};

// we implement reading/writing the lua stack in LuaAPI.cpp
// but we tell the world which types we support on the stack here
#define make_read_write(Type)                                                                         \
    template<>                                                                                        \
    typename std::remove_reference<Type>::type                                                        \
    LuaCallbacksCollection::LuaState::read<typename std::remove_reference<Type>::type>(size_t index); \
                                                                                                      \
    template<>                                                                                        \
    void LuaCallbacksCollection::LuaState::write(Type item);

make_read_write(bool);
make_read_write(int);
make_read_write(long);
make_read_write(float);
make_read_write(double);
make_read_write(std::string);
#undef make_read_write

} // namespace megamol::frontend_resources
