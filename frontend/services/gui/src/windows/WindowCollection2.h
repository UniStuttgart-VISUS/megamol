#pragma once

#include <algorithm>
#include <functional>
#include <memory>
#include <string>
#include <typeindex>
#include <typeinfo>
#include <unordered_map>

#include "AbstractWindow2.h"
#include "KeyboardMouseInput.h"

namespace megamol::gui {
class WindowCollection2 {
public:
    template<typename T>
    void RegisterWindowType() {
        registered_types_[std::type_index(typeid(T))] = T::GetTypeInfo();
    }

    bool EnumRegisteredWindows(std::function<void(WindowType const&)> const& func) const {
        std::for_each(
            registered_types_.begin(), registered_types_.end(), [&func](auto const& entry) { func(entry.second); });
    }

    bool EnumWindows(std::function<void(AbstractWindow2 const&)> const& func) const {
        std::for_each(windows_.begin(), windows_.end(), [&func](auto const& entry) { func(*entry.second); });
    }

    bool EnumWindows(std::function<void(AbstractWindow2&)> const& func) {
        std::for_each(windows_.begin(), windows_.end(), [&func](auto& entry) { func(*entry.second); });
    }

    template<typename T, typename ...Args>
    void AddWindow(std::string const& name, Args... args) {
        windows_[name] = std::make_shared<T>(name, std::forward<Args>(args)...);
    }

    void RemoveWindow(std::string const& name) {
        windows_.erase(name);
    }

    template<typename T>
    std::shared_ptr<T> GetWindow(std::string const& name) {
        return std::dynamic_pointer_cast<T>(windows_[name]);
    }

    void Update();

private:
    std::unordered_map<std::type_index, WindowType> registered_types_;

    std::unordered_map<std::string, std::shared_ptr<AbstractWindow2>> windows_;
};
} // namespace megamol::gui
