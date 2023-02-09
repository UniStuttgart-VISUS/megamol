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
    struct WindowType {
        bool unique = true;
        //WindowConfigID id = WINDOW_ID_VOLATILE;
        //std::string name;
        frontend_resources::KeyCode hotkey;
    };

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

    template<typename T>
    bool AddWindow(std::string const& name) {
        windows_[name] = std::make_shared<T>(name);
    }

    void RemoveWindow(std::string const& name) {
        windows_.erase(name);
    }

    template<typename T>
    std::shared_ptr<T> GetWindow(std::string const& name) const {
        return windows_[name];
    }

private:
    std::unordered_map<std::type_index, WindowType> registered_types_;

    std::unordered_map<std::string, std::shared_ptr<AbstractWindow2>> windows_;
};
} // namespace megamol::gui
