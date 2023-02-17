#include "WindowCollection2.h"


void megamol::gui::WindowCollection2::Update() {
    for (auto& [key, val] : windows_) {
        val->Update();
    }
}
