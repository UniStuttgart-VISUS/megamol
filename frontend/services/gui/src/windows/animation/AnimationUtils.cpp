#include "AnimationUtils.h"

using namespace megamol::gui;

void megamol::gui::animation::to_json(nlohmann::json& j, const animation::Key& k) {
    j = nlohmann::json{{"time", k.time}, {"value", k.value}, {"tangents_linked", k.tangents_linked},
        {"interpolation", k.interpolation}, {"in_tangent", nlohmann::json{k.in_tangent.x, k.in_tangent.y}},
        {"out_tangent", nlohmann::json{k.out_tangent.x, k.out_tangent.y}}};
}


void megamol::gui::animation::from_json(const nlohmann::json& j, animation::Key& k) {
    j["time"].get_to(k.time);
    j["value"].get_to(k.value);
    j["tangents_linked"].get_to(k.tangents_linked);
    j["interpolation"].get_to(k.interpolation);
    j["in_tangent"][0].get_to(k.in_tangent.x);
    j["in_tangent"][1].get_to(k.in_tangent.y);
    j["out_tangent"][0].get_to(k.out_tangent.x);
    j["out_tangent"][1].get_to(k.out_tangent.y);
}


void megamol::gui::animation::to_json(nlohmann::json& j, const animation::FloatAnimation& f) {
    j = nlohmann::json{{"name", f.GetName()}};
    auto k_array = nlohmann::json::array();
    for (auto& k : f.GetAllKeys()) {
        k_array.push_back(f[k]);
    }
    j["keys"] = k_array;
}


void megamol::gui::animation::from_json(const nlohmann::json& j, animation::FloatAnimation& f) {
    f = animation::FloatAnimation{j.at("name")};
    for (auto& j : j["keys"]) {
        animation::Key k;
        j.get_to(k);
        f.AddKey(k);
    }
}
