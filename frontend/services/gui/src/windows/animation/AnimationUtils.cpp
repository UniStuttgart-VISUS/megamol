#include "AnimationUtils.h"

using namespace megamol::gui;

void animation::to_json(nlohmann::json& j, const animation::FloatKey& k) {
    j = nlohmann::json{{"time", k.time}, {"value", k.value}, {"tangents_linked", k.tangents_linked},
        {"interpolation", k.interpolation}, {"in_tangent", nlohmann::json{k.in_tangent.x, k.in_tangent.y}},
        {"out_tangent", nlohmann::json{k.out_tangent.x, k.out_tangent.y}}};
}


void animation::from_json(const nlohmann::json& j, animation::FloatKey& k) {
    j["time"].get_to(k.time);
    j["value"].get_to(k.value);
    j["tangents_linked"].get_to(k.tangents_linked);
    j["interpolation"].get_to(k.interpolation);
    j["in_tangent"][0].get_to(k.in_tangent.x);
    j["in_tangent"][1].get_to(k.in_tangent.y);
    j["out_tangent"][0].get_to(k.out_tangent.x);
    j["out_tangent"][1].get_to(k.out_tangent.y);
}


void animation::to_json(nlohmann::json& j, const StringKey& k) {
    j = nlohmann::json{{"time", k.time}, {"value", k.value}};
}


void animation::from_json(const nlohmann::json& j, StringKey& k) {
    j["time"].get_to(k.time);
    j["value"].get_to(k.value);
}


void animation::to_json(nlohmann::json& j, const FloatAnimation& f) {
    j = nlohmann::json{{"name", f.GetName()}, {"type", "float"}};
    auto k_array = nlohmann::json::array();
    for (auto& k : f.GetAllKeys()) {
        k_array.push_back(f[k]);
    }
    j["keys"] = k_array;
}


void animation::from_json(const nlohmann::json& j, FloatAnimation& f) {
    f = FloatAnimation{j.at("name")};
    assert(j.at("type") == "float");
    for (auto& j : j["keys"]) {
        FloatKey k;
        j.get_to(k);
        f.AddKey(k);
    }
}

void animation::to_json(nlohmann::json& j, const StringAnimation& s) {
    j = nlohmann::json{{"name", s.GetName()}, {"type", "string"}};
    auto k_array = nlohmann::json::array();
    for (auto& k : s.GetAllKeys()) {
        k_array.push_back(s[k]);
    }
    j["keys"] = k_array;
}

void animation::from_json(const nlohmann::json& j, StringAnimation& s) {
    s = StringAnimation{j.at("name")};
    assert(j.at("type") == "string");
    for (auto& j : j["keys"]) {
        StringKey k;
        j.get_to(k);
        s.AddKey(k);
    }
}
