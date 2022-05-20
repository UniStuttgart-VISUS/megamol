#include "mesh/MeshDataCall.h"

#include <algorithm>
#include <memory>
#include <string>
#include <vector>

namespace megamol {
namespace mesh {
    void MeshDataCall::set_data(const std::string& name, std::shared_ptr<data_set> data) {
        if (this->data_sets.find(name) == this->data_sets.end() || data != nullptr) {
            this->data_sets[name] = data;
        }
    }

    std::shared_ptr<MeshDataCall::data_set> MeshDataCall::get_data(const std::string& name) const {
        if (this->data_sets.find(name) != this->data_sets.end()) {
            return this->data_sets.at(name);
        }

        return nullptr;
    }

    std::vector<std::string> MeshDataCall::get_data_sets() const {
        std::vector<std::string> data_sets;

        for (const auto& entry : this->data_sets) {
            data_sets.push_back(entry.first);
        }

        std::sort(data_sets.begin(), data_sets.end());

        return data_sets;
    }

    void MeshDataCall::set_mask(const std::string& name, std::shared_ptr<std::vector<float>> data) {
        if (this->masks.find(name) == this->masks.end() || data != nullptr) {
            this->masks[name] = data;
        }
    }

    std::shared_ptr<std::vector<float>> MeshDataCall::get_mask(const std::string& name) const {
        if (this->masks.find(name) != this->masks.end()) {
            return this->masks.at(name);
        }

        return nullptr;
    }

    std::vector<std::string> MeshDataCall::get_masks() const {
        std::vector<std::string> masks;

        for (const auto& entry : this->masks) {
            masks.push_back(entry.first);
        }

        std::sort(masks.begin(), masks.end());

        return masks;
    }
} // namespace mesh
} // namespace megamol
