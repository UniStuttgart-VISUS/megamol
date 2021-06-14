#include "mesh/MeshDataAccessor.h"

#include <algorithm>
#include <stdexcept>

#include "mmcore/utility/log/Log.h"


megamol::mesh::MeshDataTriangleAccessor::MeshDataTriangleAccessor(MeshDataAccessCollection::Mesh const& mesh)
        : mesh_(mesh) {
    if (mesh_.primitive_type != MeshDataAccessCollection::PrimitiveType::TRIANGLES) {
        throw std::runtime_error("Triangle type required");
    }

    switch (mesh_.indices.type) {
    case MeshDataAccessCollection::ValueType::UNSIGNED_INT: {
        index_acc_ = ValueAccessor(std::make_unique<U32Accessor>(mesh_.indices.data,
            MeshDataAccessCollection::getByteSize(MeshDataAccessCollection::ValueType::UNSIGNED_INT) * 3));
    } break;
    default:
        throw std::runtime_error("Unsupported index type");
    }

    num_triangles_ = mesh_.indices.byte_size /
                     (3 * MeshDataAccessCollection::getByteSize(MeshDataAccessCollection::ValueType::UNSIGNED_INT));

    auto const pos_fit = std::find_if(mesh_.attributes.begin(), mesh_.attributes.end(),
        [](auto const& el) { return el.semantic == MeshDataAccessCollection::AttributeSemanticType::POSITION; });
    if (pos_fit == mesh_.attributes.end())
        throw std::runtime_error("Spatial data missing");

    switch (pos_fit->component_type) {
    case MeshDataAccessCollection::ValueType::FLOAT: {
        pos_acc_ = ValueAccessor(std::make_unique<FloatAccessor>(pos_fit->data + pos_fit->offset, pos_fit->stride));
    } break;
    case MeshDataAccessCollection::ValueType::DOUBLE: {
        pos_acc_ = ValueAccessor(std::make_unique<DoubleAccessor>(pos_fit->data + pos_fit->offset, pos_fit->stride));
    } break;
    default:
        throw std::runtime_error("Only floating point type spatial data supported");
    }

    auto const normal_fit = std::find_if(mesh_.attributes.begin(), mesh_.attributes.end(),
        [](auto const& el) { return el.semantic == MeshDataAccessCollection::AttributeSemanticType::NORMAL; });

    if (normal_fit != mesh_.attributes.end()) {
        switch (normal_fit->component_type) {
        case MeshDataAccessCollection::ValueType::FLOAT: {
            normal_acc_ = std::make_unique<ValueAccessor>(
                std::make_unique<FloatAccessor>(normal_fit->data + normal_fit->offset, normal_fit->stride));
        } break;
        case MeshDataAccessCollection::ValueType::DOUBLE: {
            normal_acc_ = std::make_unique<ValueAccessor>(
                std::make_unique<DoubleAccessor>(normal_fit->data + normal_fit->offset, normal_fit->stride));
        } break;
        default:
            core::utility::log::Log::DefaultLog.WriteInfo(
                "[MeshDataTriangleAccessor] Only floating point type supported for normals");
        }
    }

    auto const color_fit = std::find_if(mesh_.attributes.begin(), mesh_.attributes.end(),
        [](auto const& el) { return el.semantic == MeshDataAccessCollection::AttributeSemanticType::COLOR; });

    if (color_fit != mesh_.attributes.end()) {
        switch (color_fit->component_type) {
        case MeshDataAccessCollection::ValueType::FLOAT: {
            color_acc_ = std::make_unique<ValueAccessor>(
                std::make_unique<FloatAccessor>(color_fit->data + color_fit->offset, color_fit->stride));
        } break;
        case MeshDataAccessCollection::ValueType::DOUBLE: {
            color_acc_ = std::make_unique<ValueAccessor>(
                std::make_unique<DoubleAccessor>(color_fit->data + color_fit->offset, color_fit->stride));
        } break;
        default:
            core::utility::log::Log::DefaultLog.WriteInfo(
                "[MeshDataTriangleAccessor] Only floating point type supported for colors");
        }
    }
}


std::array<glm::vec3, 3> megamol::mesh::MeshDataTriangleAccessor::GetPosition(uint64_t idx) const {
    auto const indices = GetIndices(idx);
    auto const a = glm::vec3(pos_acc_.GetF(indices.x, 0), pos_acc_.GetF(indices.x, 1), pos_acc_.GetF(indices.x, 2));
    auto const b = glm::vec3(pos_acc_.GetF(indices.y, 0), pos_acc_.GetF(indices.y, 1), pos_acc_.GetF(indices.y, 2));
    auto const c = glm::vec3(pos_acc_.GetF(indices.z, 0), pos_acc_.GetF(indices.z, 1), pos_acc_.GetF(indices.z, 2));

    return {a, b, c};
}


std::optional<std::array<glm::vec3, 3>> megamol::mesh::MeshDataTriangleAccessor::GetNormal(uint64_t idx) const {
    if (normal_acc_ == nullptr)
        return std::nullopt;
    auto const indices = GetIndices(idx);
    auto const a =
        glm::vec3(normal_acc_->GetF(indices.x, 0), normal_acc_->GetF(indices.x, 1), normal_acc_->GetF(indices.x, 2));
    auto const b =
        glm::vec3(normal_acc_->GetF(indices.y, 0), normal_acc_->GetF(indices.y, 1), normal_acc_->GetF(indices.y, 2));
    auto const c =
        glm::vec3(normal_acc_->GetF(indices.z, 0), normal_acc_->GetF(indices.z, 1), normal_acc_->GetF(indices.z, 2));

    return std::make_optional<std::array<glm::vec3, 3>>(std::array<glm::vec3, 3>{a, b, c});
}


std::optional<std::array<glm::vec4, 3>> megamol::mesh::MeshDataTriangleAccessor::GetColor(uint64_t idx) const {
    if (color_acc_ == nullptr)
        return std::nullopt;
    auto const indices = GetIndices(idx);
    auto const a = glm::vec4(color_acc_->GetF(indices.x, 0), color_acc_->GetF(indices.x, 1),
        color_acc_->GetF(indices.x, 2), color_acc_->GetF(indices.x, 3));
    auto const b = glm::vec4(color_acc_->GetF(indices.y, 0), color_acc_->GetF(indices.y, 1),
        color_acc_->GetF(indices.y, 2), color_acc_->GetF(indices.y, 3));
    auto const c = glm::vec4(color_acc_->GetF(indices.z, 0), color_acc_->GetF(indices.z, 1),
        color_acc_->GetF(indices.z, 2), color_acc_->GetF(indices.z, 3));

    return std::make_optional<std::array<glm::vec4, 3>>(std::array<glm::vec4, 3>{a, b, c});
}


glm::uvec3 megamol::mesh::MeshDataTriangleAccessor::GetIndices(uint64_t idx) const {
    return glm::uvec3(index_acc_.GetU32(idx, 0), index_acc_.GetU32(idx, 1), index_acc_.GetU32(idx, 2));
}

uint64_t megamol::mesh::MeshDataTriangleAccessor::GetCount() const {
    return num_triangles_;
}
