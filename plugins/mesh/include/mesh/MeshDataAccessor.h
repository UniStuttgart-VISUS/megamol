#pragma once

#include <array>
#include <memory>
#include <optional>

#include <glm/glm.hpp>

#include "mesh/MeshDataAccessCollection.h"

namespace megamol::mesh {
class ValueAccessorImpl {
public:
    virtual float GetF(uint64_t idx, uint8_t cmp) const = 0;

    virtual double GetD(uint64_t idx, uint8_t cmp) const = 0;

    virtual uint32_t GetU32(uint64_t idx, uint8_t cmp) const = 0;

private:
};

class FloatAccessor : public ValueAccessorImpl {
public:
    FloatAccessor() = default;

    FloatAccessor(uint8_t* data, uint64_t stride) : data_(data), stride_(stride) {}

    float GetF(uint64_t idx, uint8_t cmp) const override {
        auto const cmp_base = reinterpret_cast<float*>(data_ + idx * stride_);
        return cmp_base[cmp];
    }

    double GetD(uint64_t idx, uint8_t cmp) const override {
        return GetF(idx, cmp);
    }

    uint32_t GetU32(uint64_t idx, uint8_t cmp) const override {
        return GetF(idx, cmp);
    }

private:
    uint8_t* data_ = nullptr;

    uint64_t stride_ = 0;

    // uint8_t component_cnt_;
};

class DoubleAccessor : public ValueAccessorImpl {
public:
    DoubleAccessor() = default;

    DoubleAccessor(uint8_t* data, uint64_t stride) : data_(data), stride_(stride) {}

    float GetF(uint64_t idx, uint8_t cmp) const override {
        return GetD(idx, cmp);
    }

    double GetD(uint64_t idx, uint8_t cmp) const override {
        auto const cmp_base = reinterpret_cast<double*>(data_ + idx * stride_);
        return cmp_base[cmp];
    }

    uint32_t GetU32(uint64_t idx, uint8_t cmp) const override {
        return GetD(idx, cmp);
    }

private:
    uint8_t* data_ = nullptr;

    uint64_t stride_ = 0;

    // uint8_t component_cnt_;
};

class U32Accessor : public ValueAccessorImpl {
public:
    U32Accessor() = default;

    U32Accessor(uint8_t* data, uint64_t stride) : data_(data), stride_(stride) {}

    float GetF(uint64_t idx, uint8_t cmp) const override {
        return GetU32(idx, cmp);
    }

    double GetD(uint64_t idx, uint8_t cmp) const override {
        return GetU32(idx, cmp);
    }

    uint32_t GetU32(uint64_t idx, uint8_t cmp) const override {
        auto const cmp_base = reinterpret_cast<uint32_t*>(data_ + idx * stride_);
        return cmp_base[cmp];
    }

private:
    uint8_t* data_ = nullptr;

    uint64_t stride_ = 0;

    // uint8_t component_cnt_;
};

class ValueAccessor {
public:
    ValueAccessor() = default;

    ValueAccessor(std::unique_ptr<ValueAccessorImpl>&& impl) : impl_(std::move(impl)) {}

    float GetF(uint64_t idx, uint8_t cmp) const {
        return impl_->GetF(idx, cmp);
    }

    double GetD(uint64_t idx, uint8_t cmp) const {
        return impl_->GetD(idx, cmp);
    }

    uint32_t GetU32(uint64_t idx, uint8_t cmp) const {
        return impl_->GetU32(idx, cmp);
    }

private:
    std::unique_ptr<ValueAccessorImpl> impl_;
};

class MeshDataTriangleAccessor {
public:
    MeshDataTriangleAccessor(MeshDataAccessCollection::Mesh const& mesh);

    std::array<glm::vec3, 3> GetPosition(uint64_t idx) const;

    std::optional<std::array<glm::vec3, 3>> GetNormal(uint64_t idx) const;

    std::optional<std::array<glm::vec4, 3>> GetColor(uint64_t idx) const;

    glm::uvec3 GetIndices(uint64_t idx) const;

    uint64_t GetCount() const;

private:
    MeshDataAccessCollection::Mesh const& mesh_;

    ValueAccessor index_acc_;

    ValueAccessor pos_acc_;

    std::unique_ptr<ValueAccessor> normal_acc_ = nullptr;

    std::unique_ptr<ValueAccessor> color_acc_ = nullptr;

    uint64_t num_triangles_ = 0;
};
} // namespace megamol::mesh
