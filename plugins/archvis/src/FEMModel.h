/*
 * FEMModel.h
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VISUS).
 * All rights reserved.
 */

#ifndef FEM_MODEL_H_INCLUDED
#define FEM_MODEL_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#    pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include <array>
#include <tuple>
#include <vector>

#include "vislib/math/Matrix.h"
#include "vislib/math/Quaternion.h"
#include "vislib/math/Vector.h"

namespace megamol {
namespace archvis {

class FEMModel {
public:
    typedef vislib::math::Matrix<float, 4, vislib::math::COLUMN_MAJOR> Mat4x4;
    typedef vislib::math::Vector<float, 3> Vec3;
    typedef vislib::math::Vector<float, 4> Vec4;
    typedef vislib::math::Quaternion<float> Quat;

    enum ElementType {
        LINE = 2,  // 2D line element, connects 2 nodes
        PLANE = 4, // 2D surface element, connects 4 nodes
        CUBE = 8   // 3D volumetric element, connects 8 nodes
    };

    struct ElementConcept {
        virtual ElementConcept* clone() const = 0;
        virtual ElementType getType() const = 0;
        virtual std::vector<size_t> getNodeIndices() const = 0;
    };

    template <typename N> struct ElementModel : ElementConcept {
        ElementModel(ElementType type, N node_indices) : m_type(type), m_node_indices(node_indices) {}

        ElementConcept* clone() const { return new ElementModel(m_type, m_node_indices); }

        ElementType getType() const { return m_type; }

        std::vector<size_t> getNodeIndices() const {
            return std::vector<size_t>(m_node_indices.begin(), m_node_indices.end());
        }

        ElementType m_type;
        N m_node_indices;
    };

    class Element {
    public:
        template <typename N>
        Element(ElementType type, N node_indices) : m_element(new ElementModel<N>(type, node_indices)) {}

        Element() : m_element(nullptr){};

        ~Element() {
            if (m_element != nullptr) delete m_element;
        }

        Element(Element const& other) : m_element(nullptr) {
            if (other.m_element != nullptr) m_element = other.m_element->clone();
        }

        Element(Element&& other) : Element() { std::swap(m_element, other.m_element); }

        Element& operator=(Element const& rhs) {
            delete m_element;
            if (rhs.m_element != nullptr)
                m_element = rhs.m_element->clone();
            else
                m_element = nullptr;

            return *this;
        }

        Element& operator=(Element&& other) { 
            std::swap(m_element, other.m_element);

            return *this;
        };

        ElementType getType() const { return m_element->getType(); }

        std::vector<size_t> getNodeIndices() const { return m_element->getNodeIndices(); }

    private:
        ElementConcept* m_element;
    };

    FEMModel();
    ~FEMModel();

    FEMModel(std::vector<Vec3> const& nodes, std::vector<std::array<size_t, 8>> const& elements);

    void setNodes(std::vector<Vec3> const& nodes);

    void setNodes(std::vector<Vec3>&& nodes);

    void setElements(std::vector<std::array<size_t, 8>> const& elements);

    void setNodeDeformations(std::vector<Vec4> const& deformations);

    std::vector<Vec3> const& getNodes();

    size_t getElementCount();

    std::vector<Element> const& getElements();

    std::vector<Vec4> const& getNodeDeformations();

private:
    size_t m_node_cnt;
    size_t m_timesteps;

    std::vector<Vec3> m_node_positions;
    std::vector<Element> m_elements;

    std::vector<Vec4> m_deformations;
};

inline FEMModel::FEMModel()
    : m_node_cnt(0)
    , m_timesteps(0) //, m_node_positions(), m_elements(), m_deformations()
{}

inline FEMModel::~FEMModel() {}

inline FEMModel::FEMModel(
    std::vector<Vec3> const& nodes, std::vector<std::array<size_t, 8>> const& elements)
    : m_node_cnt(nodes.size()), m_timesteps(0), m_node_positions(nodes), m_elements(elements.size()), m_deformations() {
    for (size_t element_idx = 0; element_idx < elements.size(); ++element_idx) {
        m_elements[element_idx] = Element(ElementType::CUBE, elements[element_idx]);
    }
}

inline void FEMModel::setNodes(std::vector<Vec3> const& nodes) {
    m_node_positions = nodes;
    m_node_cnt = m_node_positions.size();
}

inline void FEMModel::setNodes(std::vector<Vec3>&& nodes) {
    m_node_positions = nodes;
    m_node_cnt = m_node_positions.size();
}

inline void FEMModel::setElements(std::vector<std::array<size_t, 8>> const& elements) {
    m_elements.clear();
    m_elements.reserve(elements.size());

    for (size_t element_idx = 0; element_idx < elements.size(); ++element_idx) {
        m_elements.push_back(Element(ElementType::CUBE, elements[element_idx]));
    }
}

inline void FEMModel::setNodeDeformations(std::vector<Vec4> const& deformations) {
    m_deformations = deformations;
}

inline std::vector<FEMModel::Vec3> const& FEMModel::getNodes() { return m_node_positions; }

inline size_t FEMModel::getElementCount() { return m_elements.size(); }

inline std::vector<FEMModel::Element> const& FEMModel::getElements() { return m_elements; }

inline std::vector<FEMModel::Vec4> const& FEMModel::getNodeDeformations() { return m_deformations; }

} // namespace archvis
} // namespace megamol

#endif // !FEM_MODEL_H_INCLUDED