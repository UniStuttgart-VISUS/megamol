/*
 * ScaleModel.h
 *
 * Copyright (C) 2018 by Universitaet Stuttgart (VISUS).
 * All rights reserved.
 */

#ifndef SCALEMODEL_H_INCLUDED
#define SCALEMODEL_H_INCLUDED

#include <tuple>
#include <vector>

#include "vislib/math/Matrix.h"
#include "vislib/math/Quaternion.h"
#include "vislib/math/Vector.h"

namespace megamol {
namespace archvis {

class ScaleModel {
public:
    typedef vislib::math::Matrix<float, 4, vislib::math::COLUMN_MAJOR> Mat4x4;
    typedef vislib::math::Vector<float, 3> Vec3;
    typedef vislib::math::Vector<float, 4> Vec4;
    typedef vislib::math::Quaternion<float> Quat;

    enum ElementType { STRUT = 0, DIAGONAL = 1, FLOOR = 2 };

    ScaleModel() {}

    ScaleModel(
        std::vector<Vec3> node_positions,
        std::vector<std::tuple<int, int, int, int, int>> elements,
        std::vector<int> input_elements);

    void setModelTransform(Mat4x4 transform);

    void updateNodeDisplacements(std::vector<Vec3> const& displacements);

    //TODO
    void addNodeDisplacements(std::vector<Vec3> const& displacements);

    void updateElementForces(std::vector<float> const& forces);

    int getNodeCount();

    int getElementCount();

    int getInputElementCount();

    ElementType getElementType(int element_idx);

    float getElementForce(int element_idx);

    Mat4x4 getElementTransform(int element_idx);

    Vec3 getElementCenter(int element_idx);

    std::vector<Vec3> const& accessNodePositions();

    std::vector<Vec3> const& accessNodeDisplacements();

private:
    /**
     * Given indices of two nodes and both position and displacement data,
     * compute the transform of an element connected to these nodes.
     */
    static Mat4x4 computeElementTransform(std::tuple<int, int> node_indices, std::vector<Vec3> const& node_positions,
        std::vector<Vec3> const& node_displacements);

    static Mat4x4 computeElementTransform(std::tuple<int, int, int, int> node_indices,
        std::vector<Vec3> const& node_positions, std::vector<Vec3> const& node_displacements);

    static Vec3 computeElementCenter(std::tuple<int, int> node_indices, std::vector<Vec3> const& node_positions,
        std::vector<Vec3> const& node_displacements);

    static Vec3 computeElementCenter(std::tuple<int, int, int, int> node_indices,
        std::vector<Vec3> const& node_positions, std::vector<Vec3> const& node_displacements);

    struct ElementConcept {
        virtual ElementConcept* clone() const = 0;
        virtual ElementType getType() const = 0;
        virtual Mat4x4 computeTransform(
            std::vector<Vec3> const& node_positions, std::vector<Vec3> const& node_displacements) const = 0;
        virtual Vec3 computeCenter(
            std::vector<Vec3> const& node_positions, std::vector<Vec3> const& node_displacements) const = 0;
        virtual float getForce() const = 0;
        virtual void setForce(float force) = 0;
    };

    template <typename N, typename F> struct ElementModel : ElementConcept {
        ElementModel(ElementType type, N node_indices, F forces)
            : m_type(type), m_node_indices(node_indices), m_forces(forces) {}

        ElementConcept* clone() const { return new ElementModel(m_type, m_node_indices, m_forces); }

        ElementType getType() const { return m_type; }

        Mat4x4 computeTransform(
            std::vector<Vec3> const& node_positions, std::vector<Vec3> const& node_displacements) const {
            return computeElementTransform(m_node_indices, node_positions, node_displacements);
        }

        Vec3 computeCenter(std::vector<Vec3> const& node_positions, std::vector<Vec3> const& node_displacements) const {
            return computeElementCenter(m_node_indices, node_positions, node_displacements);
        }

        float getForce() const { return m_forces; }

        void setForce(float force) { m_forces = force; }

        ElementType m_type;
        N m_node_indices;
        F m_forces;
    };

    class Element {
    public:
        template <typename N, typename F>
        Element(ElementType type, N node_indices, F forces)
            : m_element(new ElementModel<N, F>(type, node_indices, forces)) {}

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

        Element& operator=(Element&& other) = delete;

        ElementType getType() { return m_element->getType(); }

        Mat4x4 computeTransform(std::vector<Vec3> const& node_positions, std::vector<Vec3> const& node_displacements) {
            return m_element->computeTransform(node_positions, node_displacements);
        }

        Vec3 computeCenter(std::vector<Vec3> const& node_positions, std::vector<Vec3> const& node_displacements) {
            return m_element->computeCenter(node_positions, node_displacements);
        }

        float getForce() { return m_element->getForce(); }

        void setForce(float force) { m_element->setForce(force); }

    private:
        ElementConcept* m_element;
    };

    Mat4x4 m_model_transform;

    std::vector<Vec3> m_node_positions;

    std::vector<Element> m_elements;
    std::vector<int> m_input_elements;

    //TODO create "global" collections of time dependent data and history of dynamically updated values
    std::vector<Vec3>  m_node_displacements;
    std::vector<float> m_element_forces;
};

} // namespace archvis
} // namespace megamol

#endif