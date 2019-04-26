/*
* FEMDataStorage.h
*
* Copyright (C) 2019 by Universitaet Stuttgart (VISUS).
* All rights reserved.
*/

#ifndef FEM_DATA_STORAGE_H_INCLUDED
#define FEM_DATA_STORAGE_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include <vector>
#include <tuple>

#include "vislib/math/Matrix.h"
#include "vislib/math/Quaternion.h"
#include "vislib/math/Vector.h"

namespace megamol {
	namespace archvis {

		class FEMDataStorage
		{
			typedef vislib::math::Matrix<float, 4, vislib::math::COLUMN_MAJOR> Mat4x4;
			typedef vislib::math::Vector<float, 3> Vec3;
			typedef vislib::math::Vector<float, 4> Vec4;
			typedef vislib::math::Quaternion<float> Quat;

			enum ElementType {
				LINE = 2, // 2D line element, connects 2 nodes
				PLANE = 4, // 2D surface element, connects 4 nodes
				CUBE = 8 // 3D volumetric element, connects 8 nodes
			};

		public:
			FEMDataStorage();
			~FEMDataStorage();

			FEMDataStorage(
				std::vector<Vec3> const& nodes,
				std::vector<std::array<size_t, 8>> const& elements);

		private:

			struct ElementConcept
			{
				virtual ElementConcept* clone() const = 0;
				virtual ElementType getType() const = 0;
			};

			template<typename N>
			struct ElementModel : ElementConcept
			{
				ElementModel(ElementType type, N node_indices, F forces)
					: m_type(type), m_node_indices(node_indices), m_forces(forces) {}

				ElementConcept* clone() const
				{
					return new ElementModel(m_type, m_node_indices, m_forces);
				}

				ElementType getType() const { return m_type; }

				ElementType m_type;
				N           m_node_indices;
			};

			class Element
			{
			public:
				template<typename N>
				Element(ElementType type, N node_indices)
					: m_element(new ElementModel<N, F>(type, node_indices)) {}

				Element() : m_element(nullptr) {};

				~Element()
				{
					if (m_element != nullptr)
						delete m_element;
				}

				Element(Element const& other) : m_element(nullptr) {
					if (other.m_element != nullptr)
						m_element = other.m_element->clone();
				}

				Element(Element&& other) : Element() {
					std::swap(m_element, other.m_element);
				}

				Element& operator=(Element const& rhs)
				{
					delete m_element;
					if (rhs.m_element != nullptr)
						m_element = rhs.m_element->clone();
					else
						m_element = nullptr;

					return *this;
				}

				Element& operator=(Element&& other) = delete;

				ElementType getType() { return m_element->getType(); }

			private:
				ElementConcept* m_element;
			};


			size_t               m_node_cnt;
			size_t               m_timesteps;

			std::vector<Vec3>    m_node_positions;
			std::vector<Element> m_elements;

			std::vector<Vec3>    m_deformations;
		};

		inline FEMDataStorage::FEMDataStorage()
			: m_node_cnt(0), m_timesteps(0), m_node_positions(), m_elements(), m_deformations()
		{
		}

		inline FEMDataStorage::~FEMDataStorage()
		{
		}

		inline FEMDataStorage::FEMDataStorage(std::vector<Vec3> const & nodes, std::vector<std::array<size_t, 8>> const & elements)
			: m_node_cnt(nodes.size()), m_timesteps(0), m_node_positions(nodes), m_elements(elements.size()), m_deformations()
		{
			for (size_t element_idx = 0; element_idx < elements.size(); ++element_idx)
			{
				m_elements[element_idx] = Element(
						ElementType::CUBE,
						elements[element_idx]);
			}
		}

	}
}

#endif // !FEM_DATA_STORAGE_H_INCLUDED