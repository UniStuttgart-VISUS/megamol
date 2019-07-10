/*
* ScaleModel.cpp
*
* Copyright (C) 2018 by Universitaet Stuttgart (VISUS).
* All rights reserved.
*/

#include "ScaleModel.h"

using namespace megamol::archvis;


ScaleModel::ScaleModel(
    std::vector<Vec3> node_positions,
	std::vector<std::tuple<int, int, int, int, int>> elements,
	std::vector<int> input_elements)
	: m_node_positions(node_positions),
	m_node_displacements(node_positions.size(), Vec3(0.0f,0.0f,0.0f)),
	m_input_elements(input_elements)
{
	for (auto& element : elements)
	{
		int type = std::get<0>(element);

		switch (type)
		{
		case 0:
			m_elements.push_back(
				Element(
					ElementType::STRUT,
					std::tuple<int, int>(std::get<1>(element), std::get<2>(element)),
					0.0f)
			);
			break;
		case 1:
			m_elements.push_back(
				Element(
					ElementType::DIAGONAL,
					std::tuple<int, int>(std::get<1>(element), std::get<2>(element)),
					0.0f)
			);
			break;
		case 2:
			m_elements.push_back(
				Element(
					ElementType::FLOOR,
					std::tuple<int, int, int, int>(std::get<1>(element), std::get<2>(element), std::get<3>(element), std::get<4>(element)),
					0.0f)
			);
			break;
		default:
			break;
		}
	}
}

void ScaleModel::setModelTransform(Mat4x4 transform)
{
	m_model_transform = transform;
}

void ScaleModel::updateNodeDisplacements(std::vector<Vec3> const& displacements)
{
	m_node_displacements = displacements;
}

void ScaleModel::updateElementForces(std::vector<float> const& forces)
{
	//TODO check forces count matches input element count

	for (int i=0; i<forces.size(); ++i)
	{
		m_elements[m_input_elements[i]].setForce(forces[i]);
	}
}

int ScaleModel::getNodeCount()
{
	return m_node_positions.size();
}

int ScaleModel::getElementCount()
{
	return m_elements.size();
}

int ScaleModel::getInputElementCount()
{
	return m_input_elements.size();
}

ScaleModel::ElementType ScaleModel::getElementType(int element_idx)
{
	return m_elements[element_idx].getType();
}

float ScaleModel::getElementForce(int element_idx)
{
	return m_elements[element_idx].getForce();
}

ScaleModel::Mat4x4 ScaleModel::getElementTransform(int element_idx)
{
	return m_model_transform * m_elements[element_idx].computeTransform(m_node_positions, m_node_displacements);
}

ScaleModel::Vec3 ScaleModel::getElementCenter(int element_idx)
{
	Vec4 center = Vec4(m_elements[element_idx].computeCenter(m_node_positions, m_node_displacements));
	center.SetW(1.0f);

	return Vec3(m_model_transform * center);
}

std::vector<ScaleModel::Vec3> const& megamol::archvis::ScaleModel::accessNodePositions() { return m_node_positions; }

std::vector<ScaleModel::Vec3> const& megamol::archvis::ScaleModel::accessNodeDisplacements() {
    return m_node_displacements;
}

ScaleModel::Mat4x4 ScaleModel::computeElementTransform(std::tuple<int, int> node_indices,
	std::vector<Vec3> const& node_positions,
	std::vector<Vec3> const& node_displacements)
{
	Vec3 src_position = node_positions[std::get<0>(node_indices)];
	Vec3 tgt_position = node_positions[std::get<1>(node_indices)];

	Vec3 src_displaced = src_position + node_displacements[std::get<0>(node_indices)];
	Vec3 tgt_displaced = tgt_position + node_displacements[std::get<1>(node_indices)];

	// compute element rotation
	Mat4x4 object_rotation;
	Vec3 diag_vector = tgt_displaced - src_displaced;
	diag_vector.Normalise();
	Vec3 up_vector(0.0f, 1.0f, 0.0f);
	Vec3 rot_vector = up_vector.Cross(diag_vector);
	rot_vector.Normalise();
	Quat rotation(std::acos(up_vector.Dot(diag_vector)), rot_vector);
	object_rotation = rotation;

	// compute element scale
	Mat4x4 object_scale;
	float base_distance = (tgt_position - src_position).Length();
	float displaced_distance = (tgt_displaced - src_displaced).Length();

	object_scale.SetAt(1, 1, displaced_distance / base_distance);

	// compute element offset
	Mat4x4 object_translation;
	object_translation.SetAt(0, 3, src_displaced.X() );
	object_translation.SetAt(1, 3, src_displaced.Y() );
	object_translation.SetAt(2, 3, src_displaced.Z() );

	return (object_translation * object_rotation * object_scale);
}

ScaleModel::Mat4x4 ScaleModel::computeElementTransform(std::tuple<int, int, int, int> node_indices,
	std::vector<Vec3> const& node_positions,
	std::vector<Vec3> const& node_displacements)
{
	Vec3 origin_displaced = node_positions[std::get<0>(node_indices)] + node_displacements[std::get<0>(node_indices)];
	Vec3 corner_x_displaced = node_positions[std::get<1>(node_indices)] + node_displacements[std::get<1>(node_indices)];
	Vec3 corner_z_displaced = node_positions[std::get<3>(node_indices)] + node_displacements[std::get<3>(node_indices)];
	Vec3 corner_xz_displaced = node_positions[std::get<2>(node_indices)] + node_displacements[std::get<2>(node_indices)];

	// compute coordinate frame of planar surface given by four points
	Vec3 x_axis = corner_x_displaced - origin_displaced;
	x_axis.Normalise();
	Vec3 z_axis = corner_z_displaced - origin_displaced;
	z_axis.Normalise();
	Vec3 y_axis = -x_axis.Cross(z_axis);
	y_axis.Normalise();

	Mat4x4 rotational_transform;

	rotational_transform.SetAt(0, 0, x_axis.X());
	rotational_transform.SetAt(1, 0, x_axis.Y());
	rotational_transform.SetAt(2, 0, x_axis.Z());

	rotational_transform.SetAt(0, 1, y_axis.X());
	rotational_transform.SetAt(1, 1, y_axis.Y());
	rotational_transform.SetAt(2, 1, y_axis.Z());

	rotational_transform.SetAt(0, 2, z_axis.X());
	rotational_transform.SetAt(1, 2, z_axis.Y());
	rotational_transform.SetAt(2, 2, z_axis.Z());

	rotational_transform.SetAt(3, 3, 1.0f);

	// compute element offset
	Mat4x4 object_translation;
	object_translation.SetAt(0, 3, origin_displaced.X());
	object_translation.SetAt(1, 3, origin_displaced.Y());
	object_translation.SetAt(2, 3, origin_displaced.Z());

	return (object_translation * rotational_transform);
}

ScaleModel::Vec3 ScaleModel::computeElementCenter(std::tuple<int, int> node_indices,
	std::vector<Vec3> const& node_positions,
	std::vector<Vec3> const& node_displacements)
{
	Vec3 src_position = node_positions[std::get<0>(node_indices)];
	Vec3 tgt_position = node_positions[std::get<1>(node_indices)];

	Vec3 src_displaced = src_position + node_displacements[std::get<0>(node_indices)];
	Vec3 tgt_displaced = tgt_position + node_displacements[std::get<1>(node_indices)];

	return (src_displaced + tgt_displaced)*0.5f;
}

ScaleModel::Vec3 ScaleModel::computeElementCenter(std::tuple<int, int, int, int> node_indices,
	std::vector<Vec3> const& node_positions,
	std::vector<Vec3> const& node_displacements)
{
	Vec3 origin_displaced = node_positions[std::get<0>(node_indices)] + node_displacements[std::get<0>(node_indices)];
	Vec3 corner_x_displaced = node_positions[std::get<1>(node_indices)] + node_displacements[std::get<1>(node_indices)];
	Vec3 corner_z_displaced = node_positions[std::get<3>(node_indices)] + node_displacements[std::get<3>(node_indices)];
	Vec3 corner_xz_displaced = node_positions[std::get<2>(node_indices)] + node_displacements[std::get<2>(node_indices)];

	return (origin_displaced + corner_x_displaced + corner_z_displaced + corner_xz_displaced)*0.25f;
}