/*
* AbstractNGMeshDataSource.h
*
* Copyright (C) 2017 by Universitaet Stuttgart (VISUS).
* All rights reserved.
*/

#include <iostream>
#include <random>
#include <string>

#include "vislib/graphics/gl/IncludeAllGL.h"
#include "vislib/net/SocketException.h"

#include "mmcore/param/FilePathParam.h"
#include "mmcore/param/IntParam.h"

#include "stdafx.h"
#include "ArchVisMSMDataSource.h"

#define TINYGLTF_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#include "tiny_gltf.h"

using namespace megamol::archvis;
using namespace megamol::ngmesh;

ArchVisMSMDataSource::ArchVisMSMDataSource() :
	m_shaderFilename_slot("Shader", "The name of to the shader file to load"),
	m_partsList_slot("Parts list", "The path to the parts list file to load"),
	m_nodeElement_table_slot("Node/Element table", "The path to the node/element table to load"),
	m_rcv_IPAddr_slot("Receive IP adress", "The ip adress for receiving data"),
	m_snd_IPAddr_slot("Send IP adress", "The ip adress for sending data"),
	m_rcv_socket_connected(false)
{
	this->m_shaderFilename_slot << new core::param::FilePathParam("");
	this->MakeSlotAvailable(&this->m_shaderFilename_slot);

	this->m_partsList_slot << new core::param::FilePathParam("");
	this->MakeSlotAvailable(&this->m_partsList_slot);

	m_nodeElement_table_slot << new core::param::FilePathParam("");
	this->MakeSlotAvailable(&this->m_nodeElement_table_slot);
	
	m_rcv_IPAddr_slot << new core::param::IntParam(0);
	this->MakeSlotAvailable(&this->m_rcv_IPAddr_slot);

	m_snd_IPAddr_slot << new core::param::IntParam(0);
	this->MakeSlotAvailable(&this->m_snd_IPAddr_slot);

	try {
		// try to start up socket
		vislib::net::Socket::Startup();
		// create socket
		this->m_rcv_socket.Create(vislib::net::Socket::ProtocolFamily::FAMILY_INET, vislib::net::Socket::Type::TYPE_DGRAM, vislib::net::Socket::Protocol::PROTOCOL_UDP);
	}
	catch (vislib::net::SocketException e) {
		vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR, "Socket Exception during startup/create: %s", e.GetMsgA());
	}
	
	
	//std::cout << "Socket Endpoint: " << endpoint.ToStringA() << std::endl;
}

ArchVisMSMDataSource::~ArchVisMSMDataSource()
{
	vislib::net::Socket::Cleanup();
}

bool ArchVisMSMDataSource::getDataCallback(megamol::core::Call& caller)
{
	CallNGMeshRenderBatches* render_batches_call = dynamic_cast<CallNGMeshRenderBatches*>(&caller);
	if (render_batches_call == NULL)
		return false;

	if (this->m_partsList_slot.IsDirty() ||
		this->m_shaderFilename_slot.IsDirty() ||
		this->m_nodeElement_table_slot.IsDirty() )
	{
		// TODO handle different slots seperatly ?
		this->m_shaderFilename_slot.ResetDirty();
		this->m_partsList_slot.ResetDirty();
		this->m_nodeElement_table_slot.ResetDirty();

		// Clear render batches TODO: add explicit clear function?
		CallNGMeshRenderBatches::RenderBatchesData empty_render_batches;
		m_render_batches = empty_render_batches;

		m_bbox.Set(-10.0f, -10.0f, -10.0f, 10.0f, 10.0f, 10.0f); //?

		auto vislib_shader_filename = m_shaderFilename_slot.Param<megamol::core::param::FilePathParam>()->Value();
		std::string shdr_filename(vislib_shader_filename.PeekBuffer());

		auto vislib_partsList_filename = m_partsList_slot.Param<megamol::core::param::FilePathParam>()->Value();
		std::string partsList_filename(vislib_partsList_filename.PeekBuffer());

		auto vislib_nodesElement_filename = m_nodeElement_table_slot.Param<megamol::core::param::FilePathParam>()->Value();
		std::string nodesElement_filename(vislib_nodesElement_filename.PeekBuffer());

		load(shdr_filename, partsList_filename, nodesElement_filename);
	}

	if (this->m_rcv_IPAddr_slot.IsDirty())
	{
		this->m_rcv_IPAddr_slot.ResetDirty();

		try {
			vislib::net::IPAddress server_addr;
			server_addr = server_addr.Create();
			this->m_rcv_socket.Connect(vislib::net::IPEndPoint(server_addr, 9050));
			this->m_rcv_socket_connected = true;

			std::string greeting("Hello, my name is MegaMol");
			this->m_rcv_socket.Send(greeting.c_str(), greeting.length());
		}
		catch (vislib::net::SocketException e) {
			vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR, "Socket Exception during connection: %s", e.GetMsgA());
			return false;
		}
	}

	render_batches_call->setRenderBatches(&m_render_batches);

	if(this->m_rcv_socket_connected)
		this->m_rcv_socket.Receive(m_node_displacements.data(), m_node_displacements.size() * 4);

	updateMSMTransform();
	
	return true;
}

ArchVisMSMDataSource::Mat4x4 ArchVisMSMDataSource::computeElementTransform(Node src, Node tgt, Vec3 src_displacement, Vec3 tgt_displacement)
{
	Node src_displaced = { src_displacement.X() + std::get<0>(src),
							src_displacement.Y() + std::get<1>(src),
							src_displacement.Z() + std::get<2>(src) };

	Node tgt_displaced = { tgt_displacement.X() + std::get<0>(tgt),
							tgt_displacement.Y() + std::get<1>(tgt),
							tgt_displacement.Z() + std::get<2>(tgt) };

	// compute element rotation
	Mat4x4 object_rotation;
	Vec3 diag_vector = Vec3(std::get<0>(tgt_displaced) - std::get<0>(src_displaced),
							std::get<1>(tgt_displaced) - std::get<1>(src_displaced),
							std::get<2>(tgt_displaced) - std::get<2>(src_displaced));
	diag_vector.Normalise();
	Vec3 up_vector(0.0f, 1.0f, 0.0f);
	Vec3 rot_vector = up_vector.Cross(diag_vector);
	rot_vector.Normalise();
	Quat rotation(std::acos(up_vector.Dot(diag_vector)), rot_vector);
	object_rotation = rotation;

	// compute element scale
	Mat4x4 object_scale;
	float base_distance = Vec3(std::get<0>(tgt) - std::get<0>(src),
		std::get<1>(tgt) - std::get<1>(src),
		std::get<2>(tgt) - std::get<2>(src)).Length();

	float displaced_distance = Vec3(std::get<0>(tgt_displaced) - std::get<0>(src_displaced),
		std::get<1>(tgt_displaced) - std::get<1>(src_displaced),
		std::get<2>(tgt_displaced) - std::get<2>(src_displaced)).Length();

	object_scale.SetAt(1, 1, displaced_distance/base_distance);

	// compute element offset
	Mat4x4 object_translation;
	object_translation.SetAt(0, 3, std::get<0>(src_displaced));
	object_translation.SetAt(1, 3, std::get<1>(src_displaced));
	object_translation.SetAt(2, 3, std::get<2>(src_displaced));

	return (object_translation * object_rotation * object_scale);
}

ArchVisMSMDataSource::Mat4x4 ArchVisMSMDataSource::computeElementTransform(
	Node orgin,
	Node corner_x,
	Node corner_z,
	Node corner_xz,
	Vec3 origin_displacement,
	Vec3 corner_x_displacement,
	Vec3 corner_z_displacement,
	Vec3 corner_xz_displacement)
{
	Node origin_displaced = { origin_displacement.X() + std::get<0>(orgin),
		origin_displacement.Y() + std::get<1>(orgin),
		origin_displacement.Z() + std::get<2>(orgin) };

	Node corner_x_displaced = { corner_x_displacement.X() + std::get<0>(corner_x),
		corner_x_displacement.Y() + std::get<1>(corner_x),
		corner_x_displacement.Z() + std::get<2>(corner_x) };

	Node corner_z_displaced = { corner_z_displacement.X() + std::get<0>(corner_z),
		corner_z_displacement.Y() + std::get<1>(corner_z),
		corner_z_displacement.Z() + std::get<2>(corner_z) };

	Node corner_xz_displaced = { corner_xz_displacement.X() + std::get<0>(corner_xz),
		corner_xz_displacement.Y() + std::get<1>(corner_xz),
		corner_xz_displacement.Z() + std::get<2>(corner_xz) };


	// compute element rotation around z
	Mat4x4 object_rotation_z;
	Vec3 diag_vector = Vec3(std::get<0>(corner_x_displaced) - std::get<0>(origin_displaced),
		std::get<1>(corner_x_displaced) - std::get<1>(origin_displaced),
		std::get<2>(corner_x_displaced) - std::get<2>(origin_displaced));
	diag_vector.Normalise();
	Vec3 up_vector(1.0f, 0.0f, 0.0f);
	Vec3 rot_vector = up_vector.Cross(diag_vector);
	rot_vector.Normalise();
	Quat rotation(std::acos(up_vector.Dot(diag_vector)), rot_vector);
	object_rotation_z = rotation;

	// compute element scale
	Mat4x4 object_scale;

	// compute element offset
	Mat4x4 object_translation;
	object_translation.SetAt(0, 3, std::get<0>(origin_displaced));
	object_translation.SetAt(1, 3, std::get<1>(origin_displaced));
	object_translation.SetAt(2, 3, std::get<2>(origin_displaced));

	return (object_translation * object_rotation_z);
}

std::vector<std::string> ArchVisMSMDataSource::parsePartsList(std::string const& filename)
{
	std::vector<std::string> parts;

	std::ifstream file;
	file.open(filename, std::ifstream::in);

	if (file.is_open())
	{
		file.seekg(0, std::ifstream::beg);

		while (!file.eof())
		{
			parts.push_back(std::string());

			getline(file, parts.back(), '\n');
		}
	}

	return parts;
}

void ArchVisMSMDataSource::parseNodeElementTable(
	std::string const& filename,
	std::vector<Node>& nodes,
	std::vector<FloorElement>& floor_elements,
	std::vector<BeamElement>& beam_elements,
	std::vector<DiagonalElement>& diagonal_elements)
{
	std::ifstream file;
	file.open(filename, std::ifstream::in);

	if (file.is_open())
	{
		file.seekg(0, std::ifstream::beg);

		// read node and element count
		std::string line;
		std::getline(file, line, '\n');
		int node_cnt = std::stoi(line);

		std::getline(file, line, '\n');
		int element_cnt = std::stoi(line);

		unsigned int lines_read = 0;
		while (!file.eof())
		{
			std::string line;
			std::getline(file, line, '\n');
			std::stringstream ss(line);
			std::string linestart = line.substr(0, 2);

			if (std::strcmp("//", linestart.c_str()) != 0 && line.length() > 0)
			{
				if (lines_read < node_cnt)
				{
					std::string x, y, z;
					ss >> x >> y >> z;

					nodes.push_back(std::make_tuple<float, float, float>(std::stof(x), std::stof(y), std::stof(z)));
				}
				else
				{
					std::string type;
					ss >> type;

					if (std::stoi(type) == 2)
					{
						std::string idx0, idx1, idx2, idx3;

						ss >> idx0 >> idx1 >> idx2 >> idx3;

						floor_elements.push_back(std::make_tuple<int, int, int, int>(
							std::stoi(idx0), std::stoi(idx1), std::stoi(idx2), std::stoi(idx3))
						);
					}
					else if (std::stoi(type) == 0)
					{
						std::string idx0, idx1;

						ss >> idx0 >> idx1;

						beam_elements.push_back(std::make_tuple<int, int>(
							std::stoi(idx0), std::stoi(idx1))
						);
					}
					else if (std::stoi(type) == 1)
					{
						std::string idx0, idx1;

						ss >> idx0 >> idx1;

						diagonal_elements.push_back(std::make_tuple<int, int>(
							std::stoi(idx0), std::stoi(idx1))
						);
					}
				}

				lines_read++;
			}
		}
	}
}

void ArchVisMSMDataSource::updateMSMTransform()
{
	ObjectShaderParamsDataAccessor mesh_shader_params;

	mesh_shader_params.byte_size = 16 * 4 * m_render_batches.getDrawCommandData(0).draw_cnt;
	mesh_shader_params.raw_data = new uint8_t[mesh_shader_params.byte_size];

	// compute element offset
	Mat4x4 tower_model_matrix;
	tower_model_matrix.SetAt(0, 3, 0.0f);
	tower_model_matrix.SetAt(1, 3, -1.0f);
	tower_model_matrix.SetAt(2, 3, 0.0f);

	int counter = 0;
	for (auto& element : m_floor_elements)
	{
		int origin_node_idx = std::get<3>(element);
		int x_node_idx = std::get<0>(element);
		int z_node_idx = std::get<1>(element);
		int xz_node_idx = std::get<2>(element);

		Mat4x4 object_transform = tower_model_matrix * computeElementTransform(
			m_nodes[origin_node_idx], 
			m_nodes[x_node_idx],
			m_nodes[z_node_idx],
			m_nodes[xz_node_idx],
			Vec3(m_node_displacements[origin_node_idx * 3], m_node_displacements[origin_node_idx * 3 + 1], m_node_displacements[origin_node_idx * 3 + 2]),
			Vec3(m_node_displacements[x_node_idx * 3], m_node_displacements[x_node_idx * 3 + 1], m_node_displacements[x_node_idx * 3 + 2]),
			Vec3(m_node_displacements[z_node_idx * 3], m_node_displacements[z_node_idx * 3 + 1], m_node_displacements[z_node_idx * 3 + 2]),
			Vec3(m_node_displacements[xz_node_idx * 3], m_node_displacements[xz_node_idx * 3 + 1], m_node_displacements[xz_node_idx * 3 + 2]) );

		std::memcpy(mesh_shader_params.raw_data + counter * (16 * 4), object_transform.PeekComponents(), 16 * 4);

		counter++;
	}

	for (auto& element : m_beam_elements)
	{
		int src_node_idx = std::get<0>(element);
		int tgt_node_idx = std::get<1>(element);

		Mat4x4 object_transform = tower_model_matrix * computeElementTransform(m_nodes[src_node_idx], m_nodes[tgt_node_idx],
			Vec3(m_node_displacements[src_node_idx * 3], m_node_displacements[src_node_idx * 3 + 1], m_node_displacements[src_node_idx * 3 + 2]),
			Vec3(m_node_displacements[tgt_node_idx * 3], m_node_displacements[tgt_node_idx * 3 + 1], m_node_displacements[tgt_node_idx * 3 + 2]));

		std::memcpy(mesh_shader_params.raw_data + counter * (16 * 4), object_transform.PeekComponents(), 16 * 4);

		counter++;
	}

	for (auto& element : m_diagonal_elements)
	{
		int src_node_idx = std::get<0>(element);
		int tgt_node_idx = std::get<1>(element);

		Mat4x4 object_transform = tower_model_matrix * computeElementTransform(m_nodes[src_node_idx], m_nodes[tgt_node_idx],
			Vec3(m_node_displacements[src_node_idx * 3], m_node_displacements[src_node_idx * 3 + 1], m_node_displacements[src_node_idx * 3 + 2]),
			Vec3(m_node_displacements[tgt_node_idx * 3], m_node_displacements[tgt_node_idx * 3 + 1], m_node_displacements[tgt_node_idx * 3 + 2]));

		std::memcpy(mesh_shader_params.raw_data + counter * (16 * 4), object_transform.PeekComponents(), 16 * 4);

		counter++;
	}

	m_render_batches.updateObjectShaderParams(0, mesh_shader_params);

	delete[] mesh_shader_params.raw_data;
}

bool ArchVisMSMDataSource::load(std::string const& shader_filename,
	std::string const& partsList_filename,
	std::string const& nodesElement_filename)
{
	vislib::sys::Log::DefaultLog.WriteInfo("ArchVisMSM loading data from files.\n");

	
	// Load node and element data
	parseNodeElementTable(nodesElement_filename, m_nodes, m_floor_elements, m_beam_elements, m_diagonal_elements);

	m_node_displacements.resize(m_nodes.size() * 3);


	ShaderPrgmDataAccessor				shader_prgm_data;
	MeshDataAccessor					mesh_data;
	DrawCommandDataAccessor				draw_command_data;
	ObjectShaderParamsDataAccessor		mesh_shader_params;
	MaterialShaderParamsDataAccessor	mtl_shader_params;

	shader_prgm_data.char_cnt = shader_filename.length();
	shader_prgm_data.raw_string = new char[shader_prgm_data.char_cnt];
	std::strcpy(shader_prgm_data.raw_string, shader_filename.c_str());
	
	// parse parts list
	std::vector<std::string> parts_meshes_paths = parsePartsList(partsList_filename);

	// Create vector of glTF models
	std::vector<tinygltf::Model> models;
	tinygltf::TinyGLTF loader;
	std::string err;

	for (auto path : parts_meshes_paths)
	{
		models.push_back(tinygltf::Model());

		bool ret = loader.LoadASCIIFromFile(&models.back(), &err, path);
		if (!err.empty()) {
			vislib::sys::Log::DefaultLog.WriteError("Err: %s\n", err.c_str());
		}

		if (!ret) {
			vislib::sys::Log::DefaultLog.WriteError("Failed to parse glTF\n");
		}
	}

	// set mesh usage and primtive type
	mesh_data.usage = GL_STATIC_DRAW;
	mesh_data.primitive_type = GL_TRIANGLES;

	// set vertex attribute layout
	std::vector<std::string> attribute_names;
	mesh_data.vertex_descriptor.attribute_cnt = models.front().meshes.front().primitives.front().attributes.size();
	mesh_data.vertex_descriptor.attributes = new MeshDataAccessor::VertexLayoutData::Attribute[mesh_data.vertex_descriptor.attribute_cnt];
	mesh_data.vertex_descriptor.stride = 0; //we currently cannot handle a different stride per attribute buffer, assuming data to be tightly packed
	size_t attribute_counter = 0;
	for (auto& attribute : models.front().meshes.front().primitives.front().attributes)
	{
		//std::cout << "Attribute " << attribute.first << std::endl;

		attribute_names.push_back(attribute.first);

		//std::cout << models.front().accessors[attribute.second].type << std::endl;
		//std::cout << models.front().accessors[attribute.second].componentType << std::endl;
		//std::cout << models.front().accessors[attribute.second].count << std::endl;
		//std::cout << models.front().accessors[attribute.second].byteOffset << std::endl;
		//std::cout << models.front().accessors[attribute.second].normalized << std::endl;

		//auto& bufferView = models.front().bufferViews[models.front().accessors[attribute.second].bufferView];

		mesh_data.vertex_descriptor.attributes[attribute_counter].type = models.front().accessors[attribute.second].componentType;
		mesh_data.vertex_descriptor.attributes[attribute_counter].size = models.front().accessors[attribute.second].type;
		mesh_data.vertex_descriptor.attributes[attribute_counter].normalized = models.front().accessors[attribute.second].normalized;
		// we use a seperated VBO per vertex attribute, therefore offset should always be zero...
		mesh_data.vertex_descriptor.attributes[attribute_counter].offset = 0;

		++attribute_counter;
	}

	// Create mesh data that holds all models
	// For now, assume glTF will supply non-interleaved data (this seems to be mostly the case,..how do I even identify interleaved data?)
	// Also assume that all glTF files contain a single mesh and all meshes use the same vertex format (i.e. go into a single render batch)
	mesh_data.index_data.byte_size = 0;
	mesh_data.vertex_data.byte_size = 0;
	mesh_data.vertex_data.buffer_cnt = mesh_data.vertex_descriptor.attribute_cnt;

	std::vector<uint32_t> first_indices; // index of the first index of the different meshes stored in the buffer
	std::vector<uint32_t> indices_cnt;
	std::vector<uint32_t> base_vertices; // base vertices of the different meshes stored in the buffer
	first_indices.push_back(0);
	base_vertices.push_back(0);

	// Sum up required storage for index and vertex data
	for (auto& model : models)
	{
		auto index_buffer_accessor = model.accessors[model.meshes.front().primitives.front().indices];
		auto index_bufferView = model.bufferViews[index_buffer_accessor.bufferView];
		mesh_data.index_data.index_type = index_buffer_accessor.componentType;
		mesh_data.index_data.byte_size += index_bufferView.byteLength;

		for (auto& attrib : model.meshes.front().primitives.front().attributes)
		{
			auto& accessor = model.accessors[attrib.second];
			auto& bufferView = model.bufferViews[accessor.bufferView];

			mesh_data.vertex_data.byte_size += bufferView.byteLength;
		}

		// log first index and base vertex for each model
		uint32_t base_vertex = base_vertices.back() + model.accessors[model.meshes.front().primitives.front().attributes.at(attribute_names.front())].count;
		base_vertices.push_back(base_vertex);

		uint32_t first_index = first_indices.back() + index_buffer_accessor.count;
		first_indices.push_back(first_index);

		indices_cnt.push_back(index_buffer_accessor.count);
	}
	// need additional storage for storing byte offsets of individual buffers (4byte per buffer)
	mesh_data.vertex_data.byte_size += 4 * mesh_data.vertex_data.buffer_cnt;

	// Allocate shared buffer for vertex data and copy data from gltf to buffer, start after offset values
	mesh_data.vertex_data.raw_data = new uint8_t[mesh_data.vertex_data.byte_size];
	uint32_t* uint32_view = reinterpret_cast<uint32_t*>(mesh_data.vertex_data.raw_data);
	uint32_t bytes_copied = 4 * mesh_data.vertex_data.buffer_cnt;

	for (int i = 0; i < mesh_data.vertex_data.buffer_cnt; ++i)
	{
		uint32_view[i] = bytes_copied;

		for (auto& model : models)
		{
			auto& accessor = model.accessors[model.meshes.front().primitives.front().attributes.at(attribute_names[i])];
			auto& bufferView = model.bufferViews[accessor.bufferView];

			auto tgt = mesh_data.vertex_data.raw_data + bytes_copied;
			auto src = model.buffers[bufferView.buffer].data.data() + accessor.byteOffset + bufferView.byteOffset;
			auto size = bufferView.byteLength;

			std::memcpy(tgt, src, size);

			bytes_copied += size;
		}
	}

	// Allocate shared buffer for index data and copy data from gltf to buffer
	mesh_data.index_data.raw_data = new uint8_t[mesh_data.index_data.byte_size];

	bytes_copied = 0;
	for (auto& model : models)
	{
		auto& accessor = model.accessors[model.meshes.front().primitives.front().indices];
		auto& bufferView = model.bufferViews[accessor.bufferView];

		auto tgt = mesh_data.index_data.raw_data + bytes_copied;
		auto src = model.buffers[bufferView.buffer].data.data() + accessor.byteOffset + bufferView.byteOffset;
		auto size = bufferView.byteLength;

		std::memcpy(tgt, src, size);

		bytes_copied += size;
	}

	assert(bytes_copied == mesh_data.index_data.byte_size);

	// TODO Build initial transform matrices from node/element data

	std::mt19937 generator(4215);
	std::uniform_real_distribution<float> distr(0.05f, 0.1f);
	std::uniform_real_distribution<float> loc_distr(-0.9f, 0.9f);

	draw_command_data.draw_cnt = m_floor_elements.size() + m_beam_elements.size() + m_diagonal_elements.size();
	draw_command_data.data = new DrawCommandDataAccessor::DrawElementsCommand[draw_command_data.draw_cnt];

	mesh_shader_params.byte_size = 16 * 4 * draw_command_data.draw_cnt;
	mesh_shader_params.raw_data = new uint8_t[mesh_shader_params.byte_size];

	// compute element offset
	Mat4x4 tower_model_matrix;
	tower_model_matrix.SetAt(0, 3, 0.0f);
	tower_model_matrix.SetAt(1, 3, -1.0f);
	tower_model_matrix.SetAt(2, 3, 0.0f);

	int counter = 0;
	for (auto& element : m_floor_elements)
	{
		draw_command_data.data[counter].cnt = indices_cnt[2];
		draw_command_data.data[counter].instance_cnt = 1;
		draw_command_data.data[counter].first_idx = first_indices[2];
		draw_command_data.data[counter].base_vertex = base_vertices[2];
		draw_command_data.data[counter].base_instance = 0;

		vislib::math::Matrix<GLfloat, 4, vislib::math::COLUMN_MAJOR> object_transform;
		GLfloat scale = distr(generator);
		scale = 1.0f;
		object_transform.SetAt(0, 0, scale);
		object_transform.SetAt(1, 1, scale);
		object_transform.SetAt(2, 2, scale);

		object_transform.SetAt(0, 3, std::get<0>(m_nodes[std::get<3>(element)]));
		object_transform.SetAt(1, 3, std::get<1>(m_nodes[std::get<3>(element)]) -1.0f);
		object_transform.SetAt(2, 3, std::get<2>(m_nodes[std::get<3>(element)]));

		std::memcpy(mesh_shader_params.raw_data + counter*(16 * 4), object_transform.PeekComponents(), 16 * 4);

		counter++;
	}

	for (auto& element : m_beam_elements)
	{
		draw_command_data.data[counter].cnt = indices_cnt[0];
		draw_command_data.data[counter].instance_cnt = 1;
		draw_command_data.data[counter].first_idx = first_indices[0];
		draw_command_data.data[counter].base_vertex = base_vertices[0];
		draw_command_data.data[counter].base_instance = 0;

		Mat4x4 object_transform = tower_model_matrix * computeElementTransform(m_nodes[std::get<0>(element)], m_nodes[std::get<1>(element)], Vec3(), Vec3());

		std::memcpy(mesh_shader_params.raw_data + counter*(16 * 4), object_transform.PeekComponents(), 16 * 4);

		counter++;
	}

	for (auto& element : m_diagonal_elements)
	{
		draw_command_data.data[counter].cnt = indices_cnt[1];
		draw_command_data.data[counter].instance_cnt = 1;
		draw_command_data.data[counter].first_idx = first_indices[1];
		draw_command_data.data[counter].base_vertex = base_vertices[1];
		draw_command_data.data[counter].base_instance = 0;

		Mat4x4 object_transform = tower_model_matrix * computeElementTransform(m_nodes[std::get<0>(element)], m_nodes[std::get<1>(element)], Vec3(), Vec3());

		std::memcpy(mesh_shader_params.raw_data + counter*(16 * 4), object_transform.PeekComponents(), 16 * 4);

		counter++;
	}

	mtl_shader_params.elements_cnt = 0;
	
	m_render_batches.addBatch(
		shader_prgm_data,
		mesh_data,
		draw_command_data,
		mesh_shader_params,
		mtl_shader_params);
	
	return true;
}