/*
* AbstractmeshDataSource.h
*
* Copyright (C) 2017 by Universitaet Stuttgart (VISUS).
* All rights reserved.
*/

#include <iostream>
#include <random>
#include <string>

#include <sstream>
#include <iomanip>

#include "vislib/graphics/gl/IncludeAllGL.h"
#include "vislib/net/SocketException.h"
#include "vislib/graphics/gl/Verdana.inc"

#include "mmcore/param/FilePathParam.h"
#include "mmcore/param/IntParam.h"
#include "mmcore/param/StringParam.h"

#include "stdafx.h"
#include "ArchVisMSMDataSource.h"

#define TINYGLTF_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#include "tiny_gltf.h"

using namespace megamol::archvis;

ArchVisMSMDataSource::ArchVisMSMDataSource() :
	m_partsList_slot("Parts list", "The path to the parts list file to load"),
	m_nodes_slot("Node list","The filepath of the node list to load"),
	m_elements_slot("Edge list", "The filepath of the element list to load"),
	m_nodeElement_table_slot("Node/Element table", "The path to the node/element table to load"),
	m_rcv_IPAddr_slot("Receive IP adress", "The ip adress for receiving data"),
	m_rcv_port_slot("Receive port", "The port for receiving data"),
	m_snd_IPAddr_slot("Send IP adress", "The ip adress for sending data"),
	m_snd_port_slot("Send port", "The port for sending data"),
	m_rcv_socket_connected(false),
	font("Evolventa-SansSerif", core::utility::SDFFont::RenderType::RENDERTYPE_FILL),
	m_last_spawn_time(std::chrono::steady_clock::now()),
	m_last_update_time(std::chrono::steady_clock::now())
{

	this->m_partsList_slot << new core::param::FilePathParam("");
	this->MakeSlotAvailable(&this->m_partsList_slot);

	this->m_nodes_slot << new core::param::FilePathParam("");
	this->MakeSlotAvailable(&this->m_nodes_slot);

	this->m_elements_slot << new core::param::FilePathParam("");
	this->MakeSlotAvailable(&this->m_elements_slot);

	this->m_nodeElement_table_slot << new core::param::FilePathParam("");
	this->MakeSlotAvailable(&this->m_nodeElement_table_slot);
	
	m_rcv_IPAddr_slot << new core::param::StringParam("127.0.0.1");
	this->MakeSlotAvailable(&this->m_rcv_IPAddr_slot);

	m_rcv_port_slot << new core::param::IntParam(9050);
	this->MakeSlotAvailable(&this->m_rcv_port_slot);

	m_snd_IPAddr_slot << new core::param::StringParam("127.0.0.1");
	this->MakeSlotAvailable(&this->m_snd_IPAddr_slot);

	m_snd_port_slot << new core::param::IntParam(0);
	this->MakeSlotAvailable(&this->m_snd_port_slot);

	try {
		// try to start up socket
		vislib::net::Socket::Startup();
		// create receive socket
		this->m_rcv_socket.Create(vislib::net::Socket::ProtocolFamily::FAMILY_INET, vislib::net::Socket::Type::TYPE_DGRAM, vislib::net::Socket::Protocol::PROTOCOL_UDP);

		// create send socket
		this->m_snd_socket.Create(vislib::net::Socket::ProtocolFamily::FAMILY_INET, vislib::net::Socket::Type::TYPE_STREAM, vislib::net::Socket::Protocol::PROTOCOL_TCP);
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

bool megamol::archvis::ArchVisMSMDataSource::create(void)
{
	return false;
}

bool ArchVisMSMDataSource::getDataCallback(megamol::core::Call& caller)
{
	static bool fontInitialized = false;
	if (!fontInitialized) {
		if (font.Initialise(this->GetCoreInstance()))
		{
			std::cout << "Font Initialised. " << std::endl;
		}
		fontInitialized = true;
	}

	if (this->m_partsList_slot.IsDirty() ||
		this->m_nodeElement_table_slot.IsDirty() ||
		this->m_nodes_slot.IsDirty() ||
		this->m_elements_slot.IsDirty() )
	{
		// TODO handle different slots seperatly ?
		this->m_partsList_slot.ResetDirty();
		this->m_nodeElement_table_slot.ResetDirty();
		this->m_nodes_slot.ResetDirty();
		this->m_elements_slot.ResetDirty();

		auto vislib_partsList_filename = m_partsList_slot.Param<megamol::core::param::FilePathParam>()->Value();
		std::string partsList_filename(vislib_partsList_filename.PeekBuffer());

		auto vislib_nodesElement_filename = m_nodeElement_table_slot.Param<megamol::core::param::FilePathParam>()->Value();
		std::string nodesElement_filename(vislib_nodesElement_filename.PeekBuffer());


		auto vislib_nodes_filename = m_nodes_slot.Param<megamol::core::param::FilePathParam>()->Value();
		std::string nodes_filename(vislib_nodes_filename.PeekBuffer());

		auto vislib_elements_filename = m_elements_slot.Param<megamol::core::param::FilePathParam>()->Value();
		std::string elements_filename(vislib_elements_filename.PeekBuffer());

		// Load scale model data
		std::vector<Vec3> node_positions;
		parseNodeList(nodes_filename, node_positions);

		std::vector<std::tuple<int, int, int, int, int>> element_data;
		parseElementList(elements_filename, element_data);

		std::vector<int> input_elements;
		parseInputElementList("input_" + elements_filename, input_elements);

		m_scale_model = ScaleModel(node_positions, element_data, input_elements);

		Mat4x4 tower_model_matrix;
		tower_model_matrix.SetAt(0, 3, -0.13f);
		tower_model_matrix.SetAt(1, 3, -1.0f);
		tower_model_matrix.SetAt(2, 3, -0.13f);

		m_scale_model.setModelTransform(tower_model_matrix);
	}

	if (this->m_rcv_IPAddr_slot.IsDirty() || this->m_rcv_port_slot.IsDirty())
	{
		this->m_rcv_IPAddr_slot.ResetDirty();

		try {
			vislib::net::IPAddress server_addr(static_cast<const char*>(m_rcv_IPAddr_slot.Param<megamol::core::param::StringParam>()->Value()));
			unsigned short server_port = static_cast<unsigned short>(m_rcv_port_slot.Param<megamol::core::param::IntParam>()->Value());
			server_addr = server_addr.Create();
			this->m_rcv_socket.Connect(vislib::net::IPEndPoint(server_addr, server_port));
			this->m_rcv_socket_connected = true;

			std::string greeting("Hello, my name is MegaMol");
			this->m_rcv_socket.Send(greeting.c_str(), greeting.length());
		}
		catch (vislib::net::SocketException e) {
			vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR, "Socket Exception during connection: %s", e.GetMsgA());
			return false;
		}
	}

	if (this->m_snd_IPAddr_slot.IsDirty() || this->m_snd_port_slot.IsDirty())
	{
		this->m_snd_IPAddr_slot.ResetDirty();

		try {
			vislib::net::IPAddress server_addr(static_cast<const char*>(m_snd_IPAddr_slot.Param<megamol::core::param::StringParam>()->Value()));
			unsigned short server_port = static_cast<unsigned short>(m_snd_port_slot.Param<megamol::core::param::IntParam>()->Value());
			server_addr = server_addr.Create();
			this->m_snd_socket.Bind(vislib::net::IPEndPoint());
			//this->m_snd_socket.Bind(vislib::net::IPEndPoint(vislib::net::IPAddress::ANY, server_port));
			//this->m_snd_socket.Connect(vislib::net::IPEndPoint(vislib::net::IPAddress::ANY, server_port));

			std::string greeting("Hello, my name is MegaMol");
			//this->m_snd_socket.Send(greeting.c_str(), greeting.length());
		}
		catch (vislib::net::SocketException e) {
			vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR, "Socket Exception during connection: %s", e.GetMsgA());
			return false;
		}
	}

	DataPackage data(m_scale_model.getNodeCount(), m_scale_model.getInputElementCount());

	size_t bytes_received = 0;

	if(this->m_rcv_socket_connected)
		bytes_received = this->m_rcv_socket.Receive(data.data(), data.getByteSize());

	if (bytes_received > 0)
	{
		//TODO somehow skip this tedious copying...
		std::vector<Vec3> displacements;
		std::vector<float> forces;

		for (int i = 0; i < m_scale_model.getNodeCount(); ++i)
		{
			displacements.push_back(data.getNodeDisplacement(i));
		}

		for (int i = 0; i < m_scale_model.getInputElementCount(); ++i)
		{
			forces.push_back(data.getElementForces(i));
		}

		std::cout << "Timestep: " << data.getTime() << std::endl;

		m_scale_model.updateNodeDisplacements(displacements);
		m_scale_model.updateElementForces(forces);
		updateMSMTransform();
		spawnAndUpdateTextLabels();
	}
	
	return true;
}

void megamol::archvis::ArchVisMSMDataSource::release()
{
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

void ArchVisMSMDataSource::parseNodeList(
	std::string const& filename,
	std::vector<Vec3>& node_positions)
{
	std::ifstream file;
	file.open(filename, std::ifstream::in);

	if (file.is_open())
	{
		file.seekg(0, std::ifstream::beg);

		unsigned int lines_read = 0;
		while (!file.eof())
		{
			std::string line;
			std::getline(file, line, '\n');
			std::stringstream ss(line);

			std::string x, y, z;
			std::getline(ss, x, ',');
			std::getline(ss, y, ',');
			std::getline(ss, z, ',');

			// flip input y and z axis to match y-up coordinate system
			node_positions.push_back(Vec3(std::stof(x), std::stof(z), std::stof(y)));
		}
	}
}

void ArchVisMSMDataSource::parseElementList(
	std::string const& filename,
	std::vector<std::tuple<int, int, int, int, int>>& element_data)
{
	std::ifstream file;
	file.open(filename, std::ifstream::in);

	if (file.is_open())
	{
		file.seekg(0, std::ifstream::beg);

		unsigned int lines_read = 0;
		while (!file.eof())
		{
			std::string line;
			std::getline(file, line, '\n');
			std::stringstream ss(line);

			std::string idx0, idx1, idx2, idx3;

			std::getline(ss, idx0, ',');
			std::getline(ss, idx1, ',');
			std::getline(ss, idx2, ',');
			std::getline(ss, idx3, ',');

			int type = (lines_read % 13) == 0 ? 2 : (lines_read % 13) < 5 ? 0 : 1;

			// given indices start at 1, so offset by -1
			element_data.push_back(std::make_tuple<int, int, int, int, int>(std::move(type), std::stoi(idx0)-1, std::stoi(idx1)-1, std::stoi(idx2)-1, std::stoi(idx3)-1));

			++lines_read;
		}
	}
}

void ArchVisMSMDataSource::parseInputElementList(
	std::string const& filename,
	std::vector<int>& input_elements)
{
	std::ifstream file;
	file.open(filename, std::ifstream::in);

	if (file.is_open())
	{
		file.seekg(0, std::ifstream::beg);

		unsigned int lines_read = 0;
		while (!file.eof())
		{
			std::string line;
			std::getline(file, line, '\n');
			std::stringstream ss(line);

			// given indices start at 1, so offset by -1
			input_elements.push_back(std::stoi(line)-1);
		}
	}
}

void ArchVisMSMDataSource::updateMSMTransform()
{
	for (auto& particle : m_text_particles)
	{
		std::string label = particle.text;

		float x = particle.position.X();
		float y = particle.position.Y();
		float z = particle.position.Z();

	
		//glEnable(GL_DEPTH_TEST);
		//glColor4f(particle.color.X(), particle.color.Y(), particle.color.Z(), 1.0f - (particle.age/2000.0));

		float c[4] = { particle.color.X(), particle.color.Y(), particle.color.Z(), 1.0 };
		
		//glPolygonMode(GL_FRONT_AND_BACK,GL_FILL);
		//font.SetBillboard(true);
		font.DrawString(c, x, y, z, 0.03f, false, label.c_str(), core::utility::SDFFont::ALIGN_LEFT_MIDDLE);
	}
}

void ArchVisMSMDataSource::spawnAndUpdateTextLabels()
{
	double elapsed_spawn_time = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - m_last_spawn_time).count();
	double elapsed_update_time = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - m_last_update_time).count();
	m_last_update_time = std::chrono::steady_clock::now();
	
	//for (auto& particle : m_text_particles)
	//{
	//	particle.age += elapsed_update_time;
	//	particle.position.SetY(particle.position.GetY() + elapsed_update_time * 0.00005);
	//	particle.position.SetX(particle.position.GetX() + (std::signbit(particle.position.GetX()) ? -1.0f : 1.0f) * elapsed_update_time * 0.000025);
	//}
	
	//m_text_particles.erase(std::remove_if(m_text_particles.begin(), m_text_particles.end(), [](const TextLabelParticle& p) { return p.age > 2000; }), m_text_particles.end());
	
	std::list<TextLabelParticle>::iterator particle_itr = m_text_particles.begin();
	if (m_text_particles.size() > 0)
	{
		for (int i = 0; i < m_scale_model.getElementCount(); ++i)
		{
			if (m_scale_model.getElementType(i) != ScaleModel::DIAGONAL)
			{
				(*particle_itr).age += elapsed_update_time;
				(*particle_itr).position = m_scale_model.getElementCenter(i) + (m_scale_model.getElementCenter(i)*Vec3(0.25, 0.0, 0.25));

				if (particle_itr != m_text_particles.end())
					++particle_itr;// = std::next(particle_itr, 1);
			}
		}
	}


	if (elapsed_spawn_time > 33)
	{
		m_text_particles.clear();

		for (int i = 0; i < m_scale_model.getElementCount(); ++i)
		{
			if(m_scale_model.getElementType(i) != ScaleModel::DIAGONAL)
			{
				TextLabelParticle new_label;
	
				float force = m_scale_model.getElementForce(i);
				std::ostringstream out;
				out << std::setprecision(3) << force;
				new_label.text = out.str();
	
				new_label.position = m_scale_model.getElementCenter(i) + (m_scale_model.getElementCenter(i)*Vec3(0.25,0.0,0.25));
	
				new_label.age = 0.0;
	
				Vec3 white(1.0f, 1.0f, 1.0f);
				Vec3 red(1.0f, 0.0f, 0.0f);
				Vec3 blue(0.0f, 0.0f, 1.0f);
				new_label.color = (force < 0.0f) ? blue * (-force/100.0f) + white * (1.0f - (-force/100.0f)) : red * (force / 100.0f) + white * (1.0f - (force / 100.0f));
	
				m_text_particles.push_back(new_label);
			}
		}
	
		m_last_spawn_time = std::chrono::steady_clock::now();
	}
}