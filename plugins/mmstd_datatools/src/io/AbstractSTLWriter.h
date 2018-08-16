/*
 * AbstractSTLWriter.h
 *
 * Copyright (C) 2018 by MegaMol Team
 * Alle Rechte vorbehalten.
 */
#ifndef MEGAMOL_DATATOOLS_IO_ABSTRACTSTLWRITER_H_INCLUDED
#define MEGAMOL_DATATOOLS_IO_ABSTRACTSTLWRITER_H_INCLUDED
#pragma once

#include "mmcore/Module.h"
#include "mmcore/Call.h"

#include "mmcore/param/ParamSlot.h"

#include "vislib/sys/Log.h"

#include <algorithm>
#include <fstream>
#include <ios>
#include <iterator>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <type_traits>
#include <vector>

namespace megamol
{
	namespace stdplugin
	{
		namespace datatools
		{
			namespace io
			{
				class AbstractSTLWriter : public core::Module
				{
				public:
					/// <summary>
					/// Constructor
					/// </summary>
					AbstractSTLWriter();

					/// <summary>
					/// Destructor
					/// </summary>
					~AbstractSTLWriter();

				protected:
					/// <summary>
					/// Create the module
					/// </summary>
					/// <returns>True on success; false otherwise</returns>
					virtual bool create() = 0;

					/// <summary>
					/// Callback function for requesting information
					/// </summary>
					/// <param name="caller">Call for this request</param>
					/// <returns>True on success; false otherwise</returns>
					virtual bool get_extent_callback(core::Call& caller) = 0;

					/// <summary>
					/// Callback function for requesting data
					/// </summary>
					/// <param name="caller">Call for this request</param>
					/// <returns>True on success; false otherwise</returns>
					virtual bool get_data_callback(core::Call& caller) = 0;

					/// <summary>
					/// Release the module
					/// </summary>
					virtual void release() = 0;

					/// <summary>
					/// Write an STL file
					/// </summary>
					/// <param name="num_triangles">Number of triangles</param>
					/// <param name="vertices">Vertex pointer</param>
					/// <param name="normals">Normal pointer</param>
					/// <param name="indices">Index pointer</param>
					/// <typeparam name="VFT">Floating point type of the vertices</typeparam>
					/// <typeparam name="NFT">Floating point type of the normals</typeparam>
					/// <typeparam name="IT">Integer type of the indices</typeparam>
					template <typename VFT, typename NFT, typename IT = nullptr_t>
					void write(std::size_t num_triangles, const VFT* vertices, const NFT* normals, const IT* indices = nullptr) const
					{
						static_assert(std::is_integral<IT>::value || std::is_null_pointer<IT>::value, "Indices must be of integral type");

						// Get filename
						const auto& vislib_filename = AbstractSTLWriter::filename_slot.Param<core::param::FilePathParam>()->Value();
						const std::string filename(vislib_filename.PeekBuffer());

						// Decide file type
						if (this->ascii_binary_slot.Param<core::param::EnumParam>()->Value() == 0)
						{
							vislib::sys::Log::DefaultLog.WriteInfo("Writing binary STL file '%s'", filename.c_str());
							write_binary(filename, static_cast<uint32_t>(num_triangles), vertices, normals, pointer_or_identity<IT>(indices));
						}
						else
						{
							vislib::sys::Log::DefaultLog.WriteInfo("Writing ASCII STL file '%s'", filename.c_str());
							write_ascii(filename, num_triangles, vertices, normals, pointer_or_identity<IT>(indices));
						}
					}
				private:
					/// <summary>
					/// Struct for returning the incoming value or the value at the pointer's position
					/// </summary>
					/// <typeparam name="IT">Index type</typeparam>
					template <typename IT>
					struct pointer_or_identity
					{
						pointer_or_identity(const IT* pointer)
						{
							this->pointer = pointer;
						}

						std::size_t operator[](const std::size_t i) const
						{
							return static_cast<std::size_t>(this->pointer[i]);
						}

					private:
						const IT* pointer;
					};

					template <>
					struct pointer_or_identity<nullptr_t>
					{
						pointer_or_identity(const nullptr_t*)
						{ }

						std::size_t operator[](const std::size_t i) const
						{
							return i;
						}
					};

					/// <summary>
					/// Class for wrapping a pointer, using a pointer or a container as input
					/// </summary>
					/// <typeparam name="VT">Value type</typeparam>
					/// <typeparam name="CT">Container type</typeparam>
					template <typename VT, typename CT>
					class pointer_wrapper
					{
					public:
						using value_type = VT;
						using container_type = CT;

						static_assert(std::is_same<typename CT::value_type, VT>::value, "Value type of container must match the value type");

						/// <summary>
						/// Constructor for nullptr
						/// </summary>
						pointer_wrapper()
						{
							this->pointer = nullptr;
						}

						/// <summary>
						/// Constructor from pointer (does not assume ownership)
						/// </summary>
						/// <param name="pointer">Pointer</param>
						pointer_wrapper(const VT* pointer)
						{
							this->pointer = pointer;
						}

						/// <summary>
						/// Constructor from invalid pointer
						/// </summary>
						/// <param name="pointer">Pointer</param>
						pointer_wrapper(const void*)
						{ }

						/// <summary>
						/// Construct from moved container (destroy on desctruction)
						/// </summary>
						/// <param name="container">Container</param>
						pointer_wrapper(CT&& container)
						{
							std::swap(this->container, container);
							this->pointer = this->container.data();
						}

						/// <summary>
						/// Construct from other
						/// </summary>
						/// <param name="original">Source</param>
						pointer_wrapper(pointer_wrapper&& original)
						{
							std::swap(this->container, original.container);
							this->pointer = this->container.data();
						}

						/// <summary>
						/// Move operator
						/// </summary>
						/// <param name="original">Source</param>
						/// <returns>This</returns>
						pointer_wrapper& operator=(pointer_wrapper&& original)
						{
							std::swap(this->container, original.container);
							std::swap(this->pointer, original.pointer);

							return *this;
						}

						/// <summary>
						/// Return data pointer
						/// </summary>
						/// <returns>Data pointer</returns>
						const VT* get() const
						{
							return this->pointer;
						}

					private:
						/// Store pointer and container
						const VT* pointer;
						CT container;
					};

					/// <summary>
					/// Convert original data to new type
					/// </summary>
					/// <param name="original">Pointer to original data</param>
					/// <param name="length">Length of the array</param>
					/// <typeparam name="NT">Type to convert to</typeparam>
					/// <typeparam name="OT">Type to convert from</typeparam>
					/// <returns>Wrapper around converted data</returns>
					template <typename NT, typename OT>
					pointer_wrapper<NT, std::vector<NT>> convert_if_necessary(const OT* original, std::size_t length,
						typename std::enable_if<!std::is_same<OT, NT>::value>::type* = nullptr) const
					{
						std::vector<NT> converted(length);
						std::transform(original, original + length, converted.begin(), [](const OT& value) { return static_cast<NT>(value); });

						return pointer_wrapper<NT, std::vector<NT>>(std::move(converted));
					}

					template <typename NT, typename OT>
					pointer_wrapper<NT, std::vector<NT>> convert_if_necessary(const OT* original, std::size_t length,
						typename std::enable_if<std::is_same<OT, NT>::value>::type* = nullptr) const
					{
						return pointer_wrapper<NT, std::vector<NT>>(original);
					}

					/// <summary>
					/// Write a binary file
					/// </summary>
					/// <param name="filename">File name of the STL file</param>
					/// <param name="num_triangles">Number of triangles</param>
					/// <param name="vertices_ptr">Vertex pointer</param>
					/// <param name="normals_ptr">Normal pointer</param>
					/// <param name="indices">Index pointer</param>
					/// <typeparam name="VFT">Floating point type of the vertices</typeparam>
					/// <typeparam name="NFT">Floating point type of the normals</typeparam>
					/// <typeparam name="IT">Integer type of the indices</typeparam>
					template <typename VFT, typename NFT, typename IT>
					void write_binary(const std::string& filename, const uint32_t num_triangles, const VFT* vertices_ptr, const NFT* normals_ptr, const IT& indices) const
					{
						static_assert(std::is_floating_point<VFT>::value, "Vertices must be of floating point type");
						static_assert(std::is_floating_point<NFT>::value, "Normals must be of floating point type");

						if (num_triangles == 0)
						{
							vislib::sys::Log::DefaultLog.WriteWarn("Cannot write STL file. Number of triangles is zero!");
							return;
						}

						// Convert vertices and normals to float if it is not
						auto vertices_wrapper = convert_if_necessary<float>(vertices_ptr, num_triangles * 9);
						auto normals_wrapper = convert_if_necessary<float>(normals_ptr, num_triangles * 9);

						const float* vertices = vertices_wrapper.get();
						const float* normals = normals_wrapper.get();

						// Open or create file
						std::ofstream ofs(filename, std::ios_base::out | std::ios_base::binary);

						if (ofs.good())
						{
							// Write header
							std::string header_message("MegaMol by University of Stuttgart, Germany");

							std::vector<char> header_buffer(header_message.begin(), header_message.end());
							header_buffer.resize(80);

							ofs.write(header_buffer.data(), header_buffer.size());

							// Write number of triangles
							ofs.write(reinterpret_cast<const char*>(&num_triangles), sizeof(uint32_t));

							// Write vertices and normals
							const uint16_t additional_attribute = 21313;

							for (uint32_t triangle_index = 0; triangle_index < num_triangles; ++triangle_index)
							{
								ofs.write(reinterpret_cast<const char*>(&normals[indices[triangle_index * 3] * 3]), 3 * sizeof(float));
								ofs.write(reinterpret_cast<const char*>(&vertices[indices[triangle_index * 3 + 0] * 3]), 3 * sizeof(float));
								ofs.write(reinterpret_cast<const char*>(&vertices[indices[triangle_index * 3 + 1] * 3]), 3 * sizeof(float));
								ofs.write(reinterpret_cast<const char*>(&vertices[indices[triangle_index * 3 + 2] * 3]), 3 * sizeof(float));
								ofs.write(reinterpret_cast<const char*>(&additional_attribute), sizeof(uint16_t));
							}

							ofs.close();
						}
						else
						{
							std::stringstream ss;
							ss << "Binary STL file '" << filename << "' could not be written";

							throw std::runtime_error(ss.str());
						}
					}

					/// <summary>
					/// Write an ASCII file
					/// </summary>
					/// <param name="filename">File name of the STL file</param>
					/// <param name="num_triangles">Number of triangles</param>
					/// <param name="vertices">Vertex pointer</param>
					/// <param name="normals">Normal pointer</param>
					/// <param name="indices">Index pointer</param>
					/// <typeparam name="VFT">Floating point type of the vertices</typeparam>
					/// <typeparam name="NFT">Floating point type of the normals</typeparam>
					/// <typeparam name="IT">Integer type of the indices</typeparam>
					template <typename VFT, typename NFT, typename IT>
					void write_ascii(const std::string& filename, const std::size_t num_triangles, const VFT* vertices, const NFT* normals, const IT& indices) const
					{
						static_assert(std::is_floating_point<VFT>::value, "Vertices must be of floating point type");
						static_assert(std::is_floating_point<NFT>::value, "Normals must be of floating point type");

						if (num_triangles == 0)
						{
							vislib::sys::Log::DefaultLog.WriteWarn("Cannot write STL file. Number of triangles is zero!");
							return;
						}

						// Open or create file
						std::ofstream ofs(filename, std::ios_base::out);

						if (ofs.good())
						{
							// Write mesh
							ofs << "solid megamol_mesh\n";
							ofs << std::scientific;

							for (std::size_t triangle_index = 0; triangle_index < num_triangles; ++triangle_index)
							{
								ofs << "\tfacet\n";

								ofs << "\t\tnormal "
									<< normals[indices[triangle_index * 3] * 3 + 0] << " "
									<< normals[indices[triangle_index * 3] * 3 + 1] << " "
									<< normals[indices[triangle_index * 3] * 3 + 2] << "\n";

								ofs << "\t\touter loop\n";

								ofs << "\t\t\tvertex "
									<< vertices[indices[triangle_index * 3 + 0] * 3 + 0] << " "
									<< vertices[indices[triangle_index * 3 + 0] * 3 + 1] << " "
									<< vertices[indices[triangle_index * 3 + 0] * 3 + 2] << "\n";
								ofs << "\t\t\tvertex "
									<< vertices[indices[triangle_index * 3 + 1] * 3 + 0] << " "
									<< vertices[indices[triangle_index * 3 + 1] * 3 + 1] << " "
									<< vertices[indices[triangle_index * 3 + 1] * 3 + 2] << "\n";
								ofs << "\t\t\tvertex "
									<< vertices[indices[triangle_index * 3 + 2] * 3 + 0] << " "
									<< vertices[indices[triangle_index * 3 + 2] * 3 + 1] << " "
									<< vertices[indices[triangle_index * 3 + 2] * 3 + 2] << "\n";

								ofs << "\t\tendloop\n";

								ofs << "\tendfacet" << std::endl;
							}

							ofs << "endsolid megamol_mesh" << std::flush;

							ofs.close();
						}
						else
						{
							std::stringstream ss;
							ss << "ASCII STL file '" << filename << "' could not be written";

							throw std::runtime_error(ss.str());
						}
					}

					/// File name
					core::param::ParamSlot filename_slot;

					/// Option for ASCII/binary
					core::param::ParamSlot ascii_binary_slot;
				};
			}
		}
	}
}

#endif // !MEGAMOL_DATATOOLS_IO_ABSTRACTSTLWRITER_H_INCLUDED