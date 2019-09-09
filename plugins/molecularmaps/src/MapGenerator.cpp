/*
 * MapGenerator.cpp
 * Copyright (C) 2006-2016 by MegaMol Team
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "MapGenerator.h"

#include "geometry_calls/CallTriMeshData.h"
#include "mmcore/param/FloatParam.h"
#include "mmcore/param/IntParam.h"
#include "mmcore/param/ButtonParam.h"
#include "mmcore/param/EnumParam.h"
#include "mmcore/param/BoolParam.h"
#include "mmcore/param/FilePathParam.h"
#include "mmcore/param/StringParam.h"
#include "mmcore/param/Vector3fParam.h"
#include "mmcore/view/CallRender3D.h"
#include "protein_calls/MolecularDataCall.h"

using namespace megamol;
using namespace megamol::core;
using namespace megamol::geocalls;
using namespace megamol::molecularmaps;
using namespace megamol::protein_calls;

/*
 * MapGenerator::MapGenerator
 */
MapGenerator::MapGenerator(void) : Renderer3DModule(),
		aoActive("ambientOcclusion::useAO", "Flag whether or not use Ambient Occlusion for the tunnel detection."),
		aoAngleFactorParam("ambientOcclusion::angleFactor", "Factor for the angle between two sample directions"),
		aoEvalParam("ambientOcclusion::eval", "Scaling factor for the final brightness value"),
		aoFalloffParam("ambientOcclusion::falloff", "Exponent of the distance function of the ambient occlusion"),
		aoGenFactorParam("ambientOcclusion::genFactor", "The influence factor for the influence of a sphere on a single voxel"),
		aoMaxDistSample("ambientOcclusion::maxDistance", "The maximum distance between the surface and the last sample"),
		aoMinDistSample("ambientOcclusion::minDistance", "The distance between the surface and the first sample"),
		aoNumSampleDirectionsParam("ambientOcclusion::numSampleDirections", "The number of sample directions per vertex"),
		aoScalingFactorParam("ambientOcclusion::scaling", "Scaling factor for the particle radii"),
		aoThresholdParam("ambientOcclusion::threshold", "Set the thresholding factor for the shadow test."),
		aoVolSizeXParam("ambientOcclusion::volSizeX", "Size of the shadow volume in x-direction (in voxels)"),
		aoVolSizeYParam("ambientOcclusion::volSizeY", "Size of the shadow volume in y-direction (in voxels)"),
		aoVolSizeZParam("ambientOcclusion::volSizeZ", "Size of the shadow volume in z-direction (in voxels)"),
		bindingSiteColor("bindingSite::color", "The color of the selected binding site."),
		bindingSiteColoring("bindingSite::enable", "Flag whether or not use coloring of a specific binding site"),
		bindingSiteIgnoreRadius("bindingSite::ingoreRadius", "Flag whether or not use the radius around a binding site. Instead, all atoms of the site are colored"),
		bindingSiteRadius("bindingSite::radius", "The radius of the colored binding site. A negative radius means we use the circumcircle of the site as radius."),
		bindingSiteRadiusOffset("bindingSite::radiusOffset", "The offset that gets added to the radius of the computed sphere. This is ignored if radius is positive."),
		blending("rendering::blending", "Flag whether or not use blending for the surface"),
		computeButton("recompute", "Button that starts the computation of the molecular map"),
		cut_colour_param("colour::cutColour", "The path to the file that contains the colours for the cuts"),
		display_param("display mode", "Choose what to display, protein, sphere, map and debug modes"),
        draw_wireframe_param("wireframe" , "Choose whether to render meshes as wireframe or not"),
		geodesic_lines_param("geodesic lines", "Choose what kind of geodesic lines to display"),
		group_colour_param("colour::groupColour", "The path to the file that contains the colours for the groups"),
		lat_lines_count_param("grid::latLinesCount", "The number of latitude lines"),
		lat_lon_lines_param("grid::toggle lat/lon lines", "Turn latitude and longitude lines on or off"),
		lat_lon_lines_colour_param("grid::color", "The base color for the latitude/longitude grid"),
		lat_lon_lines_eq_colour_param("grid::equator color", "The color of the equator"),
		lat_lon_lines_gm_colour_param("grid::meridian color", "The color of the Greenwich meridian"),
		lighting("rendering::lighting", "Flag whether or not use lighting for the surface"),
		lon_lines_count_param("grid::lonLinesCount", "The number of longitude lines"),
        meshDataOutSlot("meshDataOut", "The output mesh data"),
		meshDataSlot("meshData", "The input mesh data"), 
		meshDataSlotWithCap("capData", "The input mesh data with the cap"),
		mirror_map_param("mirrorMap", "Choose whether the final map should be mirrored or not"),
        out_mesh_selection_slot("outputMeshMode", "Choose the mesh that is passed on to the rest of the application"),
		probeRadiusSlot("probeRadius", "The radius of the probe for protein channel detection in Angstrom"),
		proteinDataSlot("proteinData", "The input protein data"),
		radius_offset_param("radiusOffset", "The offset for the BoundingSphere radius"),
		shaderReloadButtonParam("shaderReload", "Triggers the reloading of the shader programs"),
		store_png_button("screenshot::Store Map To PNG", "Stores the molecular surface map to a PNG image file"),
		store_png_font(vislib::graphics::gl::FontInfo_Verdana),
		store_png_path("screenshot::Filename for map(PNG)", "Filename of the PNG image file to which the map will be stored"),
		zeBindingSiteSlot("bindingSite", "The input binding site data") {
	this->aoActive.SetParameter(new param::BoolParam(false));
	this->MakeSlotAvailable(&this->aoActive);

	this->aoAngleFactorParam.SetParameter(new param::FloatParam(1.0f, 0.0f));
	this->MakeSlotAvailable(&this->aoAngleFactorParam);

	this->aoEvalParam.SetParameter(new param::FloatParam(1.0f, 0.0f));
	this->MakeSlotAvailable(&this->aoEvalParam);

	this->aoFalloffParam.SetParameter(new param::FloatParam(1.0f, 0.0f));
	this->MakeSlotAvailable(&this->aoFalloffParam);

	this->aoGenFactorParam.SetParameter(new param::FloatParam(1.0f, 0.0f));
	this->MakeSlotAvailable(&this->aoGenFactorParam);

	this->aoMaxDistSample.SetParameter(new param::FloatParam(0.5f, 0.0f, 2.0f));
	this->MakeSlotAvailable(&this->aoMaxDistSample);

	this->aoMinDistSample.SetParameter(new param::FloatParam(0.5f, 0.0f));
	this->MakeSlotAvailable(&this->aoMinDistSample);

	this->aoNumSampleDirectionsParam.SetParameter(new param::IntParam(8, 1, 200));
	this->MakeSlotAvailable(&this->aoNumSampleDirectionsParam);

	this->aoScalingFactorParam.SetParameter(new param::FloatParam(1.0f, 0.0f));
	this->MakeSlotAvailable(&this->aoScalingFactorParam);

	this->aoThresholdParam.SetParameter(new param::FloatParam(0.0f, 0.0f, 1.0f));
	this->MakeSlotAvailable(&this->aoThresholdParam);

	this->aoVolSizeXParam.SetParameter(new param::IntParam(32, 2, 512));
	this->MakeSlotAvailable(&this->aoVolSizeXParam);

	this->aoVolSizeYParam.SetParameter(new param::IntParam(32, 2, 512));
	this->MakeSlotAvailable(&this->aoVolSizeYParam);

	this->aoVolSizeZParam.SetParameter(new param::IntParam(32, 2, 512));
	this->MakeSlotAvailable(&this->aoVolSizeZParam);

	this->bindingSiteColor.SetParameter(new param::StringParam("#ffffff"));
	this->MakeSlotAvailable(&this->bindingSiteColor);

	this->bindingSiteColoring.SetParameter(new param::BoolParam(false));
	this->MakeSlotAvailable(&this->bindingSiteColoring);

	this->bindingSiteIgnoreRadius.SetParameter(new param::BoolParam(false));
	this->MakeSlotAvailable(&this->bindingSiteIgnoreRadius);

	this->bindingSiteRadius.SetParameter(new param::FloatParam(2.5f));
	this->MakeSlotAvailable(&this->bindingSiteRadius);

	this->bindingSiteRadiusOffset.SetParameter(new param::FloatParam(0.0f));
	this->MakeSlotAvailable(&this->bindingSiteRadiusOffset);

	this->blending.SetParameter(new param::BoolParam(false));
	this->MakeSlotAvailable(&this->blending);

	this->computeButton.SetParameter(new param::ButtonParam(view::Key::KEY_C));
	this->MakeSlotAvailable(&this->computeButton);

	this->computed_map = false;
	this->computed_sphere = false;

	this->cut_colour_param.SetParameter(new param::FilePathParam(""));
	this->MakeSlotAvailable(&this->cut_colour_param);

	this->cuda_kernels = std::unique_ptr<CUDAKernels>(new CUDAKernels());

	param::EnumParam* display_mode_param = new param::EnumParam(static_cast<int>(DisplayMode::PROTEIN));
	DisplayMode display_mode = DisplayMode::PROTEIN;
	display_mode_param->SetTypePair(display_mode, "Show protein");
	display_mode = DisplayMode::SHADOW;
	display_mode_param->SetTypePair(display_mode, "Show shadows");
	display_mode = DisplayMode::GROUPS;
	display_mode_param->SetTypePair(display_mode, "Show groups");
	display_mode = DisplayMode::VORONOI;
	display_mode_param->SetTypePair(display_mode, "Show voronoi groups");
	display_mode = DisplayMode::CUTS;
	display_mode_param->SetTypePair(display_mode, "Show cuts");
	display_mode = DisplayMode::TUNNEL;
	display_mode_param->SetTypePair(display_mode, "Show tunnel");
	display_mode = DisplayMode::REBUILD;
	display_mode_param->SetTypePair(display_mode, "Show new surface");
	display_mode = DisplayMode::SPHERE;
	display_mode_param->SetTypePair(display_mode, "Show sphere");
	display_mode = DisplayMode::MAP;
	display_mode_param->SetTypePair(display_mode, "Show map");
	this->display_param << display_mode_param;
	this->MakeSlotAvailable(&this->display_param);

    this->draw_wireframe_param.SetParameter(new param::BoolParam(false));
    this->MakeSlotAvailable(&this->draw_wireframe_param);

	this->face_edge_offset = std::vector<std::vector<Edge>>(0);
	this->face_edge_offset_depth = std::vector<uint>(0);
	this->faces = std::vector<uint>(0);
	this->faces_rebuild = std::vector<uint>(0);

	this->geodesic_lines = std::vector<std::vector<float>>(0);
	this->geodesic_lines_colours = std::vector<std::vector<float>>(0);
	this->geodesic_lines_vbos = std::vector<GLuint>(0);

	param::EnumParam* geodesic_mode_param = new param::EnumParam(static_cast<int>(GeodesicMode::NO_LINES));
	GeodesicMode geodesic_mode = GeodesicMode::NO_LINES;
	geodesic_mode_param->SetTypePair(geodesic_mode, "No lines");
	geodesic_mode = GeodesicMode::ONE_TO_ALL;
	geodesic_mode_param->SetTypePair(geodesic_mode, "One to all");
	geodesic_mode = GeodesicMode::ALL_TO_ALL;
	geodesic_mode_param->SetTypePair(geodesic_mode, "All to all");
	this->geodesic_lines_param << geodesic_mode_param;
	this->MakeSlotAvailable(&this->geodesic_lines_param);

	this->group_colour_param.SetParameter(new param::FilePathParam(""));
	this->MakeSlotAvailable(&this->group_colour_param);

	this->lastDataHash = 0;

	this->lat_lines_count_param << new param::IntParam(11, 1, 359);
	this->MakeSlotAvailable(&this->lat_lines_count_param);

	this->lat_lon_lines_param << new param::BoolParam(true);
	this->MakeSlotAvailable(&this->lat_lon_lines_param);

	this->lat_lon_lines_colour_param << new param::StringParam("white");
	this->MakeSlotAvailable(&this->lat_lon_lines_colour_param);

	this->lat_lon_lines_eq_colour_param << new param::StringParam("red");
	this->MakeSlotAvailable(&this->lat_lon_lines_eq_colour_param);

	this->lat_lon_lines_gm_colour_param << new param::StringParam("yellow");
	this->MakeSlotAvailable(&this->lat_lon_lines_gm_colour_param);

	this->lat_lon_lines_vbo = 0;
	this->lat_lon_lines_vertex_cnt = 0;

	this->lighting.SetParameter(new param::BoolParam(true));
	this->MakeSlotAvailable(&this->lighting);

	this->lon_lines_count_param << new param::IntParam(8, 1, 179);
	this->MakeSlotAvailable(&this->lon_lines_count_param);

	this->look_at_id = 0;

	this->map_shader_init = false;
	this->map_vertex_vbo = 0;

    this->meshBoundingBox = vislib::math::Cuboid<float>();

    this->meshDataOutSlot.SetCallback(geocalls::CallTriMeshData::ClassName(), geocalls::CallTriMeshData::FunctionName(0), &MapGenerator::GetMeshData);
    this->meshDataOutSlot.SetCallback(geocalls::CallTriMeshData::ClassName(), geocalls::CallTriMeshData::FunctionName(1), &MapGenerator::GetMeshExtents);
    this->MakeSlotAvailable(&this->meshDataOutSlot);

	this->meshDataSlot.SetCompatibleCall<CallTriMeshDataDescription>();
	this->MakeSlotAvailable(&this->meshDataSlot);

	this->meshDataSlotWithCap.SetCompatibleCall<CallTriMeshDataDescription>();
	this->MakeSlotAvailable(&this->meshDataSlotWithCap);

	this->mirror_map_param.SetParameter(new param::BoolParam(false));
	this->MakeSlotAvailable(&this->mirror_map_param);

	this->normals = std::vector<float>(0);
	this->normals_rebuild = std::vector<float>(0);

    param::EnumParam* out_mesh_param = new param::EnumParam(static_cast<int>(MeshMode::MESH_ORIGINAL));
    MeshMode out_mesh_mode = MeshMode::MESH_ORIGINAL;
    out_mesh_param->SetTypePair(out_mesh_mode, "Original");
    out_mesh_mode = MeshMode::MESH_CUT;
    out_mesh_param->SetTypePair(out_mesh_mode, "Cut Mesh");
    out_mesh_mode = MeshMode::MESH_SPHERE;
    out_mesh_param->SetTypePair(out_mesh_mode, "Sphere");
    out_mesh_mode = MeshMode::MESH_MAP;
    out_mesh_param->SetTypePair(out_mesh_mode, "Map");
    this->out_mesh_selection_slot << out_mesh_param;
    this->MakeSlotAvailable(&this->out_mesh_selection_slot);

	this->probeRadiusSlot.SetParameter(new param::FloatParam(1.5f, 0.0f, 10.0f));
	this->MakeSlotAvailable(&this->probeRadiusSlot);

	this->proteinDataSlot.SetCompatibleCall<protein_calls::MolecularDataCallDescription>();
	this->MakeSlotAvailable(&this->proteinDataSlot);

	this->radius_offset_param.SetParameter(new param::FloatParam(0.1f, 0.0f, 1.0f));
	this->MakeSlotAvailable(&this->radius_offset_param);

	this->shaderReloadButtonParam << new param::ButtonParam(view::Key::KEY_F5);
	this->MakeSlotAvailable(&this->shaderReloadButtonParam);

    this->store_new_mesh = false;

	this->store_png_button.SetParameter(new param::ButtonParam(view::Key::KEY_S));
	this->MakeSlotAvailable(&this->store_png_button);

	this->store_png_path.SetParameter(new param::FilePathParam(""));
	this->MakeSlotAvailable(&this->store_png_path);

	this->tunnel_faces = std::vector<uint>(0);

	this->vertexColors = std::vector<float>(0);
	this->vertexColors_rebuild = std::vector<float>(0);
	this->vertexColors_tunnel = std::vector<float>(0);

	this->vertex_edge_offset = std::vector<std::vector<Edge>>(0);
	this->vertex_edge_offset_depth = std::vector<uint>(0);
	this->vertex_edge_offset_max_depth = 30;

	this->vertices = std::vector<float>(0);
	this->vertices_added = std::vector<uint>(0);
	this->vertices_added_tunnel_id = std::vector<uint>(0);
	this->vertices_rebuild = std::vector<float>(0);
	this->vertices_sphere = std::vector<float>(0);

	this->voronoiNeeded = false;

	this->zeBindingSiteSlot.SetCompatibleCall<protein_calls::BindingSiteCallDescription>();
	this->MakeSlotAvailable(&this->zeBindingSiteSlot);
}


/*
 * MapGenerator::~MapGenerator
 */
MapGenerator::~MapGenerator(void) {
	this->Release();
}


/*
 * MapGenerator::create
 */
bool MapGenerator::create(void) {
	this->triMeshRenderer.create();
	this->voronoiCalc.create();
	return true;
}


/*
 * MapGenerator::release
 */
void MapGenerator::release(void) {
	if (this->map_vertex_vbo != 0)
		glDeleteBuffers(1, &this->map_vertex_vbo);
}


/*
 * MapGenerator::allElementsTrue
 */
bool MapGenerator::allElementsTrue(const std::vector<bool>& p_vec) {
	for (size_t i = 0; i < p_vec.size(); i++) {
		if (!p_vec[i]) return false;
	}
	return true;
}


/*
 * MapGenerator::capColouring
 */
bool MapGenerator::capColouring(CallTriMeshData* p_cap_data_call, view::CallRender3D* p_cr3d,
		protein_calls::BindingSiteCall* p_bs) {
	// Check the calls.
	if (p_bs == nullptr || p_cr3d == nullptr) {
		return false;
	}

	// Get the extend and the data.
	p_cap_data_call->SetFrameID(static_cast<uint>(p_cr3d->Time()));
	if (!(*p_cap_data_call)(1)) return false;

	p_cap_data_call->SetFrameID(static_cast<uint>(p_cr3d->Time()));
	if (!(*p_cap_data_call)(0)) return false;

	// Check if the call contains data.
	if (p_cap_data_call->Count() <= 0) {
		return false;
	}
		
	if (p_cap_data_call->Objects()[0].GetVertexCount() <= 0) {
		return false;
	}
	const auto& mesh = p_cap_data_call->Objects()[0];

	// Copy the vertices that form the cap.
	auto vertex_cnt = mesh.GetVertexCount();
	std::vector<float> cap_vertices;
	cap_vertices.resize(vertex_cnt * 3);
	if (mesh.GetVertexDataType() == CallTriMeshData::Mesh::DT_FLOAT) {
		// float vertex data
		if (mesh.GetVertexPointerFloat() == nullptr) return false;
		std::copy(mesh.GetVertexPointerFloat(), mesh.GetVertexPointerFloat() + vertex_cnt * 3, cap_vertices.begin());

	} else if (mesh.GetVertexDataType() == CallTriMeshData::Mesh::DT_DOUBLE) {
		// double vertex data
		if (mesh.GetVertexPointerDouble() == nullptr) return false;
		std::transform(mesh.GetVertexPointerDouble(), mesh.GetVertexPointerDouble() + vertex_cnt * 3, cap_vertices.begin(), [](double v) {
			return static_cast<float>(v);
		});

	} else {
		return false;
	}

	// Get the colour of the cap. The cap always has to be the last binding site!
	uint last_idx = p_bs->GetBindingSiteCount() - 1;
	vec3f cap_colour = p_bs->GetBindingSiteColor(last_idx);

	// Loop over the faces of the mesh without the cap and look for vertices that
	// are not in the mesh with the cap. These vertices must be covered by the cap.
	size_t re_vertex_cnt = this->vertices_rebuild.size() / 3;
	size_t cap_vertex_cnt = cap_vertices.size() / 3;
	for (size_t i = 0; i < re_vertex_cnt; i++) {
		// Get the current position of the vertex.
		vec3f pos = vec3f(this->vertices_rebuild[i * 3 + 0],
			this->vertices_rebuild[i * 3 + 1],
			this->vertices_rebuild[i * 3 + 2]);
		bool found = false;

		// Loop over the mesh with the cap and get the vertex positions.
		for (size_t j = 0; j < cap_vertex_cnt; j++) {
			vec3f cap_pos = vec3f(cap_vertices[j * 3 + 0],
				cap_vertices[j * 3 + 1],
				cap_vertices[j * 3 + 2]);

			// Check if the two vertices are equal.
			if (pos == cap_pos) {
				found = true;
				break;
			}
		}

		// If no equal vertex could be found this vertex must be covered by the cap.
		if (!found) {
			this->vertexColors_rebuild[i * 3 + 0] = 0.75f * this->vertexColors_rebuild[i * 3 + 0] + 0.25f * cap_colour.GetX();
			this->vertexColors_rebuild[i * 3 + 1] = 0.75f * this->vertexColors_rebuild[i * 3 + 1] + 0.25f * cap_colour.GetY();
			this->vertexColors_rebuild[i * 3 + 2] = 0.75f * this->vertexColors_rebuild[i * 3 + 2] + 0.25f * cap_colour.GetZ();
		}
	}

	return true;
}


/*
 * MapGenerator::colourBindingSite
 */
bool MapGenerator::colourBindingSite(protein_calls::BindingSiteCall* p_bs, const vec3f& p_colour,
		protein_calls::MolecularDataCall* p_mdc, const float p_radius, float p_radiusOffset, bool p_ignoreRadius) {
	// Check if the BindingSiteCall is valid.
	if (p_bs == nullptr) {
		return false;
	}
	// no coloring concerning the radius is wanted
	if (p_ignoreRadius) {
		return false;
	}

	// The first three binding site are Serin, Histidin and Oxyanion. Get the residue indices
	// from the four aminoacids (one Serin, one Histidin and two Oxyanions).
	if (p_bs->GetBindingSiteCount() >= 3) {
		// Get the residue indices.
		auto serin = p_bs->GetBindingSite(0)->First();
		auto histidin = p_bs->GetBindingSite(1)->First();
		auto first_oxyanion = p_bs->GetBindingSite(2)->First();
		auto second_oxyanion = p_bs->GetBindingSite(2)->Last();

		// Get the residue indices for the four amino acids since the residue incides in the
		// BindingSiteCall are the original indices and not the current ones in the 
		// MolecularDataCall.
		float calpha_cnt = 0.0f;
		uint res_cnt = p_mdc->ResidueCount();
		uint serin_res_idx = res_cnt;
		uint histidin_res_idx = res_cnt;
		uint first_oxyanion_res_idx = res_cnt;
		uint second_oxyanion_res_idx = res_cnt;
		for (uint i = 0; i < res_cnt; i++) {
			// Get the new residue index for the Serin.
			if (p_mdc->Residues()[i]->OriginalResIndex() == serin.GetSecond()) {
				serin_res_idx = i;
			}

			// Get the new residue index for the Histidin.
			if (p_mdc->Residues()[i]->OriginalResIndex() == histidin.GetSecond()) {
				histidin_res_idx = i;
			}

			// Get the new residue index for the first Oxyanion.
			if (p_mdc->Residues()[i]->OriginalResIndex() == first_oxyanion.GetSecond()) {
				first_oxyanion_res_idx = i;
			}

			// Get the new residue index for the second Oxyanion.
			if (p_mdc->Residues()[i]->OriginalResIndex() == second_oxyanion.GetSecond()) {
				second_oxyanion_res_idx = i;
			}

			// Check if all indices have benn found.
			if (serin_res_idx != res_cnt && 
					histidin_res_idx != res_cnt &&
					first_oxyanion_res_idx != res_cnt &&
					second_oxyanion_res_idx != res_cnt) {
				break;
			}
		}

		// Check if no index has been found.
		if (serin_res_idx == res_cnt &&
				histidin_res_idx == res_cnt &&
				first_oxyanion_res_idx == res_cnt &&
				second_oxyanion_res_idx == res_cnt) {
			return false;
		}

		// Get the c-alpha atom index from the MolecularDataCall for the Serin.
		vec3f serin_pos;
		if (serin_res_idx != res_cnt) {
			auto serin_res = p_mdc->Residues()[serin_res_idx];
			if (serin_res->Identifier() == MolecularDataCall::Residue::AMINOACID) {
				uint serin_idx = dynamic_cast<const MolecularDataCall::AminoAcid*>(serin_res)->CAlphaIndex();
				serin_pos.Set(p_mdc->AtomPositions()[serin_idx * 3 + 0],
					p_mdc->AtomPositions()[serin_idx * 3 + 1],
					p_mdc->AtomPositions()[serin_idx * 3 + 2]);

			} else {
				return false;
			}
			calpha_cnt++;

		} else {
			return false;
		}

		// Get the c-alpha atom index from the MolecularDataCall for the Histidin.
		vec3f histidin_pos;
		if (histidin_res_idx != res_cnt) {
			auto histidin_res = p_mdc->Residues()[histidin_res_idx];
			if (histidin_res->Identifier() == MolecularDataCall::Residue::AMINOACID) {
				uint histidin_idx = dynamic_cast<const MolecularDataCall::AminoAcid*>(histidin_res)->CAlphaIndex();
				histidin_pos.Set(p_mdc->AtomPositions()[histidin_idx * 3 + 0],
					p_mdc->AtomPositions()[histidin_idx * 3 + 1],
					p_mdc->AtomPositions()[histidin_idx * 3 + 2]);

			} else {
				return false;
			}
			calpha_cnt++;

		} else {
			return false;
		}

		// We need just one of the Oxyanions.
		size_t oxy_cnt = 0;

		// Get the c-alpha atom index from the MolecularDataCall for the first Oxyanion.
		vec3f first_oxyanion_pos;
		if (first_oxyanion_res_idx != res_cnt) {
			auto first_oxyanion_res = p_mdc->Residues()[first_oxyanion_res_idx];
			if (first_oxyanion_res->Identifier() == MolecularDataCall::Residue::AMINOACID) {
				uint first_oxyanion_idx = dynamic_cast<const MolecularDataCall::AminoAcid*>(first_oxyanion_res)->CAlphaIndex();
				first_oxyanion_pos.Set(p_mdc->AtomPositions()[first_oxyanion_idx * 3 + 0],
					p_mdc->AtomPositions()[first_oxyanion_idx * 3 + 1],
					p_mdc->AtomPositions()[first_oxyanion_idx * 3 + 2]);

			} else {
				return false;
			}
			oxy_cnt++;
			calpha_cnt++;
		}

		// Get the c-alpha atom index from the MolecularDataCall for the second Oxyanion.
		vec3f second_oxyanion_pos;
		if (second_oxyanion_res_idx != res_cnt) {
			auto second_oxyanion_res = p_mdc->Residues()[second_oxyanion_res_idx];
			if (second_oxyanion_res->Identifier() == MolecularDataCall::Residue::AMINOACID) {
				uint second_oxyanion_idx = dynamic_cast<const MolecularDataCall::AminoAcid*>(second_oxyanion_res)->CAlphaIndex();
				second_oxyanion_pos.Set(p_mdc->AtomPositions()[second_oxyanion_idx * 3 + 0],
					p_mdc->AtomPositions()[second_oxyanion_idx * 3 + 1],
					p_mdc->AtomPositions()[second_oxyanion_idx * 3 + 2]);

			} else {
				return false;
			}
			oxy_cnt++;
			calpha_cnt++;
		}

		// If there is no Oxyanion return.
		if (oxy_cnt == 0) {
			return false;
		}

		// Compute the center of the sphere.
		vec3f center = (serin_pos + histidin_pos + first_oxyanion_pos + second_oxyanion_pos) / calpha_cnt;
		vec4d sphere = vec4d(static_cast<double>(center.GetX()),
			static_cast<double>(center.GetY()),
			static_cast<double>(center.GetZ()),
			static_cast<double>(p_radius + p_radiusOffset));
		
		// Perform a radius search with the sphere.
		std::vector<uint> resultFaces;
		this->octree.RadiusSearch(this->faces_rebuild, sphere, this->vertices_rebuild, resultFaces);

		// Colour all vertices in the given color.
		for (auto fid : resultFaces) {
			uint vid = this->faces_rebuild[fid * 3 + 0];
			this->vertexColors_rebuild[vid * 3 + 0] = p_colour.GetX();
			this->vertexColors_rebuild[vid * 3 + 1] = p_colour.GetY();
			this->vertexColors_rebuild[vid * 3 + 2] = p_colour.GetZ();
			vid = this->faces_rebuild[fid * 3 + 1];
			this->vertexColors_rebuild[vid * 3 + 0] = p_colour.GetX();
			this->vertexColors_rebuild[vid * 3 + 1] = p_colour.GetY();
			this->vertexColors_rebuild[vid * 3 + 2] = p_colour.GetZ();
			vid = this->faces_rebuild[fid * 3 + 2];
			this->vertexColors_rebuild[vid * 3 + 0] = p_colour.GetX();
			this->vertexColors_rebuild[vid * 3 + 1] = p_colour.GetY();
			this->vertexColors_rebuild[vid * 3 + 2] = p_colour.GetZ();
		}

	} else { 
		return false;
	}

	return true;
}


/*
 * MapGenerator::computeBoundingBox
 */
vislib::math::Cuboid<float> MapGenerator::computeBoundingBox(std::vector<float>& verts) {
    vislib::math::Cuboid<float> result;
    if (verts.size() > 2) {
        result.Set(verts[0], verts[1], verts[2], verts[0], verts[1], verts[2]);
    }
    for (size_t i = 1; i < verts.size() / 3; i++) {
        result.GrowToPoint(vislib::math::Point<float, 3>(verts[3 * i + 0], verts[3 * i + 1], verts[3 * i + 2]));
    }
    return result;
}


/*
 * MapGenerator::computeGeodesicPoint
 */
vec3f MapGenerator::computeGeodesicPoint(const float p_a, const float p_b, const float p_c, 
		const float p_lat_0, const float p_lon_0, const float p_lat_1, const float p_lon_1) {
	// Compute the point between the start point (p_lat_0, p_lon_0) and the end point
	// (p_lat_1, p_lon_1) based on the interpolation parameters a and b. The factor p_c
	// raises the line above the surface of the sphere.
	vec3f retval;
	retval.SetX(this->sphere_data.GetX() + (p_c + this->sphere_data.GetW()) *
		(p_a * cos(p_lat_0) * cos(p_lon_0) + p_b * cos(p_lat_1) * cos(p_lon_1)));
	retval.SetY(this->sphere_data.GetY() + (p_c + this->sphere_data.GetW()) *
		(p_a * cos(p_lat_0) * sin(p_lon_0) + p_b * cos(p_lat_1) * sin(p_lon_1)));
	retval.SetZ(this->sphere_data.GetZ() + (p_c + this->sphere_data.GetW()) *
		(p_a * sin(p_lat_0) + p_b * sin(p_lat_1)));

	return retval;
}


/*
 * MapGenerator::computeRotationMatrix
 */
mat4f MapGenerator::computeRotationMatrix(const vec3f& p_normal) {
	// Initialise the computation.
	mat4f rot;
	vec3f around, z_dir;
	float angle, c, s, t;

	// Get the vector around which the rotation is performen and the angle.
	z_dir.Set(0.0f, 0.0f, 1.0f);
	around = z_dir.Cross(p_normal);
	angle = acos(z_dir.Dot(p_normal));

	// Precompute certain values.
	c = cos(angle);
	s = sin(angle);
	t = 1.0f - cos(angle);

	// Set the matrix and return it.
	rot.SetAt(0, 0, t * (around.GetX() * around.GetX()) + c);
	rot.SetAt(0, 1, t * (around.GetX() * around.GetY()) - s * around.GetZ());
	rot.SetAt(0, 2, t * (around.GetX() * around.GetZ()) + s * around.GetY());
	rot.SetAt(0, 3, 0.0f);

	rot.SetAt(1, 0, t * (around.GetX() * around.GetY()) + s * around.GetZ());
	rot.SetAt(1, 1, t * (around.GetY() * around.GetY()) + c);
	rot.SetAt(1, 2, t * (around.GetY() * around.GetZ()) - s * around.GetX());
	rot.SetAt(1, 3, 0.0f);

	rot.SetAt(2, 0, t * (around.GetX() * around.GetZ()) - s * around.GetY());
	rot.SetAt(2, 1, t * (around.GetY() * around.GetZ()) + s * around.GetX());
	rot.SetAt(2, 2, t * (around.GetZ() * around.GetZ()) + c);
	rot.SetAt(2, 3, 0.0f);

	rot.SetAt(3, 0, 0.0f);
	rot.SetAt(3, 1, 0.0f);
	rot.SetAt(3, 2, 0.0f);
	rot.SetAt(3, 3, 1.0f);

	return rot;
}


/*
 * MapGenerator::convertToLatLon
 */
void MapGenerator::convertToLatLon(const vec3f& p_point, float& p_lat, float& p_lon) {
	p_lat = asinf(p_point.GetZ() /
		sqrtf(p_point.GetX() * p_point.GetX() + p_point.GetY() * p_point.GetY() +
			p_point.GetZ() * p_point.GetZ()));
	p_lon = atan2f(p_point.GetY(), p_point.GetX());
}


/*
 * MapGenerator::createBoundingSphere
 */
bool MapGenerator::createBoundingSphere(const float p_offset, float& p_radius, 
		vislib::math::Vector<float, 3>& p_center, std::vector<float>& p_vector) {
	// Set initial values.
	p_center.Set(p_vector[0], p_vector[1], p_vector[2]);
	p_radius = vislib::math::FLOAT_EPSILON;

	// Perform algorithm from:
	// http://stackoverflow.com/questions/17331203/bouncing-bubble-algorithm-for-smallest-enclosing-sphere
	vislib::math::Vector<float, 3> pos, diff;
	float len, alpha, alphaSq;
	uint idx;

	uint vertex_cnt = static_cast<uint>(p_vector.size() / 3);
	for (uint i = 0; i < 2; i++) {
		for (uint j = 0; j < vertex_cnt; j++) {
			idx = j * 3;
			pos.Set(p_vector[idx], p_vector[idx + 1],
				p_vector[idx + 2]);
			diff = pos - p_center;
			len = diff.Length();
			if (len > p_radius) {
				alpha = len / p_radius;
				alphaSq = alpha * alpha;
				p_radius = 0.5f * (alpha + 1.0f / alpha) * p_radius;
				p_center = 0.5f * ((1.0f + 1.0f / alphaSq) * p_center +
					(1.0f - 1.0f / alphaSq) * pos);
			}
		}
	}

	for (unsigned int j = 0; j < vertex_cnt; j++) {
		idx = j * 3;
		pos.Set(p_vector[idx], p_vector[idx + 1],
			p_vector[idx + 2]);
		diff = pos - p_center;
		len = diff.Length();
		if (len > p_radius) {
			p_radius = (p_radius + len) / 2.0f;
			p_center = p_center + ((len - p_radius) / len * diff);
		}
	}

	p_radius += p_offset;

	return true;
}


/*
 * MapGenerator::createCut
 */
Cut MapGenerator::createCut(const bool p_second_rebuild, const uint p_tunnel_id,
		uint& p_vertex_id, const std::vector<uint>& p_vertex_ids, const std::vector<bool>& p_tunnels) {
	Cut new_cut;
	vec3f center_normal, center_point, vertex;
	float* source_vertices_ptr;
	float* source_normals_ptr;
	std::vector<vec3f> face_vertices;
	uint vertex_id;

	// Determine the source for the vertex positions.
	if (!p_second_rebuild) {
		source_normals_ptr = this->normals.data();
		source_vertices_ptr = this->vertices.data();

	} else {
		source_normals_ptr = this->normals_rebuild.data();
		source_vertices_ptr = this->vertices_rebuild.data();
	}

	// Create the cut.
	new_cut.tunnel_id = p_tunnel_id;

	// Determine the center point of the triangle fan.
	uint colour_table_size = static_cast<uint>(this->cut_colour_table.Count());
	for (const auto id : p_vertex_ids) {
		vertex.Set(source_vertices_ptr[id * 3], source_vertices_ptr[id * 3 + 1], source_vertices_ptr[id * 3 + 2]);
		center_point += vertex;
		// Add the vertex position to the cut.
		new_cut.vertices.push_back(vertex.GetX());
		new_cut.vertices.push_back(vertex.GetY());
		new_cut.vertices.push_back(vertex.GetZ());
		// Add the normal to the cut.
		new_cut.normals.push_back(source_normals_ptr[id * 3]);
		new_cut.normals.push_back(source_normals_ptr[id * 3 + 1]);
		new_cut.normals.push_back(source_normals_ptr[id * 3 + 2]);
		// Add the colour to the cut.
		new_cut.colours.push_back(this->cut_colour_table[p_tunnel_id % colour_table_size].GetX());
		new_cut.colours.push_back(this->cut_colour_table[p_tunnel_id % colour_table_size].GetY());
		new_cut.colours.push_back(this->cut_colour_table[p_tunnel_id % colour_table_size].GetZ());
		// Add the colour to the cut colouring.
		if (!p_second_rebuild) {
			this->vertexColors_cuts[id * 3] = this->cut_colour_table[p_tunnel_id % colour_table_size].GetX();
			this->vertexColors_cuts[id * 3 + 1] = this->cut_colour_table[p_tunnel_id % colour_table_size].GetY();
			this->vertexColors_cuts[id * 3 + 2] = this->cut_colour_table[p_tunnel_id % colour_table_size].GetZ();
		}
	}
	center_point /= static_cast<float>(p_vertex_ids.size());
	// Add the center point position to the cut.
	new_cut.vertices.push_back(center_point.GetX());
	new_cut.vertices.push_back(center_point.GetY());
	new_cut.vertices.push_back(center_point.GetZ());
	// Add the colour.
	new_cut.colours.push_back(this->cut_colour_table[p_tunnel_id % colour_table_size].GetX());
	new_cut.colours.push_back(this->cut_colour_table[p_tunnel_id % colour_table_size].GetY());
	new_cut.colours.push_back(this->cut_colour_table[p_tunnel_id % colour_table_size].GetZ());

	// Add new faces and comput the sum of the face normals.
	center_normal.Set(0.0f, 0.0f, 0.0f);
	face_vertices = std::vector<vec3f>(2);
	for (size_t i = 0; i < p_vertex_ids.size() - 1; i++) {
		// Check if the edge is already there and if so invert this edge.
		bool invert = this->invertEdge(p_vertex_ids[i + 1], p_vertex_ids[i], p_tunnels);

		// Check if the first edge has to be inverted.
		if (invert) {
			// Second vertex.
			vertex_id = p_vertex_ids[i];
			new_cut.faces.push_back(vertex_id);
			face_vertices[1].Set(source_vertices_ptr[vertex_id * 3], source_vertices_ptr[vertex_id * 3 + 1],
				source_vertices_ptr[vertex_id * 3 + 2]);

			// First vertex.
			vertex_id = p_vertex_ids[i + 1];
			new_cut.faces.push_back(vertex_id);
			face_vertices[0].Set(source_vertices_ptr[vertex_id * 3], source_vertices_ptr[vertex_id * 3 + 1],
				source_vertices_ptr[vertex_id * 3 + 2]);

		} else {
			// First vertex.
			vertex_id = p_vertex_ids[i + 1];
			new_cut.faces.push_back(vertex_id);
			face_vertices[0].Set(source_vertices_ptr[vertex_id * 3], source_vertices_ptr[vertex_id * 3 + 1],
				source_vertices_ptr[vertex_id * 3 + 2]);

			// Second vertex.
			vertex_id = p_vertex_ids[i];
			new_cut.faces.push_back(vertex_id);
			face_vertices[1].Set(source_vertices_ptr[vertex_id * 3], source_vertices_ptr[vertex_id * 3 + 1],
				source_vertices_ptr[vertex_id * 3 + 2]);
		}

		// Third vertex.
		vertex_id = p_vertex_id;
		new_cut.faces.push_back(vertex_id);

		// Compute face normal.
		auto tmp_normal = (face_vertices[1] - face_vertices[0]).Cross(center_point - face_vertices[0]);
		tmp_normal.Normalise();
		center_normal += tmp_normal;
	}

	// Add last face to close the fan.
	// Check if the edge is already there and if so invert this edge.
	bool invert = this->invertEdge(p_vertex_ids[0], p_vertex_ids.back(), p_tunnels);

	// Check if the first edge has to be inverted.
	if (invert) {
		// Second vertex.
		vertex_id = p_vertex_ids.back();
		new_cut.faces.push_back(vertex_id);
		face_vertices[1].Set(source_vertices_ptr[vertex_id * 3], source_vertices_ptr[vertex_id * 3 + 1],
			source_vertices_ptr[vertex_id * 3 + 2]);

		// First vertex.
		vertex_id = p_vertex_ids[0];
		new_cut.faces.push_back(vertex_id);
		face_vertices[0].Set(source_vertices_ptr[vertex_id * 3], source_vertices_ptr[vertex_id * 3 + 1],
			source_vertices_ptr[vertex_id * 3 + 2]);
		
	} else {
		// First vertex.
		vertex_id = p_vertex_ids[0];
		new_cut.faces.push_back(vertex_id);
		face_vertices[0].Set(source_vertices_ptr[vertex_id * 3], source_vertices_ptr[vertex_id * 3 + 1],
			source_vertices_ptr[vertex_id * 3 + 2]);

		// Second vertex.
		vertex_id = p_vertex_ids.back();
		new_cut.faces.push_back(vertex_id);
		face_vertices[1].Set(source_vertices_ptr[vertex_id * 3], source_vertices_ptr[vertex_id * 3 + 1],
			source_vertices_ptr[vertex_id * 3 + 2]);
	}

	// Third vertex.
	vertex_id = p_vertex_id;
	new_cut.faces.push_back(vertex_id);

	// Compute face normal.
	auto tmp_normal = (face_vertices[1] - face_vertices[0]).Cross(center_point - face_vertices[0]);
	tmp_normal.Normalise();
	center_normal += tmp_normal;
	
	// Compute normal of the center point.
	center_normal /= static_cast<float>(new_cut.faces.size() / 3);
	new_cut.normals.push_back(center_normal.GetX());
	new_cut.normals.push_back(center_normal.GetY());
	new_cut.normals.push_back(center_normal.GetZ());

	// Increase the new vertex ID by one because we added one new vertex.
	p_vertex_id++;

	return new_cut;
}


/*
 * MapGenerator::createGeodesicLines
 */
void MapGenerator::createGeodesicLines(const GeodesicMode p_mode) {
	// Delete old geodesic lines.
	this->geodesic_lines_colours.clear();
	this->geodesic_lines_colours.shrink_to_fit();
	this->geodesic_lines.clear();
	this->geodesic_lines.shrink_to_fit();

	// Return if the mode is equal to NO_LINES
	if (p_mode == GeodesicMode::NO_LINES) {
		return;
	}

	// Set the step for the line interpolation.
	float step = 0.0001f;
	if (p_mode == GeodesicMode::ONE_TO_ALL) {
		this->geodesic_lines_colours.reserve(this->vertices_added.size());
		this->geodesic_lines.reserve(this->vertices_added.size());

	} else {
		this->geodesic_lines_colours.reserve(this->vertices_added.size() * this->vertices_added.size());
		this->geodesic_lines.reserve(this->vertices_added.size() * this->vertices_added.size());
	}

	// Determine the amount of colours we have for tunnels.
	size_t colour_table_size = this->cut_colour_table.Count();

	// Find the vertices that belong together.
	std::vector<size_t> tunnel_offset = std::vector<size_t>(this->vertices_added_tunnel_id.back() + 2,
		this->vertices_added.size());
	uint curr_tunnel_id = this->vertices_added_tunnel_id.front();
	tunnel_offset[curr_tunnel_id] = 0;
	for (size_t i = 1; i < this->vertices_added.size(); i++) {
		if (this->vertices_added_tunnel_id[i] != curr_tunnel_id) {
			curr_tunnel_id = this->vertices_added_tunnel_id[i];
			tunnel_offset[curr_tunnel_id] = i;
		}
	}

	// Create the lines.
	std::vector<float> line;
	std::vector<float> line_colour;
	float a, b, c, d, lat_0, lon_0, lat_1, lon_1, lat_2, lon_2;
	uint start_id, end_id;
	size_t start, end;
	vec3f n_0, n_1, p_0, p_1, p_2;
	for (size_t i = 0; i < tunnel_offset.size() - 1; i++) {
		// Get the first and the last vertex that belongs to the tunnel.
		start = tunnel_offset[i];
		end = tunnel_offset[i + 1];
		for (size_t j = start; j < end; j++) {
			// Get the current ID and use it to create the start point on 
			// the sphere and the normal of that point.
			start_id = this->vertices_added[j];
			p_0.Set(this->vertices_sphere[start_id * 3] - this->sphere_data.GetX(),
				this->vertices_sphere[start_id * 3 + 1] - this->sphere_data.GetY(),
				this->vertices_sphere[start_id * 3 + 2] - this->sphere_data.GetZ());
			n_0 = p_0;
			n_0.Normalise();

			// Convert the points coordinates into latitude and longitude
			// coordiantes.
			this->convertToLatLon(p_0, lat_0, lon_0);

			// Compute the line to the next cut that belongs to the same tunnel.
			for (size_t k = j + 1; k < end; k++) {
				// Delete the last line.
				line.clear();
				line.shrink_to_fit();
				line_colour.clear();
				line_colour.shrink_to_fit();

				// Get the current ID and use it to create the end point
				// on the sphere as well as the normal.
				end_id = this->vertices_added[k];
				p_1.Set(this->vertices_sphere[end_id * 3] - this->sphere_data.GetX(),
					this->vertices_sphere[end_id * 3 + 1] - this->sphere_data.GetY(),
					this->vertices_sphere[end_id * 3 + 2] - this->sphere_data.GetZ());
				n_1 = p_1;
				n_1.Normalise();

				// Convert the points coordinates into latitude and longitude
				// coordiantes.
				this->convertToLatLon(p_1, lat_1, lon_1);

				// Get the angle between the start and the end point.
				d = std::acosf(n_0.Dot(n_1));
				if (d > 180.0f) {
					d = 360.0f - d;
				}

				// Check if the angle is 180� and if so split the line into two parts.
				if (d > (180.0f - 1e-5) && d < (180.0f + 1e-5)) {
					// Reserve space for the vertices and colours of the line.
					line.reserve(static_cast<size_t>(1.0f / (2.0f * step)) * 6);
					line_colour.reserve(static_cast<size_t>(1.0f / (2.0f * step)) * 6);

					// Set the middle point on the line.
					a = 0.5f;
					b = 0.5f;
					c = 0.5f;
					d = d / 2.0f;
					p_2 = this->computeGeodesicPoint(a, b, c, lat_0, lon_0, lat_1, lon_1);

					// Convert the points coordinates into latitude and longitude
					// coordiantes.
					this->convertToLatLon(p_2, lat_2, lon_2);

					// Interpolate the line between the start point and the middle point.
					for (float t = 0.0f; t <= 1.0f; t += 2.0f * step) {
						// Get the next point.
						a = std::sin((1.0f - t) * d) / sin(d);
						b = std::sin(t * d) / sin(d);
						c = std::sin((t / 2.0f) * static_cast<float>(M_PI));
						p_2 = this->computeGeodesicPoint(a, b, c, lat_0, lon_0, lat_2, lon_2);

						// Add it to the line as well as its colour.
						line.push_back(p_2.GetX());
						line.push_back(p_2.GetY());
						line.push_back(p_2.GetZ());

						line_colour.push_back(this->cut_colour_table[i % colour_table_size].GetX());
						line_colour.push_back(this->cut_colour_table[i % colour_table_size].GetY());
						line_colour.push_back(this->cut_colour_table[i % colour_table_size].GetZ());
					}

					// Interpolate the line between the middle point and the end point.
					for (float t = 0.0f; t <= 1.0f; t += 2.0f * step) {
						// Get the next point.
						a = std::sin((1.0f - t) * d) / sin(d);
						b = std::sin(t * d) / sin(d);
						c = std::sin((t + 1.0f / 2.0f) * static_cast<float>(M_PI));
						p_2 = this->computeGeodesicPoint(a, b, c, lat_2, lon_2, lat_1, lon_1);

						// Add it to the line as well as its colour.
						line.push_back(p_2.GetX());
						line.push_back(p_2.GetY());
						line.push_back(p_2.GetZ());

						line_colour.push_back(this->cut_colour_table[i % colour_table_size].GetX());
						line_colour.push_back(this->cut_colour_table[i % colour_table_size].GetY());
						line_colour.push_back(this->cut_colour_table[i % colour_table_size].GetZ());
					}

				} else {
					// Reserve space for the vertices and colours of the line.
					line.reserve(static_cast<size_t>(1.0f / step) * 3);
					line_colour.reserve(static_cast<size_t>(1.0f / step) * 3);

					// Interpolate the line between the start point and the end point.
					for (float t = 0.0f; t <= 1.0f; t += step)
					{
						a = sin((1.0f - t) * d) / sin(d);
						b = sin(t * d) / sin(d);
						c = sin(t * static_cast<float>(M_PI));
						p_2 = this->computeGeodesicPoint(a, b, c, lat_0, lon_0, lat_1, lon_1);

						// Add it to the line as well as its colour.
						line.push_back(p_2.GetX());
						line.push_back(p_2.GetY());
						line.push_back(p_2.GetZ());

						line_colour.push_back(this->cut_colour_table[i % colour_table_size].GetX());
						line_colour.push_back(this->cut_colour_table[i % colour_table_size].GetY());
						line_colour.push_back(this->cut_colour_table[i % colour_table_size].GetZ());
					}
				}

				// Add the new line to the list of geodesic lines.
				this->geodesic_lines_colours.push_back(line_colour);
				this->geodesic_lines.push_back(line);
			}

			// If we only want to draw line from one entrance to all other entrances of the
			// same tunnel we stop this loop here.
			if (p_mode == GeodesicMode::ONE_TO_ALL) {
				break;
			} /* Inner loop over all cuts belonging to the same tunnel. */
		} /* Outer loop over all cuts belonging to the same tunnel. */
	} /* Loop over all tunnels. */
}


/*
 * MapGenerator::createSphere
 */
bool MapGenerator::createSphere(const vec3f& p_eye_dir, const vec3f& p_up_dir) {
	size_t vertex_cnt = this->vertices_rebuild.size() / 3;

	// Create the bounding sphere.
	float radius;
	vislib::math::Vector<float, 3> center;
	auto ret = createBoundingSphere(this->radius_offset_param.Param<param::FloatParam>()->Value(), 
		radius, center, this->vertices_rebuild);
	if (!ret) return false;

	// Determine the poles of the protein.
	Poles poles;
	ret = findPoles(p_eye_dir, p_up_dir, center, poles);
	if (!ret) return false;

	// Initialise the z values of the poles and their neighbouring vertices. 
	float z_val = 50.0f;
	std::vector<float> z_values = std::vector<float>(vertex_cnt, 0.0f);
	std::vector<bool> valid_vertices = std::vector<bool>(vertex_cnt, true);
	ret = initialiseZvalues(poles, z_values, valid_vertices, z_val);
	if (!ret) return false;

	// Set the zvalues of the vertices.
	ret = this->cuda_kernels->CreateZValues(20000, z_values, valid_vertices, this->vertex_edge_offset,
		this->vertex_edge_offset_depth);
	if (!ret) return false;

	// Find boundary meridian.
	std::vector<int> types = std::vector<int>(vertex_cnt, 0);
	valid_vertices = std::vector<bool>(vertex_cnt, true);
	ret = findBoundaryMeridian(poles, types, valid_vertices, z_values, p_eye_dir);
	if (!ret) return false;

	// Set the phi values of the vertices.
	std::vector<float> phi_values = std::vector<float>(vertex_cnt, 0.0f);
	ret = this->cuda_kernels->CreatePhiValues(0.01f, phi_values, valid_vertices, this->vertex_edge_offset, 
		this->vertex_edge_offset_depth, types);
	if (!ret) return false;

	// Delete the types.
	types.clear();
	types.shrink_to_fit();

	// Set the d values of the vertices.
	float theta_const;
	ret = setDvalues(vertex_cnt, theta_const, poles);
	if (!ret) return false;

	// Set the theta values of the vertices.
	std::vector<float> theta_values = std::vector<float>(vertex_cnt, 0.0f);
	valid_vertices = std::vector<bool>(vertex_cnt, true);
	theta_values[poles.north] = 0.0f;
	theta_values[poles.south] = static_cast<float>(vislib::math::PI_DOUBLE);
	valid_vertices[poles.north] = false;
	valid_vertices[poles.south] = false;
	ret = this->cuda_kernels->CreateThetaValues(theta_const, theta_values, valid_vertices, z_values, z_val);
	if (!ret) return false;

	// Delete the z values, the valid vertices.
	z_values.clear();
	z_values.shrink_to_fit();
	valid_vertices.clear();
	valid_vertices.shrink_to_fit();

	// Determine the rotation of the protein.
	vec3f pre_rot_up = vec3f(0.0f, 1.0f, 0.0f);
	vec3f rotation_axis = pre_rot_up.Cross(p_up_dir);
	rotation_axis.Normalise();
	float rotation_angle = -std::acosf(pre_rot_up.Dot(p_up_dir));
	this->rotation_quat.Set(rotation_angle, rotation_axis);

	// Determine the rotation of the sphere, since the sphere is rotated
	// compared to the original position of the protein.
	vislib::math::Quaternion<float> sphere_rotation;
	vec3f look_at_dir_pre_rot, look_at_dir_post_rot, look_at_point_sphere;

	// Compute the spherical coordinates for the look at point.
	look_at_point_sphere.SetX(center.GetX() + radius * sin(theta_values[this->look_at_id]) * cos(phi_values[this->look_at_id]));
	look_at_point_sphere.SetY(center.GetY() + radius * cos(theta_values[this->look_at_id]));
	look_at_point_sphere.SetZ(center.GetZ() + radius * sin(theta_values[this->look_at_id]) * sin(phi_values[this->look_at_id]));

	// Compute the direction of the vector from the sphere center to the
	// look at point on the protein.
	look_at_dir_pre_rot.Set(this->vertices_rebuild[this->look_at_id * 3],
		this->vertices_rebuild[this->look_at_id * 3 + 1],
		this->vertices_rebuild[this->look_at_id * 3 + 2]);
	look_at_dir_pre_rot -= center;
	look_at_dir_pre_rot.Normalise();

	// Compute the direction of the vector from the sphere center to the
	// look at point on the sphere.
	look_at_dir_post_rot = look_at_point_sphere - center;
	look_at_dir_post_rot.Normalise();

	// The rotation is always around the new up direction we just need
	// the angle. Set the rotation and multiply it with the rotation of
	// the protein to get the correct rotation for the latitude and
	// longitude lines.
	rotation_angle = -std::acosf(look_at_dir_pre_rot.Dot(look_at_dir_post_rot));
	sphere_rotation.Set(rotation_angle, p_up_dir);
	this->rotation_quat = this->rotation_quat * sphere_rotation;

	// Set the vertices of the sphere.
	this->vertices_sphere = std::vector<float>(this->vertices_rebuild.size());
	this->vertices_sphere.shrink_to_fit();
	for (size_t i = 0; i < vertex_cnt; i++) {
		// Note:
		// We do not rotate the sphere since a) that breaks everything and b)
		// nobody looks at the sphere anyway.
		this->vertices_sphere[i * 3 + 0] = center.GetX() + radius * sin(theta_values[i]) * cos(phi_values[i]);
		this->vertices_sphere[i * 3 + 1] = center.GetY() + radius * cos(theta_values[i]);
		this->vertices_sphere[i * 3 + 2] = center.GetZ() + radius * sin(theta_values[i]) * sin(phi_values[i]);
	}
	this->sphere_data.Set(center.GetX(), center.GetY(), center.GetZ(), radius);

	// Create the geodesic lines between the tunnel cuts.
	if (this->vertices_added.size() > 0) {
		this->createGeodesicLines(
			static_cast<GeodesicMode>(this->geodesic_lines_param.Param<param::EnumParam>()->Value()));
	}

	return true;
}


/*
 * MapGenerator::depthFirstSearch
 */
void MapGenerator::depthFirstSearch(const size_t p_cur, const std::vector<VoronoiEdge>& p_edges, 
		const std::vector<VoronoiEdge>& p_reversed_edges, const std::vector<size_t>& p_start_offset, 
		const std::vector<size_t>& p_end_offset, std::vector<bool>& p_visited, 
		std::vector<uint>& p_group) {
	// Mark the vertex as visited.
	p_visited[p_cur] = true;

	// Add the current vertex to the group.
	p_group.push_back(static_cast<uint>(p_cur));

	// Recur for all edges that start from the current vertex.
	size_t begin = p_start_offset[p_cur];
	size_t end = p_start_offset[p_cur + 1];
	for (size_t i = begin; i < end; i++) {
		if (!p_visited[p_edges[i].end_vertex]) {
			this->depthFirstSearch(p_edges[i].end_vertex, p_edges, p_reversed_edges,
				p_start_offset, p_end_offset, p_visited, p_group);
		}
	}

	// Recur for all edges that end at the current vertex.
	begin = p_end_offset[p_cur];
	end = p_end_offset[p_cur + 1];
	for (size_t i = begin; i < end; i++) {
		if (!p_visited[p_reversed_edges[i].start_vertex]) {
			this->depthFirstSearch(p_reversed_edges[i].start_vertex, p_edges, p_reversed_edges,
				p_start_offset, p_end_offset, p_visited, p_group);
		}
	}
}


/*
 * MapGenerator::drawMap
 */
void MapGenerator::drawMap() {
	// Enable FBO
	this->map_fbo.Enable();
	GLfloat bk_colour[4];
	glGetFloatv(GL_COLOR_CLEAR_VALUE, bk_colour);
	glClearColor(0.0, 0.0, 0.0, 0.0);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glClearColor(bk_colour[0], bk_colour[1], bk_colour[2], bk_colour[3]);
	glPushMatrix();
	glLoadIdentity();

	// Disable blending, culling and lighting
	glDisable(GL_BLEND);
	glLineWidth(1.0f);
	glDisable(GL_CULL_FACE);
	glDisable(GL_LIGHTING);
	glDisable(GL_DEPTH_TEST);

    // Enable wireframe mode if needed
    GLint oldpolymode[2];
    glGetIntegerv(GL_POLYGON_MODE, oldpolymode);
    if (this->draw_wireframe_param.Param<param::BoolParam>()->Value()) {
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
    }

	// Set data pointer
	glEnableClientState(GL_VERTEX_ARRAY);
	glEnableClientState(GL_COLOR_ARRAY);
	glBindBuffer(GL_ARRAY_BUFFER, this->map_vertex_vbo);
	glVertexPointer(3, GL_FLOAT, 0, 0);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glColorPointer(3, GL_FLOAT, 0, this->vertexColors_rebuild.data());

	// Render
	this->map_shader.Enable();
	this->map_shader.SetParameter("sphere", this->sphere_data.GetX(), this->sphere_data.GetY(), this->sphere_data.GetZ(), 
		this->sphere_data.GetW());
	this->map_shader.SetParameter("frontVertex", this->vertices_sphere[this->look_at_id * 3], 
		this->vertices_sphere[this->look_at_id * 3 + 1], this->vertices_sphere[this->look_at_id * 3 + 2]);
	this->map_shader.SetParameter("mirrorMap", this->mirror_map_param.Param<param::BoolParam>()->Value());
	glDrawElements(GL_TRIANGLES, static_cast<GLsizei>(this->faces_rebuild.size()), GL_UNSIGNED_INT, this->faces_rebuild.data());
	this->map_shader.Disable();

	// Draw FBO
	glPopMatrix();
	this->map_fbo.Disable();

    glPolygonMode(GL_FRONT, oldpolymode[0]);
    glPolygonMode(GL_BACK, oldpolymode[1]);

	this->map_fbo.DrawColourTexture(0, GL_LINEAR, GL_LINEAR, 0.9f);

	// Render geodesic lines.
	glEnable(GL_LINE_SMOOTH);
	for (size_t i = 0; i < this->geodesic_lines.size(); i++) {
		glBindBuffer(GL_ARRAY_BUFFER, this->geodesic_lines_vbos[i]);
		glVertexPointer(3, GL_FLOAT, 0, 0);
		glBindBuffer(GL_ARRAY_BUFFER, 0);
		glColorPointer(3, GL_FLOAT, 0, this->geodesic_lines_colours[i].data());
		this->geodesic_shader.Enable();
		this->geodesic_shader.SetParameter("sphere", this->sphere_data.GetX(), this->sphere_data.GetY(),
			this->sphere_data.GetZ(), this->sphere_data.GetW());
		this->geodesic_shader.SetParameter("frontVertex", this->vertices_sphere[this->look_at_id * 3],
			this->vertices_sphere[this->look_at_id * 3 + 1], this->vertices_sphere[this->look_at_id * 3 + 2]);
		glDrawArrays(GL_LINE_STRIP, 0, static_cast<GLsizei>(this->geodesic_lines[i].size() / 3));
		this->geodesic_shader.Disable();
	}
	glDisable(GL_LINE_SMOOTH);

	glDisableClientState(GL_COLOR_ARRAY);
	glDisableClientState(GL_VERTEX_ARRAY);
}


/*
 * MapGenerator::GetExtents
 */
bool MapGenerator::GetExtents(Call& call) {
	view::CallRender3D *cr3d = dynamic_cast<view::CallRender3D*>(&call);
	if (cr3d == nullptr) return false;

	CallTriMeshData *ctmd = this->meshDataSlot.CallAs<CallTriMeshData>();
	if (ctmd == nullptr) return false;

	ctmd->SetFrameID(static_cast<uint>(cr3d->Time()));
	if (!(*ctmd)(1)) return false; // GetExtent of CallTriMeshData

	cr3d->SetTimeFramesCount(ctmd->FrameCount());
	cr3d->AccessBoundingBoxes().Clear();
	auto tmpBBox = ctmd->AccessBoundingBoxes().ObjectSpaceBBox();
	cr3d->AccessBoundingBoxes().SetObjectSpaceBBox(tmpBBox);
	tmpBBox = ctmd->AccessBoundingBoxes().ObjectSpaceClipBox();
	cr3d->AccessBoundingBoxes().SetObjectSpaceClipBox(tmpBBox);
	float scale = ctmd->AccessBoundingBoxes().ClipBox().LongestEdge();
	if (scale > 0.0f) {
		scale = 2.0f / scale;
	} else {
		scale = 1.0f;
	}
	cr3d->AccessBoundingBoxes().MakeScaledWorld(scale);

	return true;
}


/*
 * MapGenerator::GetMeshData
 */
bool MapGenerator::GetMeshData(Call& call) {
    geocalls::CallTriMeshData *ctmd = dynamic_cast<geocalls::CallTriMeshData*>(&call);
    if (ctmd == nullptr) return false;
    ctmd->SetObjects(1, &this->out_mesh);

    return true;
}


/*
 * MapGenerator::GetMeshExtents
 */
bool MapGenerator::GetMeshExtents(Call& call) {
    geocalls::CallTriMeshData *ctmd = dynamic_cast<geocalls::CallTriMeshData*>(&call);
    if (ctmd == nullptr) return false;

    if (this->store_new_mesh) {
        MeshMode selected = (MeshMode) this->out_mesh_selection_slot.Param<param::EnumParam>()->Value();
        geocalls::CallTriMeshData::Mesh themesh;
        if (selected == MeshMode::MESH_ORIGINAL) {
            themesh.SetVertexData(static_cast<uint>(this->vertices.size() / 3), this->vertices.data(), this->normals.data(), this->vertexColors.data(), nullptr, false);
            themesh.SetTriangleData(static_cast<uint>(this->faces.size() / 3), this->faces.data(), false);
            this->meshBoundingBox = this->computeBoundingBox(this->vertices);
        } else if (selected == MeshMode::MESH_CUT) {
            themesh.SetVertexData(static_cast<uint>(this->vertices_rebuild.size() / 3), this->vertices_rebuild.data(), this->normals_rebuild.data(), this->vertexColors_rebuild.data(), nullptr, false);
            themesh.SetTriangleData(static_cast<uint>(this->faces_rebuild.size() / 3), this->faces_rebuild.data(), false);
            this->meshBoundingBox = this->computeBoundingBox(this->vertices_rebuild);
        } else if (selected == MeshMode::MESH_SPHERE) {
            themesh.SetVertexData(static_cast<uint>(this->vertices_sphere.size() / 3), this->vertices_sphere.data(), this->vertices_sphere.data(), this->vertexColors_rebuild.data(), nullptr, false);
            themesh.SetTriangleData(static_cast<uint>(this->faces_rebuild.size() / 3), this->faces_rebuild.data(), false);
            this->meshBoundingBox = this->computeBoundingBox(this->vertices_sphere);
        } else if (selected == MeshMode::MESH_MAP) {
            this->vertices_map = this->vertices_sphere;
            this->faces_map = this->faces_rebuild;
            vislib::math::Vector<float, 3> front(&this->vertices_map[this->look_at_id * 3]);
            vislib::math::Vector<float, 3> sp(this->sphere_data.X(), this->sphere_data.Y(), this->sphere_data.Z());
            float sr = this->sphere_data.W();
            for (size_t i = 0; i < this->vertices_map.size() / 3; i++) {
                vislib::math::Vector<float, 3> coord(&this->vertices_map[i * 3]);
                float len = (coord - sp).Length();
                if (std::abs(len / sr) > 1.0) {
                    len = 1.0f - len / sr;
                } else {
                    len = -0.1f;
                }
                auto relCoord = coord - sp;
                relCoord.Normalise();
                auto relCoord2 = front - sp;
                relCoord2.Normalise();
                float factor = 1.0f;
                if (std::signbit(relCoord.X())) {
                    factor = -1.0f;
                }
                float lambda = factor * static_cast<float>(vislib::math::PI_DOUBLE) / 2.0f;
                if (std::abs(relCoord.Z()) > 0.001f) {
                    lambda = std::atan2f(relCoord.X() , relCoord.Z());
                }
                float lambda2 = 0.0f;
                if (std::abs(relCoord2.Z()) > 0.001) {
                    lambda = std::atan2f(relCoord2.X() , relCoord2.Z());
                }
                auto fin = vislib::math::Vector<float, 3>((lambda - lambda2) / static_cast<float>(vislib::math::PI_DOUBLE), relCoord.Y(), len);

                if (fin.X() > 1.0) fin[0] -= 2.0f;
                if (fin.X() < -1.0) fin[0] += 2.0f;

                this->vertices_map[3 * i + 0] = fin.X();
                this->vertices_map[3 * i + 1] = fin.Y();
                this->vertices_map[3 * i + 2] = fin.Z();
            }

            for (size_t i = 0; i < this->faces_rebuild.size() / 3; i++) {
                std::array<vislib::math::Vector<float, 3>, 3> v = {
                    vislib::math::Vector<float, 3>(&this->vertices_map[this->faces_rebuild[i * 3 + 0]]),
                    vislib::math::Vector<float, 3>(&this->vertices_map[this->faces_rebuild[i * 3 + 1]]),
                    vislib::math::Vector<float, 3>(&this->vertices_map[this->faces_rebuild[i * 3 + 2]])
                };
                //sort vectors
                int idx0 = 0;
                int idx1 = 1;
                int idx2 = 2;
                if (v[0].X() < v[1].X()) {
                    if (v[0].X() < v[2].X()) {
                        if (v[1].X() > v[2].X()) {
                            idx1 = 2;
                            idx2 = 1;
                        }
                    } else {
                        idx0 = 2;
                        idx1 = 0;
                        idx2 = 1;
                    }
                } else {
                    if (v[1].X() < v[2].X()) {
                        if (v[0].X() < v[2].X()) {
                            idx0 = 1;
                            idx1 = 0;
                        } else {
                            idx0 = 1;
                            idx1 = 2;
                            idx2 = 0;
                        }
                    } else {
                        idx0 = 2;
                        idx2 = 0;
                    }
                }

                if (v[idx0].X() < -0.5 && v[idx1].X() < -0.5 && v[idx2].X() > 0.5) {

                    this->faces_map[i * 3 + 0] = idx0;
                    this->faces_map[i * 3 + 1] = idx1;
                    this->vertices_map.push_back(v[idx2].X() - 2.0f);
                    this->vertices_map.push_back(v[idx2].Y());
                    this->vertices_map.push_back(v[idx2].Z());
                    this->faces_map[i * 3 + 2] = static_cast<uint>((this->vertices_map.size() / 3) - 1);

                    this->vertices_map.push_back(v[idx0].X() + 2.0f);
                    this->vertices_map.push_back(v[idx0].Y());
                    this->vertices_map.push_back(v[idx0].Z());
                    this->faces_map.push_back(static_cast<uint>((this->vertices_map.size() / 3) - 1));
                    this->vertices_map.push_back(v[idx1].X() + 2.0f);
                    this->vertices_map.push_back(v[idx1].Y());
                    this->vertices_map.push_back(v[idx1].Z());
                    this->faces_map.push_back(static_cast<uint>((this->vertices_map.size() / 3) - 1));
                    this->faces_map.push_back(idx2);

                } else if (v[idx0].X() < -0.5f && v[idx1].X() > 0.5f && v[idx2].X() > 0.5f) {
                    this->faces_map[i * 3 + 0] = idx0;
                    this->vertices_map.push_back(v[idx1].X() - 2.0f);
                    this->vertices_map.push_back(v[idx1].Y());
                    this->vertices_map.push_back(v[idx1].Z());
                    this->faces_map[i * 3 + 1] = static_cast<uint>((this->vertices_map.size() / 3) - 1);
                    this->vertices_map.push_back(v[idx2].X() - 2.0f);
                    this->vertices_map.push_back(v[idx2].Y());
                    this->vertices_map.push_back(v[idx2].Z());
                    this->faces_map[i * 3 + 2] = static_cast<uint>((this->vertices_map.size() / 3) - 1);

                    this->vertices_map.push_back(v[idx0].X() - 2.0f);
                    this->vertices_map.push_back(v[idx0].Y());
                    this->vertices_map.push_back(v[idx0].Z());
                    this->faces_map.push_back(static_cast<uint>((this->vertices_map.size() / 3) - 1));
                    this->faces_map.push_back(idx2);
                    this->faces_map.push_back(idx0);
                } else {
                    this->faces_map[i * 3 + 0] = idx0;
                    this->faces_map[i * 3 + 1] = idx1;
                    this->faces_map[i * 3 + 2] = idx2;
                }
            }

            themesh.SetVertexData(static_cast<uint>(this->vertices_map.size() / 3), this->vertices_map.data(), this->vertices_map.data(), this->vertexColors_rebuild.data(), nullptr, false);
            themesh.SetTriangleData(static_cast<uint>(this->faces_map.size() / 3), this->faces_map.data(), false);
            this->meshBoundingBox = this->computeBoundingBox(this->vertices_map);
        }
        this->out_mesh = themesh;
        this->store_new_mesh = false;
    }
    ctmd->SetFrameCount(1);
    ctmd->AccessBoundingBoxes().Clear();
    ctmd->AccessBoundingBoxes().SetObjectSpaceBBox(this->meshBoundingBox);
    ctmd->AccessBoundingBoxes().SetObjectSpaceClipBox(this->meshBoundingBox);
    // Make scaled world?

    return true;
}


/*
 * MapGenerator::fillLocalMesh
 */
bool MapGenerator::fillLocalMesh(const CallTriMeshData::Mesh& mesh) {
	uint faceCnt = mesh.GetTriCount();
	uint vertexCnt = mesh.GetVertexCount();

	this->faces.resize(faceCnt * 3);
	this->faces.shrink_to_fit();

	this->face_edge_offset.resize(faceCnt);
	this->face_edge_offset.shrink_to_fit();
	for (auto& offset : this->face_edge_offset) {
		// There can't be more than 6 edges per face.
		offset.resize(6);
	}
	this->face_edge_offset_depth.resize(faceCnt + 1);
	this->face_edge_offset.shrink_to_fit();

	this->normals.resize(vertexCnt * 3);
	this->normals.shrink_to_fit();

	this->vertexColors.resize(vertexCnt * 3);
	this->vertexColors.shrink_to_fit();

	this->vertex_edge_offset.resize(vertexCnt);
	this->vertex_edge_offset.shrink_to_fit();
	for (auto& offset : this->vertex_edge_offset) {
		/**
		 * We now assume that there can not be more than 30
		 * edges per vertex. The correct size will be 
		 * determined later.
		 */
		offset.resize(this->vertex_edge_offset_max_depth);
	}
	this->vertex_edge_offset_depth.resize(vertexCnt + 1);
	this->vertex_edge_offset_depth.shrink_to_fit();

	this->vertices.resize(vertexCnt * 3);
	this->vertices.shrink_to_fit();

	// copy the vertex data
	if (mesh.GetVertexDataType() == CallTriMeshData::Mesh::DT_FLOAT) { 
		// float vertex data
		if (mesh.GetVertexPointerFloat() == nullptr) return false;
		std::copy(mesh.GetVertexPointerFloat(), mesh.GetVertexPointerFloat() + vertexCnt * 3, this->vertices.begin());

	} else if(mesh.GetVertexDataType() == CallTriMeshData::Mesh::DT_DOUBLE) { 
		// double vertex data
		if (mesh.GetVertexPointerDouble() == nullptr) return false;
		std::transform(mesh.GetVertexPointerDouble(), mesh.GetVertexPointerDouble() + vertexCnt * 3, this->vertices.begin(), [](double v) {
			return static_cast<float>(v);
		});

	} else {
		return false;
	}
	
	// copy the face data
	if (mesh.GetTriDataType() == CallTriMeshData::Mesh::DT_UINT32) { 
		// 32 bit uint face indices
		if (mesh.GetTriIndexPointerUInt32() == nullptr) return false;
		std::copy(mesh.GetTriIndexPointerUInt32(), mesh.GetTriIndexPointerUInt32() + faceCnt * 3, this->faces.begin());

	} else if (mesh.GetTriDataType() == CallTriMeshData::Mesh::DT_UINT16) { 
		// 16 bit unsigned face indices
		if (mesh.GetTriIndexPointerUInt16() == nullptr) return false;
		std::transform(mesh.GetTriIndexPointerUInt16(), mesh.GetTriIndexPointerUInt16() + faceCnt * 3, this->faces.begin(), [](unsigned short v) {
			return static_cast<uint>(v);
		});

	} else if (mesh.GetTriDataType() == CallTriMeshData::Mesh::DT_BYTE) { 
		// 8 bit unsigned face indices
		if (mesh.GetTriIndexPointerByte() == nullptr) return false;
		std::transform(mesh.GetTriIndexPointerByte(), mesh.GetTriIndexPointerByte() + faceCnt * 3, this->faces.begin(), [](unsigned char v) {
			return static_cast<uint>(v);
		});

	} else {
		return false;
	}

	// copy the normal data
	if (mesh.GetNormalDataType() == CallTriMeshData::Mesh::DT_FLOAT) { 
		// float normals
		if (mesh.GetNormalPointerFloat() == nullptr) return false;
		std::copy(mesh.GetNormalPointerFloat(), mesh.GetNormalPointerFloat() + vertexCnt * 3, this->normals.begin());

	} else if (mesh.GetNormalDataType() == CallTriMeshData::Mesh::DT_DOUBLE) { 
		// double normals
		if (mesh.GetNormalPointerDouble() == nullptr) return false;
		std::transform(mesh.GetNormalPointerDouble(), mesh.GetNormalPointerDouble() + vertexCnt * 3, this->normals.begin(), [](double v) {
			return static_cast<float>(v);
		});

	} else {
		return false;
	}

	// copy the colour data
	if (mesh.GetColourDataType() == CallTriMeshData::Mesh::DT_FLOAT) { 
		// float colours
		if (mesh.GetColourPointerFloat() == nullptr) return false;
		std::copy(mesh.GetColourPointerFloat(), mesh.GetColourPointerFloat() + vertexCnt * 3, this->vertexColors.begin());

	} else if (mesh.GetColourDataType() == CallTriMeshData::Mesh::DT_DOUBLE) { 
		// double colours
		if (mesh.GetColourPointerDouble() == nullptr) return false;
		std::transform(mesh.GetColourPointerDouble(), mesh.GetColourPointerDouble() + vertexCnt * 3, this->vertexColors.begin(), [](double v) {
			return static_cast<float>(v);
		});

	} else if (mesh.GetColourDataType() == CallTriMeshData::Mesh::DT_BYTE) { 
		// unsigned char colours
		if (mesh.GetColourPointerByte() == nullptr) return false;
		std::transform(mesh.GetColourPointerByte(), mesh.GetColourPointerByte() + vertexCnt * 3, this->vertexColors.begin(), [](unsigned char v) {
			return static_cast<float>(v) / 255.0f; // the range of uchar values has to be corrected to [0, 1]
		});

	} else {
		return false;
	}
	 
	return true;
}


/*
 * MapGenerator::findBoundaryMeridian
 */
bool MapGenerator::findBoundaryMeridian(const Poles& p_poles, std::vector<int>& p_types, 
		std::vector<bool>& p_valid_vertices, const std::vector<float>& p_zvalues,
		const vec3f& p_eye_dir) {
	// Initialise poles.
	p_valid_vertices[p_poles.north] = false;
	p_valid_vertices[p_poles.south] = false;

	// Initialise and find path.
	std::vector<uint> meridian;
	uint nxt;
	float cur_zvalue, nxt_zvalue;

	// Start at the south pole and walk along the path of steepest descent until the 
	// north pole is reached. Mark every vertex on the meridian with the ID 1.
	uint cur = p_poles.south;
	meridian.reserve(static_cast<size_t>(log(this->vertices_rebuild.size() / 3)));

	// Put the south pole into the meridian.
	meridian.push_back(cur);
	p_types[cur] = 1;
	p_valid_vertices[cur] = false;

	// Get the vertex coordinates from the south pole.
	vec3f origin = vec3f(this->vertices_rebuild[cur * 3 + 0],
		this->vertices_rebuild[cur * 3 + 1],
		this->vertices_rebuild[cur * 3 + 2]);

	// Find the neighbouring vertex that has the smallest angle to the current
	// eye direction and use this as the next vertex on the meridian.
	float min_angle = std::numeric_limits<float>::max();
	uint next;
	for (const auto& edge : this->vertex_edge_offset[cur]) {
		// Get the ID of the neighbour.
		uint nxt;
		if (edge.vertex_id_0 == cur) {
			nxt = edge.vertex_id_1;

		} else {
			nxt = edge.vertex_id_0;
		}
		
		// Get the vertex coordiantes of the neighbour.
		vec3f target = vec3f(this->vertices_rebuild[nxt * 3 + 0],
			this->vertices_rebuild[nxt * 3 + 1],
			this->vertices_rebuild[nxt * 3 + 2]);

		// Get the direction vector from the pole to the neighbour.
		vec3f dir = target - origin;
		dir.Normalise();

		// Get the angle between the direction and the eye direction and update
		// the next vertex if the angle is smaller than the minimum so far.
		float angle = dir.Angle(p_eye_dir);
		if (angle < min_angle) {
			min_angle = angle;
			next = nxt;
		}
	}

	// Start with the second vertex on the meridian as the south pole was already
	// added to it.
	cur = next;
	while (cur != p_poles.north) {
		meridian.push_back(cur);
		p_types[cur] = 1;
		p_valid_vertices[cur] = false;
		cur_zvalue = p_zvalues[cur];
		nxt_zvalue = cur_zvalue;
		for (const auto& edge : this->vertex_edge_offset[cur]) {
			if (edge.vertex_id_0 == cur) {
				if (p_zvalues[edge.vertex_id_1] < nxt_zvalue) {
					nxt_zvalue = p_zvalues[edge.vertex_id_1];
					nxt = edge.vertex_id_1;
				}

			} else {
				if (p_zvalues[edge.vertex_id_0] < nxt_zvalue) {
					nxt_zvalue = p_zvalues[edge.vertex_id_0];
					nxt = edge.vertex_id_0;
				}
			}
		}

		if (nxt == cur) return false;
		cur = nxt;
	}
	meridian.push_back(cur);
	meridian.shrink_to_fit();

	// Mark every vertex next to the meridian as left.
	for (size_t i = 1; i < meridian.size() - 1; i++) {
		cur = meridian[i];
		for (const auto& edge : this->vertex_edge_offset[cur]) {
			if (edge.vertex_id_0 == cur && p_types[edge.vertex_id_1] != 1) {
				p_types[edge.vertex_id_1] = 2;
			}
			if (edge.vertex_id_1 == cur && p_types[edge.vertex_id_0] != 1) {
				p_types[edge.vertex_id_0] = 2;
			}
		}
	}
	
	// Mark the first vertex of the south pole that is left of the meridian as
	// right.
	bool found;
	std::vector<uint> right_side;
	right_side.reserve(meridian.size() - 1);
	cur = meridian[0];
	for (const auto& edge : this->vertex_edge_offset[cur]) {
		if (edge.vertex_id_0 == cur) {
			if (p_types[edge.vertex_id_1] == 2) {
				p_types[edge.vertex_id_1] = 3;
				right_side.push_back(edge.vertex_id_1);
				break;
			}
		}

		if (edge.vertex_id_1 == cur) {
			if (p_types[edge.vertex_id_0] == 2) {
				p_types[edge.vertex_id_0] = 3;
				right_side.push_back(edge.vertex_id_0);
				break;
			}
		}
	}

	// Walk along the "right" side of the meridian and mark every vertex that was
	// previously marked as "left" as "right".
	while (!right_side.empty()) {
		cur = right_side.back();
		right_side.pop_back();
		for (const auto& edge : this->vertex_edge_offset[cur]) {
			if (edge.vertex_id_0 == cur && p_types[edge.vertex_id_1] == 2) {
				p_types[edge.vertex_id_1] = 3;
				found = false;
				for (const auto& n_edge : this->vertex_edge_offset[edge.vertex_id_0]) {
					if (n_edge.vertex_id_0 == meridian.back()) {
						found = true;
						break;
					}
					if (n_edge.vertex_id_1 == meridian.back()) {
						found = true;
						break;
					}
				}
				if (!found) {
					right_side.push_back(edge.vertex_id_1);
				}
			}

			if (edge.vertex_id_1 == cur && p_types[edge.vertex_id_0] == 2) {
				p_types[edge.vertex_id_0] = 3;
				found = false;
				for (const auto& n_edge : this->vertex_edge_offset[edge.vertex_id_1]) {
					if (n_edge.vertex_id_0 == meridian.back()) {
						found = true;
						break;
					}
					if (n_edge.vertex_id_1 == meridian.back()) {
						found = true;
						break;
					}
				}
				if (!found) {
					right_side.push_back(edge.vertex_id_0);
				}
			}
		}
	}

	// Check if there is still a vertex with the ID 2, i.e. "left" that has an edge that
	// ends on a vertex with the ID 3, i.e. "right". Mark that vertex as "right" as well.
	for (size_t i = 0; i < p_types.size(); i++) {
		if (p_types[i] == 2) {
			for (const auto& edge : this->vertex_edge_offset[i]) {
				if (cur == edge.vertex_id_0 && p_types[edge.vertex_id_1] == 3) {
					p_types[edge.vertex_id_0] = 3;
				}
				if (cur == edge.vertex_id_1 && p_types[edge.vertex_id_0] == 3) {
					p_types[edge.vertex_id_1] = 3;
				}
			}
		}
	}

	// Reset the poles.
	p_types[p_poles.north] = -1;
	p_types[p_poles.south] = -1;

	return true;
}


/*
 * MapGenerator::findCircles
 */
bool MapGenerator::findCircles(std::vector<FaceGroup>& p_groups) {
	for (auto& group : p_groups) {
		// We are only interested in shadowed groups.
		if (!group.state) continue;
		if (group.border_edges.empty()) continue;

		// Sort the edges according to the first and to the second vertex ID.
		std::vector<Edge> sorted_edges_id0 = group.border_edges;
		std::vector<Edge> sorted_edges_id1 = group.border_edges;
		if (!this->cuda_kernels->SortEdges(sorted_edges_id0, 0)) return false;
		if (!this->cuda_kernels->SortEdges(sorted_edges_id1, 1)) return false;

		// Initialise the circle detection.
		std::vector<uint> circle;
		circle.reserve(group.border_edges.size() * 2);
		std::vector<bool> marked_edges = std::vector<bool>(group.border_edges.size(), false);
		marked_edges.shrink_to_fit();
		uint start_edge_id = group.border_edges[0].edge_id;
		marked_edges[start_edge_id] = true;

		// Start the circle detection.
		uint cur_edge_id = start_edge_id;
		uint cur_vertex_id, vertex_id;
		circle.push_back(group.border_edges[0].vertex_id_1);
		circle.push_back(group.border_edges[0].vertex_id_0);
		cur_vertex_id = group.border_edges[0].vertex_id_0;
		while (!allElementsTrue(marked_edges)) {
			// Get the ranges for the edges that have the same ID in their first vertex ID.
			auto id_0_begin = std::lower_bound(sorted_edges_id0.begin(), sorted_edges_id0.end(),
				cur_vertex_id,
				[](const Edge& lhs, const uint rhs) { return lhs.vertex_id_0 < rhs; });
			auto id_0_end = std::upper_bound(sorted_edges_id0.begin(), sorted_edges_id0.end(),
				cur_vertex_id,
				[](const uint lhs, const Edge& rhs) { return lhs < rhs.vertex_id_0; });

			// Get the ranges for the edges that have the same ID in their second vertex ID.
			auto id_1_begin = std::lower_bound(sorted_edges_id1.begin(), sorted_edges_id1.end(),
				cur_vertex_id,
				[](const Edge& lhs, const uint rhs) { return lhs.vertex_id_1 < rhs; });
			auto id_1_end = std::upper_bound(sorted_edges_id1.begin(), sorted_edges_id1.end(),
				cur_vertex_id,
				[](const uint lhs, const Edge& rhs) { return lhs < rhs.vertex_id_1; });

			// Look for the next edge that shares the same vertex ID.
			std::vector<Edge>::iterator next_edge;
			bool found = false;
			for (auto it = id_0_begin; it != id_0_end; it++) {
				if (it->edge_id != cur_edge_id) {
					if (!marked_edges[it->edge_id] || it->edge_id == start_edge_id) {
						next_edge = it;
						found = true;
						break;
					}
				}
			}

			if (!found) {
				for (auto it = id_1_begin; it != id_1_end; it++) {
					if (it->edge_id != cur_edge_id) {
						if (!marked_edges[it->edge_id] || it->edge_id == start_edge_id) {
							next_edge = it;
							found = true;
							break;
						}
					}
				}

				// No next edge found empty the circle of edges and start new.
				if (!found) {
					circle.erase(circle.begin(), circle.end());
					circle.clear();

					// Not all edges are marked, there must be more circles. Set a new start edge ID.
					for (size_t j = 0; j < marked_edges.size(); j++) {
						if (!marked_edges[j]) {
							start_edge_id = group.border_edges[j].edge_id;
							circle.push_back(group.border_edges[j].vertex_id_1);
							circle.push_back(group.border_edges[j].vertex_id_0);
							cur_vertex_id = group.border_edges[j].vertex_id_0;
							cur_edge_id = start_edge_id;
							marked_edges[j] = true;
							found = true;
							break;
						}
					}
					continue;
				}
			}
			
			// Look at the first vertex ID.
			found = false;
			vertex_id = next_edge->vertex_id_0;
			if (vertex_id == cur_vertex_id) {
				if (next_edge->edge_id == start_edge_id) {
					// Found a circle add it to the group and check if we already have checked all edges.
					circle.pop_back();
					group.circles.push_back(circle);
					circle.erase(circle.begin(), circle.end());
					circle.clear();

					// Not all edges are marked, there must be more circles. Set a new start edge ID.
					for (size_t j = 0; j < marked_edges.size(); j++) {
						if (!marked_edges[j]) {
							start_edge_id = group.border_edges[j].edge_id;
							circle.push_back(group.border_edges[j].vertex_id_1);
							circle.push_back(group.border_edges[j].vertex_id_0);
							cur_vertex_id = group.border_edges[j].vertex_id_0;
							cur_edge_id = start_edge_id;
							marked_edges[j] = true;
							found = true;
							break;
						}
					}

				} else {
					cur_edge_id = next_edge->edge_id;
					marked_edges[cur_edge_id] = true;
					circle.push_back(next_edge->vertex_id_1);
					// Set the vertex ID to the vertex ID of the first vertex of the current edge.
					cur_vertex_id = next_edge->vertex_id_1;
					found = true;
				}
			}

			// Look at the second vertex ID.
			vertex_id = next_edge->vertex_id_1;
			if (vertex_id == cur_vertex_id && !found) {
				if (next_edge->edge_id == start_edge_id) {
					// Found a circle add it to the group and check if we already have checked all edges.
					circle.pop_back();
					group.circles.push_back(circle);
					circle.erase(circle.begin(), circle.end());
					circle.clear();

					// Not all edges are marked, there must be more circles. Set a new start edge ID.
					for (size_t j = 0; j < marked_edges.size(); j++) {
						if (!marked_edges[j]) {
							start_edge_id = group.border_edges[j].edge_id;
							circle.push_back(group.border_edges[j].vertex_id_1);
							circle.push_back(group.border_edges[j].vertex_id_0);
							cur_vertex_id = group.border_edges[j].vertex_id_0;
							cur_edge_id = start_edge_id;
							marked_edges[j] = true;
							break;
						}
					}

				} else {
					cur_edge_id = next_edge->edge_id;
					marked_edges[cur_edge_id] = true;
					circle.push_back(next_edge->vertex_id_0);
					// Set the vertex ID to the vertex ID of the first vertex of the current edge.
					cur_vertex_id = next_edge->vertex_id_0;
				}
			}
		}

		// Add the circle that was found last.
		if (circle.front() == circle.back()) {
			circle.pop_back();
			group.circles.push_back(circle);
		}
	}
	return true;
}


/*
 * MapGenerator::findEnclosures
 */
int MapGenerator::findEnclosures(uint p_start_face, uint p_face_id, std::map<uint, uint>& p_group,
	const std::vector<uint>& p_face_ids, const size_t p_face_id_cnt,
	std::vector<bool>& p_marked_faces, const size_t p_rounds) {
	// Initialise the growing.
	uint active_queue;
	std::vector<uint> queue_one, queue_two;
	std::vector<uint> group_cnt = std::vector<uint>(p_face_id_cnt, 0);

	// Add the start face to the group and mark it.
	p_group.insert(std::make_pair(p_start_face, p_start_face));
	p_marked_faces[p_start_face] = true;

	// Find the neighbours of the faces of the smaller circle.
	queue_one.reserve(3);
	queue_two.reserve(3);
	for (const auto& edge : this->face_edge_offset[p_start_face]) {
		if (edge.face_id_0 == p_start_face) {
			if (p_face_ids[edge.face_id_1] == p_face_id) {
				auto group_it = p_group.find(edge.face_id_1);
				if (group_it == p_group.end()) {
					queue_one.emplace_back(edge.face_id_1);
					p_group.insert(std::make_pair(edge.face_id_1, edge.face_id_1));
					p_marked_faces[edge.face_id_1] = true;
				}

			} else {
				group_cnt[p_face_ids[edge.face_id_1]]++;
			}

		} else {
			if (p_face_ids[edge.face_id_0] == p_face_id) {
				auto group_it = p_group.find(edge.face_id_0);
				if (group_it == p_group.end()) {
					queue_one.emplace_back(edge.face_id_0);
					p_group.insert(std::make_pair(edge.face_id_0, edge.face_id_0));
					p_marked_faces[edge.face_id_0] = true;
				}

			} else {
				group_cnt[p_face_ids[edge.face_id_0]]++;
			}
		}
	}

	// Check if there are any new faces that can be reached from the start face.
	if (queue_one.empty()) {
		// There are no new faces so the face is an enclosure. Look at the face IDs
		// of the surrounding faces and return the face ID that occures the most.
		uint max_cnt = 0;
		int neighbouring_group = -1;
		for (size_t j = 0; j < group_cnt.size(); j++) {
			if (group_cnt[j] > max_cnt) {
				neighbouring_group = static_cast<int>(j);
				max_cnt = group_cnt[j];
			}
		}
		return neighbouring_group;
	}

	// Get as many neighbouring faces as possible. If there are less than 
	// p_rounds possible then the faces form an enclosure. So find the surrounding
	// group and return the group ID.
	active_queue = 0;
	uint cur;
	for (size_t i = 0; i < p_rounds; i++) {
		// Set the active and inactive queues.
		std::vector<uint>* active;
		std::vector<uint>* inactive;
		if (i % 2 == 0) {
			active = &queue_one;
			inactive = &queue_two;

		} else {
			inactive = &queue_one;
			active = &queue_two;
		}

		// Loop over the faces in the current active queue and add new faces to thee
		// inactive queue.
		while (!active->empty()) {
			// Get the last face and remove it from the active queue.
			cur = active->back();
			active->pop_back();

			// Get the neighbouring faces that are not inside the group.
			for (const auto& edge : this->face_edge_offset[cur]) {
				if (edge.face_id_0 == cur) {
					if (p_face_ids[edge.face_id_1] == p_face_id) {
						auto group_it = p_group.find(edge.face_id_1);
						if (group_it == p_group.end()) {
							inactive->emplace_back(edge.face_id_1);
							p_group.insert(std::make_pair(edge.face_id_1, edge.face_id_1));
							p_marked_faces[edge.face_id_1] = true;
						}

					} else {
						group_cnt[p_face_ids[edge.face_id_1]]++;
					}

				} else {
					if (p_face_ids[edge.face_id_0] == p_face_id) {
						auto group_it = p_group.find(edge.face_id_0);
						if (group_it == p_group.end()) {
							inactive->emplace_back(edge.face_id_0);
							p_group.insert(std::make_pair(edge.face_id_0, edge.face_id_0));
							p_marked_faces[edge.face_id_0] = true;
						}

					} else {
						group_cnt[p_face_ids[edge.face_id_0]]++;
					}
				}
			}
		}

		// Check if there are any new faces that can be reached from the current faces.
		if (inactive->empty()) {
			// There are no new faces so the face is an enclosure. Look at the face IDs
			// of the surrounding faces and return the face ID that occures the most.
			uint max_cnt = 0;
			int neighbouring_group = -1;
			for (size_t j = 0; j < group_cnt.size(); j++) {
				if (group_cnt[j] > max_cnt) {
					neighbouring_group = static_cast<int>(j);
					max_cnt = group_cnt[j];
				}
			}
			return neighbouring_group;
		}
	}

	// The faces do not form an enclosure so return -1.
	return -1;
}


/*
 * MapGenerator::findPoles
 */
bool MapGenerator::findPoles(const vislib::math::Vector<float, 3>& p_eye_dir, 
		const vislib::math::Vector<float, 3>& p_up_dir, 
		const vislib::math::Vector<float, 3>& p_center, Poles& p_poles) {
	vislib::math::Vector<float, 3> pos, tmp_vec;
	float len, dist, step, min_y, max_y;
	bool finished;

	finished = false;
	step = 1.5f;
	min_y = std::numeric_limits<float>::max();
	max_y = -std::numeric_limits<float>::max();
	while (!finished) {
		for (size_t i = 0; i < this->vertices_rebuild.size() / 3; i++) {
			pos.Set(this->vertices_rebuild[i * 3 + 0], this->vertices_rebuild[i * 3 + 1], 
				this->vertices_rebuild[i * 3 + 2]);

			tmp_vec = pos - p_center;
			len = p_up_dir.Dot(tmp_vec);
			dist = (p_up_dir.Cross(tmp_vec)).Length();

			if (dist < step) {
				if (len < min_y) {
					min_y = len;
					p_poles.south = static_cast<uint>(i);
				}
				if (len > max_y)
				{
					max_y = len;
					p_poles.north = static_cast<uint>(i);
				}
			}
		}

		if (p_poles.north == p_poles.south) step += 1.5f;
		else finished = true;
	}

	// Initialise the ray from the center of the sphere to the camera.
	Ray ray = Ray(p_eye_dir, p_center);
	// Intersect the Octree to find the first face we intersect and use one of it's vertex IDs.
	auto retval = this->octree.IntersectOctree(this->faces_rebuild, ray, this->vertices_rebuild);
	if (retval == -1) return false;
	this->look_at_id = static_cast<uint>(retval);

	return true;
}


/*
 * MapGenerator::getNameOfPDB
 */
std::string MapGenerator::getNameOfPDB(MolecularDataCall & mdc) {
    std::string name(vislib::StringA(T2A(mdc.GetPDBFilename())).PeekBuffer());
    if (name.length() > 4) {
        name = name.substr(0, name.length() - 4);
    }
	return name;
}


/*
 * MapGenerator::growFaces
 */
void MapGenerator::growFaces(std::vector<uint>& p_circle_faces_0, std::vector<uint>& p_circle_faces_1, 
		std::vector<uint>& p_circle_vertices_0, std::vector<uint>& p_circle_vertices_1, 
		std::map<uint, uint>& p_group) {
	// Initialise the region growing.
	std::vector<uint>& bigger_circle_faces = p_circle_faces_1;
	std::vector<uint>& bigger_circle_vertices = p_circle_vertices_1;
	std::vector<uint> queue;
	std::vector<uint>& smaller_circle_faces = p_circle_faces_0;
	std::vector<uint>& smaller_circle_vertices = p_circle_vertices_0;

	// Get the smaller circle.
	if (p_circle_faces_0.size() > p_circle_faces_1.size()) {
		bigger_circle_faces = p_circle_faces_0;
		bigger_circle_vertices = p_circle_vertices_0;
		smaller_circle_faces = p_circle_faces_1;
		smaller_circle_vertices = p_circle_vertices_1;
	}

	// Add the smaller circle to the group.
	for (const auto face_id : smaller_circle_faces) {
		p_group.insert(std::make_pair(face_id, face_id));
	}

	// Find the neighbours of the faces of the smaller circle.
	queue.reserve(smaller_circle_faces.size());
	for (const auto face_id : smaller_circle_faces) {
		for (const auto& edge : this->face_edge_offset[face_id]) {
			auto it = std::find(smaller_circle_vertices.begin(), smaller_circle_vertices.end(), edge.vertex_id_0);
			if (it == smaller_circle_vertices.end()) {
				it = std::find(smaller_circle_vertices.begin(), smaller_circle_vertices.end(), edge.vertex_id_1);
				if (it == smaller_circle_vertices.end()) {
					if (edge.face_id_0 == face_id) {
						auto group_it = p_group.find(edge.face_id_1);
						if(group_it == p_group.end()) {
							queue.emplace_back(edge.face_id_1);
							p_group.insert(std::make_pair(edge.face_id_1, edge.face_id_1));
						}

					} else {
						auto group_it = p_group.find(edge.face_id_0);
						if (group_it == p_group.end()) {
							queue.emplace_back(edge.face_id_0);
							p_group.insert(std::make_pair(edge.face_id_0, edge.face_id_0));
						}
					}
				}
			}
		}
	}

	// Get neighbours until the other circle is reached.
	uint cur;
	while (!queue.empty()) {
		// Get the last face and remove it from the queue.
		cur = queue.back();
		queue.pop_back();

		// Get the neighbouring faces that are not inside the group.
		for (const auto& edge : this->face_edge_offset[cur]) {
			if (edge.face_id_0 == cur) {
				auto group_it = p_group.find(edge.face_id_1);
				if (group_it == p_group.end()) {
					auto it = std::find(bigger_circle_faces.begin(), bigger_circle_faces.end(), edge.face_id_1);
					if (it == bigger_circle_faces.end()) {
						queue.emplace_back(edge.face_id_1);
					}
					p_group.insert(std::make_pair(edge.face_id_1, edge.face_id_1));
				}

			} else {
				auto group_it = p_group.find(edge.face_id_0);
				if (group_it == p_group.end()) {
					auto it = std::find(bigger_circle_faces.begin(), bigger_circle_faces.end(), edge.face_id_0);
					if (it == bigger_circle_faces.end()) {
						queue.emplace_back(edge.face_id_0);
					}
					p_group.insert(std::make_pair(edge.face_id_0, edge.face_id_0));
				}
			}
		}
	}

	// Add the bigger circle to the group.
	for (const auto face_id : bigger_circle_faces) {
		auto group_it = p_group.find(face_id);
		if (group_it == p_group.end()) {
			p_group.insert(std::make_pair(face_id, face_id));
		}
	}
}


/*
 * MapGenerator::growFaces
 */
bool MapGenerator::growFaces(std::vector<uint>& p_circle_faces, std::vector<uint>& p_circle_vertices, 
		std::map<uint, uint>& p_group, const size_t p_rounds) {
	// Initialise the growing.
	uint active_queue;
	std::vector<uint> queue_one, queue_two;

	// Add the circle to the group.
	for (const auto face_id : p_circle_faces) {
		p_group.insert(std::make_pair(face_id, face_id));
	}

	// Find the neighbours of the faces of the smaller circle.
	queue_one.reserve(p_circle_faces.size());
	queue_two.reserve(p_circle_faces.size());
	for (const auto face_id : p_circle_faces) {
		for (const auto& edge : this->face_edge_offset[face_id]) {
			auto it = std::find(p_circle_vertices.begin(), p_circle_vertices.end(), edge.vertex_id_0);
			if (it == p_circle_vertices.end()) {
				it = std::find(p_circle_vertices.begin(), p_circle_vertices.end(), edge.vertex_id_1);
				if (it == p_circle_vertices.end()) {
					if (edge.face_id_0 == face_id) {
						auto group_it = p_group.find(edge.face_id_1);
						if (group_it == p_group.end()) {
							queue_one.emplace_back(edge.face_id_1);
							p_group.insert(std::make_pair(edge.face_id_1, edge.face_id_1));
						}

					} else {
						auto group_it = p_group.find(edge.face_id_0);
						if (group_it == p_group.end()) {
							queue_one.emplace_back(edge.face_id_0);
							p_group.insert(std::make_pair(edge.face_id_0, edge.face_id_0));
						}
					}
				}
			}
		}
	}

	// Check if there are any new faces that can be reached from the circle.
	if (queue_one.empty()) {
		// There are no new faces so the circle is not valid.
		return false;
	}

	// Get the first five "circles" and return false if there are less then five, true
	// otherwise.
	active_queue = 0;
	uint cur;
	for (size_t i = 0; i < p_rounds; i++) {
		// Set the active and inactive queues.
		std::vector<uint>* active;
		std::vector<uint>* inactive;
		if (i % 2 == 0) {
			active = &queue_one;
			inactive = &queue_two;

		} else {
			inactive = &queue_one;
			active = &queue_two;
		}

		// Loop over the faces in the current active queue and add new faces to thee
		// inactive queue.
		while (!active->empty()) {
			// Get the last face and remove it from the active queue.
			cur = active->back();
			active->pop_back();

			// Get the neighbouring faces that are not inside the group.
			for (const auto& edge : this->face_edge_offset[cur]) {
				if (edge.face_id_0 == cur) {
					auto group_it = p_group.find(edge.face_id_1);
					if (group_it == p_group.end()) {
						inactive->emplace_back(edge.face_id_1);
						p_group.insert(std::make_pair(edge.face_id_1, edge.face_id_1));
					}

				} else {
					auto group_it = p_group.find(edge.face_id_0);
					if (group_it == p_group.end()) {
						inactive->emplace_back(edge.face_id_0);
						p_group.insert(std::make_pair(edge.face_id_0, edge.face_id_0));
					}
				}
			}
		}

		// Check if there are any new faces that can be reached from the circle.
		if (inactive->empty()) {
			// There are no new faces so the circle is not valid.
			return false;
		}
	}

	// The circle is valid.
	return true;
}


/*
 * MapGenerator::identifyBorderEdges
 */
void MapGenerator::identifyBorderEdges(const std::vector<bool>& p_face_shadowed,
		std::vector<FaceGroup>& p_groups) {
	for (auto& group : p_groups) {
		// Skip not shadowed groups, we are only interested in tunnels.
		if (!group.state) continue;
		group.border_edges.reserve(group.face_map.size() * 3);

		// Look at all faces of the group.
		uint edge_id = 0;
		for (auto it = group.face_map.begin(); it != group.face_map.end(); it++) {
			for (const auto& edge : this->face_edge_offset[it->second]) {
				// Look at all edges that border the face.
				if (edge.face_id_0 == it->second) {
					if (edge.face_id_1 != -1) {
						if (p_face_shadowed[edge.face_id_1] != group.state) {
							group.border_edges.emplace_back(edge);
							group.border_edges.back().edge_id = edge_id++;
						}
					}

				} else {
					if (edge.face_id_0 != -1) {
						if (p_face_shadowed[edge.face_id_0] != group.state) {
							group.border_edges.emplace_back(edge);
							group.border_edges.back().edge_id = edge_id++;
						}
					}
				}
			}
		}
	}
}


/*
 * MapGenerator::identifyBorderEdges
 */
void MapGenerator::identifyBorderEdges(const std::vector<uint>& p_face_group,
	std::vector<FaceGroup>& p_groups) {
	// The faces that do not belong to any group have the ID 0. The first group
	// has the ID 1.
	size_t group_id = 0;
	for (auto& group : p_groups) {
		// Initialise the border edges of the group.
		group.border_edges.clear();
		group.border_edges.reserve(group.face_map.size() * 3);

		// Look at all faces of the group.
		uint edge_id = 0;
		for (auto it = group.face_map.begin(); it != group.face_map.end(); it++) {
			for (const auto& edge : this->face_edge_offset[it->second]) {
				// Look at all edges that border the face.
				if (edge.face_id_0 == it->second) {
					if (edge.face_id_1 != -1) {
						if (p_face_group[edge.face_id_1] != group_id) {
							group.border_edges.emplace_back(edge);
							group.border_edges.back().edge_id = edge_id++;
						}
					}

				} else {
					if (edge.face_id_0 != -1) {
						if (p_face_group[edge.face_id_0] != group_id) {
							group.border_edges.emplace_back(edge);
							group.border_edges.back().edge_id = edge_id++;
						}
					}
				}
			}
		}
		group_id++;
	}
}


/*
 * MapGenerator::initialiseMapShader
 */
bool MapGenerator::initialiseMapShader(bool shaderReload) {
	GLint error = glGetError();

	if (this->map_vertex_vbo && !shaderReload) {
		glBindBuffer(GL_ARRAY_BUFFER, this->map_vertex_vbo);
		glDeleteBuffers(1, &this->map_vertex_vbo);
		this->map_vertex_vbo = 0;
	}

	if (!shaderReload) {
	// Create vertex buffer object for sphere area.
	size_t vertex_cnt = this->vertices_sphere.size();
	glGenBuffers(1, &this->map_vertex_vbo);
	glBindBuffer(GL_ARRAY_BUFFER, this->map_vertex_vbo);
	glBufferData(GL_ARRAY_BUFFER, sizeof(float) * vertex_cnt, this->vertices_sphere.data(), GL_DYNAMIC_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	// Create vertex buffer objects for the geodesic lines.
	if (this->geodesic_lines_vbos.size() != 0) {
		// Delete old buffers.
		for (auto& vbo : this->geodesic_lines_vbos) {
			if (vbo) {
				glBindBuffer(GL_ARRAY_BUFFER, vbo);
				glDeleteBuffers(1, &vbo);
				vbo = 0;
			}
		}
	}

	// Create new buffers for the geodesic lines.
	this->geodesic_lines_vbos.resize(this->geodesic_lines.size(), 0);
	for (size_t i = 0; i < this->geodesic_lines_vbos.size(); i++) {
		glGenBuffers(1, &this->geodesic_lines_vbos[i]);
		glBindBuffer(GL_ARRAY_BUFFER, this->geodesic_lines_vbos[i]);
			glBufferData(GL_ARRAY_BUFFER, sizeof(float) *
			static_cast<unsigned long long>(this->geodesic_lines[i].size()),
			this->geodesic_lines[i].data(), GL_DYNAMIC_DRAW);
		glBindBuffer(GL_ARRAY_BUFFER, 0);
	}

	error = glGetError();
	if (error != 0) return false;
	}

	if (!this->map_shader_init || shaderReload) {
		this->map_shader_init = true;

		if (shaderReload) {
			instance()->ShaderSourceFactory().LoadBTF("mapShader", true);
		}

		vislib::graphics::gl::ShaderSource vert, frag, geom;

		// Create shader for map and build the programme.
		if (!instance()->ShaderSourceFactory().MakeShaderSource("mapShader::vertex", vert)) {
			vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
				"Unable to load vertex shader source for map shader");
			return false;
		}
		if (!instance()->ShaderSourceFactory().MakeShaderSource("mapShader::geometry", geom)) {
			vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
				"Unable to load geometry shader source for map shader");
			return false;
		}
		if (!instance()->ShaderSourceFactory().MakeShaderSource("mapShader::fragment", frag)) {
			vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
				"Unable to load fragment shader source for map shader");
			return false;
		}

		const char* buildState = "compile";
		try {
			if (!this->map_shader.Compile(vert.Code(), vert.Count(), geom.Code(), geom.Count(), frag.Code(), frag.Count())) {
				vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
					"Unable to compile map shader: Unknown error\n");
				return false;
			}
			buildState = "setup";
			this->map_shader.SetProgramParameter(GL_GEOMETRY_INPUT_TYPE_EXT, GL_TRIANGLES);
			this->map_shader.SetProgramParameter(GL_GEOMETRY_OUTPUT_TYPE_EXT, GL_TRIANGLE_STRIP);
			this->map_shader.SetProgramParameter(GL_GEOMETRY_VERTICES_OUT_EXT, 6);
			buildState = "link";
			if (!this->map_shader.Link()) {
				vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
					"Unable to link map shader: Unknown error\n");
				return false;
			}

		} catch (vislib::graphics::gl::AbstractOpenGLShader::CompileException ce) {
			vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
				"Unable to %s map shader (@%s): %s\n", buildState,
				vislib::graphics::gl::AbstractOpenGLShader::CompileException::CompileActionName(
					ce.FailedAction()), ce.GetMsgA());
			return false;

		} catch (vislib::Exception e) {
			vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
				"Unable to %s map shader: %s\n", buildState, e.GetMsgA());
			return false;

		} catch (...) {
			vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
				"Unable to %s map shader: Unknown exception\n", buildState);
			return false;
		}

		// Create shader for geodesic lines and build the programme.
		if (!instance()->ShaderSourceFactory().MakeShaderSource("geolinesShader::vertex", vert)) {
			vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
				"Unable to load vertex shader source for geodesic lines shader");
			return false;
		}
		if (!instance()->ShaderSourceFactory().MakeShaderSource("geolinesShader::geometry", geom)) {
			vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
				"Unable to load geometry shader source for geodesic lines shader");
			return false;
		}
		if (!instance()->ShaderSourceFactory().MakeShaderSource("geolinesShader::fragment", frag)) {
			vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
				"Unable to load vertex shader source for geodesic lines shader");
			return false;
		}

		buildState = "compile";
		try {
			if (!this->geodesic_shader.Compile(vert.Code(), vert.Count(), geom.Code(), geom.Count(),
				frag.Code(), frag.Count()))
			{
				vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
					"Unable to compile geodesic lines shader: Unknown error\n");
				return false;
			}
			buildState = "setup";
			this->geodesic_shader.SetProgramParameter(GL_GEOMETRY_INPUT_TYPE_EXT, GL_LINES);
			this->geodesic_shader.SetProgramParameter(GL_GEOMETRY_OUTPUT_TYPE_EXT, GL_LINE_STRIP);
			this->geodesic_shader.SetProgramParameter(GL_GEOMETRY_VERTICES_OUT_EXT, 4);
			buildState = "link";
			if (!this->geodesic_shader.Link())
			{
				vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
					"Unable to link geodesic lines shader: Unknown error\n");
				return false;
			}

		} catch (vislib::graphics::gl::AbstractOpenGLShader::CompileException ce) {
			vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
				"Unable to %s geodesic lines shader (@%s): %s\n", buildState,
				vislib::graphics::gl::AbstractOpenGLShader::CompileException::CompileActionName(
					ce.FailedAction()), ce.GetMsgA());
			return false;
		} catch (vislib::Exception e) {
			vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
				"Unable to %s geodesic lines shader: %s\n", buildState, e.GetMsgA());
			return false;
		} catch (...) {
			vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
				"Unable to %s geodesic lines shader: Unknown exception\n", buildState);
			return false;
		}
	}

	if (!this->map_fbo.IsValid() && !shaderReload) {
		uint pxW = 1570 * 4;
		uint pxH = static_cast<uint>(pxW / vislib::math::PI_DOUBLE);
		this->map_fbo.Create(pxW, pxH);
	}
	
	return true;
 }


/*
 * MapGenerator::initialiseZvalues
 */
bool MapGenerator::initialiseZvalues(const Poles& p_poles, std::vector<float>& p_zvalues, 
		std::vector<bool>& p_valid_vertices, const float p_zvalue) {
	// Initialise poles.
	p_zvalues[p_poles.north] = -2.0f * p_zvalue;
	p_zvalues[p_poles.south] = 2.0f * p_zvalue;
	p_valid_vertices[p_poles.north] = false;
	p_valid_vertices[p_poles.south] = false;

	// Loop over the neighbours of the north pole.
	for (const auto& edge : this->vertex_edge_offset[p_poles.north]) {
		if (edge.vertex_id_0 == p_poles.north) {
			p_zvalues[edge.vertex_id_1] = -p_zvalue;
			p_valid_vertices[edge.vertex_id_1] = false;

		} else {
			p_zvalues[edge.vertex_id_0] = -p_zvalue;
			p_valid_vertices[edge.vertex_id_0] = false;
		}
	}

	// Loop over the neighbours of the south pole.
	for (const auto& edge : this->vertex_edge_offset[p_poles.south]) {
		if (edge.vertex_id_0 == p_poles.south) {
			p_zvalues[edge.vertex_id_1] = p_zvalue;
			p_valid_vertices[edge.vertex_id_1] = false;

		}
		else {
			p_zvalues[edge.vertex_id_0] = p_zvalue;
			p_valid_vertices[edge.vertex_id_0] = false;
		}
	}

	return true;
}

/*
 * MapGenerator::invertEdge
 */
bool MapGenerator::invertEdge(const uint p_start_id, const uint p_end_id,
		const std::vector<bool>& p_tunnels) {
	for (const auto& edge : this->vertex_edge_offset[p_start_id]) {
		if ((edge.vertex_id_0 == p_start_id || edge.vertex_id_1 == p_start_id) &&
			(edge.vertex_id_0 == p_end_id || edge.vertex_id_1 == p_end_id)) {
			// Get the two faces of the edge.
			auto face_id_0 = edge.face_id_0;
			auto face_id_1 = edge.face_id_1;

			// Check if one face was removed, if so look at the face that still exists.
			if (p_tunnels[face_id_0]) {
				for (const auto& face_edge : this->face_edge_offset[face_id_1]) {
					// If the face contains an edge in the same direction invert the new face.
					if (face_edge.vertex_id_0 == p_start_id && face_edge.vertex_id_1 == p_end_id) {
						return true;
					}
				}

			} else if (p_tunnels[face_id_1]) {
				for (const auto& face_edge : this->face_edge_offset[face_id_0]) {
					// If the face contains an edge in the same direction invert the new face.
					if (face_edge.vertex_id_0 == p_start_id && face_edge.vertex_id_1 == p_end_id) {
						return true;
					}
				}
			}
		}
	}

	return false;
}

/*
 * MapGenerator::isGenusN
 */
bool MapGenerator::isGenusN(const uint vertexCnt, const uint faceCnt, uint& p_genus) {
	uint num_edges = (3 * faceCnt) - ((3 * faceCnt) / 2);
	int euler = vertexCnt + faceCnt - num_edges;
	int genus = 1 - (euler / 2);
	p_genus = static_cast<uint>(genus);
	return genus > 0;
}


/*
 * MapGenerator::isValidCircle
 */
bool MapGenerator::isValidCircle(const vec4d& p_sphereToTest, const std::vector<uint>& p_circle) {
	// Compute average circle position.
	vec3f avgPos = vec3f(0.0f, 0.0f, 0.0f);
	for (auto vid : p_circle) {
		// Add the vertex position to the average postion.
		avgPos[0] += this->vertices_rebuild[vid * 3 + 0];
		avgPos[1] += this->vertices_rebuild[vid * 3 + 1];
		avgPos[2] += this->vertices_rebuild[vid * 3 + 2];
	}
	avgPos /= static_cast<float>(p_circle.size());

	// Find the vertex furthest distance from the center.
	float furthestDist = -1.0f;
	float dist = 0.0f;
	for (auto vid : p_circle) {
		// Update the furthest distance.
		dist = (this->vertices_rebuild[vid * 3 + 0] - avgPos[0]) * (this->vertices_rebuild[vid * 3 + 0] - avgPos[0]) +
			(this->vertices_rebuild[vid * 3 + 1] - avgPos[1]) * (this->vertices_rebuild[vid * 3 + 1] - avgPos[1]) +
			(this->vertices_rebuild[vid * 3 + 2] - avgPos[2]) * (this->vertices_rebuild[vid * 3 + 2] - avgPos[2]);
		if (dist > furthestDist) {
			furthestDist = dist;
		}
	}
	return (furthestDist > p_sphereToTest[3] * p_sphereToTest[3]);
}

/*
 * MapGenerator::latLonLineAddColour
 */
void MapGenerator::latLonLineAddColour(const float p_angle, const bool p_is_lat) {
	// Intitialise the colour values;
	float r, g, b;
	r = g = b = 0.0f;

	// Determine the type of the line.
	if (p_is_lat && std::fabsf(p_angle) < vislib::math::FLOAT_EPSILON) {
		// We are on the equator so get the equator colour.
		utility::ColourParser::FromString(this->lat_lon_lines_eq_colour_param.Param<param::StringParam>()->Value(),
			r, g, b);

	} else if (!p_is_lat && std::fabsf(std::fabsf(p_angle) - static_cast<float>(M_PI) / 2.0f) < vislib::math::FLOAT_EPSILON) {
		// We are on the Greenwich meridian so get the Greenwich meridian colour.
		utility::ColourParser::FromString(this->lat_lon_lines_gm_colour_param.Param<param::StringParam>()->Value(),
			r, g, b);

	} else {
		// We are not on any special latitude or longitude so get the standart colour.
		utility::ColourParser::FromString(this->lat_lon_lines_colour_param.Param<param::StringParam>()->Value(),
			r, g, b);

	}

	// Add the colour to the vertex.
	this->lat_lon_lines_colours.push_back(r);
	this->lat_lon_lines_colours.push_back(g);
	this->lat_lon_lines_colours.push_back(b);
}


/*
 * MapGenerator::processAOOutput
 */
uint MapGenerator::processAOOutput(const std::vector<float>* p_ao_vals, std::vector<Cut>& p_cuts,
		const float p_threshold, std::vector<bool>& p_tunnels) {
	// Determine if a face is shadowed.
	size_t face_cnt = this->faces.size() / 3;
	uint cnt;
	std::vector<bool> face_shadowed = std::vector<bool>(face_cnt, false);
	for (size_t i = 0; i < face_cnt; i++) {
		cnt = 0;
		for (size_t j = 0; j < 3; j++) {
			auto id = this->faces[i * 3 + j];
			if (p_ao_vals->at(id * 4) <= p_threshold) cnt++;
		}
		if (cnt >= 2) face_shadowed[i] = true;
	}

	// Set the group colours to a default value of gray (193,193,193)
	this->vertexColors_group = std::vector<float>(this->vertexColors.size(), 193.0f / 255.0f);
	auto groups = this->cuda_kernels->GroupFaces(face_shadowed, this->face_edge_offset, this->face_edge_offset_depth,
		this->group_colour_table, this->faces, this->vertexColors_group);
	this->identifyBorderEdges(face_shadowed, groups);

	// Find circles and mark tunnels.
	this->findCircles(groups);
	for (const auto& group : groups) {
		// We are only interested in shadowed groups.
		if (!group.state) continue;

		if (group.circles.size() >= 2) {
			// The group represents a tunnel so mark the faces.
			for (auto it = group.face_map.begin(); it != group.face_map.end(); it++) {
				p_tunnels[it->second] = true;
			}
		}
	}

	// Create cuts and tesselate triangle fan.
	uint tunnel_id = 0;
	uint vertex_id = static_cast<uint>(this->vertices.size() / 3);
	this->vertexColors_cuts = std::vector<float>(this->vertexColors.size(), 193.0f / 255.0f);
	for (const auto& group : groups) {
		if (!group.state || group.circles.size() < 2) continue;
		for (const auto& circle : group.circles) {
			p_cuts.push_back(this->createCut(false, tunnel_id, vertex_id, circle, p_tunnels));
		}
		tunnel_id++;
	}

	return tunnel_id;
}


/*
 * MapGenerator::processTopologyOutput
 */
void MapGenerator::processTopologyOutput(std::vector<Cut>& p_cuts, uint& p_tunnel_id,
		std::vector<bool>& p_tunnels, const std::vector<VoronoiVertex>& p_voronoi_vertices,
		const std::vector<VoronoiEdge>& p_voronoi_edges, const vislib::math::Cuboid<float>& bbox) {
	// Create the Octree of the current faces.
	vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_INFO, "Creating the Voronoi Cut Octree...");
	Octree voronoiOctree;
	voronoiOctree.CreateOctreeRootNode(this->faces_rebuild, make_float3(bbox.Right(), bbox.Top(), bbox.Front()),
		make_float3(bbox.Left(), bbox.Bottom(), bbox.Back()), make_float3(2.0f), this->vertices_rebuild);

	// Compute the AO value for every voronoi vertex. Remember the faces that where intersected
	// by the rays. Also mark the voronoi vertices that have an AO value higher than the
	// threshold value.
	std::vector<bool> voronoi_tunnel = std::vector<bool>(p_voronoi_vertices.size(), false);
	std::vector<std::vector<uint>> voro_faces = std::vector<std::vector<uint>>(p_voronoi_vertices.size());
	std::vector<std::pair<size_t, VoronoiVertex>> potential_vertices;
	potential_vertices.reserve(p_voronoi_vertices.size());

	// If we are in debug mode we have to use the CPU implementation of the ambient occlusion
	// beacuse the CUDA version destroys the GPU so that it turns of and on again every second.
#ifndef _DEBUG
	// Convert the Octree to the CUDA representation.
	std::vector<CudaOctreeNode> cuda_octree_nodes;
	std::vector<std::vector<uint>> cuda_node_faces;
	auto node_cnt = voronoiOctree.ConvertToCUDAOctree(cuda_octree_nodes, cuda_node_faces);

	// Compute all rays on a very coarse sampled sphere.
	std::vector<float3> ray_dirs;
	float3 center = make_float3(0.0f);
	for (float theta = 0.0f; theta < static_cast<float>(vislib::math::PI_DOUBLE); theta += 1.0f) {
		for (float phi = 0.0f; phi < static_cast<float>(vislib::math::PI_DOUBLE * 2.0); phi += 1.0f) {
			// Compute the next ray and push it to the list of rays if it is unique.
			float3 ray_dir;
			ray_dir.x = center.x + 1.0f * sin(theta) * cos(phi);
			ray_dir.y = center.y + 1.0f * cos(theta);
			ray_dir.z = center.z + 1.0f * sin(theta) * sin(phi);

			// Look if the ray is unique.
			auto it = std::find(ray_dirs.begin(), ray_dirs.end(), ray_dir);
			if (it == ray_dirs.end()) {
				ray_dirs.push_back(ray_dir);
			}
		}
	}

	// Compute the AO value using CUDA.
	this->cuda_kernels->ComputeVoronoiAO(cuda_octree_nodes, cuda_node_faces, this->faces_rebuild,
		ray_dirs, this->vertices_rebuild, p_voronoi_vertices, static_cast<uint>(node_cnt), voronoi_tunnel,
		voro_faces, potential_vertices);
#else
	// Compute all rays on a very coarse sampled sphere.
	std::vector<vec3f> ray_dirs;
	vec3f center = vec3f(0.0f, 0.0f, 0.0f);
	for (float theta = 0.0f; theta < static_cast<float>(vislib::math::PI_DOUBLE); theta += 1.0f) {
		for (float phi = 0.0f; phi < static_cast<float>(vislib::math::PI_DOUBLE * 2.0); phi += 1.0f) {
			// Compute the next ray and push it to the list of rays if it is unique.
			vec3f ray_dir;
			ray_dir.SetX(center.GetX() + 1.0f * sin(theta) * cos(phi));
			ray_dir.SetY(center.GetY() + 1.0f * cos(theta));
			ray_dir.SetZ(center.GetZ() + 1.0f * sin(theta) * sin(phi));

			// Look if the ray is unique.
			auto it = std::find(ray_dirs.begin(), ray_dirs.end(), ray_dir);
			if (it == ray_dirs.end()) {
				ray_dirs.push_back(ray_dir);
			}
		}
	}

#pragma omp parallel for
	for (int i = 0; i < p_voronoi_vertices.size(); i++) {
		// Initialise the ao value and the position of the voronoi vertex.
		float ao_val = 0.0f;
		vec3f origin = vec3f(static_cast<float>(p_voronoi_vertices[i].vertex.GetX()),
			static_cast<float>(p_voronoi_vertices[i].vertex.GetY()),
			static_cast<float>(p_voronoi_vertices[i].vertex.GetZ()));
		for (size_t j = 0; j < ray_dirs.size(); j++) {
			// Initialise the ray from the voronoi vertex and the current direction.
			Ray ray = Ray(ray_dirs[j], origin);

			// Intersect the Octree to find the first face we intersect and use one of it's vertex IDs.
			uint face_id;
			auto retval = voronoiOctree.IntersectOctree(this->faces_rebuild, ray, this->vertices_rebuild, face_id);
			if (retval != -1) {
				// There was an intersection with a face of the mesh, increase the AO sum and remember the face.
				ao_val++;
				voro_faces[i].push_back(face_id);
			}
		}
		ao_val /= static_cast<float>(ray_dirs.size());

		// Check if the voronoi vertex is on the surface.
		if (ao_val > 0.9f) {
			// The AO value is higher than the threshold so remember the voronoi vertex and set the visited flag
			// of the ID to false.
			potential_vertices.push_back(std::make_pair(i, p_voronoi_vertices[i]));
			voronoi_tunnel[i] = true;
		}
	}
#endif

	// Remove the edges from the whole voronoi graph and compute the start and end vertex offsets for the
	// new edges.
	std::vector<VoronoiEdge> tmp_voro_edges;
	std::vector<VoronoiEdge> tmp_voro_edges_reverse;
	tmp_voro_edges.reserve(p_voronoi_edges.size());
	for (size_t i = 0; i < p_voronoi_edges.size(); i++) {
		if (voronoi_tunnel[p_voronoi_edges[i].start_vertex] || voronoi_tunnel[p_voronoi_edges[i].end_vertex]) {
			tmp_voro_edges.emplace_back(p_voronoi_edges[i]);
		}
	}

	// Sort the edges ascendig to their start and end vertex ID.
	tmp_voro_edges_reverse = tmp_voro_edges;
	std::sort(tmp_voro_edges.begin(), tmp_voro_edges.end());
	std::sort(tmp_voro_edges_reverse.begin(), tmp_voro_edges_reverse.end(), [](const VoronoiEdge& a, const VoronoiEdge& b) {
		return a.end_vertex < b.end_vertex;
	});

	// Create the vertex edge offset from the start vertex ID of the edges.
	std::vector<size_t> start_vertex_offset = std::vector<size_t>(p_voronoi_vertices.size() + 1, tmp_voro_edges.size());
	size_t cur_vertex = tmp_voro_edges[0].start_vertex;
	start_vertex_offset[cur_vertex] = 0;
	for (size_t i = 1; i < tmp_voro_edges.size(); i++) {
		if (tmp_voro_edges[i].start_vertex != cur_vertex) {
			cur_vertex = tmp_voro_edges[i].start_vertex;
			start_vertex_offset[cur_vertex] = i;
		}
	}
	for (int i = static_cast<int>(start_vertex_offset.size() - 2); i >= 0; i--) {
		if (start_vertex_offset[i] == tmp_voro_edges.size()) {
			start_vertex_offset[i] = start_vertex_offset[i + 1];
		}
	}

	// Create the vertex edge offset from the end vertex ID of the edges.
	std::vector<size_t> end_vertex_offset = std::vector<size_t>(p_voronoi_vertices.size() + 1, tmp_voro_edges_reverse.size());
	cur_vertex = tmp_voro_edges_reverse[0].end_vertex;
	end_vertex_offset[cur_vertex] = 0;
	for (size_t i = 1; i < tmp_voro_edges_reverse.size(); i++) {
		if (tmp_voro_edges_reverse[i].end_vertex != cur_vertex) {
			cur_vertex = tmp_voro_edges_reverse[i].end_vertex;
			end_vertex_offset[cur_vertex] = i;
		}
	}
	for (int i = static_cast<int>(end_vertex_offset.size() - 2); i >= 0; i--) {
		if (end_vertex_offset[i] == tmp_voro_edges_reverse.size()) {
			end_vertex_offset[i] = end_vertex_offset[i + 1];
		}
	}

	// Group the voronoi vertices together using a DFS.
	std::vector<bool> visited = std::vector<bool>(p_voronoi_vertices.size(), false);
	std::vector<std::vector<uint>> groups;
	for (size_t i = 0; i < potential_vertices.size(); i++) {
		if (!visited[potential_vertices[i].first]) {
			groups.emplace_back();
			this->depthFirstSearch(potential_vertices[i].first, tmp_voro_edges, tmp_voro_edges_reverse,
				start_vertex_offset, end_vertex_offset, visited, groups.back());
		}
	}

	// Get the faces from the radius search in the Octree and add them to the faces from 
	// the AO intersection tests.
	double epsilon = 0.275;
#pragma omp parallel for
	for (int i = 0; i < potential_vertices.size(); i++) {
		// Initialise the query by adding an epsilon value to the radius of the voronoi vertex.
		std::vector<uint> res;
		vec4d querySphere = potential_vertices[i].second.vertex;
		querySphere.SetW(querySphere.GetW() + epsilon);

		// Get the faces from the Octree and add them to the faces from the AO intersections.
		voronoiOctree.RadiusSearch(this->faces_rebuild, querySphere, this->vertices_rebuild, res);
		for (size_t j = 0; j < res.size(); j++) {
			uint face_id = res[j];
			voro_faces[potential_vertices[i].first].push_back(face_id);
		}
	}

	// Create the groups based on the DFS. Add the faces from the voronoi vertices that belong
	// to the same group together. Also add every face that does not belong to a voronoi group
	// to the group with the ID 0.
	std::vector<FaceGroup> voronoi_groups;
	voronoi_groups.reserve(p_voronoi_vertices.size() + 1);
	std::vector<uint> face_group_ids = std::vector<uint>(this->faces_rebuild.size() / 3, 0);
	uint face_group_id = 1;
	voronoi_groups.emplace_back();
	voronoi_groups.back().state = false;

	// Add the faces to the voronoi groups.
	for (size_t i = 0; i < groups.size(); i++) {
		voronoi_groups.emplace_back();
		voronoi_groups.back().state = true;
		for (size_t j = 0; j < groups[i].size(); j++) {
			for (size_t k = 0; k < voro_faces[groups[i][j]].size(); k++) {
				if (face_group_ids[voro_faces[groups[i][j]][k]] == 0) {
					voronoi_groups.back().AddFace(voro_faces[groups[i][j]][k]);
					face_group_ids[voro_faces[groups[i][j]][k]] = face_group_id;
				}
			}
		}
		face_group_id++;
	}

	// Add the faces to the non voronoi group.
	for (size_t i = 0; i < face_group_ids.size(); i++) {
		if (face_group_ids[i] == 0) {
			voronoi_groups[0].AddFace(static_cast<uint>(i));
		}
	}

	// Add the enclosures of the groups to the surrounding group.
	std::vector<bool> marked_faces = std::vector<bool>(this->faces_rebuild.size() / 3, false);
	for (size_t group_id = 0; group_id < voronoi_groups.size(); group_id++) {
		// Find the enclosures by growing the faces and finding faces than can't be grown ofen
		// enough. These groups are enclosures and can be added to the surrounding group.
		std::vector<std::pair<int, std::map<uint, uint>>> enclosures;
		for (auto face_it = voronoi_groups[group_id].face_map.begin(); face_it != voronoi_groups[group_id].face_map.end(); face_it++) {
			// If the face was looked at before grow from this face and check how many faces
			// can be reached.
			if (!marked_faces[face_it->second]) {
				// Get the group of faces and check what other group surrounds it.
				std::map<uint, uint> full_enclosure;
				auto retval = this->findEnclosures(face_it->second, static_cast<uint>(group_id),
					full_enclosure, face_group_ids, voronoi_groups.size(), marked_faces, 10);

				// If the group is an enclosure then add its faces to the list.
				if (retval != -1) {
					enclosures.emplace_back(std::make_pair(retval, full_enclosure));
				}
			}
		}

		// Remove the faces from the current group and add them to the surrounding group.
		for (const auto& enclosure : enclosures) {
			for (auto it = enclosure.second.begin(); it != enclosure.second.end(); it++) {
				voronoi_groups[enclosure.first].AddFace(it->second);
				voronoi_groups[group_id].RemoveFace(it->second);
				face_group_ids[it->second] = enclosure.first;
			}
		}
	}

	// Check for faces of the non voronoi group to see if any enclosures have been missed.
	marked_faces.assign(marked_faces.size(), false);
	// Find the enclosures by growing the faces and finding faces than can't be grown ofen
	// enough. These groups are enclosures and can be added to the surrounding group.
	std::vector<std::pair<int, std::map<uint, uint>>> enclosures;
	for (auto face_it = voronoi_groups[0].face_map.begin(); face_it != voronoi_groups[0].face_map.end(); face_it++) {
		// If the face was looked at before grow from this face and check how many faces
		// can be reached.
		if (!marked_faces[face_it->second]) {
			// Get the group of faces and check what other group surrounds it.
			std::map<uint, uint> full_enclosure;
			auto retval = this->findEnclosures(face_it->second, 0,
				full_enclosure, face_group_ids, voronoi_groups.size(), marked_faces, 50);

			// If the group is an enclosure then add its faces to the list.
			if (retval != -1) {
				enclosures.emplace_back(std::make_pair(retval, full_enclosure));
			}
		}
	}

	// Remove the faces from the current group and add them to the surrounding group.
	for (const auto& enclosure : enclosures) {
		for (auto it = enclosure.second.begin(); it != enclosure.second.end(); it++) {
			voronoi_groups[enclosure.first].AddFace(it->second);
			voronoi_groups[0].RemoveFace(it->second);
			face_group_ids[it->second] = enclosure.first;
		}
	}

	// Find the border edges for every group.
	this->identifyBorderEdges(face_group_ids, voronoi_groups);

	// Find the circles for the faces of every group.
	this->findCircles(voronoi_groups);

	// Check the circle of every group, except the first one, if they are a valid circle.
	for (auto& group : voronoi_groups) {
		if (group.state) {
			if (group.circles.size() < 2) {
				// If the group has only one circle we can saftly remove the faces from the
				// group and add them to the non voronoi group.
				for (auto it = group.face_map.begin(); it != group.face_map.end(); it++) {
					voronoi_groups[0].AddFace(it->second);
					face_group_ids[it->second] = 0;
				}

				// Reset the group to an empty group.
				group.border_edges.clear();
				group.circles.clear();
				group.face_map.clear();

			} else {
				// The group has more than one circle so the circles need to be checked for
				// validity. Only groups with more than one valid circle can represent a
				// tunnel. 
				uint valid_circles = 0;
				group.valid_circles = std::vector<bool>(group.circles.size(), false);

				// Get the vertices from the border edges.
				std::vector<uint> border_vertices;
				border_vertices.reserve(group.border_edges.size() * 2);
				for (const auto& edge : group.border_edges) {
					// Look of the vertex was already added, if not add it.
					auto it = std::find(border_vertices.begin(), border_vertices.end(), edge.vertex_id_0);
					if (it == border_vertices.end()) {
						border_vertices.push_back(edge.vertex_id_0);
					}

					// Look of the vertex was already added, if not add it.
					it = std::find(border_vertices.begin(), border_vertices.end(), edge.vertex_id_1);
					if (it == border_vertices.end()) {
						border_vertices.push_back(edge.vertex_id_1);
					}
				}

				// Get the faces from the circles.
				for (size_t i = 0; i < group.circles.size(); i++) {
					// Initialise the circle faces.
					std::vector<uint> circle_faces;
					circle_faces.reserve(group.circles[i].size());

					// Loop over all vertices of the circle and add faces that contain
					// the vertex.
					for (const auto vertex_id : group.circles[i]) {
						for (const auto& edge : this->vertex_edge_offset[vertex_id]) {
							// Check the first face of the edge.
							auto it = group.face_map.find(edge.face_id_0);
							if (it != group.face_map.end()) {
								// Check if the fase was already added.
								auto it_find = std::find(circle_faces.begin(), circle_faces.end(), it->second);
								if (it_find == circle_faces.end()) {
									circle_faces.push_back(it->second);
								}
							}

							// Check the second face of the edge.
							it = group.face_map.find(edge.face_id_1);
							if (it != group.face_map.end()) {
								// Check if the fase was already added.
								auto it_find = std::find(circle_faces.begin(), circle_faces.end(), it->second);
								if (it_find == circle_faces.end()) {
									circle_faces.push_back(it->second);
								}
							}
						}
					}

					// Check if the circle is valid. If not add the faces to the non
					// voronoi group.
					std::map<uint, uint> circle_face_map;
					if (!this->growFaces(circle_faces, border_vertices, circle_face_map, 100)) {
						for (auto it = circle_face_map.begin(); it != circle_face_map.end(); it++) {
							voronoi_groups[0].AddFace(it->second);
							group.RemoveFace(it->second);
							face_group_ids[it->second] = 0;
						}

					} else {
						valid_circles++;
						group.valid_circles[i] = true;
					}
				}

				// Check for the number of valid cirlces. If the number is smaller than
				// two the group can be deleted.
				if (valid_circles < 2) {
					for (auto it = group.face_map.begin(); it != group.face_map.end(); it++) {
						voronoi_groups[0].AddFace(it->second);
						face_group_ids[it->second] = 0;
					}

					// Reset the group to an empty group.
					group.border_edges.clear();
					group.circles.clear();
					group.face_map.clear();
					group.valid_circles.clear();
				}
			}
		}
	}

	// Check for faces of the non voronoi group to see if any enclosures have been missed.
	marked_faces.assign(marked_faces.size(), false);
	// Find the enclosures by growing the faces and finding faces than can't be grown ofen
	// enough. These groups are enclosures and can be added to the surrounding group.
	enclosures.clear();
	for (auto face_it = voronoi_groups[0].face_map.begin(); face_it != voronoi_groups[0].face_map.end(); face_it++) {
		// If the face was looked at before grow from this face and check how many faces
		// can be reached.
		if (!marked_faces[face_it->second]) {
			// Get the group of faces and check what other group surrounds it.
			std::map<uint, uint> full_enclosure;
			auto retval = this->findEnclosures(face_it->second, 0,
				full_enclosure, face_group_ids, voronoi_groups.size(), marked_faces, 50);

			// If the group is an enclosure then add its faces to the list.
			if (retval != -1) {
				enclosures.emplace_back(std::make_pair(retval, full_enclosure));
			}
		}
	}

	// Remove the faces from the current group and add them to the surrounding group.
	for (const auto& enclosure : enclosures) {
		for (auto it = enclosure.second.begin(); it != enclosure.second.end(); it++) {
			voronoi_groups[enclosure.first].AddFace(it->second);
			voronoi_groups[0].RemoveFace(it->second);
			face_group_ids[it->second] = enclosure.first;
		}
	}
	
	// Add the colours to the voronoi colour vector and create the cuts.
	this->vertexColors_voronoi = std::vector<float>(this->vertexColors_rebuild.size(), 193.0f / 255.0f);
	int colour_table_size = static_cast<int>(this->group_colour_table.Count());
	uint vertex_id = static_cast<uint>(this->vertices_rebuild.size() / 3);
	for (size_t i = 1; i < voronoi_groups.size(); i++) {
		// Colour the faces in the vertexColors_voronoi vector.
		for (auto it = voronoi_groups[i].face_map.begin(); it != voronoi_groups[i].face_map.end(); it++) {
			// Determine the colour.
			uint face_id = it->second;
			this->vertexColors_voronoi[this->faces_rebuild[face_id * 3 + 0] * 3 + 0] = this->group_colour_table[i % colour_table_size].GetX();
			this->vertexColors_voronoi[this->faces_rebuild[face_id * 3 + 0] * 3 + 1] = this->group_colour_table[i % colour_table_size].GetY();
			this->vertexColors_voronoi[this->faces_rebuild[face_id * 3 + 0] * 3 + 2] = this->group_colour_table[i % colour_table_size].GetZ();
			this->vertexColors_voronoi[this->faces_rebuild[face_id * 3 + 1] * 3 + 0] = this->group_colour_table[i % colour_table_size].GetX();
			this->vertexColors_voronoi[this->faces_rebuild[face_id * 3 + 1] * 3 + 1] = this->group_colour_table[i % colour_table_size].GetY();
			this->vertexColors_voronoi[this->faces_rebuild[face_id * 3 + 1] * 3 + 2] = this->group_colour_table[i % colour_table_size].GetZ();
			this->vertexColors_voronoi[this->faces_rebuild[face_id * 3 + 2] * 3 + 0] = this->group_colour_table[i % colour_table_size].GetX();
			this->vertexColors_voronoi[this->faces_rebuild[face_id * 3 + 2] * 3 + 1] = this->group_colour_table[i % colour_table_size].GetY();
			this->vertexColors_voronoi[this->faces_rebuild[face_id * 3 + 2] * 3 + 2] = this->group_colour_table[i % colour_table_size].GetZ();

			// Mark the face as a tunnel face.
			p_tunnels[face_id] = true;
		}

		// Create the cuts for all valid groups and circles.
		if (voronoi_groups[i].circles.size() >= 2) {
			size_t idx = 0;
			for (const auto& circle : voronoi_groups[i].circles) {
				if (voronoi_groups[i].valid_circles[idx++]) {
					p_cuts.emplace_back(this->createCut(true, p_tunnel_id, vertex_id, circle, p_tunnels));
				}
			}
			p_tunnel_id++;
		}
	}
}


/*
 * MapGenerator::rebuildSurface
 */
void MapGenerator::rebuildSurface(const std::vector<Cut>& p_cuts, const bool p_second_rebuild,
		const std::vector<bool>& p_tunnels) {
	// Create vectors for the new surface.
	std::vector<float> new_colours, new_normals, new_vertices;
	std::vector<uint> new_faces;

	// Create vectors to the old surface parameters.
	float* old_colours_ptr;
	float* old_normals_ptr;
	float* old_vertices_ptr;
	uint* old_faces_ptr;

	// Create the vertex ID offset array for the old vertices.
	std::vector<int> vertex_offset;
	int offset_cnt;

	// Create sizes for old surface paramters.
	size_t old_colours_cnt, old_face_cnt, old_normals_cnt, old_vertices_cnt;
	
	// Create sizes for parameters introduced with the cuts.
	size_t cut_colour_cnt, cut_face_cnt, cut_normal_cnt, cut_vertex_cnt;

	// Determine the old surface by looking at p_second_rebuild.
	if (!p_second_rebuild) {
		// Set the data.
		old_colours_ptr = this->vertexColors.data();
		old_faces_ptr = this->faces.data();
		old_normals_ptr = this->normals.data();
		old_vertices_ptr = this->vertices.data();
		// Set the sizes.
		old_colours_cnt = this->vertexColors.size() / 3;
		old_face_cnt = this->faces.size() / 3;
		old_normals_cnt = this->normals.size() / 3;
		old_vertices_cnt = this->vertices.size() / 3;

	} else {
		// Set the data.
		old_colours_ptr = this->vertexColors_rebuild.data();
		old_faces_ptr = this->faces_rebuild.data();
		old_normals_ptr = this->normals_rebuild.data();
		old_vertices_ptr = this->vertices_rebuild.data();
		// Set the sizes.
		old_colours_cnt = this->vertexColors_rebuild.size() / 3;
		old_face_cnt = this->faces_rebuild.size() / 3;
		old_normals_cnt = this->normals_rebuild.size() / 3;
		old_vertices_cnt = this->vertices_rebuild.size() / 3;
	}
	this->vertexColors_tunnel = std::vector<float>(old_colours_cnt * 4, 0.3f);
	this->vertexColors_tunnel.shrink_to_fit();
	// Set the offset.
	vertex_offset = std::vector<int>(old_vertices_cnt, -1);
	vertex_offset.shrink_to_fit();
	offset_cnt = 0;

	cut_colour_cnt = cut_face_cnt = cut_normal_cnt = cut_vertex_cnt = 0;
	for (const auto& cut : p_cuts) {
		cut_colour_cnt += cut.colours.size();
		cut_face_cnt += cut.faces.size();
		cut_normal_cnt += cut.normals.size();
		cut_vertex_cnt += cut.vertices.size();
	}
	
	// Reserve space for the new surface parameters.
	new_colours.reserve(old_colours_cnt * 3 + cut_colour_cnt);
	new_faces.reserve(old_face_cnt * 3 + cut_face_cnt);
	new_normals.reserve(old_normals_cnt * 3 + cut_normal_cnt);
	new_vertices.reserve(old_vertices_cnt * 3 + cut_vertex_cnt);

	// Copy the old faces into the new ones except for the faces that need to be removed.
	for (size_t i = 0; i < old_face_cnt; i++) {
		if (!p_tunnels[i]) {
			// The face is not removed so copy it.
			// Copy the vertices and assign them new IDs.
			for (size_t j = 0; j < 3; j++) {
				auto id = old_faces_ptr[i * 3 + j];
				// Check if the vertex was already added to the new vertices.
				if (vertex_offset[id] == -1) {
					// Add the new vertex ID to the face and update the vertex offsets.
					new_faces.push_back(offset_cnt);
					vertex_offset[id] = offset_cnt++;
					// Copy the colour.
					new_colours.push_back(old_colours_ptr[id * 3]);
					new_colours.push_back(old_colours_ptr[id * 3 + 1]);
					new_colours.push_back(old_colours_ptr[id * 3 + 2]);
					// Copy the normal.
					new_normals.push_back(old_normals_ptr[id * 3]);
					new_normals.push_back(old_normals_ptr[id * 3 + 1]);
					new_normals.push_back(old_normals_ptr[id * 3 + 2]);
					// Copy the vertex position.
					new_vertices.push_back(old_vertices_ptr[id * 3]);
					new_vertices.push_back(old_vertices_ptr[id * 3 + 1]);
					new_vertices.push_back(old_vertices_ptr[id * 3 + 2]);
					// Copy the colour for the tunnel rendering.
					this->vertexColors_tunnel[id * 4] = old_colours_ptr[id * 3];
					this->vertexColors_tunnel[id * 4 + 1] = old_colours_ptr[id * 3 + 1];
					this->vertexColors_tunnel[id * 4 + 2] = old_colours_ptr[id * 3 + 2];

				} else {
					// The vertex was already added, use the updated vertex ID.
					new_faces.push_back(vertex_offset[id]);
				}
			}

		} else {
			// The face is removed but we need the colour for the tunnel rendering.
			for (size_t j = 0; j < 3; j++) {
				auto id = old_faces_ptr[i * 3 + j];
				this->tunnel_faces.push_back(id);
				// Copy the colour for the tunnel rendering.
				this->vertexColors_tunnel[id * 4] = old_colours_ptr[id * 3];
				this->vertexColors_tunnel[id * 4 + 1] = old_colours_ptr[id * 3 + 1];
				this->vertexColors_tunnel[id * 4 + 2] = old_colours_ptr[id * 3 + 2];
			}
		}
	}

	// Update the vertex IDs of all vertices that where added.
	for (auto& v_id : this->vertices_added) {
		if (vertex_offset[v_id] != -1) {
			v_id = vertex_offset[v_id];
		}
	}

	// Add the colours, faces, normals and vertices from the cuts.
	for (const auto& cut : p_cuts) {
		cut_face_cnt = cut.faces.size() / 3;
		// Set the maximum vertex edge offset depth.
		if ((cut_face_cnt * 4) > this->vertex_edge_offset_max_depth) {
			this->vertex_edge_offset_max_depth = cut_face_cnt * 4;
		}
		for (size_t i = 0; i < cut_face_cnt; i++) {
			for (size_t j = 0; j < 3; j++) {
				// Get the vertex ID and check if it is an old vertex.
				auto id = cut.faces[i * 3 + j];
				if (id < static_cast<uint>(vertex_offset.size())) {
					// It is an old vertex check if it was deleted and if so add it again.
					if (vertex_offset[id] == -1) {
						// The vertex was deleted add it again with a new ID.
						new_faces.push_back(offset_cnt);
						// Update the offset vector.
						vertex_offset[id] = offset_cnt++;
						// Copy the colour.
						new_colours.push_back(old_colours_ptr[id * 3]);
						new_colours.push_back(old_colours_ptr[id * 3 + 1]);
						new_colours.push_back(old_colours_ptr[id * 3 + 2]);
						// Copy the normal.
						new_normals.push_back(old_normals_ptr[id * 3]);
						new_normals.push_back(old_normals_ptr[id * 3 + 1]);
						new_normals.push_back(old_normals_ptr[id * 3 + 2]);
						// Copy the vertex position.
						new_vertices.push_back(old_vertices_ptr[id * 3]);
						new_vertices.push_back(old_vertices_ptr[id * 3 + 1]);
						new_vertices.push_back(old_vertices_ptr[id * 3 + 2]);

					} else {
						// The vertex was not delete so add the the new ID to the faces.
						new_faces.push_back(vertex_offset[id]);
					}

				} else {
					// The vertex is not an old vertex so add it.
					vertex_offset.push_back(offset_cnt++);
					new_faces.push_back(vertex_offset.back());
					this->vertices_added.push_back(vertex_offset.back());
					this->vertices_added_tunnel_id.push_back(cut.tunnel_id);
					uint cut_id = static_cast<uint>(cut.vertices.size() / 3) - 1;
					// Copy the colour.
					new_colours.push_back(cut.colours[cut_id * 3]);
					new_colours.push_back(cut.colours[cut_id * 3 + 1]);
					new_colours.push_back(cut.colours[cut_id * 3 + 2]);
					// Copy the normal.
					new_normals.push_back(cut.normals[cut_id * 3]);
					new_normals.push_back(cut.normals[cut_id * 3 + 1]);
					new_normals.push_back(cut.normals[cut_id * 3 + 2]);
					// Copy the vertex position.
					new_vertices.push_back(cut.vertices[cut_id * 3]);
					new_vertices.push_back(cut.vertices[cut_id * 3 + 1]);
					new_vertices.push_back(cut.vertices[cut_id * 3 + 2]);
				}
			}
		}
	}

	// Copy the new values to the rebuild vectors.
	this->faces_rebuild = new_faces;
	this->faces_rebuild.shrink_to_fit();
	this->normals_rebuild = new_normals;
	this->normals_rebuild.shrink_to_fit();
	this->vertexColors_rebuild = new_colours;
	this->vertexColors_rebuild.shrink_to_fit();
	this->vertices_rebuild = new_vertices;
	this->vertices_rebuild.shrink_to_fit();
	
	// Delete temporary vectors.
	new_colours.erase(new_colours.begin(), new_colours.end());
	new_colours.clear();
	new_faces.erase(new_faces.begin(), new_faces.end());
	new_faces.clear();
	new_normals.erase(new_normals.begin(), new_normals.end());
	new_normals.clear();
	new_vertices.erase(new_vertices.begin(), new_vertices.end());
	new_vertices.clear();

	// Reset the offsets.
	this->face_edge_offset.erase(this->face_edge_offset.begin(), this->face_edge_offset.end());
	this->face_edge_offset.clear();
	this->face_edge_offset.resize(this->faces_rebuild.size() / 3);
	this->face_edge_offset.shrink_to_fit();
	for (auto& offset : this->face_edge_offset) {
		// There can't be more than 6 edges per face.
		offset.resize(6);
	}

	this->face_edge_offset_depth.erase(this->face_edge_offset_depth.begin(), this->face_edge_offset_depth.end());
	this->face_edge_offset_depth.clear();
	this->face_edge_offset_depth.resize(this->faces_rebuild.size() / 3 + 1);
	this->face_edge_offset.shrink_to_fit();

	this->vertex_edge_offset.erase(this->vertex_edge_offset.begin(), this->vertex_edge_offset.end());
	this->vertex_edge_offset.clear();
	this->vertex_edge_offset.resize(this->vertices_rebuild.size() / 3);
	this->vertex_edge_offset.shrink_to_fit();
	for (auto& offset : this->vertex_edge_offset) {
		// We set the maximum number of edges for old vertices to 30 and only
		// for new vertices we update the number to the maximum that we found.
		offset.resize(30);
	}
	for (const auto v_id : this->vertices_added) {
		this->vertex_edge_offset[v_id].resize(this->vertex_edge_offset_max_depth);
	}

	this->vertex_edge_offset_depth.erase(this->vertex_edge_offset_depth.begin(), this->vertex_edge_offset_depth.end());
	this->vertex_edge_offset_depth.clear();
	this->vertex_edge_offset_depth.resize(this->vertices_rebuild.size() / 3 + 1);
	this->vertex_edge_offset_depth.shrink_to_fit();
}


/*
 * MapGenerator::Render
 */
bool MapGenerator::Render(Call& call) {
	// Check if we need to reload the shaders.
	bool shaderReloaded = false;
	if (this->shaderReloadButtonParam.IsDirty()) {
		this->aoCalculator.loadShaders(this->GetCoreInstance());
		this->initialiseMapShader(true);
		this->shaderReloadButtonParam.ResetDirty();
		shaderReloaded = true;
	}

    MeshMode meshMode = (MeshMode)this->out_mesh_selection_slot.Param<param::EnumParam>()->Value();
    if (this->out_mesh_selection_slot.IsDirty()) {
        store_new_mesh = true;
        this->out_mesh_selection_slot.ResetDirty();
    }

	// Set up the calls for the CallTriMeshData call and the MolecularDataCall call.
	view::CallRender3D *cr3d = dynamic_cast<view::CallRender3D*>(&call);
	if (cr3d == nullptr) return false;
	CallTriMeshData *ctmd = this->meshDataSlot.CallAs<CallTriMeshData>();
	if (ctmd == nullptr) return false;
	CallTriMeshData *cctmd = this->meshDataSlotWithCap.CallAs<CallTriMeshData>();
	MolecularDataCall *mdc = this->proteinDataSlot.CallAs<MolecularDataCall>();
	if (mdc == nullptr) return false;

	// Set the frame for the MolecularDataCall .
	mdc->SetFrameID(static_cast<uint>(cr3d->Time()));
	if (!(*mdc)(1)) return false;

	// Set the frame for the CallTriMeshData and get the bounding box.
	ctmd->SetFrameID(static_cast<uint>(cr3d->Time()));
	if (!(*ctmd)(1)) return false;
	float scale = ctmd->AccessBoundingBoxes().ClipBox().LongestEdge();
	if (scale > 0.0f) {
		scale = 2.0f / scale;
	} else {
		scale = 1.0f;
	}
	::glScalef(scale, scale, scale);

	// Get the data from the MolecularDataCall.
	mdc->SetFrameID(static_cast<uint>(cr3d->Time()));
	if (!(*mdc)(0)) return false;

	// Get the data from the CallTriMeshData.
	ctmd->SetFrameID(static_cast<uint>(cr3d->Time()));
	if (!(*ctmd)(0)) return false;

	// Set up OpenGL.
	glEnable(GL_DEPTH_TEST);
	if (this->lighting.Param<param::BoolParam>()->Value()) {
		glEnable(GL_LIGHTING);

	} else {
		glDisable(GL_LIGHTING);
	}
	if (this->blending.Param<param::BoolParam>()->Value()) {
		glEnable(GL_BLEND);
		glDisable(GL_DEPTH_TEST);

	} else {
		glDisable(GL_BLEND);
	}
	glEnableClientState(GL_VERTEX_ARRAY);
	glDisableClientState(GL_NORMAL_ARRAY);
	glDisableClientState(GL_COLOR_ARRAY);
	glDisableClientState(GL_TEXTURE_COORD_ARRAY);
	glEnable(GL_COLOR_MATERIAL);
	glEnable(GL_TEXTURE_2D);
	glBindTexture(GL_TEXTURE_2D, 0);
	glLightModeli(GL_LIGHT_MODEL_TWO_SIDE, GL_TRUE);
	glEnable(GL_NORMALIZE);

	GLint cfm;
	glGetIntegerv(GL_CULL_FACE_MODE, &cfm);
	GLint pm[2];
	glGetIntegerv(GL_POLYGON_MODE, pm);
	GLint twr;
	glGetIntegerv(GL_FRONT_FACE, &twr);

	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	glDisable(GL_CULL_FACE);

	// Has the input data changed or the shader been reloaded?
	if (this->lastDataHash != ctmd->DataHash() || this->computeButton.IsDirty() || shaderReloaded || this->voronoiNeeded) {
		bool justReload = shaderReloaded && this->lastDataHash == ctmd->DataHash() && !this->computeButton.IsDirty();
		this->lastDataHash = ctmd->DataHash();
		this->computeButton.ResetDirty();
		this->voronoiNeeded = false;

		// Initialise computed flags.
		this->computed_map = false;
		this->computed_sphere = false;

        this->store_new_mesh = true;

		// Reset the latitude and longitude lines.
		glDeleteBuffers(1, &this->lat_lon_lines_vbo);
		this->lat_lon_lines_vbo = 0;

		// Get the new name of the input PDB and set the screenshot path.
        auto pdb_name =  getNameOfPDB(*mdc);
        if (!pdb_name.empty()) {
            pdb_name = splitString(pdb_name, '\\').back();
        }
        vislib::TString prev_file_path;
        if (this->store_png_path.IsDirty()) {
            prev_file_path = this->store_png_path.Param<param::FilePathParam>()->Value();
            this->store_png_path.ResetDirty();
        }
		prev_file_path.Append(A2T(pdb_name.c_str()));
		prev_file_path.Append(_T(".png"));
		this->store_png_path.Param<param::FilePathParam>()->SetValue(prev_file_path, false);

		// Get the colour tables.
		Color::ReadColorTableFromFile(cut_colour_param.Param<param::FilePathParam>()->Value(),
			cut_colour_table);
		Color::ReadColorTableFromFile(group_colour_param.Param<param::FilePathParam>()->Value(),
			group_colour_table);

		// Get the bounding box, the view direction and the up direction of the camera.
		auto bbox = cr3d->AccessBoundingBoxes().ObjectSpaceBBox();
		auto eye_dir = cr3d->GetCameraParameters()->EyeDirection();
		auto up_dir = cr3d->GetCameraParameters()->EyeUpVector();

		if (ctmd->Count() > 0 && ctmd->Objects()[0].GetVertexCount() > 0 ) {
			// Create local copy of the mesh.
			vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_INFO, "Copying the mesh...");
			if (!fillLocalMesh(ctmd->Objects()[0])) {
				vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR, "The mesh data is malformed!");
				return false;
			}

			// Create mesh topology information.
			vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_INFO, "Creating the topology information...");
			if (!this->cuda_kernels->CreateMeshTopology(this->faces, this->vertex_edge_offset,
					this->face_edge_offset, this->vertex_edge_offset_depth, this->face_edge_offset_depth)) {
				vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
					"Unable to create the Mesh Topology!"
					"\nPlease contact the developer to fix this.\n");
				return false;
			}

			// Check if we have a genus > 0.
			uint genus;
			if (isGenusN(static_cast<uint>(this->vertices.size() / 3), static_cast<uint>(this->faces.size() / 3), genus)) {
				vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_INFO, "The mesh has genus %d.", genus);

				// Check if we want to use AO (default is false, i.e. we do not want to use AO).
				uint tunnel_id = 0;
				if (this->aoActive.Param<param::BoolParam>()->Value()) {
					this->aoCalculator.clearStoredShadowData();
					AmbientOcclusionCalculator::AOSettings settings;
					settings.angleFactor = this->aoAngleFactorParam.Param<param::FloatParam>()->Value();
					settings.evalFactor = this->aoEvalParam.Param<param::FloatParam>()->Value();
					settings.falloffFactor = this->aoFalloffParam.Param<param::FloatParam>()->Value();
					settings.genFac = this->aoGenFactorParam.Param<param::FloatParam>()->Value();
					settings.maxDist = this->aoMaxDistSample.Param<param::FloatParam>()->Value();
					settings.minDist = this->aoMinDistSample.Param<param::FloatParam>()->Value();
					settings.numSampleDirections = this->aoNumSampleDirectionsParam.Param<param::IntParam>()->Value();
					settings.scaling = this->aoScalingFactorParam.Param<param::FloatParam>()->Value();
					settings.volSizeX = this->aoVolSizeXParam.Param<param::IntParam>()->Value();
					settings.volSizeY = this->aoVolSizeYParam.Param<param::IntParam>()->Value();
					settings.volSizeZ = this->aoVolSizeZParam.Param<param::IntParam>()->Value();
					this->aoCalculator.initilialize(instance(), &this->vertices, &this->normals, mdc);
					auto result = this->aoCalculator.calculateVertexShadows(settings);
					if (result != nullptr) {
						// Process the output to identify tunnels and create the cuts.
						vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_INFO, "Processing the AO output...");
						std::vector<Cut> cuts;
						std::vector<bool> tunnels = std::vector<bool>(this->faces.size() / 3, false);
						tunnel_id = processAOOutput(result, cuts, this->aoThresholdParam.Param<param::FloatParam>()->Value(),
							tunnels);

						// Rebuild the local copy with the new cuts and the tunnels removed.
						vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_INFO, "Rebuilding the surface...");
						rebuildSurface(cuts, false, tunnels);

						// Create mesh topology information after the cuts have been placed.
						vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_INFO, "Creating the topology information...");
						if (!this->cuda_kernels->CreateMeshTopology(this->faces_rebuild, this->vertex_edge_offset, 
								this->face_edge_offset, this->vertex_edge_offset_depth, 
								this->face_edge_offset_depth)) {
							vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
								"Unable to create the Mesh Topology!"
								"\nPlease contact the developer to fix this.\n");
							return false;
						}

					} else {
						vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
							"Unable to compute the Ambient Occlusion!"
							"\nPlease contact the developer to fix this.\n");
						return false;
					}

				} else {
					vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_INFO, "Not computing the Ambient Occlusion.");

					// The rebuild mesh is equal to the local copy because AO was not used.
					this->faces_rebuild = this->faces;
					this->normals_rebuild = this->normals;
					this->vertexColors_rebuild = this->vertexColors;
					this->vertices_rebuild = this->vertices;

					// Copy the colour for the tunnel rendering.
					auto colour_cnt = this->vertexColors_rebuild.size() / 3;
					this->vertexColors_tunnel = std::vector<float>(colour_cnt * 4, 0.3f);
					for (size_t i = 0; i < colour_cnt; i++) {
						this->vertexColors_tunnel[i * 4] = this->vertexColors_rebuild[i * 3];
						this->vertexColors_tunnel[i * 4 + 1] = this->vertexColors_rebuild[i * 3 + 1];
						this->vertexColors_tunnel[i * 4 + 2] = this->vertexColors_rebuild[i * 3 + 2];
					}

					// Set the group colours to a default value of gray (193,193,193)
					this->vertexColors_group = std::vector<float>(this->vertexColors.size(), 193.0f / 255.0f);
					this->vertexColors_cuts = std::vector<float>(this->vertexColors.size(), 193.0f / 255.0f);
				}

				// Only perfrom the topology based algorithm if we have a mesh of genus n after the AO.
				auto vertex_cnt = static_cast<uint>(this->vertices_rebuild.size() / 3);
				auto face_cnt = static_cast<uint>(this->faces_rebuild.size() / 3);
				if (isGenusN(vertex_cnt, face_cnt, genus)) {
					vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_INFO, "The mesh has genus %d.", genus);
					// Compute the voronoi vertices and edges.
					vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_INFO, "Computing the Voronoi diagram...");
					std::vector<VoronoiVertex> new_voronoi_vertices;
					std::vector<VoronoiEdge> new_voronoi_edges;
					if (!this->voronoiCalc.Update(mdc, new_voronoi_vertices, new_voronoi_edges, 
							this->probeRadiusSlot.Param<param::FloatParam>()->Value())) {
						vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR, 
							"Unable to compute/update the Voronoi Diagram!"
							"\nPlease contact the developer to fix this.\n");
						return false;
					}

					// Process the output of the topology based algorithm.
					vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_INFO, "Processing the topology output...");
					std::vector<Cut> cuts;
					std::vector<bool> tunnels = std::vector<bool>(this->faces_rebuild.size() / 3, false);
					this->processTopologyOutput(cuts, tunnel_id, tunnels, new_voronoi_vertices, new_voronoi_edges, bbox);

					// Rebuild the local copy with the new cuts and the tunnels removed.
					vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_INFO, "Rebuilding the surface...");
					rebuildSurface(cuts, true, tunnels);

					// Create mesh topology information after the last cuts have been placed.
					vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_INFO, "Creating the topology information...");
					if (!this->cuda_kernels->CreateMeshTopology(this->faces_rebuild, this->vertex_edge_offset, 
							this->face_edge_offset, this->vertex_edge_offset_depth, 
							this->face_edge_offset_depth)) {
						vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
							"Unable to create the Mesh Topology!"
							"\nPlease contact the developer to fix this.\n");
						return false;
					}
				}

			} else {
				// The rebuild mesh is equal to the local copy.
				vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_INFO, "The mesh has genus %d.", 0);
				this->faces_rebuild = this->faces;
				this->normals_rebuild = this->normals;
				this->vertexColors_rebuild = this->vertexColors;
				this->vertices_rebuild = this->vertices;

				// Copy the colour for the tunnel rendering.
				auto colour_cnt = this->vertexColors_rebuild.size() / 3;
				this->vertexColors_tunnel = std::vector<float>(colour_cnt * 4, 0.3f);
				for (size_t i = 0; i < colour_cnt; i++) {
					this->vertexColors_tunnel[i * 4] = this->vertexColors_rebuild[i * 3];
					this->vertexColors_tunnel[i * 4 + 1] = this->vertexColors_rebuild[i * 3 + 1];
					this->vertexColors_tunnel[i * 4 + 2] = this->vertexColors_rebuild[i * 3 + 2];
				}

				// Set the group colours to a default value of gray (193,193,193)
				this->vertexColors_group = std::vector<float>(this->vertexColors.size(), 193.0f / 255.0f);
				this->vertexColors_cuts = std::vector<float>(this->vertexColors.size(), 193.0f / 255.0f);
				this->vertexColors_voronoi = std::vector<float>(this->vertexColors.size(), 193.0f / 255.0f);
			}

			// Print the genus of the mesh after the voronoi tunnel detection is finished.
			auto vertex_cnt = static_cast<uint>(this->vertices_rebuild.size() / 3);
			auto face_cnt = static_cast<uint>(this->faces_rebuild.size() / 3);
			isGenusN(vertex_cnt, face_cnt, genus);
			vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_INFO, "The mesh has genus %d.", genus);

			// Create Octree.
			vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_INFO, "Creating the Octree...");
			this->octree.CreateOctreeRootNode(this->faces_rebuild, make_float3(bbox.Right(), bbox.Top(), bbox.Front()),
				make_float3(bbox.Left(), bbox.Bottom(), bbox.Back()), make_float3(2.0f), this->vertices_rebuild);

			// Color the selected binding site
			if (this->bindingSiteColoring.Param<param::BoolParam>()->Value()) {
				float radius = this->bindingSiteRadius.Param<param::FloatParam>()->Value();
				auto bsColorString = this->bindingSiteColor.Param<param::StringParam>()->Value();
				float radiusOffset = this->bindingSiteRadiusOffset.Param<param::FloatParam>()->Value();
				bool ignoreRadius = this->bindingSiteIgnoreRadius.Param<param::BoolParam>()->Value();
				vec3f bsColor;
				utility::ColourParser::FromString(bsColorString, 3, bsColor.PeekComponents());

				protein_calls::BindingSiteCall * bs = this->zeBindingSiteSlot.CallAs<protein_calls::BindingSiteCall>();
				
				if (bs != nullptr) {
					(*bs)(protein_calls::BindingSiteCall::CallForGetData);
					if (!this->colourBindingSite(bs, bsColor, mdc, radius, radiusOffset, ignoreRadius)) {
						vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
							"Unable to create the binding site sphere!"
							"\nPlease contact the developer to fix this.\n");
						return false;
					}
				}
			}

			// Color the vertices that are shadowed by the cap.
			if (cctmd != nullptr) {
				protein_calls::BindingSiteCall* bs = this->zeBindingSiteSlot.CallAs<protein_calls::BindingSiteCall>();
				if (bs != nullptr) {
					(*bs)(protein_calls::BindingSiteCall::CallForGetData);
					if (!this->capColouring(cctmd, cr3d, bs)) {
						vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
							"Unable to show the shadow of the cap!"
							"\nPlease contact the developer to fix this.\n");
						return false;
					}
				}
			}

			// Create sphere.
			vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_INFO, "Creating the sphere...");
			if (!createSphere(cr3d->GetCameraParameters()->EyeDirection(), cr3d->GetCameraParameters()->EyeUpVector())) {
				return false;
			}
			this->computed_sphere = true;

			// Initialise map shader
			vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_INFO, "Initialising the map shader...");
			if (!initialiseMapShader()) return false;
			this->computed_map = true;

		} else {
			vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_WARN, "The mesh input data is empty!");
			return false;
		}
	}

	// Update the voronoi diagram filtering when the probe radius changes
	if (this->probeRadiusSlot.IsDirty()) {
		this->probeRadiusSlot.ResetDirty();
		this->voronoiNeeded = true;
	}

	// Do we want to save the map as an image?
	if (this->store_png_button.IsDirty() && this->computed_map) {
		this->store_png_button.ResetDirty();
		
		// Reset the fbo size if necessary.
		if (!this->store_png_fbo.IsValid()) {
			this->store_png_fbo.Release();
			// The size of the map is determined by the radius of the bounding sphere. Since we can't just
			// create an image with a floating point width or an odd width we need to round up or down 
			// based on the radius. The base width is 1500 and we add 10 * radius of the sphere to it. Then
			// the width needs to be rounded to the closest even number. Since that does not always work,
			// e.g. a width of 1738 doesn't produce a correct png image, the rounding is changed to the 
			// nearest even 100. The 1738 then becomes 1700. The height is the half of the width, Hobo-Dyer
			// (aspect ratio of 2.0).
			uint width = 1500 + static_cast<uint>(this->sphere_data.GetW() * 10.0f);
			uint diff = width - ((width / 100) * 100);
			if (diff < 50) width = (width / 100) * 100;
			else width = ((width / 100) * 100) + 100;
			uint height = width / 2;
			this->store_png_fbo.Create(width, height);
		}

		// Render the map to the fbo and write the content to a file.
		this->store_png_fbo.Enable();
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		drawMap();
		if (this->lat_lon_lines_param.Param<param::BoolParam>()->Value()) {
			uint numLat = static_cast<uint>(this->lat_lines_count_param.Param<param::IntParam>()->Value());
			uint numLon = static_cast<uint>(this->lon_lines_count_param.Param<param::IntParam>()->Value());
			this->renderLatLonLines(numLat, numLon, 120, 120, true);
		}

		if (this->store_png_font.Initialise()) {
			// Generate the text to be rendered based on the length of the equator in Angstr�m.
			vislib::StringA text = "Equator length: ";
			float equator_length = 2.0f * static_cast<float>(vislib::math::PI_DOUBLE) * 
				this->sphere_data.GetW();
			text.Append(std::to_string(equator_length).c_str());
			// Note:
			// The vislib font renderer is unable to render the Angst�rm symbol. Therefore we render a
			// simple A instead of the symbol. If at any time in the future the vislib is able to render
			// the symbol simply change the A with the Anstr�m symbol and all is well.
			//char32_t angstrom = U'\U000000c5';
			//text.Append(angstrom);
			text.Append("A");
			
			// Disable the depth test, culling, texture and ligthing
			glDisable(GL_DEPTH_TEST);
			glDisable(GL_CULL_FACE);
			glDisable(GL_TEXTURE_2D);
			glDisable(GL_LIGHTING);

			// Set the 2D orthographic projection.
			glMatrixMode(GL_PROJECTION);
			glLoadIdentity();
			glOrtho(0, this->store_png_fbo.GetWidth(), this->store_png_fbo.GetHeight(), 0, -1.0, 1.0);
			
			// Reset the modelview matrix
			glMatrixMode(GL_MODELVIEW);
			glLoadIdentity();

			// Render the text.
			glColor3f(1.0f, 0.0f, 0.0f);
			float font_size = static_cast<float>(this->store_png_fbo.GetWidth() - 1200) / 100.0f * 3.0f;
			float text_len = this->store_png_font.LineWidth(font_size, text) + font_size;
			float x = this->store_png_fbo.GetWidth() - text_len;
			float y = this->store_png_fbo.GetHeight() - font_size;
			this->store_png_font.DrawString(x, y, text_len, -1.0f, font_size, false, text,
				vislib::graphics::AbstractFont::ALIGN_LEFT_MIDDLE);
		}
		this->store_png_fbo.Disable();
		this->store_png_fbo.DrawColourTexture(0, GL_LINEAR, GL_LINEAR, 0.9f);
		this->store_png_data.SetCount(this->store_png_fbo.GetWidth() * this->store_png_fbo.GetHeight() * 3);
		this->store_png_fbo.GetColourTexture(&this->store_png_data[0], 0, GL_RGB, GL_UNSIGNED_BYTE);
		this->store_png_image.Image() = new vislib::graphics::BitmapImage(this->store_png_fbo.GetWidth(),
			this->store_png_fbo.GetHeight(), 3U, vislib::graphics::BitmapImage::CHANNELTYPE_BYTE,
			static_cast<const void*>(&this->store_png_data[0]));
		this->store_png_image.Image()->FlipVertical();
		if (this->store_png_image.Save(T2A(this->store_png_path.Param<param::FilePathParam>()->Value()))) {
			vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_INFO,
				"%s: Stored molecular surface map to file: %s",
				this->ClassName(), T2A(this->store_png_path.Param<param::FilePathParam>()->Value()));

		} else {
			vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
				"%s: Could not store molecular surface map to file: %s",
				this->ClassName(), T2A(this->store_png_path.Param<param::FilePathParam>()->Value()));
		}
		delete this->store_png_image.Image();
	}

	// If we render the tunnels we need blending.
	bool blending = false;
	// If we render the AO texture we don't need lighting.
	bool lighting = true;
	// If we render the geodesic lines they need to be rendered after the sphere.
	bool render_geo_lines = false;
	// If we render the latitude and longitude lines they need to be rendered 
	// after the sphere.
	bool render_lat_lon_lines = false;
	// If we are in sphere mode we have to disable the lighting
	bool sphereMode = false;

	// Create the geodesic lines between the tunnel cuts.
	if (this->geodesic_lines_param.IsDirty()) {
		this->geodesic_lines_param.ResetDirty();
		if (this->vertices_added.size() > 0) {
			this->createGeodesicLines(
				static_cast<GeodesicMode>(this->geodesic_lines_param.Param<param::EnumParam>()->Value()));
			if (!this->initialiseMapShader()) {
				return false;
			}
		}
	}

	// Reset the latitude and longitude lines.
	if (this->lat_lon_lines_colour_param.IsDirty() || this->lat_lon_lines_eq_colour_param.IsDirty() ||
		this->lat_lon_lines_gm_colour_param.IsDirty() || this->lat_lines_count_param.IsDirty() ||
		this->lon_lines_count_param.IsDirty()) {
		// Reset the param slots.
		this->lat_lon_lines_colour_param.ResetDirty();
		this->lat_lon_lines_eq_colour_param.ResetDirty();
		this->lat_lon_lines_gm_colour_param.ResetDirty();
		this->lat_lines_count_param.ResetDirty();
		this->lon_lines_count_param.ResetDirty();

		glDeleteBuffers(1, &this->lat_lon_lines_vbo);
		this->lat_lon_lines_vbo = 0;
	}

	// Determine the data that is rendered based on the Display mode and
	// update the renderer with the new data.
	if (this->display_param.Param<param::EnumParam>()->Value() == DisplayMode::PROTEIN) {
        if (this->display_param.Param<param::EnumParam>()->Value() == DisplayMode::PROTEIN) {
            this->triMeshRenderer.update(&this->faces, &this->vertices, &this->vertexColors, &this->normals);
        }
	} else if (this->display_param.Param<param::EnumParam>()->Value() == DisplayMode::SHADOW) {
		// Get the AO texture.
		auto vector = this->aoCalculator.getVertexShadows();
		if (vector == nullptr) {
			// Texture is empty create one.
			this->aoCalculator.initilialize(this->GetCoreInstance(), &this->vertices, &this->normals, mdc);
			AmbientOcclusionCalculator::AOSettings settings;
			settings.angleFactor = this->aoAngleFactorParam.Param<param::FloatParam>()->Value();
			settings.evalFactor = this->aoEvalParam.Param<param::FloatParam>()->Value();
			settings.falloffFactor = this->aoFalloffParam.Param<param::FloatParam>()->Value();
			settings.genFac = this->aoGenFactorParam.Param<param::FloatParam>()->Value();
			settings.maxDist = this->aoMaxDistSample.Param<param::FloatParam>()->Value();
			settings.minDist = this->aoMinDistSample.Param<param::FloatParam>()->Value();
			settings.numSampleDirections = this->aoNumSampleDirectionsParam.Param<param::IntParam>()->Value();
			settings.scaling = this->aoScalingFactorParam.Param<param::FloatParam>()->Value();
			settings.volSizeX = this->aoVolSizeXParam.Param<param::IntParam>()->Value();
			settings.volSizeY = this->aoVolSizeYParam.Param<param::IntParam>()->Value();
			settings.volSizeZ = this->aoVolSizeZParam.Param<param::IntParam>()->Value();
			vector = this->aoCalculator.calculateVertexShadows(settings);
			if (vector == nullptr) {
				// The computation did not work so set the render to nullptr so nothing is rendered.
				this->triMeshRenderer.update(nullptr, nullptr, nullptr, nullptr);

			} else {
				// We have something to show.
				this->triMeshRenderer.update(&this->faces, &this->vertices, vector, &this->normals, 4);
				lighting = false;
			}

		} else {
			// We have something to show.
			this->triMeshRenderer.update(&this->faces, &this->vertices, vector, &this->normals, 4);
			lighting = false;
		}

	} else if (this->display_param.Param<param::EnumParam>()->Value() == DisplayMode::GROUPS) {
		this->triMeshRenderer.update(&this->faces, &this->vertices, &this->vertexColors_group, &this->normals);

	} else if (this->display_param.Param<param::EnumParam>()->Value() == DisplayMode::VORONOI) {
		this->triMeshRenderer.update(&this->faces, &this->vertices, &this->vertexColors_voronoi,
			&this->normals);

	} else if (this->display_param.Param<param::EnumParam>()->Value() == DisplayMode::CUTS) {
		this->triMeshRenderer.update(&this->faces, &this->vertices, &this->vertexColors_cuts, &this->normals);

	} else if (this->display_param.Param<param::EnumParam>()->Value() == DisplayMode::TUNNEL) {
		// Render the tunnel with an alpha value of 1.0f.
		renderTunnels();
		// Render the rest of the mesh with an alpha value of 0.3f and activate blending.
		this->triMeshRenderer.update(&this->faces, &this->vertices, &this->vertexColors_tunnel, &this->normals, 4);
		blending = true;

	} else if (this->display_param.Param<param::EnumParam>()->Value() == DisplayMode::REBUILD) {
		this->triMeshRenderer.update(&this->faces_rebuild, &this->vertices_rebuild,
			&this->vertexColors_rebuild, &this->normals_rebuild);

	} else if (this->display_param.Param<param::EnumParam>()->Value() == DisplayMode::SPHERE) {
		if (this->computed_sphere) {
			this->triMeshRenderer.update(&this->faces_rebuild, &this->vertices_sphere,
				&this->vertexColors_rebuild, &this->normals_rebuild);
			render_geo_lines = true;
			render_lat_lon_lines = this->lat_lon_lines_param.Param<param::BoolParam>()->Value();
		}
		sphereMode = true;

	} else if (this->display_param.Param<param::EnumParam>()->Value() == DisplayMode::MAP) {
		this->triMeshRenderer.update(nullptr, nullptr, nullptr, nullptr);
		if (this->computed_map) {
			drawMap();
			if (this->lat_lon_lines_param.Param<param::BoolParam>()->Value()) {
				uint numLat = static_cast<uint>(this->lat_lines_count_param.Param<param::IntParam>()->Value());
				uint numLon = static_cast<uint>(this->lon_lines_count_param.Param<param::IntParam>()->Value());
				this->renderLatLonLines(numLat, numLon, 120, 120, true);
			}
		}

	} else {
		// This should not happen but just in case render the protein.
		this->triMeshRenderer.update(&this->faces, &this->vertices, 
			&this->vertexColors, &this->normals);
	}

	// If we need blending set the correct OpenGL states.
	if (blending) {
		glBlendFuncSeparate(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, GL_ONE, GL_ONE_MINUS_SRC_ALPHA);
		glEnable(GL_BLEND);
		glCullFace(GL_BACK);
		glEnable(GL_CULL_FACE);
		glDisable(GL_DEPTH_TEST);
	}

	// If we do not need lighting turn it off.
	if (!lighting || sphereMode) {
		glDisable(GL_LIGHTING);
	}

	// Render the protein.
    if (this->draw_wireframe_param.Param<param::BoolParam>()->Value()) {
        this->triMeshRenderer.RenderWireFrame(*cr3d);
    } else {
        this->triMeshRenderer.Render(*cr3d);
    }

	// Render the geodesic lines.
	if (render_geo_lines) {
		glDisableClientState(GL_NORMAL_ARRAY);
		this->renderGeodesicLines();
		glEnableClientState(GL_NORMAL_ARRAY);
	}

	// Render the latitude and longitude lines.
	if (render_lat_lon_lines) {
		uint numLat = static_cast<uint>(this->lat_lines_count_param.Param<param::IntParam>()->Value());
		uint numLon = static_cast<uint>(this->lon_lines_count_param.Param<param::IntParam>()->Value());
		this->renderLatLonLines(numLat, numLon, 120, 120, false);
	}

	// Disable the states again.
	if (blending) {
		glEnable(GL_DEPTH_TEST);
		glDisable(GL_CULL_FACE);
		glDisable(GL_BLEND);
	}

	// Turn the lighting back on if it was turned of.
	if (!lighting || sphereMode) {
		glEnable(GL_LIGHTING);
	}

	// Reset the OpenGL state
	::glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

	glCullFace(cfm);
	glFrontFace(twr);
	glPolygonMode(GL_FRONT, pm[0]);
	glPolygonMode(GL_BACK, pm[1]);

	glEnable(GL_CULL_FACE);
	glDisableClientState(GL_VERTEX_ARRAY);
	glDisableClientState(GL_NORMAL_ARRAY);
	glDisable(GL_POINT_SIZE);
	glEnable(GL_BLEND);
	glEnable(GL_DEPTH_TEST);

	return true;
}


/*
 * MapGenerator::renderGeodesicLines
 */
void MapGenerator::renderGeodesicLines() {
	glEnableClientState(GL_COLOR_ARRAY);
	glEnableClientState(GL_VERTEX_ARRAY);

	for(size_t i = 0; i < this->geodesic_lines.size(); i++) {
		::glColorPointer(3, GL_FLOAT, 0, this->geodesic_lines_colours[i].data());
		::glVertexPointer(3, GL_FLOAT, 0, this->geodesic_lines[i].data());
		::glDrawArrays(GL_LINE_STRIP, 0, static_cast<GLsizei>(this->geodesic_lines[i].size() / 3));
	}

	glDisableClientState(GL_VERTEX_ARRAY);
	glDisableClientState(GL_COLOR_ARRAY);
}


/*
 * MapGenerator::renderLatLonLines
 */
void MapGenerator::renderLatLonLines(const uint p_num_lat, const uint p_num_lon, const uint p_tess_lat, 
		const uint p_tess_lon, const bool p_project) {
	// Precompute the data if necessary.
	if (!glIsBuffer(this->lat_lon_lines_vbo)) {
		// Intialise the new lines.
		this->lat_lon_lines_vertex_cnt = 0;
		vislib::Array<float> lines;
		lines.AssertCapacity(((p_num_lat + 1) * 2 * (p_tess_lat + 1) + (p_num_lon + 1) * 2 * (p_tess_lon + 1)) * 3);
		this->lat_lon_lines_colours.clear();
		this->lat_lon_lines_colours.shrink_to_fit();
		this->lat_lon_lines_colours.reserve(((p_num_lat + 1) * 2 * (p_tess_lat + 1) + 
			(p_num_lon + 1) * 2 * (p_tess_lon + 1)) * 3);

		// Intitialise the computation of the latitudes.
		float phi = 0.0f;
		float lambda = 0.0f;
		float stepPhi = static_cast<float>(vislib::math::PI_DOUBLE / p_num_lat);
		float stepLambda = static_cast<float>((2.0 * vislib::math::PI_DOUBLE) / p_tess_lat);
		float radius = this->sphere_data.GetW() + this->radius_offset_param.Param<param::FloatParam>()->Value();
		vec3f first, tmpVec, center;
		center.Set(this->sphere_data.GetX(), this->sphere_data.GetY(), this->sphere_data.GetZ());

		// Compute the latitudes.
		for (uint i = 0; i < static_cast<uint>(ceilf(p_num_lat / 2.0f)); i++) {
			// Compute the upper latitudes.
			lambda = 0.0f;
			for (uint j = 0; j < p_tess_lat; j++) {
				// Compute the position.
				tmpVec.SetX(cos(lambda) * radius * cos(phi));
				tmpVec.SetY(sin(phi) * radius);
				tmpVec.SetZ(sin(lambda) * radius * cos(phi));

				// Apply the rotation and move the position onto the sphere.
				tmpVec = this->rotation_quat * tmpVec;
				tmpVec += center;

				// Add the vertex and its colour to the line.
				lines.Add(tmpVec.X());
				lines.Add(tmpVec.Y());
				lines.Add(tmpVec.Z());
				this->latLonLineAddColour(phi, true);
				this->lat_lon_lines_vertex_cnt++;

				// If it is not the first vertex add it twice otherwise remember it.
				if (j > 0) {
					lines.Add(tmpVec.X());
					lines.Add(tmpVec.Y());
					lines.Add(tmpVec.Z());
					this->latLonLineAddColour(phi, true);
					this->lat_lon_lines_vertex_cnt++;

				} else {
					first.SetX(tmpVec.X());
					first.SetY(tmpVec.Y());
					first.SetZ(tmpVec.Z());
				}

				// Increase the angle.
				lambda += stepLambda;
			}

			// Add the first vertex and its colour to the line.
			lines.Add(first.X());
			lines.Add(first.Y());
			lines.Add(first.Z());
			this->latLonLineAddColour(phi, true);
			this->lat_lon_lines_vertex_cnt++;

			// Compute the lower latitudes.
			lambda = 0.0f;
			for (uint j = 0; j < p_tess_lat; j++) {
				// Compute the position.
				tmpVec.SetX(cos(lambda) * radius * cos(-phi));
				tmpVec.SetY(sin(-phi) * radius);
				tmpVec.SetZ(sin(lambda) * radius * cos(-phi));

				// Apply the rotation and move the position onto the sphere.
				tmpVec = this->rotation_quat * tmpVec;
				tmpVec += center;

				// Add the vertex and its colour to the line.
				lines.Add(tmpVec.X());
				lines.Add(tmpVec.Y());
				lines.Add(tmpVec.Z());
				this->latLonLineAddColour(phi, true);
				this->lat_lon_lines_vertex_cnt++;

				// If it is not the first vertex add it twice otherwise remember it.
				if (j > 0) {
					lines.Add(tmpVec.X());
					lines.Add(tmpVec.Y());
					lines.Add(tmpVec.Z());
					this->lat_lon_lines_vertex_cnt++;
					this->latLonLineAddColour(phi, true);

				} else {
					first.SetX(tmpVec.X());
					first.SetY(tmpVec.Y());
					first.SetZ(tmpVec.Z());
				}

				// Increase the angle.
				lambda += stepLambda;
			}

			// Add the first vertex and its colour to the line.
			lines.Add(first.X());
			lines.Add(first.Y());
			lines.Add(first.Z());
			this->lat_lon_lines_vertex_cnt++;
			this->latLonLineAddColour(phi, true);

			// Move to the next latitute.
			phi += stepPhi;
		}

		// Intitialise the computation of the longitudes.
		lambda = static_cast<float>(vislib::math::PI_DOUBLE / 2.0);
		stepPhi = static_cast<float>((2.0 * vislib::math::PI_DOUBLE) / p_tess_lon);
		stepLambda = static_cast<float>(vislib::math::PI_DOUBLE / p_num_lon);

		// Compute the longitudes.
		for (uint i = 0; i < p_num_lon; i++) {
			phi = static_cast<float>(vislib::math::PI_DOUBLE / 2.0);
			for (uint j = 0; j < p_tess_lon; j++) {
				// Compute the position.
				tmpVec.SetX(cos(lambda) * radius * cos(phi));
				tmpVec.SetY(sin(phi) * radius);
				tmpVec.SetZ(sin(lambda) * radius * cos(phi));

				// Apply the rotation and move the position onto the sphere.
				tmpVec = this->rotation_quat * tmpVec;
				tmpVec += center;

				// Add the vertex and its colour to the line.
				lines.Add(tmpVec.X());
				lines.Add(tmpVec.Y());
				lines.Add(tmpVec.Z());
				this->latLonLineAddColour(lambda, false);
				this->lat_lon_lines_vertex_cnt++;

				// If it is not the first vertex add it twice otherwise remember it.
				if (j > 0) {
					lines.Add(tmpVec.X());
					lines.Add(tmpVec.Y());
					lines.Add(tmpVec.Z());
					this->latLonLineAddColour(lambda, false);
					this->lat_lon_lines_vertex_cnt++;

				} else {
					first.SetX(tmpVec.X());
					first.SetY(tmpVec.Y());
					first.SetZ(tmpVec.Z());
				}

				// Increase the angle.
				phi += stepPhi;
			}

			// Add the first vertex and its colour to the line.
			lines.Add(first.X());
			lines.Add(first.Y());
			lines.Add(first.Z());
			this->latLonLineAddColour(lambda, false);
			this->lat_lon_lines_vertex_cnt++;

			// Move to the next longitude
			lambda += stepLambda;
		}

		// Create the buffer for OpenGL.
		glGenBuffers(1, &this->lat_lon_lines_vbo);
		glBindBuffer(GL_ARRAY_BUFFER, this->lat_lon_lines_vbo);
		glBufferData(GL_ARRAY_BUFFER, sizeof(float) * lines.Count(), lines.PeekElements(), GL_DYNAMIC_DRAW);
		glBindBuffer(GL_ARRAY_BUFFER, 0);
	}

	// Set the OpenGL states.
	glPushAttrib(GL_ENABLE_BIT);
	glDisable(GL_LIGHTING);
	glLineWidth(1.0);

	// Bind the buffer and the colour data.
	glEnableClientState(GL_VERTEX_ARRAY);
	glEnableClientState(GL_COLOR_ARRAY);
	glBindBuffer(GL_ARRAY_BUFFER, this->lat_lon_lines_vbo);
	glVertexPointer(3, GL_FLOAT, 0, 0);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glColorPointer(3, GL_FLOAT, 0, this->lat_lon_lines_colours.data());

	// Check if we render onto the 3D sphere or the 2D map.
	if (p_project) {
		// We render onto the map so use the geodesic lines shader and disable the depth test.
		glDisable(GL_DEPTH_TEST);
		this->geodesic_shader.Enable();
		this->geodesic_shader.SetParameter("sphere", this->sphere_data.GetX(), this->sphere_data.GetY(),
			this->sphere_data.GetZ(), this->sphere_data.GetW());
		this->geodesic_shader.SetParameter("frontVertex", this->vertices_sphere[this->look_at_id * 3],
			this->vertices_sphere[this->look_at_id * 3 + 1], this->vertices_sphere[this->look_at_id * 3 + 2]);
	}

	// Render the lines.
	glDrawArrays(GL_LINES, 0, GLsizei(this->lat_lon_lines_vertex_cnt));

	// Check if we render onto the 3D sphere or the 2D map.
	if (p_project) {
		// Disable the shader and enable the depth test.
		this->geodesic_shader.Disable();
		glEnable(GL_DEPTH_TEST);
	}

	// Reset the OpenGL states.
	glDisableClientState(GL_COLOR_ARRAY);
	glDisableClientState(GL_VERTEX_ARRAY);
	glPopAttrib();
}


/*
 * MapGenerator::renderTunnels
 */
void MapGenerator::renderTunnels() {
	glEnableClientState(GL_COLOR_ARRAY);
	glEnableClientState(GL_NORMAL_ARRAY);
	glEnableClientState(GL_VERTEX_ARRAY);

	glColorPointer(3, GL_FLOAT, 0, this->vertexColors.data());
	glVertexPointer(3, GL_FLOAT, 0, this->vertices.data());
	glNormalPointer(GL_FLOAT, 0, this->normals.data());
	glDrawElements(GL_TRIANGLES, static_cast<unsigned int>(this->tunnel_faces.size()), 
		GL_UNSIGNED_INT, this->tunnel_faces.data());

	glDisableClientState(GL_VERTEX_ARRAY);
	glDisableClientState(GL_NORMAL_ARRAY);
	glDisableClientState(GL_COLOR_ARRAY);
}


/*
 * MapGenerator::setDvalues
 */
bool MapGenerator::setDvalues(const size_t p_vertex_cnt, float& p_theta, const Poles& p_poles) {
	// Intialise vectors for storing the vertices that do not have a d value assigned.
	std::vector<uint> d_values = std::vector<uint>(p_vertex_cnt, 0);
	std::vector<bool> processed = std::vector<bool>(unsigned int(d_values.size()), false);
	std::vector<unsigned int> to_process;
	to_process.reserve(unsigned int(d_values.size()));

	// Initialise the neighbours of the north pole with the da value of 1.
	for (const auto& edge : this->vertex_edge_offset[p_poles.north]) {
		if (edge.vertex_id_0 == p_poles.north) {
			d_values[edge.vertex_id_1] = 1;
			processed[edge.vertex_id_1] = true;
			// Intialise the neighbours of the neighbours of the north pole.
			for (const auto& n_edge : this->vertex_edge_offset[edge.vertex_id_1]) {
				if (n_edge.vertex_id_0 == edge.vertex_id_1 &&
					n_edge.vertex_id_1 != p_poles.north &&
					d_values[n_edge.vertex_id_1] != 1) {
					to_process.push_back(n_edge.vertex_id_1);
					d_values[n_edge.vertex_id_1] = 2;
				}

				if (n_edge.vertex_id_1 == edge.vertex_id_1 &&
					n_edge.vertex_id_0 != p_poles.north &&
					d_values[n_edge.vertex_id_0] != 1) {
					to_process.push_back(n_edge.vertex_id_0);
					d_values[n_edge.vertex_id_0] = 2;
				}				
			}

		} else {
			d_values[edge.vertex_id_0] = 1;
			processed[edge.vertex_id_0] = true;
			// Intialise the neighbours of the neighbours of the north pole.
			for (const auto& n_edge : this->vertex_edge_offset[edge.vertex_id_0]) {
				if (n_edge.vertex_id_0 == edge.vertex_id_0 &&
					n_edge.vertex_id_1 != p_poles.north &&
					d_values[n_edge.vertex_id_1] != 1) {
					to_process.push_back(n_edge.vertex_id_1);
					d_values[n_edge.vertex_id_1] = 2;
				}

				if (n_edge.vertex_id_1 == edge.vertex_id_0 &&
					n_edge.vertex_id_0 != p_poles.north &&
					d_values[n_edge.vertex_id_0] != 1) {
					to_process.push_back(n_edge.vertex_id_0);
					d_values[n_edge.vertex_id_0] = 2;
				}
			}
		}
	}
	processed[p_poles.north] = true;

	uint id, d;
	while (!to_process.empty()) {
		id = to_process.front();
		to_process.erase(to_process.begin());
		processed[id] = true;
		d = d_values[id];

		for (const auto& edge : this->vertex_edge_offset[id]) {
			if (edge.vertex_id_0 == id && !processed[edge.vertex_id_1]) {
				if (std::find(to_process.begin(), to_process.end(), edge.vertex_id_1) == to_process.end()) {
					to_process.push_back(edge.vertex_id_1);
					d_values[edge.vertex_id_1] = d + 1;
				}
			}

			if (edge.vertex_id_1 == id && !processed[edge.vertex_id_0]) {
				if (std::find(to_process.begin(), to_process.end(), edge.vertex_id_0) == to_process.end()) {
					to_process.push_back(edge.vertex_id_0);
					d_values[edge.vertex_id_0] = d + 1;
				}
			}
		}
	}

	p_theta = static_cast<float>(vislib::math::PI_DOUBLE) / static_cast<float>(d_values[p_poles.south]);

	return true;
}


/*
 * MapGenerator::splitString
 */
std::vector<std::string> MapGenerator::splitString(const std::string & p_string, const char p_delim) {
	std::vector<std::string> elements;
	splitString(p_string, p_delim, elements);
	return elements;
}


/*
 * MapGenerator::splitString
 */
std::vector<std::string> MapGenerator::splitString(const std::string & p_string, const char p_delim, 
		std::vector<std::string>& p_elements) {
	std::stringstream ss(p_string);
	std::string item;
	while (std::getline(ss, item, p_delim)) {
		p_elements.push_back(item);
	}
	return p_elements;
}
