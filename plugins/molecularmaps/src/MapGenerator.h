/*
 * MapGenerator.h
 * Copyright (C) 2006-2016 by MegaMol Team
 * Alle Rechte vorbehalten.
 */
#ifndef MMMOLMAPPLG_MAPGENERATOR_H_INCLUDED
#define MMMOLMAPPLG_MAPGENERATOR_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#    pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "mmcore/CallerSlot.h"
#include "mmcore/CoreInstance.h"
#include "mmcore/misc/PngBitmapCodec.h"
#include "mmcore/param/ParamSlot.h"
#include "mmcore/utility/ColourParser.h"
#include "mmcore_gl/utility/ShaderSourceFactory.h"
#include "mmcore_gl/view/Renderer3DModuleGL.h"

#include "geometry_calls_gl/CallTriMeshDataGL.h"

#include "vislib_gl/graphics/gl/OutlineFont.h"
#include "vislib_gl/graphics/gl/Verdana.inc"

#include "AmbientOcclusionCalculator.h"
#include "CUDAKernels.cuh"
#include "Color.h"
#include "Octree.h"
#include "TriangleMeshRenderer.h"
#include "VoronoiChannelCalculator.h"

#include "glowl/BufferObject.hpp"

#include <filesystem>

namespace megamol {
namespace molecularmaps {

class MapGenerator : public core_gl::view::Renderer3DModuleGL {
public:
    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char* ClassName(void) {
        return "MapGenerator";
    }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static const char* Description(void) {
        return "Generates a molecular map out of an incoming triangle mesh";
    }

    /**
     * Answers whether this module is available on the current system.
     *
     * @return 'true' if the module is available, 'false' otherwise.
     */
    static bool IsAvailable(void) {
        return true;
    }

    /** Ctor */
    MapGenerator(void);

    /** Dtor */
    virtual ~MapGenerator(void);

protected:
    /**
     * Implementation of 'Create'.
     *
     * @return 'true' on success, 'false' otherwise.
     */
    virtual bool create(void);

    /**
     * Implementation of 'Release'.
     */
    virtual void release(void);

private:
    /**
     * Checks if every element in a vector is true.
     *
     * @param p_vec the vector to check
     *
     * @return true if all elements are true, false otherwise
     */
    bool allElementsTrue(const std::vector<bool>& p_vec);

    /**
     * Use the mesh without the cap and the mesh with the cap to find vertices that
     * are shadowed by the cap. Mix the current colour with the coulour from the cap
     * to highlight these vertices.
     *
     * @param p_cap_data_call The call that contains the full mesh with the cap.
     * @param p_cr3d The call from the Render3d modul.
     * @param p_bs Pointer to the incoming binding site call.
     *
     * @return false if an error occured, true otherwise.
     */
    bool capColouring(megamol::geocalls_gl::CallTriMeshDataGL* p_cap_data_call, megamol::core_gl::view::CallRender3DGL& p_cr3d,
        protein_calls::BindingSiteCall* p_bs);

    /**
     * Colours the mesh in a certain radius around a binding site.
     *
     * @param p_bs Pointer to the incoming binding site call.
     * @param p_colour The colour the mesh vertices should have.
     * @param p_mdc Pointer to the incoming molecular data call.
     * @param p_radius The radius in which the coloring should happen. If this is negative
     * we take the radius of the circumcircle of the binding site.
     * @param p_radiusOffset The offset that gets added to the computed radius if the given
     * radius is negative
     *
     * @return false if an error occured, true otherwise.
     */
    bool colourBindingSite(protein_calls::BindingSiteCall* p_bs, const vec3f& p_colour,
        protein_calls::MolecularDataCall* p_mdc, const float p_radius, float p_radiusOffset = 0.0f,
        bool p_ignoreRadius = false);

    /**
     * Computes the bounding box of a given set of vertices
     *
     * @param verts The coordinates of the vertices stored as continous vector
     * @return The bounding box of the vertices
     */
    vislib::math::Cuboid<float> computeBoundingBox(std::vector<float>& verts);

    /**
     * Compute the next point on the geodesic line based on the a and b interpolation
     * parameters. See
     * http://fraserchapman.blogspot.de/2008/09/intermediate-points-on-great-circle.html
     * for more details.
     *
     * @param p_a the first interpolation parameter
     * @param p_b the second interpolation parameter
     * @param p_c the offset that is added to the radius to raise the line
     * @param p_lat_0 the latitude of the start point
     * @param p_lon_0 the longitude of the start point
     * @param p_lat_1 the latitude of the end point
     * @param p_lon_1 the longitude of the end point
     *
     * @return the point in 3D coordinates.
     */
    vec3f computeGeodesicPoint(const float p_a, const float p_b, const float p_c, const float p_lat_0,
        const float p_lon_0, const float p_lat_1, const float p_lon_1);

    /**
     * Find the rotation matrix to get from (0,1,0) to the given normal.
     *
     * @param p_normal the normal of the oriented hemisphere
     *
     * @return the rotation matrix to get from (0,1,0) to the normal
     */
    mat4f computeRotationMatrix(const vec3f& p_normal);

    /**
     * Convert 3D sphere coorinates into latitude and longitude coordinates.
     *
     * @param p_point the point on the sphere
     * @param p_lat the latitude of the point
     * @param p_lon the logitude of the point
     */
    void convertToLatLon(const vec3f& p_point, float& p_lat, float& p_lon);

    /**
     * Create the the bounding sphere of the protein based on the bouncing
     * bubble algorithm.
     *
     * @param p_offset The offset that is added to the radius of the sphere.
     * @param p_radius Will be the radius of the sphere.
     * @param p_center Will be the center point of the sphere.
     * @param p_vector The vector containing the atom positions
     *
     * @return Always true.
     */
    bool createBoundingSphere(
        const float p_offset, float& p_radius, vislib::math::Vector<float, 3>& p_center, std::vector<float>& p_vector);

    /**
     * Create a triangle fan for each circle that was identified.
     *
     * @param p_second_rebuild determines if the old surface is stored in
     * the local copy or the rebuild vectors
     * @param p_tunnel_id the ID of the tunnel (determines the colour of the fan)
     * @param p_vertex_id the ID for new vertices
     * @param p_vertex_ids the vertex IDs that from the circle
     * @param p_tunnels contains the faces that belong to tunnels
     *
     * @return a cut that stores all information about the fan
     */
    Cut createCut(const bool p_second_rebuild, const uint p_tunnel_id, uint& p_vertex_id,
        const std::vector<uint>& p_vertex_ids, const std::vector<bool>& p_tunnels);

    /**
     * Create a geodeisc lines for all tunnel entrances. The mode decides
     * how many lines are created. NO_LINES does not create any lines,
     * ONE_TO_ALL creates a line from on entrace to all the other entrances
     * of the same tunnel and ALL_TO_ALL creates lines from every entrance
     * to all other entrances of the same tunnel.
     *
     * @param p_mode the mode that decides how many lines are created.
     */
    void createGeodesicLines(const GeodesicMode p_mode);

    /**
     * Create a sphere out of the local copy. The algorithm used to create
     * the sphere is from Rahi & Sharp.
     *
     * @param p_eye_dir the eye direction of the current camera
     * @param p_up_dir the up direction of the current camera
     *
     * @return true if the sphere was created, false otherwise
     */
    bool createSphere(const vec3f& p_eye_dir, const vec3f& p_up_dir);

    /**
     * Given the list of vertices and edges the function performs a depth
     * first search from the given start vertex. Adds every node that it
     * finds to the given group
     *
     * @param p_cur the current vertex
     * @param p_edges the edges of the graph sorted by the start vertex
     * @param p_reversed_edges the edges of the graph sorted by the end vertex
     * @param p_start_offset the list that contains the index of the first
     * edge that starts from the vertex
     * @param p_end_offset the list that contains the index of the first
     * edge that ends at the vertex
     * @param p_visited the list that marks visited vertices
     * @param p_group will contain all nodes that can be reached from the
     * start vertex
     */
    void depthFirstSearch(const size_t p_cur, const std::vector<VoronoiEdge>& p_edges,
        const std::vector<VoronoiEdge>& p_reversed_edges, const std::vector<size_t>& p_start_offset,
        const std::vector<size_t>& p_end_offset, std::vector<bool>& p_visited, std::vector<uint>& p_group);

    /**
     * Draw the map based on the vertices of the sphere.
     */
    void drawMap(void);


    /**
     * The get extents callback. The module should set the members of
     * 'call' to tell the caller the extents of its data (bounding boxes
     * and times).
     *
     * @param call The calling call.
     * @return The return value of the function.
     */
    virtual bool GetExtents(core_gl::view::CallRender3DGL& call);

    /**
     * The get data callback for the resulting mesh. The module should set the members of
     * 'call' to give the caller access to its data.
     *
     *  @param call The calling call.
     * @return The return value of the function.
     */
    virtual bool GetMeshData(core::Call& call);

    /**
     * The get extents callback for the resulting mesh. The module should set the members of
     * 'call' to tell the caller the extents of its data (bounding boxes
     * and times).
     *
     * @param call The calling call.
     * @return The return value of the function.
     */
    virtual bool GetMeshExtents(core::Call& call);

    /**
     * Fills the local mesh data vectors with a given mesh
     *
     * @param emsh The triangle mesh
     * @return True on success. False otherwise.
     */
    bool fillLocalMesh(const geocalls_gl::CallTriMeshDataGL::Mesh& mesh);

    /**
     * Queries the index of the value attribute from a given mesh
     *
     * @param mesh The mesh to query the index from.
     * @return The queried index or -1 if it does not exist.
     */
    int findValueAttributeIndex(const geocalls_gl::CallTriMeshDataGL::Mesh& mesh);

    /**
     * Determine the boundary meridian of the protein. And set the types
     * of vertices to -1: for the poles, 0: for vertices that are not on
     * the meridian or a neighbour, 1: for vertices that are on the meridian,
     * 2: for vertices that are on the "right" side of the meridian and
     * 3: for vertices that are on the "left" side of the meridian. The
     * meridian is found by following the path of steepest descent.
     *
     * @param p_poles The IDs of the north and south pole.
     * @param p_types Will contain the types of vertices.
     * @param p_valid_vertices The poles an the vertices on the meridia
     * are not valid.
     * @param p_zvalues The z values of the vertices.
     * @param p_eye_dir the eye direction of the current camera
     *
     * @return True if a path was found, false otherwise.
     */
    bool findBoundaryMeridian(const Poles& p_poles, std::vector<int>& p_types, std::vector<bool>& p_valid_vertices,
        const std::vector<float>& p_zvalues, const vec3f& p_eye_dir);

    /**
     * Uses the border edges of each shadowed group to find circles. Starts
     * with one edge and looks for another edge that has the same vertex ID.
     * Connects edges until the first edge is reached again. If more than one
     * circle is found for a group it is marked as a tunnel.
     *
     * @param p_groups the groups created bay the AO algorithm
     *
     * @return false if the edges could not be sorted, true otherwise
     */
    bool findCircles(std::vector<FaceGroup>& p_groups);

    /**
     * Adds faces that have the same group ID as the start ID until no further
     * faces can be added or the maximum number of rounds is reached. If the
     * maximum number of rounds is reached the faces are not an enclosure and
     * the function therefore returns -1. If the faces are an enclosure then the
     * ID of the group with the most faces bordering the enclosure is returned.
     *
     * @param p_start_face the ID of the start face from which the group will grow
     * @param p_face_id the group ID of the start face
     * @param p_group will contain all faces that could be reached
     * @param p_face_ids the group IDs for all faces
     * @param p_face_id_cnt the nbumber of different group IDs
     * @param p_marked_faces faces that have been added to a group will be true
     * @param p_rounds the maximum number of rounds
     *
     * @reuturn the ID of the group surrounding the faces or -1 if the group is
     * not surrounded
     */
    int findEnclosures(uint p_start_face, uint p_face_id, std::map<uint, uint>& p_group,
        const std::vector<uint>& p_face_ids, const size_t p_face_id_cnt, std::vector<bool>& p_marked_faces,
        const size_t p_rounds);

    /**
     * Determine the poles of the sphere based on the current camera.
     * The north pole is the highest vertex along the up vector of the
     * camera and the south pole is the lowest vertex.
     *
     * @param p_eye_dir the eye direction of the current camera
     * @param p_up_dir the up direction of the current camera
     * @param p_center the center of the bounding sphere
     * @param p_poles Will contain the vertices that correspond to the
     * poles
     *
     * @return false if the look at point was not found, otherwise true
     */
    bool findPoles(const vislib::math::Vector<float, 3>& p_eye_dir, const vislib::math::Vector<float, 3>& p_up_dir,
        const vislib::math::Vector<float, 3>& p_center, Poles& p_poles);

    /**
     * Get the name of the pdb file to add it as the default name to the screenshot
     * file path. Uses an evil hack for convinience.
     *
     * @return the name of the loaded pdb file from the PDBLoader.
     */
    std::string getNameOfPDB(protein_calls::MolecularDataCall& mdc);

    /**
     * Group the surfaces between two circles of faces.
     *
     * @param p_circle_faces_0 the faces on the first circle
     * @param p_circle_faces_1 the faces on the second circle
     * @param p_circle_vertices_0 the vertices that form the first circle
     * @param p_circle_vertices_1 the vertices that form the second circle
     * @param p_group will contain the faces that are between the two circles
     */
    void growFaces(std::vector<uint>& p_circle_faces_0, std::vector<uint>& p_circle_faces_1,
        std::vector<uint>& p_circle_vertices_0, std::vector<uint>& p_circle_vertices_1, std::map<uint, uint>& p_group);

    /**
     * Try to filter out invalid circles by grouping the faces from a given circle
     * and check how many new faces can be added. If there are no new faces then the
     * circle is not valid, i.e. it does not belong to a tunnel. If there are less
     * then the given number of rounds "circles" of faces than the circle is also
     * considered invalid. A valid circle would be able to acess every face on the
     * mesh. This is very  expensive so we want to stop valid circles as early as
     * possible.
     *
     * @param p_circle_faces the faces on the circle
     * @param p_circle_vertices the vertices that form the circle
     * @param p_group will contain the faces that can be reached from the circle
     * @param p_rounds the number of rounds after which the circle is acepted as valid.
     *
     * @return true if the circle is valid, false otherwise
     */
    bool growFaces(std::vector<uint>& p_circle_faces, std::vector<uint>& p_circle_vertices,
        std::map<uint, uint>& p_group, const size_t p_rounds);

    /**
     * Identify the border edges by checking all faces in the group
     * if they have an edge that borders a face of a different state.
     * Only creates border edges for shadowed groups.
     *
     * @param p_face_shadowed the state of the faces
     * @param the groups created by the AO algorithm
     */
    void identifyBorderEdges(const std::vector<bool>& p_face_shadowed, std::vector<FaceGroup>& p_groups);

    /**
     * Identify the border edges by checking all faces in the group
     * if they have an edge that borders a face of a different group.
     * Only creates border edges for groups where the state is true.
     *
     * @param p_face_group the group IDs for all faces
     * @param p_groups p_groups the groups created by the voronoi vertices
     */
    void identifyBorderEdges(const std::vector<uint>& p_face_group, std::vector<FaceGroup>& p_groups);

    /**
     * Initialise the map shader and the corresponding buffer.
     *
     * @param shaderReload False: Does the complete initialization. True: only reloads the shaders.
     * @return true if no error occures, false otherwise
     */
    bool initialiseMapShader(bool shaderReload = false);

    /**
     * Initialise the z values of the poles and their neighbouring
     * vertices.
     *
     * @param p_poles The IDs of the north and south pole.
     * @param p_zvalues The z values of the vertices.
     * @param p_valid_vertices The poles and their neighbours are not
     * valid vertices.
     * @param p_zvalue The maximum z value.
     *
     * @return Always true.
     */
    bool initialiseZvalues(
        const Poles& p_poles, std::vector<float>& p_zvalues, std::vector<bool>& p_valid_vertices, const float p_zvalue);

    /**
     * Checks if an face exists with an edge in the same direction as the
     * new edge. If so the function returns true, if not the function
     * returns false.
     *
     * @param p_start_id the ID of the start vertex of the edge
     * @param p_end_id the ID of the end vertex of the edge
     * @param p_tunnels contains the faces that belong to tunnels
     *
     * @return true if an edge in the same direction already exists, false
     * otherwise
     */
    bool invertEdge(const uint p_start_id, const uint p_end_id, const std::vector<bool>& p_tunnels);

    /**
     * Computes whether a mesh has a genus greater than 0.
     *
     * @param vertexCnt The number of vertices of the mesh.
     * @param faceCnt The number of faces of the mesh.
     * @param p_genus will contain the genus of the mesh
     *
     * @return True, if the genus is greater than 0. False otherwise.
     */
    bool isGenusN(const uint vertexCnt, const uint faceCnt, uint& p_genus);

    /**
     * Computes whether a circle is a valid one compared to the represented
     * sphere.
     *
     * @param p_sphereToTest The represented sphere
     * @param p_circle The circle represented as
     *
     * @return True if the circle is valid, false otherwise.
     */
    bool isValidCircle(const vec4d& p_sphereToTest, const std::vector<uint>& p_circle);

    /**
     * Add the colour to the latitude or longitude line based on the position
     * of the line. The equator and the Greenwich meridian get another colour.
     *
     * @param p_angle the current angle
     * @param p_is_lat if the current colour is added to a latitude line
     */
    void latLonLineAddColour(const float p_angle, const bool p_is_lat);

    /**
     * Processes the output of the Ambient Occlusion algorithm. Identifys
     * the faces that are shadowed by checking if they have more than one
     * shadowed vertex. The faces are then grouped and used as input for
     * the tunnel detection. Faces that gelong to a tunnel are marked and
     * the vertices at the border are used as input for the cut creation.
     *
     * @param p_ao_vals the output of the Ambient Occlusion algorithm
     * @param p_cuts will contain the cuts that are created
     * @param p_threshold the factor that determines if a vertex is shadowed
     * or not
     * @param p_tunnels will be true for the faces that belong to tunnels
     *
     * @return the number of tunnels that where identyfied
     */
    uint processAOOutput(const std::vector<float>* p_ao_vals, std::vector<Cut>& p_cuts, const float p_threshold,
        std::vector<bool>& p_tunnels);

    /**
     * Process the output of the voronoi computation. Identifys the faces that
     * form a tunnel by performing an AO algorithm on the voronoi vertices and
     * then looking which faces can be reached from the remaining voronoi
     * vertices. Merging those faces and removing enclosures leads to one group
     * of faces for each tunnel network. Those faces can then be removed and the
     * openings can be closed by triangle fans.
     *
     * @param p_cuts will contain the cuts that are created
     * @param p_tunnel_id the current number of tunnels that where created by the
     * processAOOutput function. Will also be used to identify the tunnels created
     * by this function
     * @param p_tunnels will be true for the faces that belong to tunnels
     * @param p_voronoi_vertices the voronoi vertices from the voronoi diagram
     * computation
     * @param p_voronoi_edges the voronoi edges from the voronoi diagram
     * computation
     * @param bbox the bounding box of the protein that is used to create the
     * Octree of the faces
     */
    void processTopologyOutput(std::vector<Cut>& p_cuts, uint& p_tunnel_id, std::vector<bool>& p_tunnels,
        const std::vector<VoronoiVertex>& p_voronoi_vertices, const std::vector<VoronoiEdge>& p_voronoi_edges,
        const vislib::math::Cuboid<float>& bbox);

    /**
     * Rebuild the surface based on the old surface and the cuts, as well as
     * the detected tunnels that need to be removed.
     *
     * @param p_cuts a list of cuts that belong to the same tunnel, each
     * cut contains the colours, faces, normals and vertices
     * @param p_second_rebuild determines if the old surface is stored in
     * the local copy or the rebuild vectors
     * @param p_tunnels a list of faces that need to be deleted
     */
    void rebuildSurface(
        const std::vector<Cut>& p_cuts, const bool p_second_rebuild, const std::vector<bool>& p_tunnels);

    /**
     * The Open GL Render callback.
     *
     * @param call The calling call.
     * @return The return value of the function.
     */
    virtual bool Render(core_gl::view::CallRender3DGL& call);

    /**
     * Render the geodesic lines in 3D.
     */
    void renderGeodesicLines();

    /**
     * Render the latitude and longitude lines.
     */
    void renderLatLonLines(
        const uint p_num_lat, const uint p_num_lon, const uint p_tess_lat, const uint p_tess_lon, const bool p_project);

    /**
     * Render the tunnel faces only.
     */
    void renderTunnels();

    /**
     * The north pole starts with a value of 0. All it's neighbours
     * get the value 1. All their neighbours the value 2, etc.
     * The south pole gets the value d.
     * This value determins the constant theta, which is equal to
     * pi/d.
     * All neighbours of the north pole have the theta value pi/d,
     * the neighbours of the south pole have the theta value pi - pi/d.
     *
     * @param p_vertex_cnt The number of vertices
     * @param p_theta Will contain the constant theta.
     * @param p_poles The IDs of the north and south pole.
     *
     * @return Always true.
     */
    bool setDvalues(const size_t p_vertex_cnt, float& p_theta, const Poles& p_poles);

    /**
     * Split the string at the given delimiter and return a vector
     * that contains the parts without the delimiter.
     *
     * @param p_string the whole string
     * @param p_delim the delimiter
     *
     * @return the parts of the string without the delimiter
     */
    std::vector<std::string> splitString(const std::string& p_string, const char p_delim);

    /**
     * splitString calls this function to perform the splitting.
     *
     * @param p_string the whole string
     * @param p_delim the delimiter
     * @param p_elements will contain the parts of the input string
     *
     * @return the parts of the string without the delimiter
     */
    std::vector<std::string> splitString(
        const std::string& p_string, const char p_delim, std::vector<std::string>& p_elements);

    /**
     * Writes the value image to disk.
     *
     * @param path_to_image The filepath for the image
     * @param ctmd The call containing the relevant data to write.
     * @param input_image The image that has to be rewritten.
     */
    void writeValueImage(const vislib::TString& path_to_image, const geocalls_gl::CallTriMeshDataGL& ctmd,
        vislib::Array<unsigned char>& input_image);

    /** Turn the Ambient Occlusion on or off */
    core::param::ParamSlot aoActive;

    /** The param slot that defines the factor for the ambient occlusion vector angles */
    core::param::ParamSlot aoAngleFactorParam;

    /** Calculator class for the ambient occlusion */
    AmbientOcclusionCalculator aoCalculator;

    /** The param slot for the final ambient occlusion scaling factor */
    core::param::ParamSlot aoEvalParam;

    /** The param slot that defines the exponent p of the ambient occlusion distance function */
    core::param::ParamSlot aoFalloffParam;

    /** The param slot that defines the influence of a sphere on a voxel for the ambient occlusion */
    core::param::ParamSlot aoGenFactorParam;

    /** The param sloat for the maximal distance between a vertex and the furthest ambient occlusion sample */
    core::param::ParamSlot aoMaxDistSample;

    /** The param slot for the minimal distance between a vertex and the nearest ambient occlusion sample */
    core::param::ParamSlot aoMinDistSample;

    /** The param slot for the number of sample direction of the ambient occlusion */
    core::param::ParamSlot aoNumSampleDirectionsParam;

    /** The sphere radius scaling param slot for the ambient occlusion */
    core::param::ParamSlot aoScalingFactorParam;

    /** The threshold value for the ambient occlusion. */
    core::param::ParamSlot aoThresholdParam;

    /** The param slot for the number of voxels in x-direction for the ambient occlusion */
    core::param::ParamSlot aoVolSizeXParam;

    /** The param slot for the number of voxels in y-direction for the ambient occlusion */
    core::param::ParamSlot aoVolSizeYParam;

    /** The param slot for the number of voxels in z-direction for the ambient occlusion */
    core::param::ParamSlot aoVolSizeZParam;

    /** The param slot for the color of the selected binding site */
    core::param::ParamSlot bindingSiteColor;

    /** The param slot thata is used to enable the coloring of a specific binding site */
    core::param::ParamSlot bindingSiteColoring;

    /** The param slot that is used to disable the radius computation for binding sites */
    core::param::ParamSlot bindingSiteIgnoreRadius;

    /** The param slot for the radius of the colored binding site */
    core::param::ParamSlot bindingSiteRadius;

    /** The radius offset for the binding site coloring */
    core::param::ParamSlot bindingSiteRadiusOffset;

    /** Flag whether or not use lighting for the surface */
    core::param::ParamSlot blending;

    /** SSBO for the ids */
    std::unique_ptr<glowl::BufferObject> bufferIDs;

    /** SSBO for the buffers */
    std::unique_ptr<glowl::BufferObject> bufferValues;

    /** min and max values of the buffer */
    std::pair<float, float> bufferMinMax;

    /** enables the shutdown of megamol when a screenshot is stored */
    core::param::ParamSlot close_after_screen_store_param;

    /** Button parameter that triggers the computation */
    core::param::ParamSlot computeButton;

    /** Signals if the map computation was sucessfull. */
    bool computed_map;

    /** Signals if the sphere computation was sucessfull. */
    bool computed_sphere;

    /** Unique pointer to the CUDA kernels */
    std::unique_ptr<CUDAKernels> cuda_kernels;

    /** Sets the path to the colour file for the cuts. */
    core::param::ParamSlot cut_colour_param;

    /** The colour table for the cuts. */
    vislib::Array<vec3f> cut_colour_table;

    /** Determines what to render */
    core::param::ParamSlot display_param;

    /** Switch for the wireframe rendering */
    core::param::ParamSlot draw_wireframe_param;

    /** Vertex indices of the mesh faces */
    std::vector<uint> faces;

    /** Rebuilt version of the mesh face vertex indices */
    std::vector<uint> faces_rebuild;

    /** Faces of the map (only for writeout) */
    std::vector<uint> faces_map;

    /** The edges that belong to a certain face */
    std::vector<std::vector<Edge>> face_edge_offset;

    /** The number of edges per face */
    std::vector<uint> face_edge_offset_depth;

    /** Store the vertices for all geodesic lines. */
    std::vector<std::vector<float>> geodesic_lines;

    /** Store the colours for all geodesic lines. */
    std::vector<std::vector<float>> geodesic_lines_colours;

    /** Sets the type of geodesic lines that are drawn. */
    core::param::ParamSlot geodesic_lines_param;

    /** The geodesic lines shader programme. */
    vislib_gl::graphics::gl::GLSLGeometryShader geodesic_shader;

    /** Store the OpenGL vertex buffer for the geodesic lines. */
    std::vector<GLuint> geodesic_lines_vbos;

    /** Sets the path to the colour file for the groups. */
    core::param::ParamSlot group_colour_param;

    /** The colour table for the groups. */
    vislib::Array<vec3f> group_colour_table;

    /** The data hash of the recently loaded data */
    SIZE_T lastDataHash;

    /** The number of latitude lines. */
    core::param::ParamSlot lat_lines_count_param;

    /** Flag whether or not to draw the latitude and longitude lines. */
    core::param::ParamSlot lat_lon_lines_param;

    /** The colour of the latitude and longitude lines. */
    core::param::ParamSlot lat_lon_lines_colour_param;

    /** The colours of the latitude and longitude lines. */
    std::vector<float> lat_lon_lines_colours;

    /** The colour of the equator. */
    core::param::ParamSlot lat_lon_lines_eq_colour_param;

    /** The colour of the greenwich meridian. */
    core::param::ParamSlot lat_lon_lines_gm_colour_param;

    /** The buffer for the latitude and longitude lines vertices. */
    GLuint lat_lon_lines_vbo;

    /** The number of vertices that create the latitude and longitude lines. */
    size_t lat_lon_lines_vertex_cnt;

    /** Flag whether or not use lighting for the surface */
    core::param::ParamSlot lighting;

    /** The number of longitude lines. */
    core::param::ParamSlot lon_lines_count_param;

    /** The id of the vertex we are looking at */
    uint look_at_id;

    /** The buffer for the sphere vertices that are used in the map shader */
    GLuint map_vertex_vbo;

    /** The framebufferobject for the map shader */
    vislib_gl::graphics::gl::FramebufferObject map_fbo;

    /** The map shader programme */
    vislib_gl::graphics::gl::GLSLGeometryShader map_shader;

    /** The state of the shaders */
    bool map_shader_init;

    /** The bounding box of the mesh */
    vislib::math::Cuboid<float> meshBoundingBox;

    /** The output slot for the mesh data */
    core::CalleeSlot meshDataOutSlot;

    /** The input mesh data slot */
    core::CallerSlot meshDataSlot;

    /** The input mesh data slot, also contains the cap that was removed in the meshDataSlot */
    core::CallerSlot meshDataSlotWithCap;

    /** The faces of the output mesh */
    std::vector<uint> mesh_faces;

    /** The vertex positions of the output mesh */
    std::vector<float> mesh_vertices;

    /** Parameter slot to trigger the mirroring of the final map */
    core::param::ParamSlot mirror_map_param;

    /** The mesh vertex normals */
    std::vector<float> normals;

    /** Rebuilt version of the vertex normals */
    std::vector<float> normals_rebuild;

    /** The Octree that contains all faces of the surface. */
    Octree octree;

    /** Mesh that gets outputted via a call for possible further processing */
    geocalls_gl::CallTriMeshDataGL::Mesh out_mesh;

    /** Parameter slot for the selection of the output mesh */
    core::param::ParamSlot out_mesh_selection_slot;

    /** The input slot for the probe radius necessary for filtering the voronoi diagram. */
    core::param::ParamSlot probeRadiusSlot;

    /** The input protein data slot */
    core::CallerSlot proteinDataSlot;

    /** Parameter slot for enabling and disabling the rendering of the equator length */
    core::param::ParamSlot render_equator_length;

    /** The quaternion that represents the rotation of the protein. */
    vislib::math::Quaternion<float> rotation_quat;

    /** The parameter slot for the shader reload button */
    core::param::ParamSlot shaderReloadButtonParam;

    /** Center of the sphere */
    vislib::math::Vector<float, 4> sphere_data;

    /** Flag determining whether a new mesh should be stored */
    bool store_new_mesh;

    /** Button to store the map as an PNG. */
    core::param::ParamSlot store_png_button;

    /** The image data in which the map is stored int. */
    vislib::Array<unsigned char> store_png_data;

    /** Renders the radius onto bottom left corner of the map. */
    vislib_gl::graphics::gl::OutlineFont store_png_font;

    /** The image itself that is stored. */
    sg::graphics::PngBitmapCodec store_png_image;

    /** The values image that is stored */
    sg::graphics::PngBitmapCodec store_values_image;

    /** The path to which the image is stored. */
    core::param::ParamSlot store_png_path;

    /** The fbo that is used to render the map image */
    vislib_gl::graphics::gl::FramebufferObject store_png_fbo;

    /** The fbo that is used to render the value image */
    vislib_gl::graphics::gl::FramebufferObject store_values_fbo;

    /** The path to which the values image is stored */
    core::param::ParamSlot store_png_values_path;

    /** The param slot thar defines the offset for the bounding sphere radius */
    core::param::ParamSlot radius_offset_param;

    /** Renderer for the triangle mesh */
    TriangleMeshRenderer triMeshRenderer;

    /** The faces that form a tunnel. */
    std::vector<uint> tunnel_faces;

    /** The colors of the mesh vertices. */
    std::vector<float> vertexColors;

    /** The colours of the cuts. */
    std::vector<float> vertexColors_cuts;

    /** The colours of the groups. */
    std::vector<float> vertexColors_group;

    /** The colours of the groups that are detected based on the voronoi vertices. */
    std::vector<float> vertexColors_voronoi;

    /** Rebuilt version of the vertex colors */
    std::vector<float> vertexColors_rebuild;

    /** The colour values for the tunnel rendering. */
    std::vector<float> vertexColors_tunnel;

    /** The edges that contain the vertex */
    std::vector<std::vector<Edge>> vertex_edge_offset;

    /** The number of edges that contain the vertex */
    std::vector<uint> vertex_edge_offset_depth;

    /** The maximum number of edges that we expect per vertex. */
    size_t vertex_edge_offset_max_depth;

    /** The mesh vertex positions */
    std::vector<float> vertices;

    /** The IDs of newly added vertices. */
    std::vector<uint> vertices_added;

    /** The tunnel IDs the new vertices blong to. */
    std::vector<uint> vertices_added_tunnel_id;

    /** Rebuilt version of the vertex positions */
    std::vector<float> vertices_rebuild;

    /** Old ids of the rebuild vertices */
    std::vector<int> vertices_rebuild_ids;

    /** Sphere vertices */
    std::vector<float> vertices_sphere;

    /** Map vertices (only for writeout) */
    std::vector<float> vertices_map;

    /** Calculator for the voronoi protein channels */
    VoronoiChannelCalculator voronoiCalc;

    /** A flag determining whether the voronoi diagram was needed or not */
    bool voronoiNeeded;

    /** Parameter enabling the writing of the value image */
    core::param::ParamSlot writeValueImageParam;

    /** The slot for the binding site information */
    core::CallerSlot zeBindingSiteSlot;
};

} /* end namespace molecularmaps */
} /* end namespace megamol */

#endif
