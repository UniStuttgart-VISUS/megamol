/*
 * TriangleMeshRenderer.cpp
 * Copyright (C) 2006-2016 by MegaMol Team
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "TriangleMeshRenderer.h"

using namespace megamol;
using namespace megamol::molecularmaps;

/*
 * TriangleMeshRenderer::TriangleMeshRenderer
 */
TriangleMeshRenderer::TriangleMeshRenderer(void) : AbstractLocalRenderer(), faces(nullptr), 
	vertex_colors(nullptr), vertex_normals(nullptr), vertices(nullptr){
}

/*
 * TriangleMeshRenderer::~TriangleMeshRenderer
 */
TriangleMeshRenderer::~TriangleMeshRenderer(void) {
	this->Release();
}

/*
 * TriangleMeshRenderer::create
 */
bool TriangleMeshRenderer::create(void) {

	return true;
}

/*
 * TriangleMeshRenderer::release
 */
void TriangleMeshRenderer::release(void) {
	
}

/*
 * TriangleMeshRenderer::Render
 */
bool TriangleMeshRenderer::Render(core::view::CallRender3D& call) {

	if (this->faces == nullptr) return false;
	if (this->vertices == nullptr) return false;
	if (this->vertex_colors == nullptr) return false;
	if (this->vertex_normals == nullptr) return false;
	if (this->numValuesPerColor != 3 && this->numValuesPerColor != 4) return false;

	glEnableClientState(GL_COLOR_ARRAY);
	glEnableClientState(GL_NORMAL_ARRAY);
	glEnableClientState(GL_VERTEX_ARRAY);

	glColorPointer(this->numValuesPerColor, GL_FLOAT, 0, this->vertex_colors->data());
	glVertexPointer(3, GL_FLOAT, 0, this->vertices->data());
	glNormalPointer(GL_FLOAT, 0, this->vertex_normals->data());
	glDrawElements(GL_TRIANGLES, static_cast<unsigned int>(this->faces->size()), GL_UNSIGNED_INT, this->faces->data());

	glDisableClientState(GL_VERTEX_ARRAY);
	glDisableClientState(GL_NORMAL_ARRAY);
	glDisableClientState(GL_COLOR_ARRAY);

	return true;
}

/*
 * TriangleMeshRenderer::update
 */
bool TriangleMeshRenderer::update(const std::vector<uint> * faces, const std::vector<float> * vertices, const std::vector<float> * vertex_colors,
	const std::vector<float> * vertex_normals, unsigned int numValuesPerColor) {

	this->faces = faces;
	this->vertices = vertices;
	this->vertex_colors = vertex_colors;
	this->vertex_normals = vertex_normals;

	if (numValuesPerColor == 3 || numValuesPerColor == 4) {
		this->numValuesPerColor = numValuesPerColor;
	} else {
		return false;
	}

	return true;
}