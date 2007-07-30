#ifndef _VISLOGO_H_INCLUDED
#define _VISLOGO_H_INCLUDED

typedef struct VLQuad_ {
	unsigned int v[4];
} VLQuad;

typedef struct VLVec3_ {
	union {
		double f[3];
		struct {
			double x, y, z;
		};
	};
} VLVec3;

unsigned int VisLogoCountFaces();
unsigned int VisLogoCountVertices();
VLQuad* VisLogoFace(unsigned int fid);
VLVec3* VisLogoVertex(unsigned int vid);
VLVec3* VisLogoFaceColor(unsigned int fid);
VLVec3* VisLogoFaceNormal(unsigned int fid);
VLVec3* VisLogoVertexColor(unsigned int vid);
VLVec3* VisLogoVertexNormal(unsigned int vid);
void VisLogoDoStuff();
void VisLogoTwistLogo();

#endif
