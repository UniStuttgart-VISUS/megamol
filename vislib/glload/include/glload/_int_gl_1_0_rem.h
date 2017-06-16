#ifndef OPENGL_GEN_CORE_REM1_0_H
#define OPENGL_GEN_CORE_REM1_0_H
#include "_int_gl_load_ie.h"

#ifdef __cplusplus
extern "C" {
#endif /*__cplusplus*/
typedef void (CODEGEN_FUNCPTR *  PFNGLACCUMPROC)(GLenum op, GLfloat value);
typedef void (CODEGEN_FUNCPTR *  PFNGLALPHAFUNCPROC)(GLenum func, GLfloat ref);
typedef void (CODEGEN_FUNCPTR *  PFNGLBEGINPROC)(GLenum mode);
typedef void (CODEGEN_FUNCPTR *  PFNGLBITMAPPROC)(GLsizei width, GLsizei height, GLfloat xorig, GLfloat yorig, GLfloat xmove, GLfloat ymove, const GLubyte * bitmap);
typedef void (CODEGEN_FUNCPTR *  PFNGLCALLLISTPROC)(GLuint list);
typedef void (CODEGEN_FUNCPTR *  PFNGLCALLLISTSPROC)(GLsizei n, GLenum type, const GLvoid * lists);
typedef void (CODEGEN_FUNCPTR *  PFNGLCLEARACCUMPROC)(GLfloat red, GLfloat green, GLfloat blue, GLfloat alpha);
typedef void (CODEGEN_FUNCPTR *  PFNGLCLEARINDEXPROC)(GLfloat c);
typedef void (CODEGEN_FUNCPTR *  PFNGLCLIPPLANEPROC)(GLenum plane, const GLdouble * equation);
typedef void (CODEGEN_FUNCPTR *  PFNGLCOLOR3BPROC)(GLbyte red, GLbyte green, GLbyte blue);
typedef void (CODEGEN_FUNCPTR *  PFNGLCOLOR3BVPROC)(const GLbyte * v);
typedef void (CODEGEN_FUNCPTR *  PFNGLCOLOR3DPROC)(GLdouble red, GLdouble green, GLdouble blue);
typedef void (CODEGEN_FUNCPTR *  PFNGLCOLOR3DVPROC)(const GLdouble * v);
typedef void (CODEGEN_FUNCPTR *  PFNGLCOLOR3FPROC)(GLfloat red, GLfloat green, GLfloat blue);
typedef void (CODEGEN_FUNCPTR *  PFNGLCOLOR3FVPROC)(const GLfloat * v);
typedef void (CODEGEN_FUNCPTR *  PFNGLCOLOR3IPROC)(GLint red, GLint green, GLint blue);
typedef void (CODEGEN_FUNCPTR *  PFNGLCOLOR3IVPROC)(const GLint * v);
typedef void (CODEGEN_FUNCPTR *  PFNGLCOLOR3SPROC)(GLshort red, GLshort green, GLshort blue);
typedef void (CODEGEN_FUNCPTR *  PFNGLCOLOR3SVPROC)(const GLshort * v);
typedef void (CODEGEN_FUNCPTR *  PFNGLCOLOR3UBPROC)(GLubyte red, GLubyte green, GLubyte blue);
typedef void (CODEGEN_FUNCPTR *  PFNGLCOLOR3UBVPROC)(const GLubyte * v);
typedef void (CODEGEN_FUNCPTR *  PFNGLCOLOR3UIPROC)(GLuint red, GLuint green, GLuint blue);
typedef void (CODEGEN_FUNCPTR *  PFNGLCOLOR3UIVPROC)(const GLuint * v);
typedef void (CODEGEN_FUNCPTR *  PFNGLCOLOR3USPROC)(GLushort red, GLushort green, GLushort blue);
typedef void (CODEGEN_FUNCPTR *  PFNGLCOLOR3USVPROC)(const GLushort * v);
typedef void (CODEGEN_FUNCPTR *  PFNGLCOLOR4BPROC)(GLbyte red, GLbyte green, GLbyte blue, GLbyte alpha);
typedef void (CODEGEN_FUNCPTR *  PFNGLCOLOR4BVPROC)(const GLbyte * v);
typedef void (CODEGEN_FUNCPTR *  PFNGLCOLOR4DPROC)(GLdouble red, GLdouble green, GLdouble blue, GLdouble alpha);
typedef void (CODEGEN_FUNCPTR *  PFNGLCOLOR4DVPROC)(const GLdouble * v);
typedef void (CODEGEN_FUNCPTR *  PFNGLCOLOR4FPROC)(GLfloat red, GLfloat green, GLfloat blue, GLfloat alpha);
typedef void (CODEGEN_FUNCPTR *  PFNGLCOLOR4FVPROC)(const GLfloat * v);
typedef void (CODEGEN_FUNCPTR *  PFNGLCOLOR4IPROC)(GLint red, GLint green, GLint blue, GLint alpha);
typedef void (CODEGEN_FUNCPTR *  PFNGLCOLOR4IVPROC)(const GLint * v);
typedef void (CODEGEN_FUNCPTR *  PFNGLCOLOR4SPROC)(GLshort red, GLshort green, GLshort blue, GLshort alpha);
typedef void (CODEGEN_FUNCPTR *  PFNGLCOLOR4SVPROC)(const GLshort * v);
typedef void (CODEGEN_FUNCPTR *  PFNGLCOLOR4UBPROC)(GLubyte red, GLubyte green, GLubyte blue, GLubyte alpha);
typedef void (CODEGEN_FUNCPTR *  PFNGLCOLOR4UBVPROC)(const GLubyte * v);
typedef void (CODEGEN_FUNCPTR *  PFNGLCOLOR4UIPROC)(GLuint red, GLuint green, GLuint blue, GLuint alpha);
typedef void (CODEGEN_FUNCPTR *  PFNGLCOLOR4UIVPROC)(const GLuint * v);
typedef void (CODEGEN_FUNCPTR *  PFNGLCOLOR4USPROC)(GLushort red, GLushort green, GLushort blue, GLushort alpha);
typedef void (CODEGEN_FUNCPTR *  PFNGLCOLOR4USVPROC)(const GLushort * v);
typedef void (CODEGEN_FUNCPTR *  PFNGLCOLORMATERIALPROC)(GLenum face, GLenum mode);
typedef void (CODEGEN_FUNCPTR *  PFNGLCOPYPIXELSPROC)(GLint x, GLint y, GLsizei width, GLsizei height, GLenum type);
typedef void (CODEGEN_FUNCPTR *  PFNGLDELETELISTSPROC)(GLuint list, GLsizei range);
typedef void (CODEGEN_FUNCPTR *  PFNGLDRAWPIXELSPROC)(GLsizei width, GLsizei height, GLenum format, GLenum type, const GLvoid * pixels);
typedef void (CODEGEN_FUNCPTR *  PFNGLEDGEFLAGPROC)(GLboolean flag);
typedef void (CODEGEN_FUNCPTR *  PFNGLEDGEFLAGVPROC)(const GLboolean * flag);
typedef void (CODEGEN_FUNCPTR *  PFNGLENDPROC)();
typedef void (CODEGEN_FUNCPTR *  PFNGLENDLISTPROC)();
typedef void (CODEGEN_FUNCPTR *  PFNGLEVALCOORD1DPROC)(GLdouble u);
typedef void (CODEGEN_FUNCPTR *  PFNGLEVALCOORD1DVPROC)(const GLdouble * u);
typedef void (CODEGEN_FUNCPTR *  PFNGLEVALCOORD1FPROC)(GLfloat u);
typedef void (CODEGEN_FUNCPTR *  PFNGLEVALCOORD1FVPROC)(const GLfloat * u);
typedef void (CODEGEN_FUNCPTR *  PFNGLEVALCOORD2DPROC)(GLdouble u, GLdouble v);
typedef void (CODEGEN_FUNCPTR *  PFNGLEVALCOORD2DVPROC)(const GLdouble * u);
typedef void (CODEGEN_FUNCPTR *  PFNGLEVALCOORD2FPROC)(GLfloat u, GLfloat v);
typedef void (CODEGEN_FUNCPTR *  PFNGLEVALCOORD2FVPROC)(const GLfloat * u);
typedef void (CODEGEN_FUNCPTR *  PFNGLEVALMESH1PROC)(GLenum mode, GLint i1, GLint i2);
typedef void (CODEGEN_FUNCPTR *  PFNGLEVALMESH2PROC)(GLenum mode, GLint i1, GLint i2, GLint j1, GLint j2);
typedef void (CODEGEN_FUNCPTR *  PFNGLEVALPOINT1PROC)(GLint i);
typedef void (CODEGEN_FUNCPTR *  PFNGLEVALPOINT2PROC)(GLint i, GLint j);
typedef void (CODEGEN_FUNCPTR *  PFNGLFEEDBACKBUFFERPROC)(GLsizei size, GLenum type, GLfloat * buffer);
typedef void (CODEGEN_FUNCPTR *  PFNGLFOGFPROC)(GLenum pname, GLfloat param);
typedef void (CODEGEN_FUNCPTR *  PFNGLFOGFVPROC)(GLenum pname, const GLfloat * params);
typedef void (CODEGEN_FUNCPTR *  PFNGLFOGIPROC)(GLenum pname, GLint param);
typedef void (CODEGEN_FUNCPTR *  PFNGLFOGIVPROC)(GLenum pname, const GLint * params);
typedef void (CODEGEN_FUNCPTR *  PFNGLFRUSTUMPROC)(GLdouble left, GLdouble right, GLdouble bottom, GLdouble top, GLdouble zNear, GLdouble zFar);
typedef GLuint (CODEGEN_FUNCPTR *  PFNGLGENLISTSPROC)(GLsizei range);
typedef void (CODEGEN_FUNCPTR *  PFNGLGETCLIPPLANEPROC)(GLenum plane, GLdouble * equation);
typedef void (CODEGEN_FUNCPTR *  PFNGLGETLIGHTFVPROC)(GLenum light, GLenum pname, GLfloat * params);
typedef void (CODEGEN_FUNCPTR *  PFNGLGETLIGHTIVPROC)(GLenum light, GLenum pname, GLint * params);
typedef void (CODEGEN_FUNCPTR *  PFNGLGETMAPDVPROC)(GLenum target, GLenum query, GLdouble * v);
typedef void (CODEGEN_FUNCPTR *  PFNGLGETMAPFVPROC)(GLenum target, GLenum query, GLfloat * v);
typedef void (CODEGEN_FUNCPTR *  PFNGLGETMAPIVPROC)(GLenum target, GLenum query, GLint * v);
typedef void (CODEGEN_FUNCPTR *  PFNGLGETMATERIALFVPROC)(GLenum face, GLenum pname, GLfloat * params);
typedef void (CODEGEN_FUNCPTR *  PFNGLGETMATERIALIVPROC)(GLenum face, GLenum pname, GLint * params);
typedef void (CODEGEN_FUNCPTR *  PFNGLGETPIXELMAPFVPROC)(GLenum map, GLfloat * values);
typedef void (CODEGEN_FUNCPTR *  PFNGLGETPIXELMAPUIVPROC)(GLenum map, GLuint * values);
typedef void (CODEGEN_FUNCPTR *  PFNGLGETPIXELMAPUSVPROC)(GLenum map, GLushort * values);
typedef void (CODEGEN_FUNCPTR *  PFNGLGETPOLYGONSTIPPLEPROC)(GLubyte * mask);
typedef void (CODEGEN_FUNCPTR *  PFNGLGETTEXENVFVPROC)(GLenum target, GLenum pname, GLfloat * params);
typedef void (CODEGEN_FUNCPTR *  PFNGLGETTEXENVIVPROC)(GLenum target, GLenum pname, GLint * params);
typedef void (CODEGEN_FUNCPTR *  PFNGLGETTEXGENDVPROC)(GLenum coord, GLenum pname, GLdouble * params);
typedef void (CODEGEN_FUNCPTR *  PFNGLGETTEXGENFVPROC)(GLenum coord, GLenum pname, GLfloat * params);
typedef void (CODEGEN_FUNCPTR *  PFNGLGETTEXGENIVPROC)(GLenum coord, GLenum pname, GLint * params);
typedef void (CODEGEN_FUNCPTR *  PFNGLINDEXMASKPROC)(GLuint mask);
typedef void (CODEGEN_FUNCPTR *  PFNGLINDEXDPROC)(GLdouble c);
typedef void (CODEGEN_FUNCPTR *  PFNGLINDEXDVPROC)(const GLdouble * c);
typedef void (CODEGEN_FUNCPTR *  PFNGLINDEXFPROC)(GLfloat c);
typedef void (CODEGEN_FUNCPTR *  PFNGLINDEXFVPROC)(const GLfloat * c);
typedef void (CODEGEN_FUNCPTR *  PFNGLINDEXIPROC)(GLint c);
typedef void (CODEGEN_FUNCPTR *  PFNGLINDEXIVPROC)(const GLint * c);
typedef void (CODEGEN_FUNCPTR *  PFNGLINDEXSPROC)(GLshort c);
typedef void (CODEGEN_FUNCPTR *  PFNGLINDEXSVPROC)(const GLshort * c);
typedef void (CODEGEN_FUNCPTR *  PFNGLINITNAMESPROC)();
typedef GLboolean (CODEGEN_FUNCPTR *  PFNGLISLISTPROC)(GLuint list);
typedef void (CODEGEN_FUNCPTR *  PFNGLLIGHTMODELFPROC)(GLenum pname, GLfloat param);
typedef void (CODEGEN_FUNCPTR *  PFNGLLIGHTMODELFVPROC)(GLenum pname, const GLfloat * params);
typedef void (CODEGEN_FUNCPTR *  PFNGLLIGHTMODELIPROC)(GLenum pname, GLint param);
typedef void (CODEGEN_FUNCPTR *  PFNGLLIGHTMODELIVPROC)(GLenum pname, const GLint * params);
typedef void (CODEGEN_FUNCPTR *  PFNGLLIGHTFPROC)(GLenum light, GLenum pname, GLfloat param);
typedef void (CODEGEN_FUNCPTR *  PFNGLLIGHTFVPROC)(GLenum light, GLenum pname, const GLfloat * params);
typedef void (CODEGEN_FUNCPTR *  PFNGLLIGHTIPROC)(GLenum light, GLenum pname, GLint param);
typedef void (CODEGEN_FUNCPTR *  PFNGLLIGHTIVPROC)(GLenum light, GLenum pname, const GLint * params);
typedef void (CODEGEN_FUNCPTR *  PFNGLLINESTIPPLEPROC)(GLint factor, GLushort pattern);
typedef void (CODEGEN_FUNCPTR *  PFNGLLISTBASEPROC)(GLuint base);
typedef void (CODEGEN_FUNCPTR *  PFNGLLOADIDENTITYPROC)();
typedef void (CODEGEN_FUNCPTR *  PFNGLLOADMATRIXDPROC)(const GLdouble * m);
typedef void (CODEGEN_FUNCPTR *  PFNGLLOADMATRIXFPROC)(const GLfloat * m);
typedef void (CODEGEN_FUNCPTR *  PFNGLLOADNAMEPROC)(GLuint name);
typedef void (CODEGEN_FUNCPTR *  PFNGLMAP1DPROC)(GLenum target, GLdouble u1, GLdouble u2, GLint stride, GLint order, const GLdouble * points);
typedef void (CODEGEN_FUNCPTR *  PFNGLMAP1FPROC)(GLenum target, GLfloat u1, GLfloat u2, GLint stride, GLint order, const GLfloat * points);
typedef void (CODEGEN_FUNCPTR *  PFNGLMAP2DPROC)(GLenum target, GLdouble u1, GLdouble u2, GLint ustride, GLint uorder, GLdouble v1, GLdouble v2, GLint vstride, GLint vorder, const GLdouble * points);
typedef void (CODEGEN_FUNCPTR *  PFNGLMAP2FPROC)(GLenum target, GLfloat u1, GLfloat u2, GLint ustride, GLint uorder, GLfloat v1, GLfloat v2, GLint vstride, GLint vorder, const GLfloat * points);
typedef void (CODEGEN_FUNCPTR *  PFNGLMAPGRID1DPROC)(GLint un, GLdouble u1, GLdouble u2);
typedef void (CODEGEN_FUNCPTR *  PFNGLMAPGRID1FPROC)(GLint un, GLfloat u1, GLfloat u2);
typedef void (CODEGEN_FUNCPTR *  PFNGLMAPGRID2DPROC)(GLint un, GLdouble u1, GLdouble u2, GLint vn, GLdouble v1, GLdouble v2);
typedef void (CODEGEN_FUNCPTR *  PFNGLMAPGRID2FPROC)(GLint un, GLfloat u1, GLfloat u2, GLint vn, GLfloat v1, GLfloat v2);
typedef void (CODEGEN_FUNCPTR *  PFNGLMATERIALFPROC)(GLenum face, GLenum pname, GLfloat param);
typedef void (CODEGEN_FUNCPTR *  PFNGLMATERIALFVPROC)(GLenum face, GLenum pname, const GLfloat * params);
typedef void (CODEGEN_FUNCPTR *  PFNGLMATERIALIPROC)(GLenum face, GLenum pname, GLint param);
typedef void (CODEGEN_FUNCPTR *  PFNGLMATERIALIVPROC)(GLenum face, GLenum pname, const GLint * params);
typedef void (CODEGEN_FUNCPTR *  PFNGLMATRIXMODEPROC)(GLenum mode);
typedef void (CODEGEN_FUNCPTR *  PFNGLMULTMATRIXDPROC)(const GLdouble * m);
typedef void (CODEGEN_FUNCPTR *  PFNGLMULTMATRIXFPROC)(const GLfloat * m);
typedef void (CODEGEN_FUNCPTR *  PFNGLNEWLISTPROC)(GLuint list, GLenum mode);
typedef void (CODEGEN_FUNCPTR *  PFNGLNORMAL3BPROC)(GLbyte nx, GLbyte ny, GLbyte nz);
typedef void (CODEGEN_FUNCPTR *  PFNGLNORMAL3BVPROC)(const GLbyte * v);
typedef void (CODEGEN_FUNCPTR *  PFNGLNORMAL3DPROC)(GLdouble nx, GLdouble ny, GLdouble nz);
typedef void (CODEGEN_FUNCPTR *  PFNGLNORMAL3DVPROC)(const GLdouble * v);
typedef void (CODEGEN_FUNCPTR *  PFNGLNORMAL3FPROC)(GLfloat nx, GLfloat ny, GLfloat nz);
typedef void (CODEGEN_FUNCPTR *  PFNGLNORMAL3FVPROC)(const GLfloat * v);
typedef void (CODEGEN_FUNCPTR *  PFNGLNORMAL3IPROC)(GLint nx, GLint ny, GLint nz);
typedef void (CODEGEN_FUNCPTR *  PFNGLNORMAL3IVPROC)(const GLint * v);
typedef void (CODEGEN_FUNCPTR *  PFNGLNORMAL3SPROC)(GLshort nx, GLshort ny, GLshort nz);
typedef void (CODEGEN_FUNCPTR *  PFNGLNORMAL3SVPROC)(const GLshort * v);
typedef void (CODEGEN_FUNCPTR *  PFNGLORTHOPROC)(GLdouble left, GLdouble right, GLdouble bottom, GLdouble top, GLdouble zNear, GLdouble zFar);
typedef void (CODEGEN_FUNCPTR *  PFNGLPASSTHROUGHPROC)(GLfloat token);
typedef void (CODEGEN_FUNCPTR *  PFNGLPIXELMAPFVPROC)(GLenum map, GLsizei mapsize, const GLfloat * values);
typedef void (CODEGEN_FUNCPTR *  PFNGLPIXELMAPUIVPROC)(GLenum map, GLsizei mapsize, const GLuint * values);
typedef void (CODEGEN_FUNCPTR *  PFNGLPIXELMAPUSVPROC)(GLenum map, GLsizei mapsize, const GLushort * values);
typedef void (CODEGEN_FUNCPTR *  PFNGLPIXELTRANSFERFPROC)(GLenum pname, GLfloat param);
typedef void (CODEGEN_FUNCPTR *  PFNGLPIXELTRANSFERIPROC)(GLenum pname, GLint param);
typedef void (CODEGEN_FUNCPTR *  PFNGLPIXELZOOMPROC)(GLfloat xfactor, GLfloat yfactor);
typedef void (CODEGEN_FUNCPTR *  PFNGLPOLYGONSTIPPLEPROC)(const GLubyte * mask);
typedef void (CODEGEN_FUNCPTR *  PFNGLPOPATTRIBPROC)();
typedef void (CODEGEN_FUNCPTR *  PFNGLPOPMATRIXPROC)();
typedef void (CODEGEN_FUNCPTR *  PFNGLPOPNAMEPROC)();
typedef void (CODEGEN_FUNCPTR *  PFNGLPUSHATTRIBPROC)(GLbitfield mask);
typedef void (CODEGEN_FUNCPTR *  PFNGLPUSHMATRIXPROC)();
typedef void (CODEGEN_FUNCPTR *  PFNGLPUSHNAMEPROC)(GLuint name);
typedef void (CODEGEN_FUNCPTR *  PFNGLRASTERPOS2DPROC)(GLdouble x, GLdouble y);
typedef void (CODEGEN_FUNCPTR *  PFNGLRASTERPOS2DVPROC)(const GLdouble * v);
typedef void (CODEGEN_FUNCPTR *  PFNGLRASTERPOS2FPROC)(GLfloat x, GLfloat y);
typedef void (CODEGEN_FUNCPTR *  PFNGLRASTERPOS2FVPROC)(const GLfloat * v);
typedef void (CODEGEN_FUNCPTR *  PFNGLRASTERPOS2IPROC)(GLint x, GLint y);
typedef void (CODEGEN_FUNCPTR *  PFNGLRASTERPOS2IVPROC)(const GLint * v);
typedef void (CODEGEN_FUNCPTR *  PFNGLRASTERPOS2SPROC)(GLshort x, GLshort y);
typedef void (CODEGEN_FUNCPTR *  PFNGLRASTERPOS2SVPROC)(const GLshort * v);
typedef void (CODEGEN_FUNCPTR *  PFNGLRASTERPOS3DPROC)(GLdouble x, GLdouble y, GLdouble z);
typedef void (CODEGEN_FUNCPTR *  PFNGLRASTERPOS3DVPROC)(const GLdouble * v);
typedef void (CODEGEN_FUNCPTR *  PFNGLRASTERPOS3FPROC)(GLfloat x, GLfloat y, GLfloat z);
typedef void (CODEGEN_FUNCPTR *  PFNGLRASTERPOS3FVPROC)(const GLfloat * v);
typedef void (CODEGEN_FUNCPTR *  PFNGLRASTERPOS3IPROC)(GLint x, GLint y, GLint z);
typedef void (CODEGEN_FUNCPTR *  PFNGLRASTERPOS3IVPROC)(const GLint * v);
typedef void (CODEGEN_FUNCPTR *  PFNGLRASTERPOS3SPROC)(GLshort x, GLshort y, GLshort z);
typedef void (CODEGEN_FUNCPTR *  PFNGLRASTERPOS3SVPROC)(const GLshort * v);
typedef void (CODEGEN_FUNCPTR *  PFNGLRASTERPOS4DPROC)(GLdouble x, GLdouble y, GLdouble z, GLdouble w);
typedef void (CODEGEN_FUNCPTR *  PFNGLRASTERPOS4DVPROC)(const GLdouble * v);
typedef void (CODEGEN_FUNCPTR *  PFNGLRASTERPOS4FPROC)(GLfloat x, GLfloat y, GLfloat z, GLfloat w);
typedef void (CODEGEN_FUNCPTR *  PFNGLRASTERPOS4FVPROC)(const GLfloat * v);
typedef void (CODEGEN_FUNCPTR *  PFNGLRASTERPOS4IPROC)(GLint x, GLint y, GLint z, GLint w);
typedef void (CODEGEN_FUNCPTR *  PFNGLRASTERPOS4IVPROC)(const GLint * v);
typedef void (CODEGEN_FUNCPTR *  PFNGLRASTERPOS4SPROC)(GLshort x, GLshort y, GLshort z, GLshort w);
typedef void (CODEGEN_FUNCPTR *  PFNGLRASTERPOS4SVPROC)(const GLshort * v);
typedef void (CODEGEN_FUNCPTR *  PFNGLRECTDPROC)(GLdouble x1, GLdouble y1, GLdouble x2, GLdouble y2);
typedef void (CODEGEN_FUNCPTR *  PFNGLRECTDVPROC)(const GLdouble * v1, const GLdouble * v2);
typedef void (CODEGEN_FUNCPTR *  PFNGLRECTFPROC)(GLfloat x1, GLfloat y1, GLfloat x2, GLfloat y2);
typedef void (CODEGEN_FUNCPTR *  PFNGLRECTFVPROC)(const GLfloat * v1, const GLfloat * v2);
typedef void (CODEGEN_FUNCPTR *  PFNGLRECTIPROC)(GLint x1, GLint y1, GLint x2, GLint y2);
typedef void (CODEGEN_FUNCPTR *  PFNGLRECTIVPROC)(const GLint * v1, const GLint * v2);
typedef void (CODEGEN_FUNCPTR *  PFNGLRECTSPROC)(GLshort x1, GLshort y1, GLshort x2, GLshort y2);
typedef void (CODEGEN_FUNCPTR *  PFNGLRECTSVPROC)(const GLshort * v1, const GLshort * v2);
typedef GLint (CODEGEN_FUNCPTR *  PFNGLRENDERMODEPROC)(GLenum mode);
typedef void (CODEGEN_FUNCPTR *  PFNGLROTATEDPROC)(GLdouble angle, GLdouble x, GLdouble y, GLdouble z);
typedef void (CODEGEN_FUNCPTR *  PFNGLROTATEFPROC)(GLfloat angle, GLfloat x, GLfloat y, GLfloat z);
typedef void (CODEGEN_FUNCPTR *  PFNGLSCALEDPROC)(GLdouble x, GLdouble y, GLdouble z);
typedef void (CODEGEN_FUNCPTR *  PFNGLSCALEFPROC)(GLfloat x, GLfloat y, GLfloat z);
typedef void (CODEGEN_FUNCPTR *  PFNGLSELECTBUFFERPROC)(GLsizei size, GLuint * buffer);
typedef void (CODEGEN_FUNCPTR *  PFNGLSHADEMODELPROC)(GLenum mode);
typedef void (CODEGEN_FUNCPTR *  PFNGLTEXCOORD1DPROC)(GLdouble s);
typedef void (CODEGEN_FUNCPTR *  PFNGLTEXCOORD1DVPROC)(const GLdouble * v);
typedef void (CODEGEN_FUNCPTR *  PFNGLTEXCOORD1FPROC)(GLfloat s);
typedef void (CODEGEN_FUNCPTR *  PFNGLTEXCOORD1FVPROC)(const GLfloat * v);
typedef void (CODEGEN_FUNCPTR *  PFNGLTEXCOORD1IPROC)(GLint s);
typedef void (CODEGEN_FUNCPTR *  PFNGLTEXCOORD1IVPROC)(const GLint * v);
typedef void (CODEGEN_FUNCPTR *  PFNGLTEXCOORD1SPROC)(GLshort s);
typedef void (CODEGEN_FUNCPTR *  PFNGLTEXCOORD1SVPROC)(const GLshort * v);
typedef void (CODEGEN_FUNCPTR *  PFNGLTEXCOORD2DPROC)(GLdouble s, GLdouble t);
typedef void (CODEGEN_FUNCPTR *  PFNGLTEXCOORD2DVPROC)(const GLdouble * v);
typedef void (CODEGEN_FUNCPTR *  PFNGLTEXCOORD2FPROC)(GLfloat s, GLfloat t);
typedef void (CODEGEN_FUNCPTR *  PFNGLTEXCOORD2FVPROC)(const GLfloat * v);
typedef void (CODEGEN_FUNCPTR *  PFNGLTEXCOORD2IPROC)(GLint s, GLint t);
typedef void (CODEGEN_FUNCPTR *  PFNGLTEXCOORD2IVPROC)(const GLint * v);
typedef void (CODEGEN_FUNCPTR *  PFNGLTEXCOORD2SPROC)(GLshort s, GLshort t);
typedef void (CODEGEN_FUNCPTR *  PFNGLTEXCOORD2SVPROC)(const GLshort * v);
typedef void (CODEGEN_FUNCPTR *  PFNGLTEXCOORD3DPROC)(GLdouble s, GLdouble t, GLdouble r);
typedef void (CODEGEN_FUNCPTR *  PFNGLTEXCOORD3DVPROC)(const GLdouble * v);
typedef void (CODEGEN_FUNCPTR *  PFNGLTEXCOORD3FPROC)(GLfloat s, GLfloat t, GLfloat r);
typedef void (CODEGEN_FUNCPTR *  PFNGLTEXCOORD3FVPROC)(const GLfloat * v);
typedef void (CODEGEN_FUNCPTR *  PFNGLTEXCOORD3IPROC)(GLint s, GLint t, GLint r);
typedef void (CODEGEN_FUNCPTR *  PFNGLTEXCOORD3IVPROC)(const GLint * v);
typedef void (CODEGEN_FUNCPTR *  PFNGLTEXCOORD3SPROC)(GLshort s, GLshort t, GLshort r);
typedef void (CODEGEN_FUNCPTR *  PFNGLTEXCOORD3SVPROC)(const GLshort * v);
typedef void (CODEGEN_FUNCPTR *  PFNGLTEXCOORD4DPROC)(GLdouble s, GLdouble t, GLdouble r, GLdouble q);
typedef void (CODEGEN_FUNCPTR *  PFNGLTEXCOORD4DVPROC)(const GLdouble * v);
typedef void (CODEGEN_FUNCPTR *  PFNGLTEXCOORD4FPROC)(GLfloat s, GLfloat t, GLfloat r, GLfloat q);
typedef void (CODEGEN_FUNCPTR *  PFNGLTEXCOORD4FVPROC)(const GLfloat * v);
typedef void (CODEGEN_FUNCPTR *  PFNGLTEXCOORD4IPROC)(GLint s, GLint t, GLint r, GLint q);
typedef void (CODEGEN_FUNCPTR *  PFNGLTEXCOORD4IVPROC)(const GLint * v);
typedef void (CODEGEN_FUNCPTR *  PFNGLTEXCOORD4SPROC)(GLshort s, GLshort t, GLshort r, GLshort q);
typedef void (CODEGEN_FUNCPTR *  PFNGLTEXCOORD4SVPROC)(const GLshort * v);
typedef void (CODEGEN_FUNCPTR *  PFNGLTEXENVFPROC)(GLenum target, GLenum pname, GLfloat param);
typedef void (CODEGEN_FUNCPTR *  PFNGLTEXENVFVPROC)(GLenum target, GLenum pname, const GLfloat * params);
typedef void (CODEGEN_FUNCPTR *  PFNGLTEXENVIPROC)(GLenum target, GLenum pname, GLint param);
typedef void (CODEGEN_FUNCPTR *  PFNGLTEXENVIVPROC)(GLenum target, GLenum pname, const GLint * params);
typedef void (CODEGEN_FUNCPTR *  PFNGLTEXGENDPROC)(GLenum coord, GLenum pname, GLdouble param);
typedef void (CODEGEN_FUNCPTR *  PFNGLTEXGENDVPROC)(GLenum coord, GLenum pname, const GLdouble * params);
typedef void (CODEGEN_FUNCPTR *  PFNGLTEXGENFPROC)(GLenum coord, GLenum pname, GLfloat param);
typedef void (CODEGEN_FUNCPTR *  PFNGLTEXGENFVPROC)(GLenum coord, GLenum pname, const GLfloat * params);
typedef void (CODEGEN_FUNCPTR *  PFNGLTEXGENIPROC)(GLenum coord, GLenum pname, GLint param);
typedef void (CODEGEN_FUNCPTR *  PFNGLTEXGENIVPROC)(GLenum coord, GLenum pname, const GLint * params);
typedef void (CODEGEN_FUNCPTR *  PFNGLTRANSLATEDPROC)(GLdouble x, GLdouble y, GLdouble z);
typedef void (CODEGEN_FUNCPTR *  PFNGLTRANSLATEFPROC)(GLfloat x, GLfloat y, GLfloat z);
typedef void (CODEGEN_FUNCPTR *  PFNGLVERTEX2DPROC)(GLdouble x, GLdouble y);
typedef void (CODEGEN_FUNCPTR *  PFNGLVERTEX2DVPROC)(const GLdouble * v);
typedef void (CODEGEN_FUNCPTR *  PFNGLVERTEX2FPROC)(GLfloat x, GLfloat y);
typedef void (CODEGEN_FUNCPTR *  PFNGLVERTEX2FVPROC)(const GLfloat * v);
typedef void (CODEGEN_FUNCPTR *  PFNGLVERTEX2IPROC)(GLint x, GLint y);
typedef void (CODEGEN_FUNCPTR *  PFNGLVERTEX2IVPROC)(const GLint * v);
typedef void (CODEGEN_FUNCPTR *  PFNGLVERTEX2SPROC)(GLshort x, GLshort y);
typedef void (CODEGEN_FUNCPTR *  PFNGLVERTEX2SVPROC)(const GLshort * v);
typedef void (CODEGEN_FUNCPTR *  PFNGLVERTEX3DPROC)(GLdouble x, GLdouble y, GLdouble z);
typedef void (CODEGEN_FUNCPTR *  PFNGLVERTEX3DVPROC)(const GLdouble * v);
typedef void (CODEGEN_FUNCPTR *  PFNGLVERTEX3FPROC)(GLfloat x, GLfloat y, GLfloat z);
typedef void (CODEGEN_FUNCPTR *  PFNGLVERTEX3FVPROC)(const GLfloat * v);
typedef void (CODEGEN_FUNCPTR *  PFNGLVERTEX3IPROC)(GLint x, GLint y, GLint z);
typedef void (CODEGEN_FUNCPTR *  PFNGLVERTEX3IVPROC)(const GLint * v);
typedef void (CODEGEN_FUNCPTR *  PFNGLVERTEX3SPROC)(GLshort x, GLshort y, GLshort z);
typedef void (CODEGEN_FUNCPTR *  PFNGLVERTEX3SVPROC)(const GLshort * v);
typedef void (CODEGEN_FUNCPTR *  PFNGLVERTEX4DPROC)(GLdouble x, GLdouble y, GLdouble z, GLdouble w);
typedef void (CODEGEN_FUNCPTR *  PFNGLVERTEX4DVPROC)(const GLdouble * v);
typedef void (CODEGEN_FUNCPTR *  PFNGLVERTEX4FPROC)(GLfloat x, GLfloat y, GLfloat z, GLfloat w);
typedef void (CODEGEN_FUNCPTR *  PFNGLVERTEX4FVPROC)(const GLfloat * v);
typedef void (CODEGEN_FUNCPTR *  PFNGLVERTEX4IPROC)(GLint x, GLint y, GLint z, GLint w);
typedef void (CODEGEN_FUNCPTR *  PFNGLVERTEX4IVPROC)(const GLint * v);
typedef void (CODEGEN_FUNCPTR *  PFNGLVERTEX4SPROC)(GLshort x, GLshort y, GLshort z, GLshort w);
typedef void (CODEGEN_FUNCPTR *  PFNGLVERTEX4SVPROC)(const GLshort * v);

extern GLLOADAPI PFNGLACCUMPROC _funcptr_glAccum;
#define glAccum _funcptr_glAccum
extern GLLOADAPI PFNGLALPHAFUNCPROC _funcptr_glAlphaFunc;
#define glAlphaFunc _funcptr_glAlphaFunc
extern GLLOADAPI PFNGLBEGINPROC _funcptr_glBegin;
#define glBegin _funcptr_glBegin
extern GLLOADAPI PFNGLBITMAPPROC _funcptr_glBitmap;
#define glBitmap _funcptr_glBitmap
extern GLLOADAPI PFNGLCALLLISTPROC _funcptr_glCallList;
#define glCallList _funcptr_glCallList
extern GLLOADAPI PFNGLCALLLISTSPROC _funcptr_glCallLists;
#define glCallLists _funcptr_glCallLists
extern GLLOADAPI PFNGLCLEARACCUMPROC _funcptr_glClearAccum;
#define glClearAccum _funcptr_glClearAccum
extern GLLOADAPI PFNGLCLEARINDEXPROC _funcptr_glClearIndex;
#define glClearIndex _funcptr_glClearIndex
extern GLLOADAPI PFNGLCLIPPLANEPROC _funcptr_glClipPlane;
#define glClipPlane _funcptr_glClipPlane
extern GLLOADAPI PFNGLCOLOR3BPROC _funcptr_glColor3b;
#define glColor3b _funcptr_glColor3b
extern GLLOADAPI PFNGLCOLOR3BVPROC _funcptr_glColor3bv;
#define glColor3bv _funcptr_glColor3bv
extern GLLOADAPI PFNGLCOLOR3DPROC _funcptr_glColor3d;
#define glColor3d _funcptr_glColor3d
extern GLLOADAPI PFNGLCOLOR3DVPROC _funcptr_glColor3dv;
#define glColor3dv _funcptr_glColor3dv
extern GLLOADAPI PFNGLCOLOR3FPROC _funcptr_glColor3f;
#define glColor3f _funcptr_glColor3f
extern GLLOADAPI PFNGLCOLOR3FVPROC _funcptr_glColor3fv;
#define glColor3fv _funcptr_glColor3fv
extern GLLOADAPI PFNGLCOLOR3IPROC _funcptr_glColor3i;
#define glColor3i _funcptr_glColor3i
extern GLLOADAPI PFNGLCOLOR3IVPROC _funcptr_glColor3iv;
#define glColor3iv _funcptr_glColor3iv
extern GLLOADAPI PFNGLCOLOR3SPROC _funcptr_glColor3s;
#define glColor3s _funcptr_glColor3s
extern GLLOADAPI PFNGLCOLOR3SVPROC _funcptr_glColor3sv;
#define glColor3sv _funcptr_glColor3sv
extern GLLOADAPI PFNGLCOLOR3UBPROC _funcptr_glColor3ub;
#define glColor3ub _funcptr_glColor3ub
extern GLLOADAPI PFNGLCOLOR3UBVPROC _funcptr_glColor3ubv;
#define glColor3ubv _funcptr_glColor3ubv
extern GLLOADAPI PFNGLCOLOR3UIPROC _funcptr_glColor3ui;
#define glColor3ui _funcptr_glColor3ui
extern GLLOADAPI PFNGLCOLOR3UIVPROC _funcptr_glColor3uiv;
#define glColor3uiv _funcptr_glColor3uiv
extern GLLOADAPI PFNGLCOLOR3USPROC _funcptr_glColor3us;
#define glColor3us _funcptr_glColor3us
extern GLLOADAPI PFNGLCOLOR3USVPROC _funcptr_glColor3usv;
#define glColor3usv _funcptr_glColor3usv
extern GLLOADAPI PFNGLCOLOR4BPROC _funcptr_glColor4b;
#define glColor4b _funcptr_glColor4b
extern GLLOADAPI PFNGLCOLOR4BVPROC _funcptr_glColor4bv;
#define glColor4bv _funcptr_glColor4bv
extern GLLOADAPI PFNGLCOLOR4DPROC _funcptr_glColor4d;
#define glColor4d _funcptr_glColor4d
extern GLLOADAPI PFNGLCOLOR4DVPROC _funcptr_glColor4dv;
#define glColor4dv _funcptr_glColor4dv
extern GLLOADAPI PFNGLCOLOR4FPROC _funcptr_glColor4f;
#define glColor4f _funcptr_glColor4f
extern GLLOADAPI PFNGLCOLOR4FVPROC _funcptr_glColor4fv;
#define glColor4fv _funcptr_glColor4fv
extern GLLOADAPI PFNGLCOLOR4IPROC _funcptr_glColor4i;
#define glColor4i _funcptr_glColor4i
extern GLLOADAPI PFNGLCOLOR4IVPROC _funcptr_glColor4iv;
#define glColor4iv _funcptr_glColor4iv
extern GLLOADAPI PFNGLCOLOR4SPROC _funcptr_glColor4s;
#define glColor4s _funcptr_glColor4s
extern GLLOADAPI PFNGLCOLOR4SVPROC _funcptr_glColor4sv;
#define glColor4sv _funcptr_glColor4sv
extern GLLOADAPI PFNGLCOLOR4UBPROC _funcptr_glColor4ub;
#define glColor4ub _funcptr_glColor4ub
extern GLLOADAPI PFNGLCOLOR4UBVPROC _funcptr_glColor4ubv;
#define glColor4ubv _funcptr_glColor4ubv
extern GLLOADAPI PFNGLCOLOR4UIPROC _funcptr_glColor4ui;
#define glColor4ui _funcptr_glColor4ui
extern GLLOADAPI PFNGLCOLOR4UIVPROC _funcptr_glColor4uiv;
#define glColor4uiv _funcptr_glColor4uiv
extern GLLOADAPI PFNGLCOLOR4USPROC _funcptr_glColor4us;
#define glColor4us _funcptr_glColor4us
extern GLLOADAPI PFNGLCOLOR4USVPROC _funcptr_glColor4usv;
#define glColor4usv _funcptr_glColor4usv
extern GLLOADAPI PFNGLCOLORMATERIALPROC _funcptr_glColorMaterial;
#define glColorMaterial _funcptr_glColorMaterial
extern GLLOADAPI PFNGLCOPYPIXELSPROC _funcptr_glCopyPixels;
#define glCopyPixels _funcptr_glCopyPixels
extern GLLOADAPI PFNGLDELETELISTSPROC _funcptr_glDeleteLists;
#define glDeleteLists _funcptr_glDeleteLists
extern GLLOADAPI PFNGLDRAWPIXELSPROC _funcptr_glDrawPixels;
#define glDrawPixels _funcptr_glDrawPixels
extern GLLOADAPI PFNGLEDGEFLAGPROC _funcptr_glEdgeFlag;
#define glEdgeFlag _funcptr_glEdgeFlag
extern GLLOADAPI PFNGLEDGEFLAGVPROC _funcptr_glEdgeFlagv;
#define glEdgeFlagv _funcptr_glEdgeFlagv
extern GLLOADAPI PFNGLENDPROC _funcptr_glEnd;
#define glEnd _funcptr_glEnd
extern GLLOADAPI PFNGLENDLISTPROC _funcptr_glEndList;
#define glEndList _funcptr_glEndList
extern GLLOADAPI PFNGLEVALCOORD1DPROC _funcptr_glEvalCoord1d;
#define glEvalCoord1d _funcptr_glEvalCoord1d
extern GLLOADAPI PFNGLEVALCOORD1DVPROC _funcptr_glEvalCoord1dv;
#define glEvalCoord1dv _funcptr_glEvalCoord1dv
extern GLLOADAPI PFNGLEVALCOORD1FPROC _funcptr_glEvalCoord1f;
#define glEvalCoord1f _funcptr_glEvalCoord1f
extern GLLOADAPI PFNGLEVALCOORD1FVPROC _funcptr_glEvalCoord1fv;
#define glEvalCoord1fv _funcptr_glEvalCoord1fv
extern GLLOADAPI PFNGLEVALCOORD2DPROC _funcptr_glEvalCoord2d;
#define glEvalCoord2d _funcptr_glEvalCoord2d
extern GLLOADAPI PFNGLEVALCOORD2DVPROC _funcptr_glEvalCoord2dv;
#define glEvalCoord2dv _funcptr_glEvalCoord2dv
extern GLLOADAPI PFNGLEVALCOORD2FPROC _funcptr_glEvalCoord2f;
#define glEvalCoord2f _funcptr_glEvalCoord2f
extern GLLOADAPI PFNGLEVALCOORD2FVPROC _funcptr_glEvalCoord2fv;
#define glEvalCoord2fv _funcptr_glEvalCoord2fv
extern GLLOADAPI PFNGLEVALMESH1PROC _funcptr_glEvalMesh1;
#define glEvalMesh1 _funcptr_glEvalMesh1
extern GLLOADAPI PFNGLEVALMESH2PROC _funcptr_glEvalMesh2;
#define glEvalMesh2 _funcptr_glEvalMesh2
extern GLLOADAPI PFNGLEVALPOINT1PROC _funcptr_glEvalPoint1;
#define glEvalPoint1 _funcptr_glEvalPoint1
extern GLLOADAPI PFNGLEVALPOINT2PROC _funcptr_glEvalPoint2;
#define glEvalPoint2 _funcptr_glEvalPoint2
extern GLLOADAPI PFNGLFEEDBACKBUFFERPROC _funcptr_glFeedbackBuffer;
#define glFeedbackBuffer _funcptr_glFeedbackBuffer
extern GLLOADAPI PFNGLFOGFPROC _funcptr_glFogf;
#define glFogf _funcptr_glFogf
extern GLLOADAPI PFNGLFOGFVPROC _funcptr_glFogfv;
#define glFogfv _funcptr_glFogfv
extern GLLOADAPI PFNGLFOGIPROC _funcptr_glFogi;
#define glFogi _funcptr_glFogi
extern GLLOADAPI PFNGLFOGIVPROC _funcptr_glFogiv;
#define glFogiv _funcptr_glFogiv
extern GLLOADAPI PFNGLFRUSTUMPROC _funcptr_glFrustum;
#define glFrustum _funcptr_glFrustum
extern GLLOADAPI PFNGLGENLISTSPROC _funcptr_glGenLists;
#define glGenLists _funcptr_glGenLists
extern GLLOADAPI PFNGLGETCLIPPLANEPROC _funcptr_glGetClipPlane;
#define glGetClipPlane _funcptr_glGetClipPlane
extern GLLOADAPI PFNGLGETLIGHTFVPROC _funcptr_glGetLightfv;
#define glGetLightfv _funcptr_glGetLightfv
extern GLLOADAPI PFNGLGETLIGHTIVPROC _funcptr_glGetLightiv;
#define glGetLightiv _funcptr_glGetLightiv
extern GLLOADAPI PFNGLGETMAPDVPROC _funcptr_glGetMapdv;
#define glGetMapdv _funcptr_glGetMapdv
extern GLLOADAPI PFNGLGETMAPFVPROC _funcptr_glGetMapfv;
#define glGetMapfv _funcptr_glGetMapfv
extern GLLOADAPI PFNGLGETMAPIVPROC _funcptr_glGetMapiv;
#define glGetMapiv _funcptr_glGetMapiv
extern GLLOADAPI PFNGLGETMATERIALFVPROC _funcptr_glGetMaterialfv;
#define glGetMaterialfv _funcptr_glGetMaterialfv
extern GLLOADAPI PFNGLGETMATERIALIVPROC _funcptr_glGetMaterialiv;
#define glGetMaterialiv _funcptr_glGetMaterialiv
extern GLLOADAPI PFNGLGETPIXELMAPFVPROC _funcptr_glGetPixelMapfv;
#define glGetPixelMapfv _funcptr_glGetPixelMapfv
extern GLLOADAPI PFNGLGETPIXELMAPUIVPROC _funcptr_glGetPixelMapuiv;
#define glGetPixelMapuiv _funcptr_glGetPixelMapuiv
extern GLLOADAPI PFNGLGETPIXELMAPUSVPROC _funcptr_glGetPixelMapusv;
#define glGetPixelMapusv _funcptr_glGetPixelMapusv
extern GLLOADAPI PFNGLGETPOLYGONSTIPPLEPROC _funcptr_glGetPolygonStipple;
#define glGetPolygonStipple _funcptr_glGetPolygonStipple
extern GLLOADAPI PFNGLGETTEXENVFVPROC _funcptr_glGetTexEnvfv;
#define glGetTexEnvfv _funcptr_glGetTexEnvfv
extern GLLOADAPI PFNGLGETTEXENVIVPROC _funcptr_glGetTexEnviv;
#define glGetTexEnviv _funcptr_glGetTexEnviv
extern GLLOADAPI PFNGLGETTEXGENDVPROC _funcptr_glGetTexGendv;
#define glGetTexGendv _funcptr_glGetTexGendv
extern GLLOADAPI PFNGLGETTEXGENFVPROC _funcptr_glGetTexGenfv;
#define glGetTexGenfv _funcptr_glGetTexGenfv
extern GLLOADAPI PFNGLGETTEXGENIVPROC _funcptr_glGetTexGeniv;
#define glGetTexGeniv _funcptr_glGetTexGeniv
extern GLLOADAPI PFNGLINDEXMASKPROC _funcptr_glIndexMask;
#define glIndexMask _funcptr_glIndexMask
extern GLLOADAPI PFNGLINDEXDPROC _funcptr_glIndexd;
#define glIndexd _funcptr_glIndexd
extern GLLOADAPI PFNGLINDEXDVPROC _funcptr_glIndexdv;
#define glIndexdv _funcptr_glIndexdv
extern GLLOADAPI PFNGLINDEXFPROC _funcptr_glIndexf;
#define glIndexf _funcptr_glIndexf
extern GLLOADAPI PFNGLINDEXFVPROC _funcptr_glIndexfv;
#define glIndexfv _funcptr_glIndexfv
extern GLLOADAPI PFNGLINDEXIPROC _funcptr_glIndexi;
#define glIndexi _funcptr_glIndexi
extern GLLOADAPI PFNGLINDEXIVPROC _funcptr_glIndexiv;
#define glIndexiv _funcptr_glIndexiv
extern GLLOADAPI PFNGLINDEXSPROC _funcptr_glIndexs;
#define glIndexs _funcptr_glIndexs
extern GLLOADAPI PFNGLINDEXSVPROC _funcptr_glIndexsv;
#define glIndexsv _funcptr_glIndexsv
extern GLLOADAPI PFNGLINITNAMESPROC _funcptr_glInitNames;
#define glInitNames _funcptr_glInitNames
extern GLLOADAPI PFNGLISLISTPROC _funcptr_glIsList;
#define glIsList _funcptr_glIsList
extern GLLOADAPI PFNGLLIGHTMODELFPROC _funcptr_glLightModelf;
#define glLightModelf _funcptr_glLightModelf
extern GLLOADAPI PFNGLLIGHTMODELFVPROC _funcptr_glLightModelfv;
#define glLightModelfv _funcptr_glLightModelfv
extern GLLOADAPI PFNGLLIGHTMODELIPROC _funcptr_glLightModeli;
#define glLightModeli _funcptr_glLightModeli
extern GLLOADAPI PFNGLLIGHTMODELIVPROC _funcptr_glLightModeliv;
#define glLightModeliv _funcptr_glLightModeliv
extern GLLOADAPI PFNGLLIGHTFPROC _funcptr_glLightf;
#define glLightf _funcptr_glLightf
extern GLLOADAPI PFNGLLIGHTFVPROC _funcptr_glLightfv;
#define glLightfv _funcptr_glLightfv
extern GLLOADAPI PFNGLLIGHTIPROC _funcptr_glLighti;
#define glLighti _funcptr_glLighti
extern GLLOADAPI PFNGLLIGHTIVPROC _funcptr_glLightiv;
#define glLightiv _funcptr_glLightiv
extern GLLOADAPI PFNGLLINESTIPPLEPROC _funcptr_glLineStipple;
#define glLineStipple _funcptr_glLineStipple
extern GLLOADAPI PFNGLLISTBASEPROC _funcptr_glListBase;
#define glListBase _funcptr_glListBase
extern GLLOADAPI PFNGLLOADIDENTITYPROC _funcptr_glLoadIdentity;
#define glLoadIdentity _funcptr_glLoadIdentity
extern GLLOADAPI PFNGLLOADMATRIXDPROC _funcptr_glLoadMatrixd;
#define glLoadMatrixd _funcptr_glLoadMatrixd
extern GLLOADAPI PFNGLLOADMATRIXFPROC _funcptr_glLoadMatrixf;
#define glLoadMatrixf _funcptr_glLoadMatrixf
extern GLLOADAPI PFNGLLOADNAMEPROC _funcptr_glLoadName;
#define glLoadName _funcptr_glLoadName
extern GLLOADAPI PFNGLMAP1DPROC _funcptr_glMap1d;
#define glMap1d _funcptr_glMap1d
extern GLLOADAPI PFNGLMAP1FPROC _funcptr_glMap1f;
#define glMap1f _funcptr_glMap1f
extern GLLOADAPI PFNGLMAP2DPROC _funcptr_glMap2d;
#define glMap2d _funcptr_glMap2d
extern GLLOADAPI PFNGLMAP2FPROC _funcptr_glMap2f;
#define glMap2f _funcptr_glMap2f
extern GLLOADAPI PFNGLMAPGRID1DPROC _funcptr_glMapGrid1d;
#define glMapGrid1d _funcptr_glMapGrid1d
extern GLLOADAPI PFNGLMAPGRID1FPROC _funcptr_glMapGrid1f;
#define glMapGrid1f _funcptr_glMapGrid1f
extern GLLOADAPI PFNGLMAPGRID2DPROC _funcptr_glMapGrid2d;
#define glMapGrid2d _funcptr_glMapGrid2d
extern GLLOADAPI PFNGLMAPGRID2FPROC _funcptr_glMapGrid2f;
#define glMapGrid2f _funcptr_glMapGrid2f
extern GLLOADAPI PFNGLMATERIALFPROC _funcptr_glMaterialf;
#define glMaterialf _funcptr_glMaterialf
extern GLLOADAPI PFNGLMATERIALFVPROC _funcptr_glMaterialfv;
#define glMaterialfv _funcptr_glMaterialfv
extern GLLOADAPI PFNGLMATERIALIPROC _funcptr_glMateriali;
#define glMateriali _funcptr_glMateriali
extern GLLOADAPI PFNGLMATERIALIVPROC _funcptr_glMaterialiv;
#define glMaterialiv _funcptr_glMaterialiv
extern GLLOADAPI PFNGLMATRIXMODEPROC _funcptr_glMatrixMode;
#define glMatrixMode _funcptr_glMatrixMode
extern GLLOADAPI PFNGLMULTMATRIXDPROC _funcptr_glMultMatrixd;
#define glMultMatrixd _funcptr_glMultMatrixd
extern GLLOADAPI PFNGLMULTMATRIXFPROC _funcptr_glMultMatrixf;
#define glMultMatrixf _funcptr_glMultMatrixf
extern GLLOADAPI PFNGLNEWLISTPROC _funcptr_glNewList;
#define glNewList _funcptr_glNewList
extern GLLOADAPI PFNGLNORMAL3BPROC _funcptr_glNormal3b;
#define glNormal3b _funcptr_glNormal3b
extern GLLOADAPI PFNGLNORMAL3BVPROC _funcptr_glNormal3bv;
#define glNormal3bv _funcptr_glNormal3bv
extern GLLOADAPI PFNGLNORMAL3DPROC _funcptr_glNormal3d;
#define glNormal3d _funcptr_glNormal3d
extern GLLOADAPI PFNGLNORMAL3DVPROC _funcptr_glNormal3dv;
#define glNormal3dv _funcptr_glNormal3dv
extern GLLOADAPI PFNGLNORMAL3FPROC _funcptr_glNormal3f;
#define glNormal3f _funcptr_glNormal3f
extern GLLOADAPI PFNGLNORMAL3FVPROC _funcptr_glNormal3fv;
#define glNormal3fv _funcptr_glNormal3fv
extern GLLOADAPI PFNGLNORMAL3IPROC _funcptr_glNormal3i;
#define glNormal3i _funcptr_glNormal3i
extern GLLOADAPI PFNGLNORMAL3IVPROC _funcptr_glNormal3iv;
#define glNormal3iv _funcptr_glNormal3iv
extern GLLOADAPI PFNGLNORMAL3SPROC _funcptr_glNormal3s;
#define glNormal3s _funcptr_glNormal3s
extern GLLOADAPI PFNGLNORMAL3SVPROC _funcptr_glNormal3sv;
#define glNormal3sv _funcptr_glNormal3sv
extern GLLOADAPI PFNGLORTHOPROC _funcptr_glOrtho;
#define glOrtho _funcptr_glOrtho
extern GLLOADAPI PFNGLPASSTHROUGHPROC _funcptr_glPassThrough;
#define glPassThrough _funcptr_glPassThrough
extern GLLOADAPI PFNGLPIXELMAPFVPROC _funcptr_glPixelMapfv;
#define glPixelMapfv _funcptr_glPixelMapfv
extern GLLOADAPI PFNGLPIXELMAPUIVPROC _funcptr_glPixelMapuiv;
#define glPixelMapuiv _funcptr_glPixelMapuiv
extern GLLOADAPI PFNGLPIXELMAPUSVPROC _funcptr_glPixelMapusv;
#define glPixelMapusv _funcptr_glPixelMapusv
extern GLLOADAPI PFNGLPIXELTRANSFERFPROC _funcptr_glPixelTransferf;
#define glPixelTransferf _funcptr_glPixelTransferf
extern GLLOADAPI PFNGLPIXELTRANSFERIPROC _funcptr_glPixelTransferi;
#define glPixelTransferi _funcptr_glPixelTransferi
extern GLLOADAPI PFNGLPIXELZOOMPROC _funcptr_glPixelZoom;
#define glPixelZoom _funcptr_glPixelZoom
extern GLLOADAPI PFNGLPOLYGONSTIPPLEPROC _funcptr_glPolygonStipple;
#define glPolygonStipple _funcptr_glPolygonStipple
extern GLLOADAPI PFNGLPOPATTRIBPROC _funcptr_glPopAttrib;
#define glPopAttrib _funcptr_glPopAttrib
extern GLLOADAPI PFNGLPOPMATRIXPROC _funcptr_glPopMatrix;
#define glPopMatrix _funcptr_glPopMatrix
extern GLLOADAPI PFNGLPOPNAMEPROC _funcptr_glPopName;
#define glPopName _funcptr_glPopName
extern GLLOADAPI PFNGLPUSHATTRIBPROC _funcptr_glPushAttrib;
#define glPushAttrib _funcptr_glPushAttrib
extern GLLOADAPI PFNGLPUSHMATRIXPROC _funcptr_glPushMatrix;
#define glPushMatrix _funcptr_glPushMatrix
extern GLLOADAPI PFNGLPUSHNAMEPROC _funcptr_glPushName;
#define glPushName _funcptr_glPushName
extern GLLOADAPI PFNGLRASTERPOS2DPROC _funcptr_glRasterPos2d;
#define glRasterPos2d _funcptr_glRasterPos2d
extern GLLOADAPI PFNGLRASTERPOS2DVPROC _funcptr_glRasterPos2dv;
#define glRasterPos2dv _funcptr_glRasterPos2dv
extern GLLOADAPI PFNGLRASTERPOS2FPROC _funcptr_glRasterPos2f;
#define glRasterPos2f _funcptr_glRasterPos2f
extern GLLOADAPI PFNGLRASTERPOS2FVPROC _funcptr_glRasterPos2fv;
#define glRasterPos2fv _funcptr_glRasterPos2fv
extern GLLOADAPI PFNGLRASTERPOS2IPROC _funcptr_glRasterPos2i;
#define glRasterPos2i _funcptr_glRasterPos2i
extern GLLOADAPI PFNGLRASTERPOS2IVPROC _funcptr_glRasterPos2iv;
#define glRasterPos2iv _funcptr_glRasterPos2iv
extern GLLOADAPI PFNGLRASTERPOS2SPROC _funcptr_glRasterPos2s;
#define glRasterPos2s _funcptr_glRasterPos2s
extern GLLOADAPI PFNGLRASTERPOS2SVPROC _funcptr_glRasterPos2sv;
#define glRasterPos2sv _funcptr_glRasterPos2sv
extern GLLOADAPI PFNGLRASTERPOS3DPROC _funcptr_glRasterPos3d;
#define glRasterPos3d _funcptr_glRasterPos3d
extern GLLOADAPI PFNGLRASTERPOS3DVPROC _funcptr_glRasterPos3dv;
#define glRasterPos3dv _funcptr_glRasterPos3dv
extern GLLOADAPI PFNGLRASTERPOS3FPROC _funcptr_glRasterPos3f;
#define glRasterPos3f _funcptr_glRasterPos3f
extern GLLOADAPI PFNGLRASTERPOS3FVPROC _funcptr_glRasterPos3fv;
#define glRasterPos3fv _funcptr_glRasterPos3fv
extern GLLOADAPI PFNGLRASTERPOS3IPROC _funcptr_glRasterPos3i;
#define glRasterPos3i _funcptr_glRasterPos3i
extern GLLOADAPI PFNGLRASTERPOS3IVPROC _funcptr_glRasterPos3iv;
#define glRasterPos3iv _funcptr_glRasterPos3iv
extern GLLOADAPI PFNGLRASTERPOS3SPROC _funcptr_glRasterPos3s;
#define glRasterPos3s _funcptr_glRasterPos3s
extern GLLOADAPI PFNGLRASTERPOS3SVPROC _funcptr_glRasterPos3sv;
#define glRasterPos3sv _funcptr_glRasterPos3sv
extern GLLOADAPI PFNGLRASTERPOS4DPROC _funcptr_glRasterPos4d;
#define glRasterPos4d _funcptr_glRasterPos4d
extern GLLOADAPI PFNGLRASTERPOS4DVPROC _funcptr_glRasterPos4dv;
#define glRasterPos4dv _funcptr_glRasterPos4dv
extern GLLOADAPI PFNGLRASTERPOS4FPROC _funcptr_glRasterPos4f;
#define glRasterPos4f _funcptr_glRasterPos4f
extern GLLOADAPI PFNGLRASTERPOS4FVPROC _funcptr_glRasterPos4fv;
#define glRasterPos4fv _funcptr_glRasterPos4fv
extern GLLOADAPI PFNGLRASTERPOS4IPROC _funcptr_glRasterPos4i;
#define glRasterPos4i _funcptr_glRasterPos4i
extern GLLOADAPI PFNGLRASTERPOS4IVPROC _funcptr_glRasterPos4iv;
#define glRasterPos4iv _funcptr_glRasterPos4iv
extern GLLOADAPI PFNGLRASTERPOS4SPROC _funcptr_glRasterPos4s;
#define glRasterPos4s _funcptr_glRasterPos4s
extern GLLOADAPI PFNGLRASTERPOS4SVPROC _funcptr_glRasterPos4sv;
#define glRasterPos4sv _funcptr_glRasterPos4sv
extern GLLOADAPI PFNGLRECTDPROC _funcptr_glRectd;
#define glRectd _funcptr_glRectd
extern GLLOADAPI PFNGLRECTDVPROC _funcptr_glRectdv;
#define glRectdv _funcptr_glRectdv
extern GLLOADAPI PFNGLRECTFPROC _funcptr_glRectf;
#define glRectf _funcptr_glRectf
extern GLLOADAPI PFNGLRECTFVPROC _funcptr_glRectfv;
#define glRectfv _funcptr_glRectfv
extern GLLOADAPI PFNGLRECTIPROC _funcptr_glRecti;
#define glRecti _funcptr_glRecti
extern GLLOADAPI PFNGLRECTIVPROC _funcptr_glRectiv;
#define glRectiv _funcptr_glRectiv
extern GLLOADAPI PFNGLRECTSPROC _funcptr_glRects;
#define glRects _funcptr_glRects
extern GLLOADAPI PFNGLRECTSVPROC _funcptr_glRectsv;
#define glRectsv _funcptr_glRectsv
extern GLLOADAPI PFNGLRENDERMODEPROC _funcptr_glRenderMode;
#define glRenderMode _funcptr_glRenderMode
extern GLLOADAPI PFNGLROTATEDPROC _funcptr_glRotated;
#define glRotated _funcptr_glRotated
extern GLLOADAPI PFNGLROTATEFPROC _funcptr_glRotatef;
#define glRotatef _funcptr_glRotatef
extern GLLOADAPI PFNGLSCALEDPROC _funcptr_glScaled;
#define glScaled _funcptr_glScaled
extern GLLOADAPI PFNGLSCALEFPROC _funcptr_glScalef;
#define glScalef _funcptr_glScalef
extern GLLOADAPI PFNGLSELECTBUFFERPROC _funcptr_glSelectBuffer;
#define glSelectBuffer _funcptr_glSelectBuffer
extern GLLOADAPI PFNGLSHADEMODELPROC _funcptr_glShadeModel;
#define glShadeModel _funcptr_glShadeModel
extern GLLOADAPI PFNGLTEXCOORD1DPROC _funcptr_glTexCoord1d;
#define glTexCoord1d _funcptr_glTexCoord1d
extern GLLOADAPI PFNGLTEXCOORD1DVPROC _funcptr_glTexCoord1dv;
#define glTexCoord1dv _funcptr_glTexCoord1dv
extern GLLOADAPI PFNGLTEXCOORD1FPROC _funcptr_glTexCoord1f;
#define glTexCoord1f _funcptr_glTexCoord1f
extern GLLOADAPI PFNGLTEXCOORD1FVPROC _funcptr_glTexCoord1fv;
#define glTexCoord1fv _funcptr_glTexCoord1fv
extern GLLOADAPI PFNGLTEXCOORD1IPROC _funcptr_glTexCoord1i;
#define glTexCoord1i _funcptr_glTexCoord1i
extern GLLOADAPI PFNGLTEXCOORD1IVPROC _funcptr_glTexCoord1iv;
#define glTexCoord1iv _funcptr_glTexCoord1iv
extern GLLOADAPI PFNGLTEXCOORD1SPROC _funcptr_glTexCoord1s;
#define glTexCoord1s _funcptr_glTexCoord1s
extern GLLOADAPI PFNGLTEXCOORD1SVPROC _funcptr_glTexCoord1sv;
#define glTexCoord1sv _funcptr_glTexCoord1sv
extern GLLOADAPI PFNGLTEXCOORD2DPROC _funcptr_glTexCoord2d;
#define glTexCoord2d _funcptr_glTexCoord2d
extern GLLOADAPI PFNGLTEXCOORD2DVPROC _funcptr_glTexCoord2dv;
#define glTexCoord2dv _funcptr_glTexCoord2dv
extern GLLOADAPI PFNGLTEXCOORD2FPROC _funcptr_glTexCoord2f;
#define glTexCoord2f _funcptr_glTexCoord2f
extern GLLOADAPI PFNGLTEXCOORD2FVPROC _funcptr_glTexCoord2fv;
#define glTexCoord2fv _funcptr_glTexCoord2fv
extern GLLOADAPI PFNGLTEXCOORD2IPROC _funcptr_glTexCoord2i;
#define glTexCoord2i _funcptr_glTexCoord2i
extern GLLOADAPI PFNGLTEXCOORD2IVPROC _funcptr_glTexCoord2iv;
#define glTexCoord2iv _funcptr_glTexCoord2iv
extern GLLOADAPI PFNGLTEXCOORD2SPROC _funcptr_glTexCoord2s;
#define glTexCoord2s _funcptr_glTexCoord2s
extern GLLOADAPI PFNGLTEXCOORD2SVPROC _funcptr_glTexCoord2sv;
#define glTexCoord2sv _funcptr_glTexCoord2sv
extern GLLOADAPI PFNGLTEXCOORD3DPROC _funcptr_glTexCoord3d;
#define glTexCoord3d _funcptr_glTexCoord3d
extern GLLOADAPI PFNGLTEXCOORD3DVPROC _funcptr_glTexCoord3dv;
#define glTexCoord3dv _funcptr_glTexCoord3dv
extern GLLOADAPI PFNGLTEXCOORD3FPROC _funcptr_glTexCoord3f;
#define glTexCoord3f _funcptr_glTexCoord3f
extern GLLOADAPI PFNGLTEXCOORD3FVPROC _funcptr_glTexCoord3fv;
#define glTexCoord3fv _funcptr_glTexCoord3fv
extern GLLOADAPI PFNGLTEXCOORD3IPROC _funcptr_glTexCoord3i;
#define glTexCoord3i _funcptr_glTexCoord3i
extern GLLOADAPI PFNGLTEXCOORD3IVPROC _funcptr_glTexCoord3iv;
#define glTexCoord3iv _funcptr_glTexCoord3iv
extern GLLOADAPI PFNGLTEXCOORD3SPROC _funcptr_glTexCoord3s;
#define glTexCoord3s _funcptr_glTexCoord3s
extern GLLOADAPI PFNGLTEXCOORD3SVPROC _funcptr_glTexCoord3sv;
#define glTexCoord3sv _funcptr_glTexCoord3sv
extern GLLOADAPI PFNGLTEXCOORD4DPROC _funcptr_glTexCoord4d;
#define glTexCoord4d _funcptr_glTexCoord4d
extern GLLOADAPI PFNGLTEXCOORD4DVPROC _funcptr_glTexCoord4dv;
#define glTexCoord4dv _funcptr_glTexCoord4dv
extern GLLOADAPI PFNGLTEXCOORD4FPROC _funcptr_glTexCoord4f;
#define glTexCoord4f _funcptr_glTexCoord4f
extern GLLOADAPI PFNGLTEXCOORD4FVPROC _funcptr_glTexCoord4fv;
#define glTexCoord4fv _funcptr_glTexCoord4fv
extern GLLOADAPI PFNGLTEXCOORD4IPROC _funcptr_glTexCoord4i;
#define glTexCoord4i _funcptr_glTexCoord4i
extern GLLOADAPI PFNGLTEXCOORD4IVPROC _funcptr_glTexCoord4iv;
#define glTexCoord4iv _funcptr_glTexCoord4iv
extern GLLOADAPI PFNGLTEXCOORD4SPROC _funcptr_glTexCoord4s;
#define glTexCoord4s _funcptr_glTexCoord4s
extern GLLOADAPI PFNGLTEXCOORD4SVPROC _funcptr_glTexCoord4sv;
#define glTexCoord4sv _funcptr_glTexCoord4sv
extern GLLOADAPI PFNGLTEXENVFPROC _funcptr_glTexEnvf;
#define glTexEnvf _funcptr_glTexEnvf
extern GLLOADAPI PFNGLTEXENVFVPROC _funcptr_glTexEnvfv;
#define glTexEnvfv _funcptr_glTexEnvfv
extern GLLOADAPI PFNGLTEXENVIPROC _funcptr_glTexEnvi;
#define glTexEnvi _funcptr_glTexEnvi
extern GLLOADAPI PFNGLTEXENVIVPROC _funcptr_glTexEnviv;
#define glTexEnviv _funcptr_glTexEnviv
extern GLLOADAPI PFNGLTEXGENDPROC _funcptr_glTexGend;
#define glTexGend _funcptr_glTexGend
extern GLLOADAPI PFNGLTEXGENDVPROC _funcptr_glTexGendv;
#define glTexGendv _funcptr_glTexGendv
extern GLLOADAPI PFNGLTEXGENFPROC _funcptr_glTexGenf;
#define glTexGenf _funcptr_glTexGenf
extern GLLOADAPI PFNGLTEXGENFVPROC _funcptr_glTexGenfv;
#define glTexGenfv _funcptr_glTexGenfv
extern GLLOADAPI PFNGLTEXGENIPROC _funcptr_glTexGeni;
#define glTexGeni _funcptr_glTexGeni
extern GLLOADAPI PFNGLTEXGENIVPROC _funcptr_glTexGeniv;
#define glTexGeniv _funcptr_glTexGeniv
extern GLLOADAPI PFNGLTRANSLATEDPROC _funcptr_glTranslated;
#define glTranslated _funcptr_glTranslated
extern GLLOADAPI PFNGLTRANSLATEFPROC _funcptr_glTranslatef;
#define glTranslatef _funcptr_glTranslatef
extern GLLOADAPI PFNGLVERTEX2DPROC _funcptr_glVertex2d;
#define glVertex2d _funcptr_glVertex2d
extern GLLOADAPI PFNGLVERTEX2DVPROC _funcptr_glVertex2dv;
#define glVertex2dv _funcptr_glVertex2dv
extern GLLOADAPI PFNGLVERTEX2FPROC _funcptr_glVertex2f;
#define glVertex2f _funcptr_glVertex2f
extern GLLOADAPI PFNGLVERTEX2FVPROC _funcptr_glVertex2fv;
#define glVertex2fv _funcptr_glVertex2fv
extern GLLOADAPI PFNGLVERTEX2IPROC _funcptr_glVertex2i;
#define glVertex2i _funcptr_glVertex2i
extern GLLOADAPI PFNGLVERTEX2IVPROC _funcptr_glVertex2iv;
#define glVertex2iv _funcptr_glVertex2iv
extern GLLOADAPI PFNGLVERTEX2SPROC _funcptr_glVertex2s;
#define glVertex2s _funcptr_glVertex2s
extern GLLOADAPI PFNGLVERTEX2SVPROC _funcptr_glVertex2sv;
#define glVertex2sv _funcptr_glVertex2sv
extern GLLOADAPI PFNGLVERTEX3DPROC _funcptr_glVertex3d;
#define glVertex3d _funcptr_glVertex3d
extern GLLOADAPI PFNGLVERTEX3DVPROC _funcptr_glVertex3dv;
#define glVertex3dv _funcptr_glVertex3dv
extern GLLOADAPI PFNGLVERTEX3FPROC _funcptr_glVertex3f;
#define glVertex3f _funcptr_glVertex3f
extern GLLOADAPI PFNGLVERTEX3FVPROC _funcptr_glVertex3fv;
#define glVertex3fv _funcptr_glVertex3fv
extern GLLOADAPI PFNGLVERTEX3IPROC _funcptr_glVertex3i;
#define glVertex3i _funcptr_glVertex3i
extern GLLOADAPI PFNGLVERTEX3IVPROC _funcptr_glVertex3iv;
#define glVertex3iv _funcptr_glVertex3iv
extern GLLOADAPI PFNGLVERTEX3SPROC _funcptr_glVertex3s;
#define glVertex3s _funcptr_glVertex3s
extern GLLOADAPI PFNGLVERTEX3SVPROC _funcptr_glVertex3sv;
#define glVertex3sv _funcptr_glVertex3sv
extern GLLOADAPI PFNGLVERTEX4DPROC _funcptr_glVertex4d;
#define glVertex4d _funcptr_glVertex4d
extern GLLOADAPI PFNGLVERTEX4DVPROC _funcptr_glVertex4dv;
#define glVertex4dv _funcptr_glVertex4dv
extern GLLOADAPI PFNGLVERTEX4FPROC _funcptr_glVertex4f;
#define glVertex4f _funcptr_glVertex4f
extern GLLOADAPI PFNGLVERTEX4FVPROC _funcptr_glVertex4fv;
#define glVertex4fv _funcptr_glVertex4fv
extern GLLOADAPI PFNGLVERTEX4IPROC _funcptr_glVertex4i;
#define glVertex4i _funcptr_glVertex4i
extern GLLOADAPI PFNGLVERTEX4IVPROC _funcptr_glVertex4iv;
#define glVertex4iv _funcptr_glVertex4iv
extern GLLOADAPI PFNGLVERTEX4SPROC _funcptr_glVertex4s;
#define glVertex4s _funcptr_glVertex4s
extern GLLOADAPI PFNGLVERTEX4SVPROC _funcptr_glVertex4sv;
#define glVertex4sv _funcptr_glVertex4sv

#ifdef __cplusplus
}
#endif /*__cplusplus*/
#endif /*OPENGL_GEN_CORE_REM1_0_H*/
