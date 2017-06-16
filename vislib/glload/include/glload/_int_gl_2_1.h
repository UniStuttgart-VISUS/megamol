#ifndef OPENGL_GEN_CORE_2_1_H
#define OPENGL_GEN_CORE_2_1_H
#include "_int_gl_load_ie.h"

#ifdef __cplusplus
extern "C" {
#endif /*__cplusplus*/
#define GL_COMPRESSED_SRGB 0x8C48
#define GL_COMPRESSED_SRGB_ALPHA 0x8C49
#define GL_FLOAT_MAT2x3 0x8B65
#define GL_FLOAT_MAT2x4 0x8B66
#define GL_FLOAT_MAT3x2 0x8B67
#define GL_FLOAT_MAT3x4 0x8B68
#define GL_FLOAT_MAT4x2 0x8B69
#define GL_FLOAT_MAT4x3 0x8B6A
#define GL_PIXEL_PACK_BUFFER 0x88EB
#define GL_PIXEL_PACK_BUFFER_BINDING 0x88ED
#define GL_PIXEL_UNPACK_BUFFER 0x88EC
#define GL_PIXEL_UNPACK_BUFFER_BINDING 0x88EF
#define GL_SRGB 0x8C40
#define GL_SRGB8 0x8C41
#define GL_SRGB8_ALPHA8 0x8C43
#define GL_SRGB_ALPHA 0x8C42

typedef void (CODEGEN_FUNCPTR *  PFNGLUNIFORMMATRIX2X3FVPROC)(GLint location, GLsizei count, GLboolean transpose, const GLfloat * value);
typedef void (CODEGEN_FUNCPTR *  PFNGLUNIFORMMATRIX2X4FVPROC)(GLint location, GLsizei count, GLboolean transpose, const GLfloat * value);
typedef void (CODEGEN_FUNCPTR *  PFNGLUNIFORMMATRIX3X2FVPROC)(GLint location, GLsizei count, GLboolean transpose, const GLfloat * value);
typedef void (CODEGEN_FUNCPTR *  PFNGLUNIFORMMATRIX3X4FVPROC)(GLint location, GLsizei count, GLboolean transpose, const GLfloat * value);
typedef void (CODEGEN_FUNCPTR *  PFNGLUNIFORMMATRIX4X2FVPROC)(GLint location, GLsizei count, GLboolean transpose, const GLfloat * value);
typedef void (CODEGEN_FUNCPTR *  PFNGLUNIFORMMATRIX4X3FVPROC)(GLint location, GLsizei count, GLboolean transpose, const GLfloat * value);

extern GLLOADAPI PFNGLUNIFORMMATRIX2X3FVPROC _funcptr_glUniformMatrix2x3fv;
#define glUniformMatrix2x3fv _funcptr_glUniformMatrix2x3fv
extern GLLOADAPI PFNGLUNIFORMMATRIX2X4FVPROC _funcptr_glUniformMatrix2x4fv;
#define glUniformMatrix2x4fv _funcptr_glUniformMatrix2x4fv
extern GLLOADAPI PFNGLUNIFORMMATRIX3X2FVPROC _funcptr_glUniformMatrix3x2fv;
#define glUniformMatrix3x2fv _funcptr_glUniformMatrix3x2fv
extern GLLOADAPI PFNGLUNIFORMMATRIX3X4FVPROC _funcptr_glUniformMatrix3x4fv;
#define glUniformMatrix3x4fv _funcptr_glUniformMatrix3x4fv
extern GLLOADAPI PFNGLUNIFORMMATRIX4X2FVPROC _funcptr_glUniformMatrix4x2fv;
#define glUniformMatrix4x2fv _funcptr_glUniformMatrix4x2fv
extern GLLOADAPI PFNGLUNIFORMMATRIX4X3FVPROC _funcptr_glUniformMatrix4x3fv;
#define glUniformMatrix4x3fv _funcptr_glUniformMatrix4x3fv

#ifdef __cplusplus
}
#endif /*__cplusplus*/
#endif /*OPENGL_GEN_CORE_2_1_H*/
