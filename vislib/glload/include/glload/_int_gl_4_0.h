#ifndef OPENGL_GEN_CORE_4_0_H
#define OPENGL_GEN_CORE_4_0_H
#include "_int_gl_load_ie.h"

#ifdef __cplusplus
extern "C" {
#endif /*__cplusplus*/
#define GL_INT_SAMPLER_CUBE_MAP_ARRAY 0x900E
#define GL_MAX_PROGRAM_TEXTURE_GATHER_OFFSET 0x8E5F
#define GL_MIN_PROGRAM_TEXTURE_GATHER_OFFSET 0x8E5E
#define GL_MIN_SAMPLE_SHADING_VALUE 0x8C37
#define GL_PROXY_TEXTURE_CUBE_MAP_ARRAY 0x900B
#define GL_SAMPLER_CUBE_MAP_ARRAY 0x900C
#define GL_SAMPLER_CUBE_MAP_ARRAY_SHADOW 0x900D
#define GL_SAMPLE_SHADING 0x8C36
#define GL_TEXTURE_BINDING_CUBE_MAP_ARRAY 0x900A
#define GL_UNSIGNED_INT_SAMPLER_CUBE_MAP_ARRAY 0x900F

typedef void (CODEGEN_FUNCPTR *  PFNGLBEGINQUERYINDEXEDPROC)(GLenum target, GLuint index, GLuint id);
typedef void (CODEGEN_FUNCPTR *  PFNGLBINDTRANSFORMFEEDBACKPROC)(GLenum target, GLuint id);
typedef void (CODEGEN_FUNCPTR *  PFNGLBLENDEQUATIONSEPARATEIPROC)(GLuint buf, GLenum modeRGB, GLenum modeAlpha);
typedef void (CODEGEN_FUNCPTR *  PFNGLBLENDEQUATIONIPROC)(GLuint buf, GLenum mode);
typedef void (CODEGEN_FUNCPTR *  PFNGLBLENDFUNCSEPARATEIPROC)(GLuint buf, GLenum srcRGB, GLenum dstRGB, GLenum srcAlpha, GLenum dstAlpha);
typedef void (CODEGEN_FUNCPTR *  PFNGLBLENDFUNCIPROC)(GLuint buf, GLenum src, GLenum dst);
typedef void (CODEGEN_FUNCPTR *  PFNGLDELETETRANSFORMFEEDBACKSPROC)(GLsizei n, const GLuint * ids);
typedef void (CODEGEN_FUNCPTR *  PFNGLDRAWARRAYSINDIRECTPROC)(GLenum mode, const GLvoid * indirect);
typedef void (CODEGEN_FUNCPTR *  PFNGLDRAWELEMENTSINDIRECTPROC)(GLenum mode, GLenum type, const GLvoid * indirect);
typedef void (CODEGEN_FUNCPTR *  PFNGLDRAWTRANSFORMFEEDBACKPROC)(GLenum mode, GLuint id);
typedef void (CODEGEN_FUNCPTR *  PFNGLDRAWTRANSFORMFEEDBACKSTREAMPROC)(GLenum mode, GLuint id, GLuint stream);
typedef void (CODEGEN_FUNCPTR *  PFNGLENDQUERYINDEXEDPROC)(GLenum target, GLuint index);
typedef void (CODEGEN_FUNCPTR *  PFNGLGENTRANSFORMFEEDBACKSPROC)(GLsizei n, GLuint * ids);
typedef void (CODEGEN_FUNCPTR *  PFNGLGETACTIVESUBROUTINENAMEPROC)(GLuint program, GLenum shadertype, GLuint index, GLsizei bufsize, GLsizei * length, GLchar * name);
typedef void (CODEGEN_FUNCPTR *  PFNGLGETACTIVESUBROUTINEUNIFORMNAMEPROC)(GLuint program, GLenum shadertype, GLuint index, GLsizei bufsize, GLsizei * length, GLchar * name);
typedef void (CODEGEN_FUNCPTR *  PFNGLGETACTIVESUBROUTINEUNIFORMIVPROC)(GLuint program, GLenum shadertype, GLuint index, GLenum pname, GLint * values);
typedef void (CODEGEN_FUNCPTR *  PFNGLGETPROGRAMSTAGEIVPROC)(GLuint program, GLenum shadertype, GLenum pname, GLint * values);
typedef void (CODEGEN_FUNCPTR *  PFNGLGETQUERYINDEXEDIVPROC)(GLenum target, GLuint index, GLenum pname, GLint * params);
typedef GLuint (CODEGEN_FUNCPTR *  PFNGLGETSUBROUTINEINDEXPROC)(GLuint program, GLenum shadertype, const GLchar * name);
typedef GLint (CODEGEN_FUNCPTR *  PFNGLGETSUBROUTINEUNIFORMLOCATIONPROC)(GLuint program, GLenum shadertype, const GLchar * name);
typedef void (CODEGEN_FUNCPTR *  PFNGLGETUNIFORMSUBROUTINEUIVPROC)(GLenum shadertype, GLint location, GLuint * params);
typedef void (CODEGEN_FUNCPTR *  PFNGLGETUNIFORMDVPROC)(GLuint program, GLint location, GLdouble * params);
typedef GLboolean (CODEGEN_FUNCPTR *  PFNGLISTRANSFORMFEEDBACKPROC)(GLuint id);
typedef void (CODEGEN_FUNCPTR *  PFNGLMINSAMPLESHADINGPROC)(GLfloat value);
typedef void (CODEGEN_FUNCPTR *  PFNGLPATCHPARAMETERFVPROC)(GLenum pname, const GLfloat * values);
typedef void (CODEGEN_FUNCPTR *  PFNGLPATCHPARAMETERIPROC)(GLenum pname, GLint value);
typedef void (CODEGEN_FUNCPTR *  PFNGLPAUSETRANSFORMFEEDBACKPROC)();
typedef void (CODEGEN_FUNCPTR *  PFNGLRESUMETRANSFORMFEEDBACKPROC)();
typedef void (CODEGEN_FUNCPTR *  PFNGLUNIFORM1DPROC)(GLint location, GLdouble x);
typedef void (CODEGEN_FUNCPTR *  PFNGLUNIFORM1DVPROC)(GLint location, GLsizei count, const GLdouble * value);
typedef void (CODEGEN_FUNCPTR *  PFNGLUNIFORM2DPROC)(GLint location, GLdouble x, GLdouble y);
typedef void (CODEGEN_FUNCPTR *  PFNGLUNIFORM2DVPROC)(GLint location, GLsizei count, const GLdouble * value);
typedef void (CODEGEN_FUNCPTR *  PFNGLUNIFORM3DPROC)(GLint location, GLdouble x, GLdouble y, GLdouble z);
typedef void (CODEGEN_FUNCPTR *  PFNGLUNIFORM3DVPROC)(GLint location, GLsizei count, const GLdouble * value);
typedef void (CODEGEN_FUNCPTR *  PFNGLUNIFORM4DPROC)(GLint location, GLdouble x, GLdouble y, GLdouble z, GLdouble w);
typedef void (CODEGEN_FUNCPTR *  PFNGLUNIFORM4DVPROC)(GLint location, GLsizei count, const GLdouble * value);
typedef void (CODEGEN_FUNCPTR *  PFNGLUNIFORMMATRIX2DVPROC)(GLint location, GLsizei count, GLboolean transpose, const GLdouble * value);
typedef void (CODEGEN_FUNCPTR *  PFNGLUNIFORMMATRIX2X3DVPROC)(GLint location, GLsizei count, GLboolean transpose, const GLdouble * value);
typedef void (CODEGEN_FUNCPTR *  PFNGLUNIFORMMATRIX2X4DVPROC)(GLint location, GLsizei count, GLboolean transpose, const GLdouble * value);
typedef void (CODEGEN_FUNCPTR *  PFNGLUNIFORMMATRIX3DVPROC)(GLint location, GLsizei count, GLboolean transpose, const GLdouble * value);
typedef void (CODEGEN_FUNCPTR *  PFNGLUNIFORMMATRIX3X2DVPROC)(GLint location, GLsizei count, GLboolean transpose, const GLdouble * value);
typedef void (CODEGEN_FUNCPTR *  PFNGLUNIFORMMATRIX3X4DVPROC)(GLint location, GLsizei count, GLboolean transpose, const GLdouble * value);
typedef void (CODEGEN_FUNCPTR *  PFNGLUNIFORMMATRIX4DVPROC)(GLint location, GLsizei count, GLboolean transpose, const GLdouble * value);
typedef void (CODEGEN_FUNCPTR *  PFNGLUNIFORMMATRIX4X2DVPROC)(GLint location, GLsizei count, GLboolean transpose, const GLdouble * value);
typedef void (CODEGEN_FUNCPTR *  PFNGLUNIFORMMATRIX4X3DVPROC)(GLint location, GLsizei count, GLboolean transpose, const GLdouble * value);
typedef void (CODEGEN_FUNCPTR *  PFNGLUNIFORMSUBROUTINESUIVPROC)(GLenum shadertype, GLsizei count, const GLuint * indices);

extern GLLOADAPI PFNGLBEGINQUERYINDEXEDPROC _funcptr_glBeginQueryIndexed;
#define glBeginQueryIndexed _funcptr_glBeginQueryIndexed
extern GLLOADAPI PFNGLBINDTRANSFORMFEEDBACKPROC _funcptr_glBindTransformFeedback;
#define glBindTransformFeedback _funcptr_glBindTransformFeedback
extern GLLOADAPI PFNGLBLENDEQUATIONSEPARATEIPROC _funcptr_glBlendEquationSeparatei;
#define glBlendEquationSeparatei _funcptr_glBlendEquationSeparatei
extern GLLOADAPI PFNGLBLENDEQUATIONIPROC _funcptr_glBlendEquationi;
#define glBlendEquationi _funcptr_glBlendEquationi
extern GLLOADAPI PFNGLBLENDFUNCSEPARATEIPROC _funcptr_glBlendFuncSeparatei;
#define glBlendFuncSeparatei _funcptr_glBlendFuncSeparatei
extern GLLOADAPI PFNGLBLENDFUNCIPROC _funcptr_glBlendFunci;
#define glBlendFunci _funcptr_glBlendFunci
extern GLLOADAPI PFNGLDELETETRANSFORMFEEDBACKSPROC _funcptr_glDeleteTransformFeedbacks;
#define glDeleteTransformFeedbacks _funcptr_glDeleteTransformFeedbacks
extern GLLOADAPI PFNGLDRAWARRAYSINDIRECTPROC _funcptr_glDrawArraysIndirect;
#define glDrawArraysIndirect _funcptr_glDrawArraysIndirect
extern GLLOADAPI PFNGLDRAWELEMENTSINDIRECTPROC _funcptr_glDrawElementsIndirect;
#define glDrawElementsIndirect _funcptr_glDrawElementsIndirect
extern GLLOADAPI PFNGLDRAWTRANSFORMFEEDBACKPROC _funcptr_glDrawTransformFeedback;
#define glDrawTransformFeedback _funcptr_glDrawTransformFeedback
extern GLLOADAPI PFNGLDRAWTRANSFORMFEEDBACKSTREAMPROC _funcptr_glDrawTransformFeedbackStream;
#define glDrawTransformFeedbackStream _funcptr_glDrawTransformFeedbackStream
extern GLLOADAPI PFNGLENDQUERYINDEXEDPROC _funcptr_glEndQueryIndexed;
#define glEndQueryIndexed _funcptr_glEndQueryIndexed
extern GLLOADAPI PFNGLGENTRANSFORMFEEDBACKSPROC _funcptr_glGenTransformFeedbacks;
#define glGenTransformFeedbacks _funcptr_glGenTransformFeedbacks
extern GLLOADAPI PFNGLGETACTIVESUBROUTINENAMEPROC _funcptr_glGetActiveSubroutineName;
#define glGetActiveSubroutineName _funcptr_glGetActiveSubroutineName
extern GLLOADAPI PFNGLGETACTIVESUBROUTINEUNIFORMNAMEPROC _funcptr_glGetActiveSubroutineUniformName;
#define glGetActiveSubroutineUniformName _funcptr_glGetActiveSubroutineUniformName
extern GLLOADAPI PFNGLGETACTIVESUBROUTINEUNIFORMIVPROC _funcptr_glGetActiveSubroutineUniformiv;
#define glGetActiveSubroutineUniformiv _funcptr_glGetActiveSubroutineUniformiv
extern GLLOADAPI PFNGLGETPROGRAMSTAGEIVPROC _funcptr_glGetProgramStageiv;
#define glGetProgramStageiv _funcptr_glGetProgramStageiv
extern GLLOADAPI PFNGLGETQUERYINDEXEDIVPROC _funcptr_glGetQueryIndexediv;
#define glGetQueryIndexediv _funcptr_glGetQueryIndexediv
extern GLLOADAPI PFNGLGETSUBROUTINEINDEXPROC _funcptr_glGetSubroutineIndex;
#define glGetSubroutineIndex _funcptr_glGetSubroutineIndex
extern GLLOADAPI PFNGLGETSUBROUTINEUNIFORMLOCATIONPROC _funcptr_glGetSubroutineUniformLocation;
#define glGetSubroutineUniformLocation _funcptr_glGetSubroutineUniformLocation
extern GLLOADAPI PFNGLGETUNIFORMSUBROUTINEUIVPROC _funcptr_glGetUniformSubroutineuiv;
#define glGetUniformSubroutineuiv _funcptr_glGetUniformSubroutineuiv
extern GLLOADAPI PFNGLGETUNIFORMDVPROC _funcptr_glGetUniformdv;
#define glGetUniformdv _funcptr_glGetUniformdv
extern GLLOADAPI PFNGLISTRANSFORMFEEDBACKPROC _funcptr_glIsTransformFeedback;
#define glIsTransformFeedback _funcptr_glIsTransformFeedback
extern GLLOADAPI PFNGLMINSAMPLESHADINGPROC _funcptr_glMinSampleShading;
#define glMinSampleShading _funcptr_glMinSampleShading
extern GLLOADAPI PFNGLPATCHPARAMETERFVPROC _funcptr_glPatchParameterfv;
#define glPatchParameterfv _funcptr_glPatchParameterfv
extern GLLOADAPI PFNGLPATCHPARAMETERIPROC _funcptr_glPatchParameteri;
#define glPatchParameteri _funcptr_glPatchParameteri
extern GLLOADAPI PFNGLPAUSETRANSFORMFEEDBACKPROC _funcptr_glPauseTransformFeedback;
#define glPauseTransformFeedback _funcptr_glPauseTransformFeedback
extern GLLOADAPI PFNGLRESUMETRANSFORMFEEDBACKPROC _funcptr_glResumeTransformFeedback;
#define glResumeTransformFeedback _funcptr_glResumeTransformFeedback
extern GLLOADAPI PFNGLUNIFORM1DPROC _funcptr_glUniform1d;
#define glUniform1d _funcptr_glUniform1d
extern GLLOADAPI PFNGLUNIFORM1DVPROC _funcptr_glUniform1dv;
#define glUniform1dv _funcptr_glUniform1dv
extern GLLOADAPI PFNGLUNIFORM2DPROC _funcptr_glUniform2d;
#define glUniform2d _funcptr_glUniform2d
extern GLLOADAPI PFNGLUNIFORM2DVPROC _funcptr_glUniform2dv;
#define glUniform2dv _funcptr_glUniform2dv
extern GLLOADAPI PFNGLUNIFORM3DPROC _funcptr_glUniform3d;
#define glUniform3d _funcptr_glUniform3d
extern GLLOADAPI PFNGLUNIFORM3DVPROC _funcptr_glUniform3dv;
#define glUniform3dv _funcptr_glUniform3dv
extern GLLOADAPI PFNGLUNIFORM4DPROC _funcptr_glUniform4d;
#define glUniform4d _funcptr_glUniform4d
extern GLLOADAPI PFNGLUNIFORM4DVPROC _funcptr_glUniform4dv;
#define glUniform4dv _funcptr_glUniform4dv
extern GLLOADAPI PFNGLUNIFORMMATRIX2DVPROC _funcptr_glUniformMatrix2dv;
#define glUniformMatrix2dv _funcptr_glUniformMatrix2dv
extern GLLOADAPI PFNGLUNIFORMMATRIX2X3DVPROC _funcptr_glUniformMatrix2x3dv;
#define glUniformMatrix2x3dv _funcptr_glUniformMatrix2x3dv
extern GLLOADAPI PFNGLUNIFORMMATRIX2X4DVPROC _funcptr_glUniformMatrix2x4dv;
#define glUniformMatrix2x4dv _funcptr_glUniformMatrix2x4dv
extern GLLOADAPI PFNGLUNIFORMMATRIX3DVPROC _funcptr_glUniformMatrix3dv;
#define glUniformMatrix3dv _funcptr_glUniformMatrix3dv
extern GLLOADAPI PFNGLUNIFORMMATRIX3X2DVPROC _funcptr_glUniformMatrix3x2dv;
#define glUniformMatrix3x2dv _funcptr_glUniformMatrix3x2dv
extern GLLOADAPI PFNGLUNIFORMMATRIX3X4DVPROC _funcptr_glUniformMatrix3x4dv;
#define glUniformMatrix3x4dv _funcptr_glUniformMatrix3x4dv
extern GLLOADAPI PFNGLUNIFORMMATRIX4DVPROC _funcptr_glUniformMatrix4dv;
#define glUniformMatrix4dv _funcptr_glUniformMatrix4dv
extern GLLOADAPI PFNGLUNIFORMMATRIX4X2DVPROC _funcptr_glUniformMatrix4x2dv;
#define glUniformMatrix4x2dv _funcptr_glUniformMatrix4x2dv
extern GLLOADAPI PFNGLUNIFORMMATRIX4X3DVPROC _funcptr_glUniformMatrix4x3dv;
#define glUniformMatrix4x3dv _funcptr_glUniformMatrix4x3dv
extern GLLOADAPI PFNGLUNIFORMSUBROUTINESUIVPROC _funcptr_glUniformSubroutinesuiv;
#define glUniformSubroutinesuiv _funcptr_glUniformSubroutinesuiv

#ifdef __cplusplus
}
#endif /*__cplusplus*/
#endif /*OPENGL_GEN_CORE_4_0_H*/
