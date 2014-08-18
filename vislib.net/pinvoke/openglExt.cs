using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Reflection;
using System.Runtime.InteropServices;

namespace vislib.pinvoke {

    /// <summary>
    /// OpenGL extention handler class
    /// </summary>
    public class openglExt {

        /// <summary>
        /// Initialises all extensions defined in this object
        /// </summary>
        public static void InitExtensions() {
            foreach (FieldInfo field in typeof(openglExt).GetFields(BindingFlags.Static | BindingFlags.Public)) {
                IntPtr fptr = opengl32.wglGetProcAddress(field.Name);
                if (fptr != IntPtr.Zero) {
                    field.SetValue(null, Marshal.GetDelegateForFunctionPointer(fptr, field.FieldType));
                } else {
                    field.SetValue(null, null);
                }
            }
        }

        public delegate void glBindAttribLocationDelegate(uint program, uint index, [MarshalAs(UnmanagedType.LPStr), In] string name);

        public static glBindAttribLocationDelegate glBindAttribLocationARB = null;

        public delegate uint glCreateProgramObjectDelegate();

        public static glCreateProgramObjectDelegate glCreateProgramObjectARB = null;

        public delegate uint glCreateShaderObjectDelegate(uint type);

        public static glCreateShaderObjectDelegate glCreateShaderObjectARB = null;

        public delegate void glShaderSourceDelegate(uint shader, int count, IntPtr[] source, IntPtr length);

        public static glShaderSourceDelegate glShaderSourceARB = null;

        public delegate void glCompileShaderDelegate(uint s);

        public static glCompileShaderDelegate glCompileShaderARB = null;

        public delegate void glAttachObjectDelegate(uint p, uint hVertShade);

        public static glAttachObjectDelegate glAttachObjectARB = null;

        public delegate void glUseProgramObjectARBDelegete(uint p);

        public static glUseProgramObjectARBDelegete glUseProgramObjectARB = null;

        public delegate void glLinkProgramARBDelegate(uint p);

        public static glLinkProgramARBDelegate glLinkProgramARB = null;

        public delegate void glDeleteObjectARBDelegate(uint p);

        public static glDeleteObjectARBDelegate glDeleteObjectARB = null;

        public delegate void glGetObjectParameterivARBDelegate(uint p, uint mode, out int status);

        public static glGetObjectParameterivARBDelegate glGetObjectParameterivARB = null;

        public delegate void glVertexAttribPointerDelegate(uint index, int size, uint type, int normalized, int stride, IntPtr pointer);

        public delegate void glEnableVertexAttribArrayDelegate(uint index);

        public delegate void glDisableVertexAttribArrayDelegate(uint index);

        public static glVertexAttribPointerDelegate glVertexAttribPointer = null;

        public static glEnableVertexAttribArrayDelegate glEnableVertexAttribArray = null;

        public static glDisableVertexAttribArrayDelegate glDisableVertexAttribArray = null;

        public delegate int glGetUniformLocationARBDelegate(uint p, [MarshalAs(UnmanagedType.LPStr), In]string name);

        public static glGetUniformLocationARBDelegate glGetUniformLocationARB = null;

        public delegate void glUniform1fARBDelegate(int loc, float v1);

        public static glUniform1fARBDelegate glUniform1fARB = null;

        public delegate void glUniform3fARBDelegate(int loc, float v1, float v2, float v3);

        public static glUniform3fARBDelegate glUniform3fARB = null;

        public delegate void glUniform4fARBDelegate(int loc, float v1, float v2, float v3, float v4);

        public static glUniform4fARBDelegate glUniform4fARB = null;

        public delegate void glGetInfoLogARBDelegate(uint hObj, int len, ref int written, [MarshalAs(UnmanagedType.LPStr)]StringBuilder sb);

        public static glGetInfoLogARBDelegate glGetInfoLogARB = null;

    }

}
