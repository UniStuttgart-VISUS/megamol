using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Runtime.InteropServices;

namespace vislib.pinvoke {

    /// <summary>
    /// class holding opengl32 p/invoke definitions
    /// </summary>
    public static class opengl32 {

        /// <summary>
        /// The library file name
        /// </summary>
        public const string LIBNAME = "opengl32.dll";

        #region structs

        public const uint DEPTH_BUFFER_BIT = 0x00000100;
        public const uint COLOR_BUFFER_BIT = 0x00004000;

        public const uint FLAT = 0x1D00;
        public const uint SMOOTH = 0x1D01;

        public const uint CURRENT_COLOR = 0x0B00;
        public const uint CURRENT_INDEX = 0x0B01;
        public const uint CURRENT_NORMAL = 0x0B02;
        public const uint CURRENT_TEXTURE_COORDS = 0x0B03;
        public const uint CURRENT_RASTER_COLOR = 0x0B04;
        public const uint CURRENT_RASTER_INDEX = 0x0B05;
        public const uint CURRENT_RASTER_TEXTURE_COORDS = 0x0B06;
        public const uint CURRENT_RASTER_POSITION = 0x0B07;
        public const uint CURRENT_RASTER_POSITION_VALID = 0x0B08;
        public const uint CURRENT_RASTER_DISTANCE = 0x0B09;
        public const uint POINT_SMOOTH = 0x0B10;
        public const uint POINT_SIZE = 0x0B11;
        public const uint POINT_SIZE_RANGE = 0x0B12;
        public const uint POINT_SIZE_GRANULARITY = 0x0B13;
        public const uint LINE_SMOOTH = 0x0B20;
        public const uint LINE_WIDTH = 0x0B21;
        public const uint LINE_WIDTH_RANGE = 0x0B22;
        public const uint LINE_WIDTH_GRANULARITY = 0x0B23;
        public const uint LINE_STIPPLE = 0x0B24;
        public const uint LINE_STIPPLE_PATTERN = 0x0B25;
        public const uint LINE_STIPPLE_REPEAT = 0x0B26;
        public const uint LIST_MODE = 0x0B30;
        public const uint MAX_LIST_NESTING = 0x0B31;
        public const uint LIST_BASE = 0x0B32;
        public const uint LIST_INDEX = 0x0B33;
        public const uint POLYGON_MODE = 0x0B40;
        public const uint POLYGON_SMOOTH = 0x0B41;
        public const uint POLYGON_STIPPLE = 0x0B42;
        public const uint EDGE_FLAG = 0x0B43;
        public const uint CULL_FACE = 0x0B44;
        public const uint CULL_FACE_MODE = 0x0B45;
        public const uint FRONT_FACE = 0x0B46;
        public const uint LIGHTING = 0x0B50;
        public const uint LIGHT0 = 0x4000;
        public const uint LIGHT_MODEL_LOCAL_VIEWER = 0x0B51;
        public const uint LIGHT_MODEL_TWO_SIDE = 0x0B52;
        public const uint LIGHT_MODEL_AMBIENT = 0x0B53;
        public const uint SHADE_MODEL = 0x0B54;
        public const uint COLOR_MATERIAL_FACE = 0x0B55;
        public const uint COLOR_MATERIAL_PARAMETER = 0x0B56;
        public const uint COLOR_MATERIAL = 0x0B57;
        public const uint FOG = 0x0B60;
        public const uint FOG_INDEX = 0x0B61;
        public const uint FOG_DENSITY = 0x0B62;
        public const uint FOG_START = 0x0B63;
        public const uint FOG_END = 0x0B64;
        public const uint FOG_MODE = 0x0B65;
        public const uint FOG_COLOR = 0x0B66;
        public const uint DEPTH_RANGE = 0x0B70;
        public const uint DEPTH_TEST = 0x0B71;
        public const uint DEPTH_WRITEMASK = 0x0B72;
        public const uint DEPTH_CLEAR_VALUE = 0x0B73;
        public const uint DEPTH_FUNC = 0x0B74;
        public const uint ACCUM_CLEAR_VALUE = 0x0B80;
        public const uint STENCIL_TEST = 0x0B90;
        public const uint STENCIL_CLEAR_VALUE = 0x0B91;
        public const uint STENCIL_FUNC = 0x0B92;
        public const uint STENCIL_VALUE_MASK = 0x0B93;
        public const uint STENCIL_FAIL = 0x0B94;
        public const uint STENCIL_PASS_DEPTH_FAIL = 0x0B95;
        public const uint STENCIL_PASS_DEPTH_PASS = 0x0B96;
        public const uint STENCIL_REF = 0x0B97;
        public const uint STENCIL_WRITEMASK = 0x0B98;
        public const uint MATRIX_MODE = 0x0BA0;
        public const uint NORMALIZE = 0x0BA1;
        public const uint VIEWPORT = 0x0BA2;
        public const uint MODELVIEW_STACK_DEPTH = 0x0BA3;
        public const uint PROJECTION_STACK_DEPTH = 0x0BA4;
        public const uint TEXTURE_STACK_DEPTH = 0x0BA5;
        public const uint MODELVIEW_MATRIX = 0x0BA6;
        public const uint PROJECTION_MATRIX = 0x0BA7;
        public const uint TEXTURE_MATRIX = 0x0BA8;
        public const uint ATTRIB_STACK_DEPTH = 0x0BB0;
        public const uint CLIENT_ATTRIB_STACK_DEPTH = 0x0BB1;
        public const uint ALPHA_TEST = 0x0BC0;
        public const uint ALPHA_TEST_FUNC = 0x0BC1;
        public const uint ALPHA_TEST_REF = 0x0BC2;
        public const uint DITHER = 0x0BD0;
        public const uint BLEND_DST = 0x0BE0;
        public const uint BLEND_SRC = 0x0BE1;
        public const uint BLEND = 0x0BE2;
        public const uint LOGIC_OP_MODE = 0x0BF0;
        public const uint INDEX_LOGIC_OP = 0x0BF1;
        public const uint COLOR_LOGIC_OP = 0x0BF2;
        public const uint AUX_BUFFERS = 0x0C00;
        public const uint DRAW_BUFFER = 0x0C01;
        public const uint READ_BUFFER = 0x0C02;
        public const uint SCISSOR_BOX = 0x0C10;
        public const uint SCISSOR_TEST = 0x0C11;
        public const uint INDEX_CLEAR_VALUE = 0x0C20;
        public const uint INDEX_WRITEMASK = 0x0C21;
        public const uint COLOR_CLEAR_VALUE = 0x0C22;
        public const uint COLOR_WRITEMASK = 0x0C23;
        public const uint INDEX_MODE = 0x0C30;
        public const uint RGBA_MODE = 0x0C31;
        public const uint DOUBLEBUFFER = 0x0C32;
        public const uint STEREO = 0x0C33;
        public const uint RENDER_MODE = 0x0C40;
        public const uint PERSPECTIVE_CORRECTION_HINT = 0x0C50;
        public const uint POINT_SMOOTH_HINT = 0x0C51;
        public const uint LINE_SMOOTH_HINT = 0x0C52;
        public const uint POLYGON_SMOOTH_HINT = 0x0C53;
        public const uint FOG_HINT = 0x0C54;
        public const uint TEXTURE_GEN_S = 0x0C60;
        public const uint TEXTURE_GEN_T = 0x0C61;
        public const uint TEXTURE_GEN_R = 0x0C62;
        public const uint TEXTURE_GEN_Q = 0x0C63;
        public const uint PIXEL_MAP_I_TO_I = 0x0C70;
        public const uint PIXEL_MAP_S_TO_S = 0x0C71;
        public const uint PIXEL_MAP_I_TO_R = 0x0C72;
        public const uint PIXEL_MAP_I_TO_G = 0x0C73;
        public const uint PIXEL_MAP_I_TO_B = 0x0C74;
        public const uint PIXEL_MAP_I_TO_A = 0x0C75;
        public const uint PIXEL_MAP_R_TO_R = 0x0C76;
        public const uint PIXEL_MAP_G_TO_G = 0x0C77;
        public const uint PIXEL_MAP_B_TO_B = 0x0C78;
        public const uint PIXEL_MAP_A_TO_A = 0x0C79;
        public const uint PIXEL_MAP_I_TO_I_SIZE = 0x0CB0;
        public const uint PIXEL_MAP_S_TO_S_SIZE = 0x0CB1;
        public const uint PIXEL_MAP_I_TO_R_SIZE = 0x0CB2;
        public const uint PIXEL_MAP_I_TO_G_SIZE = 0x0CB3;
        public const uint PIXEL_MAP_I_TO_B_SIZE = 0x0CB4;
        public const uint PIXEL_MAP_I_TO_A_SIZE = 0x0CB5;
        public const uint PIXEL_MAP_R_TO_R_SIZE = 0x0CB6;
        public const uint PIXEL_MAP_G_TO_G_SIZE = 0x0CB7;
        public const uint PIXEL_MAP_B_TO_B_SIZE = 0x0CB8;
        public const uint PIXEL_MAP_A_TO_A_SIZE = 0x0CB9;
        public const uint UNPACK_SWAP_BYTES = 0x0CF0;
        public const uint UNPACK_LSB_FIRST = 0x0CF1;
        public const uint UNPACK_ROW_LENGTH = 0x0CF2;
        public const uint UNPACK_SKIP_ROWS = 0x0CF3;
        public const uint UNPACK_SKIP_PIXELS = 0x0CF4;
        public const uint UNPACK_ALIGNMENT = 0x0CF5;
        public const uint PACK_SWAP_BYTES = 0x0D00;
        public const uint PACK_LSB_FIRST = 0x0D01;
        public const uint PACK_ROW_LENGTH = 0x0D02;
        public const uint PACK_SKIP_ROWS = 0x0D03;
        public const uint PACK_SKIP_PIXELS = 0x0D04;
        public const uint PACK_ALIGNMENT = 0x0D05;
        public const uint MAP_COLOR = 0x0D10;
        public const uint MAP_STENCIL = 0x0D11;
        public const uint INDEX_SHIFT = 0x0D12;
        public const uint INDEX_OFFSET = 0x0D13;
        public const uint RED_SCALE = 0x0D14;
        public const uint RED_BIAS = 0x0D15;
        public const uint ZOOM_X = 0x0D16;
        public const uint ZOOM_Y = 0x0D17;
        public const uint GREEN_SCALE = 0x0D18;
        public const uint GREEN_BIAS = 0x0D19;
        public const uint BLUE_SCALE = 0x0D1A;
        public const uint BLUE_BIAS = 0x0D1B;
        public const uint ALPHA_SCALE = 0x0D1C;
        public const uint ALPHA_BIAS = 0x0D1D;
        public const uint DEPTH_SCALE = 0x0D1E;
        public const uint DEPTH_BIAS = 0x0D1F;
        public const uint MAX_EVAL_ORDER = 0x0D30;
        public const uint MAX_LIGHTS = 0x0D31;
        public const uint MAX_CLIP_PLANES = 0x0D32;
        public const uint MAX_TEXTURE_SIZE = 0x0D33;
        public const uint MAX_PIXEL_MAP_TABLE = 0x0D34;
        public const uint MAX_ATTRIB_STACK_DEPTH = 0x0D35;
        public const uint MAX_MODELVIEW_STACK_DEPTH = 0x0D36;
        public const uint MAX_NAME_STACK_DEPTH = 0x0D37;
        public const uint MAX_PROJECTION_STACK_DEPTH = 0x0D38;
        public const uint MAX_TEXTURE_STACK_DEPTH = 0x0D39;
        public const uint MAX_VIEWPORT_DIMS = 0x0D3A;
        public const uint MAX_CLIENT_ATTRIB_STACK_DEPTH = 0x0D3B;
        public const uint SUBPIXEL_BITS = 0x0D50;
        public const uint INDEX_BITS = 0x0D51;
        public const uint RED_BITS = 0x0D52;
        public const uint GREEN_BITS = 0x0D53;
        public const uint BLUE_BITS = 0x0D54;
        public const uint ALPHA_BITS = 0x0D55;
        public const uint DEPTH_BITS = 0x0D56;
        public const uint STENCIL_BITS = 0x0D57;
        public const uint ACCUM_RED_BITS = 0x0D58;
        public const uint ACCUM_GREEN_BITS = 0x0D59;
        public const uint ACCUM_BLUE_BITS = 0x0D5A;
        public const uint ACCUM_ALPHA_BITS = 0x0D5B;
        public const uint NAME_STACK_DEPTH = 0x0D70;
        public const uint AUTO_NORMAL = 0x0D80;
        public const uint MAP1_COLOR_4 = 0x0D90;
        public const uint MAP1_INDEX = 0x0D91;
        public const uint MAP1_NORMAL = 0x0D92;
        public const uint MAP1_TEXTURE_COORD_1 = 0x0D93;
        public const uint MAP1_TEXTURE_COORD_2 = 0x0D94;
        public const uint MAP1_TEXTURE_COORD_3 = 0x0D95;
        public const uint MAP1_TEXTURE_COORD_4 = 0x0D96;
        public const uint MAP1_VERTEX_3 = 0x0D97;
        public const uint MAP1_VERTEX_4 = 0x0D98;
        public const uint MAP2_COLOR_4 = 0x0DB0;
        public const uint MAP2_INDEX = 0x0DB1;
        public const uint MAP2_NORMAL = 0x0DB2;
        public const uint MAP2_TEXTURE_COORD_1 = 0x0DB3;
        public const uint MAP2_TEXTURE_COORD_2 = 0x0DB4;
        public const uint MAP2_TEXTURE_COORD_3 = 0x0DB5;
        public const uint MAP2_TEXTURE_COORD_4 = 0x0DB6;
        public const uint MAP2_VERTEX_3 = 0x0DB7;
        public const uint MAP2_VERTEX_4 = 0x0DB8;
        public const uint MAP1_GRID_DOMAIN = 0x0DD0;
        public const uint MAP1_GRID_SEGMENTS = 0x0DD1;
        public const uint MAP2_GRID_DOMAIN = 0x0DD2;
        public const uint MAP2_GRID_SEGMENTS = 0x0DD3;
        public const uint TEXTURE_1D = 0x0DE0;
        public const uint TEXTURE_2D = 0x0DE1;
        public const uint FEEDBACK_BUFFER_POINTER = 0x0DF0;
        public const uint FEEDBACK_BUFFER_SIZE = 0x0DF1;
        public const uint FEEDBACK_BUFFER_TYPE = 0x0DF2;
        public const uint SELECTION_BUFFER_POINTER = 0x0DF3;
        public const uint SELECTION_BUFFER_SIZE = 0x0DF4;

        public const uint AMBIENT = 0x1200;
        public const uint DIFFUSE = 0x1201;
        public const uint SPECULAR = 0x1202;
        public const uint POSITION = 0x1203;

        public const uint NEVER = 0x0200;
        public const uint LESS = 0x0201;
        public const uint EQUAL = 0x0202;
        public const uint LEQUAL = 0x0203;
        public const uint GREATER = 0x0204;
        public const uint NOTEQUAL = 0x0205;
        public const uint GEQUAL = 0x0206;
        public const uint ALWAYS = 0x0207;

        public const uint POINTS = 0x0000;
        public const uint LINES = 0x0001;
        public const uint LINE_LOOP = 0x0002;
        public const uint LINE_STRIP = 0x0003;
        public const uint TRIANGLES = 0x0004;
        public const uint TRIANGLE_STRIP = 0x0005;
        public const uint TRIANGLE_FAN = 0x0006;
        public const uint QUADS = 0x0007;
        public const uint QUAD_STRIP = 0x0008;
        public const uint POLYGON = 0x0009;

        public const uint MODELVIEW = 0x1700;
        public const uint PROJECTION = 0x1701;
        public const uint TEXTURE = 0x1702;

        public const uint DONT_CARE = 0x1100;
        public const uint FASTEST = 0x1101;
        public const uint NICEST = 0x1102;

        public const uint BYTE = 0x1400;
        public const uint UNSIGNED_BYTE = 0x1401;
        public const uint SHORT = 0x1402;
        public const uint UNSIGNED_SHORT = 0x1403;
        public const uint INT = 0x1404;
        public const uint UNSIGNED_INT = 0x1405;
        public const uint FLOAT = 0x1406;
        public const uint GL_2_BYTES = 0x1407;
        public const uint GL_3_BYTES = 0x1408;
        public const uint GL_4_BYTES = 0x1409;
        public const uint DOUBLE = 0x140A;

        public const uint VERTEX_ARRAY = 0x8074;
        public const uint NORMAL_ARRAY = 0x8075;
        public const uint COLOR_ARRAY = 0x8076;
        public const uint INDEX_ARRAY = 0x8077;
        public const uint TEXTURE_COORD_ARRAY = 0x8078;

        public const uint FRAGMENT_SHADER = 0x8B30;
        public const uint VERTEX_SHADER = 0x8B31;

        public const uint FRAGMENT_PROGRAM_ARB = 0x8804;
        public const uint VERTEX_PROGRAM_ARB = 0x8620;

        public const uint GL_OBJECT_DELETE_STATUS_ARB = 0x8B80;
        public const uint GL_OBJECT_COMPILE_STATUS_ARB = 0x8B81;
        public const uint GL_OBJECT_LINK_STATUS_ARB = 0x8B82;
        public const uint GL_OBJECT_INFO_LOG_LENGTH_ARB = 0x8B84;

        public const uint BACK = 0x0405;
        public const uint FRONT = 0x0404;
        public const uint RGB = 0x1907;

        #endregion

        #region functions

        [DllImport(LIBNAME)]
        public static extern void glClearColor(float red, float green, float blue, float alpha);

        [DllImport(LIBNAME)]
        public static extern void glEnable(uint cap);

        [DllImport(LIBNAME)]
        public static extern void glDisable(uint cap);

        [DllImport(LIBNAME)]
        public static extern void glDepthFunc(uint func);

        [DllImport(LIBNAME)]
        public static extern void glClear(uint mask);

        [DllImport(LIBNAME)]
        public static extern void glLoadIdentity();

        [DllImport(LIBNAME)]
        public static extern void glTranslatef(float x, float y, float z);

        [DllImport(LIBNAME)]
        public static extern void glScalef(float x, float y, float z);
        
        [DllImport(LIBNAME)]
        public static extern void glRotatef(float angle, float x, float y, float z);

        [DllImport(LIBNAME)]
        public static extern void glBegin(uint mode);

        [DllImport(LIBNAME)]
        public static extern void glMatrixMode(uint mode);

        [DllImport(LIBNAME)]
        public static extern void glColor3f(float red, float green, float blue);

        [DllImport(LIBNAME)]
        public static extern void glColor4f(float red, float green, float blue, float alpha);

        [DllImport(LIBNAME)]
        public static extern void glColor3ub(byte red, byte green, byte blue);

        [DllImport(LIBNAME)]
        public static extern void glColor4ub(byte red, byte green, byte blue, byte alpha);

        [DllImport(LIBNAME)]
        public static extern void glVertex3f(float x, float y, float z);

        [DllImport(LIBNAME)]
        public static extern void glVertex3d(double x, double y, double z);

        [DllImport(LIBNAME)]
        public static extern void glEnd();

        //[DllImport(LIBNAME)]
        //public static extern void glListBase(uint base_notkeyword);
        //[DllImport(LIBNAME)]
        //public static extern void glCallLists(int n, uint type, int[] lists);

        [DllImport(LIBNAME)]
        public static extern void glFlush();

        [DllImport(LIBNAME)]
        public static extern void glViewport(int x, int y, int width, int height);

        [DllImport(LIBNAME)]
        public static extern void glShadeModel(uint mode);

        [DllImport(LIBNAME)]
        public static extern void glClearDepth(double depth);

        [DllImport(LIBNAME)]
        public static extern void glHint(uint target, uint mode);

        [DllImport(LIBNAME)]
        public static extern void glEnableClientState(uint array);

        [DllImport(LIBNAME)]
        public static extern void glDisableClientState(uint array);

        [DllImport(LIBNAME)]
        public static extern void glVertexPointer(int size, uint type, int stride, IntPtr pointer);

        [DllImport(LIBNAME)]
        public static extern void glDrawArrays(uint mode, int first, int count);

        [DllImport(LIBNAME)]
        public static extern int glGetError();

        /// <summary>
        /// The wglMakeCurrent function makes a specified OpenGL rendering context the calling thread's current rendering context.
        /// </summary>
        /// <param name="hdc">Handle to a device context. Subsequent OpenGL calls made by the calling thread are drawn on the device identified by hdc.</param>
        /// <param name="hrc">Handle to an OpenGL rendering context that the function sets as the calling thread's rendering context. 
        /// If hglrc is NULL, the function makes the calling thread's current rendering context no longer current, and releases the device context that is used by the rendering context. In this case, hdc is ignored.
        /// </param>
        /// <returns>When the wglMakeCurrent function succeeds, the return value is TRUE</returns>
        [DllImport(LIBNAME, SetLastError = true)]
        public static extern int wglMakeCurrent(IntPtr hdc, IntPtr hrc);

        /// <summary>
        /// The wglDeleteContext function deletes a specified OpenGL rendering context.
        /// </summary>
        /// <param name="hdc">Handle to an OpenGL rendering context that the function will delete.</param>
        /// <returns>If the function succeeds, the return value is TRUE.</returns>
        [DllImport(LIBNAME, SetLastError=true)]
        public static extern int wglDeleteContext(IntPtr hdc);

        /// <summary>
        /// The wglCreateContext function creates a new OpenGL rendering context, which is suitable for drawing on the device referenced by hdc. The rendering context has the same pixel format as the device context.
        /// </summary>
        /// <param name="hdc">Handle to a device context for which the function creates a suitable OpenGL rendering context.</param>
        /// <returns>If the function succeeds, the return value is a valid handle to an OpenGL rendering context.</returns>
        [DllImport(LIBNAME, SetLastError = true)]
        public static extern IntPtr wglCreateContext(IntPtr hdc);

        [DllImport(LIBNAME)]
        public static extern IntPtr wglGetProcAddress([In, MarshalAs(UnmanagedType.LPStr)] string name);

        [DllImport(LIBNAME)]
        public static extern void glLightfv(uint light, uint mode, float[] value);

        [DllImport(LIBNAME)]
        public static extern void glLightModelfv(uint model, float[] value);

        [DllImport(LIBNAME)]
        public static extern void glReadBuffer(uint mode);

        [DllImport(LIBNAME)]
        public static extern void glReadPixels(int x, int y, int width, int height, uint format, uint type, IntPtr pixels);

        [DllImport(LIBNAME)]
        public static extern void glPixelStorei(uint pname, int param);

        #endregion

    }

}
