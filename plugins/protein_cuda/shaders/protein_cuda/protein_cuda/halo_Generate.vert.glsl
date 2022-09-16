          void main(void) {
            vec4 vertexPos = gl_Vertex;


            gl_Position = gl_ModelViewProjectionMatrix * vertexPos;

            gl_FrontColor = gl_Color;
            gl_BackColor = gl_Color;
          }
