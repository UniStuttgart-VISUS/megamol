#version 110

          uniform vec4 haloColor;
          void main(void) {
            vec3 col = abs(gl_Color.rgb - vec3(0.5, 0.5, 0.5));
            float epsilon = 0.01;
            if( col.r < epsilon && col.g < epsilon && col.b < epsilon)
              discard;
            else
              gl_FragColor = haloColor;
          }
