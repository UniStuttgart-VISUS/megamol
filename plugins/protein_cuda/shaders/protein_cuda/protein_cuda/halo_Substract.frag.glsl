#version 110

          uniform sampler2D originalTex;
          uniform sampler2D blurredTex;
          void main(void) {
            vec4 orig = texture2D(originalTex, gl_TexCoord[0].xy);
            vec3 col = abs(orig.rgb);
            float epsilon = 0.01;
            if( col.r < epsilon && col.g < epsilon && col.b < epsilon)
              gl_FragColor = texture2D(blurredTex, gl_TexCoord[0].xy);
            else
              discard;
          }
