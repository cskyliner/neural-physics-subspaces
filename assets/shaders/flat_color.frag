#version 410 core

layout(location = 0) flat in  vec3 v_Color;

uniform float u_Alpha = 1.0;

layout(location = 0) out vec4 f_Color;

void main() {
    f_Color = vec4(v_Color, u_Alpha);
}
