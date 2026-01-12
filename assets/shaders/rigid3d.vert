#version 410 core

layout(location = 0) in vec3 a_Position;
layout(location = 1) in vec3 a_Color;

layout(location = 0) flat out vec3 v_Color;
layout(location = 1) out vec3 v_FragPos;

uniform mat4 u_Projection;
uniform mat4 u_View;
uniform mat4 u_Model;

void main() {
    vec4 worldPos = u_Model * vec4(a_Position, 1.0);
    gl_Position = u_Projection * u_View * worldPos;
    v_FragPos = worldPos.xyz;
    v_Color = a_Color;
}
