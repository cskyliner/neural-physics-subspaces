#version 410 core

layout(location = 0) flat in vec3 v_Color;
layout(location = 1) in vec3 v_FragPos;

out vec4 f_Color;

void main() {
    // 使用导数计算面法线（flat shading）
    vec3 fdx = dFdx(v_FragPos);
    vec3 fdy = dFdy(v_FragPos);
    vec3 normal = normalize(cross(fdx, fdy));
    
    // 简单的光照模型
    vec3 lightDir = normalize(vec3(0.5, 1.0, 0.3));
    vec3 viewDir = normalize(-v_FragPos);
    
    // 环境光
    float ambient = 0.3;
    
    // 漫反射
    float diff = max(dot(normal, lightDir), 0.0);
    
    // 高光
    vec3 halfDir = normalize(lightDir + viewDir);
    float spec = pow(max(dot(normal, halfDir), 0.0), 32.0) * 0.3;
    
    // 边缘光（Fresnel效果）
    float fresnel = pow(1.0 - max(dot(viewDir, normal), 0.0), 2.0) * 0.2;
    
    vec3 result = v_Color * (ambient + diff * 0.7) + vec3(spec) + vec3(fresnel);
    f_Color = vec4(result, 1.0);
}
