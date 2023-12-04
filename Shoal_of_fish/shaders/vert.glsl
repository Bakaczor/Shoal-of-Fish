#version 330 core

layout(location = 0) in vec2 Point;
layout(location = 1) in float Id;
out vec3 FragmentColor;

vec3 hsv2rgb(vec3 c) {
    vec4 K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
    vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
    return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
}

void main() {
    gl_Position = vec4(Point, 0.0, 1.0);

    float hue = fract(0.618033988749895 * Id);
    float saturation = 0.9;
    float value = 0.9;
    FragmentColor = hsv2rgb(vec3(hue, saturation, value));
}
