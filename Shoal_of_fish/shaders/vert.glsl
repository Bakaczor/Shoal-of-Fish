#version 330 core

layout(location = 0) in vec2 Position;
layout(location = 1) in vec2 Velocity;
out vec4 FragmentColor;

void main() {
    gl_Position = vec4(Position, 0.0, 1.0);
    FragmentColor = vec4(255, 0, 0, 255);
}
