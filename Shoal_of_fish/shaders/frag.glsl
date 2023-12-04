#version 330 core

in vec3 FragmentColor;
out vec3 FinalColor;

void main() {
    FinalColor.r = FragmentColor.r;
    FinalColor.g = FragmentColor.g;
    FinalColor.b = FragmentColor.b;
}
