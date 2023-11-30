#version 330 core

in vec4 FragmentColor;
out vec4 FinalColor;

void main() {
    FinalColor.r = FragmentColor.r;
    FinalColor.g = FragmentColor.g;
    FinalColor.b = FragmentColor.b;
    FinalColor.a = FragmentColor.a;
}
