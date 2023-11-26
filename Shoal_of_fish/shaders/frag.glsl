#version 330 core

in vec4 FragmentColor;
out vec4 ScreenColor;

void main() {
    ScreenColor.r = abs(FragmentColor.r);
    ScreenColor.g = abs(FragmentColor.g);
    ScreenColor.b = abs(FragmentColor.b);
}
