#include "ImageGeneratorApp.hpp"
#include "ofMain.h"

int main() {
    ofSetupOpenGL(512, 512, OF_WINDOW);
    ofRunApp(new ImageGeneratorApp());
}
