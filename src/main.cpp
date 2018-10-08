#include "ImageGeneratorApp.hpp"
#include "ofMain.h"

int main() {
    ofSetupOpenGL(768, 512, OF_WINDOW);
    ofRunApp(new ImageGeneratorApp());
}
