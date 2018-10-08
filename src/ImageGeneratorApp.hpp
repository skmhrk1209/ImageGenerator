#pragma once

#include <tensorflow/cc/client/client_session.h>
#include <tensorflow/cc/ops/array_ops.h>
#include <tensorflow/cc/ops/image_ops.h>
#include <tensorflow/cc/ops/io_ops.h>
#include <tensorflow/cc/ops/math_ops.h>
#include <tensorflow/cc/ops/random_ops.h>
#include <tensorflow/cc/ops/standard_ops.h>
#include <tensorflow/core/framework/tensor.h>
#include <tensorflow/core/public/session.h>
#include <boost/asio.hpp>
#include "ofMain.h"
#include "ofxDatGui.h"

namespace tf = tensorflow;

class ImageGeneratorApp : public ofBaseApp {
   public:
    ImageGeneratorApp() {}
    ~ImageGeneratorApp() {
        if (process) process->join();
    }

    void setup();
    void update();
    void draw();

   private:
    std::unordered_map<std::string, std::pair<tf::GraphDef, tf::GraphDef>> graphs;
    std::unique_ptr<ofxDatGui> globalGui;
    ofxDatGuiDropdown* dropdown;
    ofxDatGuiToggle* toggle;
    std::unique_ptr<ofxDatGui> localGui;
    ofxDatGuiSlider* slider;
    std::unique_ptr<std::thread> process;
    std::vector<ofFloatImage> images;
    boost::asio::io_service glService;
};
