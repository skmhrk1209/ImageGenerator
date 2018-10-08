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
    std::unique_ptr<ofxDatGui> gui;
    std::unique_ptr<ofImage> image;
    std::unique_ptr<std::thread> process;
    boost::asio::io_service glService;
};
