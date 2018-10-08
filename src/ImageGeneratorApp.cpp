#include "ImageGeneratorApp.hpp"

const auto proj_dir = "/Users/hirokisakuma/documents/openFrameworks_v0.9.8/apps/myapps/ImageGenerator/";
const auto latent_size = 128;
const auto batch_size = 64;

auto concat = [](const auto&... args) {
    std::stringstream sstream;
    static_cast<void>(std::initializer_list<int>{(sstream << args, 0)...});
    return sstream.str();
};

void ImageGeneratorApp::setup() {
    gui = std::make_unique<ofxDatGui>(128, 384);
    gui->setWidth(256);

    toggle = gui->addToggle("generate");
    toggle->setStripeVisible(false);
    toggle->onToggleEvent([=](const ofxDatGuiToggleEvent& event) {
        if (process) process->join();
        process = std::make_unique<std::thread>([=]() {
            event.target->setEnabled(false);
            //================================ generate image by Tensorflow ================================//
            std::vector<tf::Tensor> inputs;
            {
                auto scope = tf::Scope::NewRootScope();
                auto session = std::unique_ptr<tf::Session>(tf::NewSession(tf::SessionOptions()));

                auto random_normal_begin = tf::ops::RandomNormal(scope, {latent_size}, tf::DT_FLOAT);
                auto random_normal_end = tf::ops::RandomNormal(scope, {latent_size}, tf::DT_FLOAT);

                std::vector<tf::Output> random_normals;

                auto lerp = [&](const auto& a, const auto& b, const auto& t) {
                    return tf::ops::Add(scope, a, tf::ops::Multiply(scope, tf::ops::Subtract(scope, b, a), t));
                };

                for (auto i = 0; i < batch_size; ++i) {
                    random_normals.push_back(lerp(random_normal_begin, random_normal_end, float(i) / batch_size));
                }

                tf::ops::Stack(scope.WithOpName("random_normals"), random_normals);
                tf::ops::Const(scope.WithOpName("const_false"), false);

                tf::GraphDef graph;
                scope.ToGraphDef(&graph);
                session->Create(graph);

                TF_CHECK_OK(session->Run({}, {"random_normals", "const_false"}, {}, &inputs));
            }

            std::vector<tf::Tensor> outputs;
            {
                auto scope = tf::Scope::NewRootScope();
                auto session = std::unique_ptr<tf::Session>(tf::NewSession(tf::SessionOptions()));

                tf::GraphDef graph;
                tf::ReadBinaryProto(tf::Env::Default(), concat(proj_dir, "bin/data/frozen_graph.pb"), &graph);
                session->Create(graph);

                TF_CHECK_OK(session->Run({{"celeba_dcgan_model/latents", inputs[0]}, {"celeba_dcgan_model/training", inputs[1]}},
                                         {"celeba_dcgan_model/fakes"}, {}, &outputs));
            }

            glService.post([=]() {
                images.clear();
                for (auto i = 0; i < outputs[0].dim_size(0); ++i) {
                    auto data = outputs[0].SubSlice(i).tensor<float, 3>().data();
                    images.emplace_back();
                    images.back().setFromPixels(data, 128, 128, OF_IMAGE_COLOR);
                }
            });

            /*
            for (auto i = 0; i < outputs[0].dim_size(0); ++i) {
                auto scope = tf::Scope::NewRootScope();
                auto session = std::unique_ptr<tf::Session>(tf::NewSession(tf::SessionOptions()));

                auto image = tf::ops::Const(scope, outputs[0].SubSlice(i));
                auto scaled = tf::ops::Multiply(scope, image, 255.0f);
                auto casted = tf::ops::Cast(scope, scaled, tf::DT_UINT8);
                auto encoded = tf::ops::EncodePng(scope, casted);

                tf::ops::WriteFile(scope.WithOpName("write_file"), concat(proj_dir, "bin/data/", i, ".png"), encoded);

                tf::GraphDef graph;
                scope.ToGraphDef(&graph);
                session->Create(graph);
                TF_CHECK_OK(session->Run({}, {}, {"write_file"}, {}));
            }
            */

            event.target->setChecked(false);
            event.target->setEnabled(true);
        });
    });

    slider = gui->addSlider("interpolation", 0.0, 1.0);
    slider->setStripeVisible(false);

    ofBackground(20);

    font.load("Verdana", 8);
}

void ImageGeneratorApp::update() {
    glService.reset();
    glService.run();
}

void ImageGeneratorApp::draw() {
    ofSetColor(255);
    ofNoFill();
    
    font.drawString("Progressive Growing of GANs", 20, 20);
    font.drawString("Celeb Image Generator", 20, 40);

    if (!images.empty()) {
        int index = (batch_size - 1) * gui->getSlider("interpolation")->getValue();
        images[index].draw(128, 128, 256, 256);
    }

    ofDrawRectangle(128, 128, 256, 256);
}
