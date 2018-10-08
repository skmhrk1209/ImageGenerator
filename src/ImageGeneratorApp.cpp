#include "ImageGeneratorApp.hpp"

const auto proj_dir = "/Users/hirokisakuma/documents/openFrameworks_v0.9.8/apps/myapps/ImageGenerator/";
const auto batch_size = 64;

auto concat = [](const auto&... args) {
    std::stringstream sstream;
    static_cast<void>(std::initializer_list<int>{(sstream << args, 0)...});
    return sstream.str();
};

void ImageGeneratorApp::setup() {
    gui = std::make_unique<ofxDatGui>(128, 384);
    gui->addToggle("generate")->onToggleEvent([=](const ofxDatGuiToggleEvent& event) {
        if (process) process->join();
        process = std::make_unique<std::thread>([=]() {
            event.target->setEnabled(false);
            //================================ generate image by Tensorflow ================================//
            std::vector<tf::Tensor> inputs;
            {
                auto scope = tf::Scope::NewRootScope();
                auto session = std::unique_ptr<tf::Session>(tf::NewSession(tf::SessionOptions()));

                auto random_normal_begin = tf::ops::RandomNormal(scope, {128}, tf::DT_FLOAT);
                auto random_normal_end = tf::ops::RandomNormal(scope, {128}, tf::DT_FLOAT);

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

            glService.post([=]() {
                image = std::make_unique<ofImage>(
                    concat(proj_dir, "bin/data/", static_cast<int>(batch_size * gui->getSlider("interpolation")->getValue()) % batch_size, ".png"));
            });

            event.target->setChecked(false);
            event.target->setEnabled(true);
        });
    });
    gui->addSlider("interpolation", 0.0, 1.0)->onSliderEvent([=](const ofxDatGuiSliderEvent& event) {
        image = std::make_unique<ofImage>(concat(proj_dir, "bin/data/", static_cast<int>(batch_size * event.value) % batch_size, ".png"));
    });
    gui->setWidth(256);

    ofBackground(20);
}

void ImageGeneratorApp::update() {
    glService.reset();
    glService.run();
}

void ImageGeneratorApp::draw() {
    if (image) image->draw(128, 128, 256, 256);
}
