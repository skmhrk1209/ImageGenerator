#include "ImageGeneratorApp.hpp"

const auto proj_dir = "/Users/hirokisakuma/documents/openFrameworks_v0.9.8/apps/myapps/ImageGenerator/";
const auto batch_size = 64;
const auto latent_size = 128;

auto concat = [](const auto&... args) {
    std::stringstream sstream;
    static_cast<void>(std::initializer_list<int>{(sstream << args, 0)...});
    return sstream.str();
};

void ImageGeneratorApp::setup() {
    globalGui = std::make_unique<ofxDatGui>(ofxDatGuiAnchor::TOP_LEFT);
    globalGui->setWidth(200);

    dropdown = globalGui->addDropdown("models", {"celeba_dcgan_model", "model_1_celebahq128_resnet19", "model_2_celebahq128_resnet19",
                                                 "model_5_celebahq128_resnet19", "model_6_celebahq128_resnet19", "model_9_celebahq128_resnet19"});
    dropdown->setStripeVisible(false);
    for (auto i = 0; i < dropdown->size(); ++i) {
        dropdown->getChildAt(i)->setStripeVisible(false);
    }
    dropdown->onDropdownEvent([=](const ofxDatGuiDropdownEvent& event) {
        //================================ build graph by Tensorflow ================================//
        auto model = event.target->getChildAt(event.child)->getLabel();

        if (graphs.find(model) != graphs.end()) return;

        tf::GraphDef input_graph;
        {
            auto scope = tf::Scope::NewRootScope();

            auto latents_begin = tf::ops::RandomNormal(scope, {latent_size}, tf::DT_FLOAT);
            auto latents_end = tf::ops::RandomNormal(scope, {latent_size}, tf::DT_FLOAT);

            std::vector<tf::Output> latents;

            auto lerp = [&](const auto& a, const auto& b, const auto& t) {
                return tf::ops::Add(scope, a, tf::ops::Multiply(scope, tf::ops::Subtract(scope, b, a), t));
            };

            for (auto i = 0; i < batch_size; ++i) {
                latents.push_back(lerp(latents_begin, latents_end, float(i) / batch_size));
            }

            tf::ops::Stack(scope.WithOpName("latents"), latents);

            scope.ToGraphDef(&input_graph);
        }

        tf::GraphDef output_graph;
        { tf::ReadBinaryProto(tf::Env::Default(), concat(proj_dir, model, "/frozen_graph.pb"), &output_graph); }

        graphs.emplace(model, std::make_pair(input_graph, output_graph));
    });

    toggle = globalGui->addToggle("generate");
    toggle->setStripeVisible(false);
    toggle->onToggleEvent([=](const ofxDatGuiToggleEvent& event) {
        if (process) process->join();
        process = std::make_unique<std::thread>([=]() {
            event.target->setEnabled(false);
            //================================ generate image by Tensorflow ================================//

            auto model = dropdown->getSelected()->getLabel();

            std::vector<tf::Tensor> inputs;
            {
                auto session = std::unique_ptr<tf::Session>(tf::NewSession(tf::SessionOptions()));
                session->Create(graphs[model].first);

                TF_CHECK_OK(session->Run({}, {"latents"}, {}, &inputs));
            }

            std::vector<tf::Tensor> outputs;
            {
                auto session = std::unique_ptr<tf::Session>(tf::NewSession(tf::SessionOptions()));
                session->Create(graphs[model].second);

                TF_CHECK_OK(session->Run({{"latents", inputs[0]}}, {"fakes"}, {}, &outputs));
            }

            glService.post([=]() {
                images.clear();
                for (auto i = 0; i < outputs[0].dim_size(0); ++i) {
                    auto data = outputs[0].SubSlice(i).tensor<float, 3>().data();
                    images.emplace_back();
                    images.back().setFromPixels(data, 128, 128, OF_IMAGE_COLOR);
                }
            });

            event.target->setChecked(false);
            event.target->setEnabled(true);
        });
    });

    localGui = std::make_unique<ofxDatGui>(256, 384);
    localGui->setWidth(256);

    slider = localGui->addSlider("interpolation", 0.0, 1.0);
    slider->setStripeVisible(false);

    ofBackground(30);
}

void ImageGeneratorApp::update() {
    glService.reset();
    glService.run();
}

void ImageGeneratorApp::draw() {
    ofSetColor(255);
    ofNoFill();

    if (!images.empty()) {
        int index = (batch_size - 1) * localGui->getSlider("interpolation")->getValue();
        images[index].draw(256, 128, 256, 256);
    }

    ofDrawRectangle(256, 128, 256, 256);
}
