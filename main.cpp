#include <iostream>
#include <vector>
#include <memory>
#include <chrono>   

#include <assert.h>

#include <opencv2/core/mat.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <cuda_runtime_api.h>

#include "NvCaffeParser.h"

using namespace nvinfer1;
using namespace nvcaffeparser1;

float m_nms_threshold = 0.4;
float data0[8] = { -248,-248,263,263,-120,-120,135,135 };
float data1[8] = { -56,-56,71,71,-24,-24,39,39 };
float data2[8] = { -8,-8,23,23,0,0,15,15 };

float resize_scale = 1;

class Anchor {
public:
	bool operator<(const Anchor &t) const {
		return score < t.score;
	}

	bool operator>(const Anchor &t) const {
		return score > t.score;
	}

	float& operator[](int i) {
		assert(0 <= i && i <= 4);

		if (i == 0)
			return finalbox.x;
		if (i == 1)
			return finalbox.y;
		if (i == 2)
			return finalbox.width;
		if (i == 3)
			return finalbox.height;
	}

	float operator[](int i) const {
		assert(0 <= i && i <= 4);

		if (i == 0)
			return finalbox.x;
		if (i == 1)
			return finalbox.y;
		if (i == 2)
			return finalbox.width;
		if (i == 3)
			return finalbox.height;
	}

	cv::Rect_< float > anchor; // x1,y1,x2,y2
	float reg[4]; // offset reg
	cv::Point center; // anchor feat center
	float score; // cls score
	std::vector<cv::Point2f> pts; // pred pts

	cv::Rect_< float > finalbox; // final box res
};

void nms_cpu(std::vector<Anchor>& boxes, float threshold, std::vector<Anchor>& filterOutBoxes) {
	filterOutBoxes.clear();
	if (boxes.size() == 0)
		return;
	std::vector<size_t> idx(boxes.size());

	for (unsigned i = 0; i < idx.size(); i++)
	{
		idx[i] = i;
	}

	//descending sort
	sort(boxes.begin(), boxes.end(), std::greater<Anchor>());

	while (idx.size() > 0)
	{
		int good_idx = idx[0];
		filterOutBoxes.push_back(boxes[good_idx]);

		std::vector<size_t> tmp = idx;
		idx.clear();
		for (unsigned i = 1; i < tmp.size(); i++)
		{
			int tmp_i = tmp[i];
			float inter_x1 = std::max(boxes[good_idx][0], boxes[tmp_i][0]);
			float inter_y1 = std::max(boxes[good_idx][1], boxes[tmp_i][1]);
			float inter_x2 = std::min(boxes[good_idx][2], boxes[tmp_i][2]);
			float inter_y2 = std::min(boxes[good_idx][3], boxes[tmp_i][3]);

			float w = std::max((inter_x2 - inter_x1 + 1), 0.0F);
			float h = std::max((inter_y2 - inter_y1 + 1), 0.0F);

			float inter_area = w * h;
			float area_1 = (boxes[good_idx][2] - boxes[good_idx][0] + 1) * (boxes[good_idx][3] - boxes[good_idx][1] + 1);
			float area_2 = (boxes[tmp_i][2] - boxes[tmp_i][0] + 1) * (boxes[tmp_i][3] - boxes[tmp_i][1] + 1);
			float o = inter_area / (area_1 + area_2 - inter_area);
			if (o <= threshold)
				idx.push_back(tmp_i);
		}
	}
}

class CRect2f {
public:
	CRect2f(float x1, float y1, float x2, float y2) {
		val[0] = x1;
		val[1] = y1;
		val[2] = x2;
		val[3] = y2;
	}

	float& operator[](int i) {
		return val[i];
	}

	float operator[](int i) const {
		return val[i];
	}

	float val[4];

	void print() {
		printf("rect %f %f %f %f\n", val[0], val[1], val[2], val[3]);
	}
};

class AnchorGenerator {
public:
	void Init(int stride, int num, float* data)
	{
		anchor_stride = stride; // anchor tile stride
		preset_anchors.push_back(CRect2f(data[0], data[1], data[2], data[3]));
		preset_anchors.push_back(CRect2f(data[4], data[5], data[6], data[7]));
		anchor_num = num; // anchor type num
	}
	// filter anchors and return valid anchors
	int FilterAnchor(float* cls, float* reg, float* pts, int w, int h, int c, std::vector<Anchor>& result)
	{
		printf("########### FilterAnchor w = %d h = %d", w, h);
		int pts_length = 0;

		pts_length = c / anchor_num / 2;

		printf("######### anchor_num = %d\n", anchor_num);

		for (int i = 0; i < h; ++i) {
			for (int j = 0; j < w; ++j) {
				int id = i * w + j;
				for (int a = 0; a < anchor_num; ++a)
				{
					float score = cls[(anchor_num + a)*w*h + id];
                    //printf("##### score = %f\n", score);
					if (score >= m_cls_threshold) {
						CRect2f box(j * anchor_stride + preset_anchors[a][0],
							i * anchor_stride + preset_anchors[a][1],
							j * anchor_stride + preset_anchors[a][2],
							i * anchor_stride + preset_anchors[a][3]);
						//printf("%f %f %f %f\n", box[0], box[1], box[2], box[3]);
						CRect2f delta(reg[(a * 4 + 0)*w*h + id],
							reg[(a * 4 + 1)*w*h + id],
							reg[(a * 4 + 2)*w*h + id],
							reg[(a * 4 + 3)*w*h + id]);

						Anchor res;
						res.anchor = cv::Rect_< float >(box[0], box[1], box[2], box[3]);
						bbox_pred(box, delta, res.finalbox);
						//printf("bbox pred\n");
						res.score = score;
						res.center = cv::Point(j, i);

						//printf("center %d %d\n", j, i);

						if (1) {
							std::vector<cv::Point2f> pts_delta(pts_length);
							for (int p = 0; p < pts_length; ++p) {
								pts_delta[p].x = pts[(a*pts_length * 2 + p * 2)*w*h + id];
								pts_delta[p].y = pts[(a*pts_length * 2 + p * 2 + 1)*w*h + id];
							}
							//printf("ready landmark_pred\n");
							landmark_pred(box, pts_delta, res.pts);
							//printf("landmark_pred\n");
						}
						result.push_back(res);
					}
				}
			}
		}
		return 0;
	}

private:
	void bbox_pred(const CRect2f& anchor, const CRect2f& delta, cv::Rect_< float >& box)
	{
		float w = anchor[2] - anchor[0] + 1;
		float h = anchor[3] - anchor[1] + 1;
		float x_ctr = anchor[0] + 0.5 * (w - 1);
		float y_ctr = anchor[1] + 0.5 * (h - 1);

		float dx = delta[0];
		float dy = delta[1];
		float dw = delta[2];
		float dh = delta[3];

		float pred_ctr_x = dx * w + x_ctr;
		float pred_ctr_y = dy * h + y_ctr;
		float pred_w = std::exp(dw) * w;
		float pred_h = std::exp(dh) * h;

		box = cv::Rect_< float >(pred_ctr_x - 0.5 * (pred_w - 1.0),
			pred_ctr_y - 0.5 * (pred_h - 1.0),
			pred_ctr_x + 0.5 * (pred_w - 1.0),
			pred_ctr_y + 0.5 * (pred_h - 1.0));
	}

	void landmark_pred(const CRect2f anchor, const std::vector<cv::Point2f>& delta, std::vector<cv::Point2f>& pts)
	{
		float w = anchor[2] - anchor[0] + 1;
		float h = anchor[3] - anchor[1] + 1;
		float x_ctr = anchor[0] + 0.5 * (w - 1);
		float y_ctr = anchor[1] + 0.5 * (h - 1);

		pts.resize(delta.size());
		for (int i = 0; i < delta.size(); ++i) {
			pts[i].x = delta[i].x*w + x_ctr;
			pts[i].y = delta[i].y*h + y_ctr;
		}
	}

	int anchor_stride; // anchor tile stride
	std::vector<CRect2f> preset_anchors;
	int anchor_num; // anchor type num
	float m_cls_threshold = 0.4;
};



float* cls[3];
float* reg[3];
float* pts[3];
AnchorGenerator ac[3];

class Logger : public ILogger
{
    void log(Severity severity, const char* msg) override
    {
        if (severity!=Severity::kINFO) std::cout << msg << std::endl;
    }
};

static std::vector<float> cvMatToCHW(cv::Mat &img, 
        int destChannel,
        int destHeight, 
        int destWidth)
{
    float scaleX = ((float)destWidth)/img.cols;
	float scaleY = ((float)destHeight)/img.rows;

    cv::Mat destImg;
    if ( scaleX != scaleY ){
		if (scaleX > scaleY)
		{
			printf("scaleX %f scaleY %f\n", scaleX, scaleY);
            resize_scale = scaleY;
			int fillSize = destWidth/scaleY - img.cols;
			printf("fillSize %d\n", fillSize);
			cv::copyMakeBorder(img, destImg, 0, 0, 0, fillSize, cv::BORDER_CONSTANT, cv::Scalar(0,0,0));
		}else
		{
            resize_scale = scaleX;
			int fillSize = destHeight/scaleX - img.rows;
			cv::copyMakeBorder(img, destImg, fillSize, 0, 0, 0, cv::BORDER_CONSTANT, cv::Scalar(0,0,0));
		}
	}else
    {
        destImg = img;
    }

    cv::imwrite("tem.jpg", destImg);

    cv::resize(destImg, destImg, cv::Size(destWidth, destHeight));
    
    destImg.convertTo(destImg, CV_32FC3);

    //HWC TO CHW
    std::vector<cv::Mat> input_channels(destImg.channels());
    printf("### destImg.channels() = %d\n", destImg.channels());
    cv::split(destImg, input_channels);

    std::vector<float> result(destChannel * destHeight * destWidth);
    auto data = result.data();
    int channelLength = destHeight * destWidth;
    for (int i = 0; i < destChannel; ++i) {
        printf("### input_channels[i] = %d\n", input_channels[i].cols);
        memcpy(data,input_channels[i].data,channelLength*sizeof(float));
        data += channelLength;
    }

    return std::move(result);
}

static int getDimsSize(Dims dim, int elementByteSize){
    int size = 1;

    for ( int i = 0; i < dim.nbDims; i++ ){
		printf("d[%d] = %lu\n", i, dim.d[i]);
        size *= dim.d[i];
    }

    return size * elementByteSize;
}

Logger gLogger;

const std::string deployFile = "models/mnet.prototxt";
const std::string modelFile = "models/mnet.caffemodel";

std::vector<float> input;

int main(){
	ac[0].Init(32, 2, data0);
	ac[1].Init(16, 2, data1);
	ac[2].Init(8, 2, data2);

    const std::vector<std::string> outputs = {
        "face_rpn_cls_prob_reshape_stride32",
        "face_rpn_bbox_pred_stride32",
        "face_rpn_landmark_pred_stride32",

        "face_rpn_cls_prob_reshape_stride16",
        "face_rpn_bbox_pred_stride16",
        "face_rpn_landmark_pred_stride16",

        "face_rpn_cls_prob_reshape_stride8",
        "face_rpn_bbox_pred_stride8",
        "face_rpn_landmark_pred_stride8"
    };
    IBuilder* builder = createInferBuilder(gLogger);
    INetworkDefinition* network = builder->createNetwork();

    ICaffeParser* parser = createCaffeParser();

    DataType modelDataType =  DataType::kFLOAT;
    const IBlobNameToTensor* blobNameToTensor =	parser->parse(deployFile.c_str(),
                                                              modelFile.c_str(),
                                                              *network,
                                                              modelDataType);

    assert(blobNameToTensor != nullptr);

    for (auto& s : outputs){
        std::cout << "register output : " << s << std::endl;
        network->markOutput(*blobNameToTensor->find(s.c_str()));
    }

    // Build the engine
	builder->setMaxBatchSize(1);
	builder->setMaxWorkspaceSize(1 << 20);

    ICudaEngine* engine = builder->buildCudaEngine(*network);
	
	assert(engine);

    std::vector<std::vector<int>> output_layer_size;
    for (auto& s : outputs){
        output_layer_size.push_back({
            engine->getBindingDimensions(engine->getBindingIndex(s.c_str())).d[0],
            engine->getBindingDimensions(engine->getBindingIndex(s.c_str())).d[1],
            engine->getBindingDimensions(engine->getBindingIndex(s.c_str())).d[2]
        });

        std::cout << "output " << s << " size: \n";
        std::cout << engine->getBindingDimensions(engine->getBindingIndex(s.c_str())).nbDims << std::endl;
        std::cout << engine->getBindingDimensions(engine->getBindingIndex(s.c_str())).d[0] << std::endl;
        std::cout << engine->getBindingDimensions(engine->getBindingIndex(s.c_str())).d[1] << std::endl;
        std::cout << engine->getBindingDimensions(engine->getBindingIndex(s.c_str())).d[2] << std::endl;
    }

    std::cout << "engine.getNbBindings() = " << engine->getNbBindings() << std::endl;

    assert( engine->getNbBindings() == 10 );

    void * buffers[10] = {0};
    std::vector<std::unique_ptr<char []>> outputHostBuffers;

    for ( int i = 0; i < engine->getNbBindings(); i++ ){
         std::cout << i << " byteSize = " << getDimsSize(engine->getBindingDimensions(i), sizeof(float)) << std::endl;
         cudaMalloc(&buffers[i], getDimsSize(engine->getBindingDimensions(i), sizeof(float)));
         std::cout << "buffers[i] = " << buffers[i] << std::endl;

         outputHostBuffers.push_back(std::unique_ptr<char[]>(new char[getDimsSize(engine->getBindingDimensions(i), sizeof(float))]));
    }

    IExecutionContext* context = engine->createExecutionContext();
    assert(context != nullptr);

    cv::Mat img = cv::imread("../test.jpg");
	cv::cvtColor(img, img, CV_BGR2RGB);

    input = cvMatToCHW(img, 3, 720, 1080);

	std::cout << "input size" << input.size() << std::endl;
	std::cout << "input size1 " << getDimsSize(engine->getBindingDimensions(0), sizeof(float)) << std::endl;


    cudaMemcpyAsync(buffers[0], input.data(), 
            getDimsSize(engine->getBindingDimensions(0), sizeof(float)), 
            cudaMemcpyHostToDevice, nullptr);

    for (size_t i = 0; i < 1; i++)
    {
        auto start = std::chrono::system_clock::now();
        context->execute(1, buffers);

        for ( int outputIndex = 1; outputIndex < engine->getNbBindings(); outputIndex++ ){
            cudaMemcpyAsync(outputHostBuffers[outputIndex].get(), 
                buffers[outputIndex], 
                getDimsSize(engine->getBindingDimensions(outputIndex), sizeof(float)), 
                cudaMemcpyDeviceToHost, nullptr);
        }

        cudaStreamSynchronize(nullptr);
        auto end   = std::chrono::system_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        std::cout << duration.count() << " ms" << std::endl;
    }

	// for (int i = 0; i < output_layer_size[i * 3 + 0][2] * output_layer_size[i * 3 + 0][1]; i++ ){
	// 	std::cout << " " << (float*)outputHostBuffers[1].get()[i];
	// }

	std::cout << "output_layer_size[i * 3 + 0][2] = " << output_layer_size[0][2] << "\n";
	std::cout << "output_layer_size[i * 3 + 0][1] = " << output_layer_size[0][1] << "\n";

    std::vector<Anchor> proposals;

    for (int i = 0; i < 3; i++){
        ac[i].FilterAnchor(
            (float*)outputHostBuffers[i * 3 + 3].get(), 
            (float*)outputHostBuffers[i * 3 + 1].get(), 
            (float*)outputHostBuffers[i * 3 + 2].get(), 
            output_layer_size[i * 3 + 0][2],
			output_layer_size[i * 3 + 0][1], 
            output_layer_size[i * 3 + 2][0], 
            proposals);
    }

    std::vector<Anchor> faces;
	nms_cpu(proposals, m_nms_threshold, faces);
    std::cout << "######### faces.size() = " << faces.size() << std::endl;
	std::sort(faces.begin(), faces.end(), [&](Anchor a, Anchor b)
	{
		return a.finalbox.area() > b.finalbox.area();
	});
	printf("##### 1\n");
	for (auto &face : faces)
	{
		face.finalbox.width /= resize_scale;
		face.finalbox.x /= resize_scale;
		face.finalbox.height /= resize_scale;
		face.finalbox.y /= resize_scale;
		for (int i = 0; i < 5; ++i)
		{
			face.pts[i].x /= resize_scale;
			face.pts[i].y /= resize_scale;
		}
	}

	auto img1 = cv::imread("tem.jpg");

    for (int i = 0; i < faces.size(); i++)
	{
		cv::rectangle(img1, cv::Point((int)faces[i].finalbox.x, (int)faces[i].finalbox.y), cv::Point((int)faces[i].finalbox.width, (int)faces[i].finalbox.height), cv::Scalar(0, 255, 255), 2, 8, 0);
		for (int j = 0; j < faces[i].pts.size(); ++j) {
			cv::circle(img1, cv::Point((int)faces[i].pts[j].x, (int)faces[i].pts[j].y), 1, cv::Scalar(225, 0, 225), 2, 8);
		}
	}

	// printf("##### 4\n");

	cv::imwrite("result.png", img1);

	// printf("##### 5\n");
}