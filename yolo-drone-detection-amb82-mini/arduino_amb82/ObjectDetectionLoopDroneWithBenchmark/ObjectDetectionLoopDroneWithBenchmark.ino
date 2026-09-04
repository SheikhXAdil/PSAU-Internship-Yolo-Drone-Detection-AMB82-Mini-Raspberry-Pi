/*
 Example guide:
 https://ameba-doc-arduino-sdk.readthedocs-hosted.com/en/latest/ameba_pro2/amb82-mini/Example_Guides/Neural%20Network/Object%20Detection.html

 NN Model Selection
 Select Neural Network(NN) task and models using modelSelect(nntask, objdetmodel, facedetmodel, facerecogmodel).
 Replace with NA_MODEL if they are not necessary for your selected NN Task.

 NN task
 =======
 OBJECT_DETECTION/ FACE_DETECTION/ FACE_RECOGNITION

 Models
 =======
 YOLOv3 model         DEFAULT_YOLOV3TINY   / CUSTOMIZED_YOLOV3TINY
 YOLOv4 model         DEFAULT_YOLOV4TINY   / CUSTOMIZED_YOLOV4TINY
 YOLOv7 model         DEFAULT_YOLOV7TINY   / CUSTOMIZED_YOLOV7TINY
 SCRFD model          DEFAULT_SCRFD        / CUSTOMIZED_SCRFD
 MobileFaceNet model  DEFAULT_MOBILEFACENET/ CUSTOMIZED_MOBILEFACENET
 No model             NA_MODEL
 */

#include "WiFi.h"
#include "StreamIO.h"
#include "VideoStream.h"
#include "RTSP.h"
#include "NNObjectDetection.h"
#include "VideoStreamOverlay.h"
#include "ObjectClassList.h"

#define CHANNEL   0
#define CHANNELNN 3

// Lower resolution for NN processing
#define NNWIDTH  576
#define NNHEIGHT 320

VideoSetting config(VIDEO_FHD, 30, VIDEO_H264, 0);
VideoSetting configNN(NNWIDTH, NNHEIGHT, 10, VIDEO_RGB, 0);
NNObjectDetection ObjDet;
RTSP rtsp;
StreamIO videoStreamer(1, 1);
StreamIO videoStreamerNN(1, 1);

char ssid[] = "Adil's PC";    // your network SSID (name)
char pass[] = "Pakistan12345";        // your network password
int status = WL_IDLE_STATUS;

IPAddress ip;
int rtsp_portnum;

// ============================================================
// BENCHMARKING
// ============================================================

#define BENCHMARK_FRAMES 300

unsigned long benchmarkFrameCount = 0;

unsigned long benchmarkStartTime = 0;
unsigned long benchmarkEndTime = 0;

unsigned long inferenceStartTime = 0;
unsigned long inferenceEndTime = 0;

unsigned long totalInferenceTime = 0;
unsigned long minInferenceTime = 0xFFFFFFFF;
unsigned long maxInferenceTime = 0;

bool benchmarkRunning = false;

void setup()
{
    Serial.begin(115200);

    // attempt to connect to Wifi network:
    while (status != WL_CONNECTED) {
        Serial.print("Attempting to connect to WPA SSID: ");
        Serial.println(ssid);
        status = WiFi.begin(ssid, pass);

        // wait 2 seconds for connection:
        delay(2000);
    }
    ip = WiFi.localIP();

    // Configure camera video channels with video format information
    // Adjust the bitrate based on your WiFi network quality
    config.setBitrate(2 * 1024 * 1024);    // Recommend to use 2Mbps for RTSP streaming to prevent network congestion
    Camera.configVideoChannel(CHANNEL, config);
    Camera.configVideoChannel(CHANNELNN, configNN);
    Camera.videoInit();

    // Configure RTSP with corresponding video format information
    rtsp.configVideo(config);
    rtsp.begin();
    rtsp_portnum = rtsp.getPort();

    // Configure object detection with corresponding video format information
    // Select Neural Network(NN) task and models
    ObjDet.configVideo(configNN);
    ObjDet.modelSelect(OBJECT_DETECTION, CUSTOMIZED_YOLOV7TINY, NA_MODEL, NA_MODEL);
    ObjDet.begin();

    // Configure StreamIO object to stream data from video channel to RTSP
    videoStreamer.registerInput(Camera.getStream(CHANNEL));
    videoStreamer.registerOutput(rtsp);
    if (videoStreamer.begin() != 0) {
        Serial.println("StreamIO link start failed");
    }

    // Start data stream from video channel
    Camera.channelBegin(CHANNEL);

    // Configure StreamIO object to stream data from RGB video channel to object detection
    videoStreamerNN.registerInput(Camera.getStream(CHANNELNN));
    videoStreamerNN.setStackSize();
    videoStreamerNN.setTaskPriority();
    videoStreamerNN.registerOutput(ObjDet);
    if (videoStreamerNN.begin() != 0) {
        Serial.println("StreamIO link start failed");
    }

    // Start video channel for NN
    Camera.channelBegin(CHANNELNN);

    // Start OSD drawing on RTSP video channel
    OSD.configVideo(CHANNEL, config);
    OSD.begin();

    Serial.print("Network URL for RTSP Streaming: ");
    Serial.print("rtsp://");
    Serial.print(ip);
    Serial.print(":");
    Serial.println(rtsp_portnum);
    Serial.println(" ");
}

void loop()
{
    // Start benchmark timer
    if (!benchmarkStarted)
    {
        benchmarkStarted = true;

        benchmarkStart = millis();

        frameCount = 0;
        totalLatency = 0;
        minLatency = 0xFFFFFFFF;
        maxLatency = 0;

        Serial.println();
        Serial.println("========================================");
        Serial.println("Starting object detection benchmark...");
        Serial.print("Benchmark frames: ");
        Serial.println(BENCHMARK_FRAMES);
        Serial.println("========================================");
    }


    // Start timing the complete detection pipeline
    unsigned long startTime = micros();


    // ========================================================
    // OBJECT DETECTION
    // ========================================================

    std::vector<ObjectDetectionResult> results = ObjDet.getResult();


    uint16_t im_h = config.height();
    uint16_t im_w = config.width();


    // ========================================================
    // OSD / DRAWING
    // ========================================================

    OSD.createBitmap(CHANNEL);

    int resultCount = ObjDet.getResultCount();

    if (resultCount > 0)
    {
        for (int i = 0; i < resultCount; i++)
        {
            int obj_type = results[i].type();

            if (itemList[obj_type].filter)
            {
                ObjectDetectionResult item = results[i];

                // Result coordinates are normalized 0.00 - 1.00
                int xmin = (int)(item.xMin() * im_w);
                int xmax = (int)(item.xMax() * im_w);
                int ymin = (int)(item.yMin() * im_h);
                int ymax = (int)(item.yMax() * im_h);


                // Draw bounding box
                OSD.drawRect(
                    CHANNEL,
                    xmin,
                    ymin,
                    xmax,
                    ymax,
                    3,
                    OSD_COLOR_WHITE
                );


                // Draw identification text
                char text_str[20];

                snprintf(
                    text_str,
                    sizeof(text_str),
                    "%s %d",
                    itemList[obj_type].objectName,
                    item.score()
                );

                OSD.drawText(
                    CHANNEL,
                    xmin,
                    ymin - OSD.getTextHeight(CHANNEL),
                    text_str,
                    OSD_COLOR_CYAN
                );
            }
        }
    }

    OSD.update(CHANNEL);


    // ========================================================
    // END TIMING
    // ========================================================

    unsigned long endTime = micros();

    unsigned long latency = endTime - startTime;


    // ========================================================
    // UPDATE BENCHMARK STATISTICS
    // ========================================================

    totalLatency += latency;

    if (latency < minLatency)
        minLatency = latency;

    if (latency > maxLatency)
        maxLatency = latency;

    frameCount++;


    // ========================================================
    // PRINT RESULTS AFTER 300 FRAMES
    // ========================================================

    if (frameCount >= BENCHMARK_FRAMES)
    {
        benchmarkEnd = millis();

        unsigned long totalTime =
            benchmarkEnd - benchmarkStart;


        float averageLatency =
            totalLatency / (float)frameCount / 1000.0;


        float minimumLatency =
            minLatency / 1000.0;


        float maximumLatency =
            maxLatency / 1000.0;


        float fps =
            frameCount * 1000.0 / totalTime;


        Serial.println();
        Serial.println("========================================");
        Serial.println("       OBJECT DETECTION BENCHMARK");
        Serial.println("========================================");

        Serial.print("Frames processed: ");
        Serial.println(frameCount);

        Serial.print("Total benchmark time: ");
        Serial.print(totalTime / 1000.0);
        Serial.println(" seconds");

        Serial.println();

        Serial.print("Average latency: ");
        Serial.print(averageLatency, 2);
        Serial.println(" ms");

        Serial.print("Minimum latency: ");
        Serial.print(minimumLatency, 2);
        Serial.println(" ms");

        Serial.print("Maximum latency: ");
        Serial.print(maximumLatency, 2);
        Serial.println(" ms");

        Serial.println();

        Serial.print("Average FPS: ");
        Serial.println(fps, 2);

        Serial.println("========================================");
        Serial.println();


        // Stop benchmark
        benchmarkStarted = false;

        // Optional: reset statistics for another 300-frame test
        frameCount = 0;
        totalLatency = 0;
        minLatency = 0xFFFFFFFF;
        maxLatency = 0;
    }


    // Wait for next result
    delay(100);
}
