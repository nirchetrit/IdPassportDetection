import logo from './logo.svg';
import './App.css';
import * as tf from '@tensorflow/tfjs';
import React, {useEffect, useRef, useState} from 'react';

tf.setBackend('webgl');
const classes_dir = {
    1: {
        name: 'id/pass',
        id: 1
    }
};

function App() {
    const [model, setModel] = useState();
    const [text, setText] = useState('');
    const [text1, setText1] = useState('');
    const canvasRef = useRef();
    const videoRef = useRef();

    //useEffect for loading the model and warming it up...
    useEffect(() => {
        tf.loadGraphModel("https://cdn.jsdelivr.net/gh/nirchetrit/IdPassportDetection@latest/src/model/model.json").then(model => {
            setModel(model);
            console.log('loaded the model');
            console.log('warming up..');
            model.executeAsync(tf.zeros([1, 320, 320, 3]).asType('int32')).then(() => {
                console.log('finish warming up');
                setText(tf.getBackend());
                setText1('you can detect now');
            });
        });
    }, []);


    useEffect(() => {
        if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
            navigator.mediaDevices.getUserMedia({
                audio: false,
                video: {
                    facingMode: {
                        exact: 'environment'
                    }
                }
            }).then(stream => {
                window.stream = stream;
                videoRef.current.srcObject = stream;
                if (model && videoRef.current) {
                    console.log('starting the detection');

                }
            });
        }
    }, []);
    const getDetectedObjFromPredictions = (threshold, scores, boxes, classes, classesDir) => {
        const detectedObj = [];
        const video_frame = document.getElementById('frame');
        if (scores[0] > threshold) {
            const bbox = [];
            const minY = boxes[0] * video_frame.offsetHeight;
            const minX = boxes[1] * video_frame.offsetWidth;
            const maxY = boxes[2] * video_frame.offsetHeight;
            const maxX = boxes[3] * video_frame.offsetWidth;
            bbox[0] = minX;
            bbox[1] = minY;
            bbox[2] = maxX - minX;
            bbox[3] = maxY - minY;
            detectedObj.push({
                class: classes_dir[classes[0]].name,
                score: scores[0].toFixed(2),
                bbox: bbox
            });
        }
        return detectedObj;
    };
    const renderPredictions = (predictions) => {
        const ctx = canvasRef.current.getContext("2d");
        ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);

        // Font options.
        const font = "16px sans-serif";
        ctx.font = font;
        ctx.textBaseline = "top";


        const boxes = predictions[1].dataSync();
        const classes = predictions[2].dataSync();
        const scores = predictions[4].dataSync();


        const detectedObj = getDetectedObjFromPredictions(0.5, scores, boxes, classes, classes_dir);
        detectedObj.forEach(item => {
            const x = item['bbox'][0];
            const y = item['bbox'][1];
            const width = item['bbox'][2];
            const height = item['bbox'][3];
            // Draw the bounding box.
            ctx.strokeStyle = "#00FFFF";
            ctx.lineWidth = 4;
            ctx.strokeRect(x, y, width, height);

            // Draw the label background.
            ctx.fillStyle = "#00FFFF";
            const textWidth = ctx.measureText(item["label"] + " " + (100 * item["score"]).toFixed(2) + "%").width;
            const textHeight = parseInt(font, 10); // base 10
            ctx.fillRect(x, y, textWidth + 4, textHeight + 4);
            ctx.fillStyle = "#000000";
            ctx.fillText(item["class"] + " " + (100 * item["score"]) + "%", x, y);
        });

    };
    const preprocessInput = (input) => {
        const raw = tf.browser.fromPixels(input);
        const expanded = raw.expandDims();
        return expanded;
    };
    const runModel = async (video) => {
        if (!video) {
            return;
        }
        tf.engine().startScope();
        const preprocessedVideo = preprocessInput(video);
        model.executeAsync(preprocessedVideo).then(predictions => {
            renderPredictions(predictions);
        });
        requestAnimationFrame(() => {
            runModel(video);
        });
        tf.engine().endScope();
    };

    return (
        <div className="App">
            <h1>Running on: {text}</h1>
            <h1>{text1}</h1>
            <video
                style={{height: '320px', width: "320px"}}
                className="size"
                autoPlay
                playsInline
                muted
                ref={videoRef}
                id="frame"
                style={{
                    position: 'absolute',
                    top: '300px',
                    left: '300px'
                }}

            />
            <canvas
                className="size"
                ref={canvasRef}
                width="320"
                height="320"
                style={{
                    position: 'absolute',
                    top: '300px',
                    left: '300px'
                }}
            />
            <button
                style={{
                    position: 'absolute',
                    top: '100px',
                    left: '100px'
                }}
                onClick={() => {
                    runModel(videoRef.current);
                }}>Click to run
            </button>
        </div>
    );
}

export default App;
