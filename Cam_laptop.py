/.h
#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QSerialPort>
#include <QSerialPortInfo>
#include <QLabel>
#include <QProcess>
#include <QTcpServer>
#include <QTcpSocket>
#include <QByteArray>

QT_BEGIN_NAMESPACE
namespace Ui {
class MainWindow;
}
QT_END_NAMESPACE

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = nullptr);
    ~MainWindow() override;

private slots:
    void on_btnStart_clicked();
    void on_btnStop_clicked();
    void on_btnCamera_clicked();
    void on_btnLow_clicked();
    void on_btnMid_clicked();
    void on_btnHigh_clicked();

    void handleCameraClient();
    void readCameraFrame();

private:
    void stopCameraStream();
    void showBlackCameraView();

    Ui::MainWindow *ui;

    QSerialPort *serial = nullptr;
    QString serialBuffer;

    QLabel *cameraLabel = nullptr;
    QProcess *cameraProcess = nullptr;
    QTcpServer *cameraServer = nullptr;
    QTcpSocket *cameraSocket = nullptr;
    QByteArray cameraFrameBuffer;
    quint32 expectedFrameSize = 0;
    bool cameraRunning = false;
};

#endif // MAINWINDOW_H

/.cpp

#include "mainwindow.h"
#include "./ui_mainwindow.h"

#include <QDebug>
#include <QThread>
#include <QVBoxLayout>
#include <QPixmap>
#include <QHostAddress>
#include <QCoreApplication>
#include <QDir>
#include <QFile>
#include <QFileInfo>

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
{
    ui->setupUi(this);

    cameraLabel = new QLabel(ui->Camera);
    cameraLabel->setAlignment(Qt::AlignCenter);
    cameraLabel->setStyleSheet("background-color: black;");

    auto *cameraLayout = new QVBoxLayout(ui->Camera);
    cameraLayout->setContentsMargins(0, 0, 0, 0);
    cameraLayout->addWidget(cameraLabel);

    showBlackCameraView();

    cameraServer = new QTcpServer(this);

    connect(cameraServer, &QTcpServer::newConnection,
            this, &MainWindow::handleCameraClient);

    if (!cameraServer->listen(QHostAddress::LocalHost, 5006)) {
        qDebug() << "Cannot start camera server:" << cameraServer->errorString();
    }

    serial = new QSerialPort(this);

    serial->setPortName("COM3");
    serial->setBaudRate(QSerialPort::Baud115200);
    serial->setDataBits(QSerialPort::Data8);
    serial->setParity(QSerialPort::NoParity);
    serial->setStopBits(QSerialPort::OneStop);

    if (serial->open(QIODevice::ReadWrite)) {
        qDebug() << "Serial OK";
    } else {
        qDebug() << "Not COM";
    }

    QThread::msleep(2000);
}

MainWindow::~MainWindow()
{
    stopCameraStream();

    if (serial && serial->isOpen()) {
        serial->close();
    }

    delete ui;
}

void MainWindow::on_btnStart_clicked()
{
    qDebug() << "Start";

    if (serial && serial->isOpen()) {
        serial->write("1");
        serial->flush();
        ui->textBrowser->append(
            "<span style='color:green;'>Running</span>");
    }
}

void MainWindow::on_btnStop_clicked()
{
    qDebug() << "Stop";

    if (serial && serial->isOpen()) {
        serial->write("0");
        serial->flush();
        ui->textBrowser->append(
            "<span style='color:red;'>Stop</span>");
    }
}

void MainWindow::on_btnCamera_clicked()
{
    if (cameraRunning) {
        stopCameraStream();
        return;
    }

    cameraProcess = new QProcess(this);

    QString scriptPath = QDir(QCoreApplication::applicationDirPath())
                             .absoluteFilePath("../../Cam_laptop.py");

    if (!QFile::exists(scriptPath)) {
        scriptPath = "D:/C++/Giao_dien_ktlt/Cam_laptop.py";
    }

    cameraProcess->setWorkingDirectory(QFileInfo(scriptPath).absolutePath());

    cameraProcess->start("python", QStringList()
                                       << scriptPath
                                       << "--qt-stream");

    if (!cameraProcess->waitForStarted(3000)) {
        ui->textBrowser->append(
            "<span style='color:red;'>Cannot start Python camera</span>");
        cameraProcess->deleteLater();
        cameraProcess = nullptr;
        return;
    }

    cameraRunning = true;
    ui->btnCamera->setText("Stop Camera");
    ui->textBrowser->append(
        "<span style='color:green;'>Hand camera started</span>");
}

void MainWindow::handleCameraClient()
{
    if (cameraSocket) {
        cameraSocket->disconnectFromHost();
        cameraSocket->deleteLater();
    }

    cameraSocket = cameraServer->nextPendingConnection();
    cameraFrameBuffer.clear();
    expectedFrameSize = 0;

    connect(cameraSocket, &QTcpSocket::readyRead,
            this, &MainWindow::readCameraFrame);
}

void MainWindow::readCameraFrame()
{
    if (!cameraSocket) {
        return;
    }

    cameraFrameBuffer.append(cameraSocket->readAll());

    while (true) {
        if (expectedFrameSize == 0) {
            if (cameraFrameBuffer.size() < 4) {
                return;
            }

            const uchar *data =
                reinterpret_cast<const uchar *>(cameraFrameBuffer.constData());

            expectedFrameSize =
                (quint32(data[0]) << 24) |
                (quint32(data[1]) << 16) |
                (quint32(data[2]) << 8) |
                quint32(data[3]);

            cameraFrameBuffer.remove(0, 4);
        }

        if (cameraFrameBuffer.size() < int(expectedFrameSize)) {
            return;
        }

        QByteArray imageData = cameraFrameBuffer.left(expectedFrameSize);
        cameraFrameBuffer.remove(0, expectedFrameSize);
        expectedFrameSize = 0;

        QPixmap pixmap;

        if (pixmap.loadFromData(imageData, "JPG")) {
            cameraLabel->setPixmap(
                pixmap.scaled(cameraLabel->size(),
                              Qt::KeepAspectRatio,
                              Qt::SmoothTransformation));
        }
    }
}

void MainWindow::stopCameraStream()
{
    if (cameraSocket) {
        cameraSocket->disconnectFromHost();
        cameraSocket->deleteLater();
        cameraSocket = nullptr;
    }

    if (cameraProcess) {
        cameraProcess->terminate();

        if (!cameraProcess->waitForFinished(1500)) {
            cameraProcess->kill();
        }

        cameraProcess->deleteLater();
        cameraProcess = nullptr;
    }

    cameraFrameBuffer.clear();
    expectedFrameSize = 0;
    cameraRunning = false;

    if (ui && ui->btnCamera) {
        ui->btnCamera->setText("Start Camera");
    }

    showBlackCameraView();

    if (ui && ui->textBrowser) {
        ui->textBrowser->append(
            "<span style='color:red;'>Hand camera stopped</span>");
    }
}

void MainWindow::showBlackCameraView()
{
    if (!cameraLabel || !ui || !ui->Camera) {
        return;
    }

    QPixmap black(ui->Camera->size());
    black.fill(Qt::black);
    cameraLabel->setPixmap(black);
}

void MainWindow::on_btnLow_clicked()
{
    qDebug() << "50rpm";

    if (serial && serial->isOpen()) {
        serial->write("L");
        serial->flush();
        serial->write("1");
        serial->flush();

        ui->textBrowser->append(
            "<span style='color:#CC9966;'>Low speed</span>");
    }
}

void MainWindow::on_btnMid_clicked()
{
    qDebug() << "90rpm";

    if (serial && serial->isOpen()) {
        serial->write("M");
        serial->flush();
        serial->write("1");
        serial->flush();

        ui->textBrowser->append(
            "<span style='color:#CC9933;'>Medium speed</span>");
    }
}

void MainWindow::on_btnHigh_clicked()
{
    qDebug() << "120rpm";

    if (serial && serial->isOpen()) {
        serial->write("H");
        serial->flush();
        serial->write("1");
        serial->flush();

        ui->textBrowser->append(
            "<span style='color:#CC9900;'>High speed</span>");
    }
}

/.py
import argparse
import socket
import struct
import time
import urllib.request
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


CAMERA_INDEX = 0
MODEL_PATH = Path(__file__).with_name("hand_landmarker.task")
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]


def ensure_model():
    if MODEL_PATH.exists():
        return True

    print("Downloading hand_landmarker.task...")

    try:
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    except Exception as error:
        print("Cannot download model:", error)
        print("Download manually and put it here:")
        print(MODEL_URL)
        print(MODEL_PATH)
        return False

    return True


def is_thumb_up(landmarks, handedness):
    if handedness == "Right":
        return landmarks[4].x > landmarks[3].x

    return landmarks[4].x < landmarks[3].x


def count_fingers(landmarks, handedness):
    count = 0

    if handedness and is_thumb_up(landmarks, handedness):
        count += 1

    finger_pairs = [
        (8, 6),
        (12, 10),
        (16, 14),
        (20, 18),
    ]

    for tip_id, pip_id in finger_pairs:
        if landmarks[tip_id].y < landmarks[pip_id].y:
            count += 1

    return count


def draw_landmarks(frame, landmarks):
    height, width = frame.shape[:2]
    points = []

    for landmark in landmarks:
        x = int(landmark.x * width)
        y = int(landmark.y * height)
        points.append((x, y))

    for start, end in HAND_CONNECTIONS:
        cv2.line(frame, points[start], points[end], (0, 255, 255), 2)

    for point in points:
        cv2.circle(frame, point, 4, (0, 255, 0), -1)


def draw_text(frame, finger_count, handedness):
    cv2.rectangle(frame, (20, 20), (285, 120), (0, 0, 0), -1)

    cv2.putText(
        frame,
        f"Fingers: {finger_count}",
        (35, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.3,
        (0, 255, 0),
        3,
    )

    if handedness:
        cv2.putText(
            frame,
            handedness,
            (35, 108),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 255, 255),
            2,
        )


def send_frame(sock, frame):
    ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])

    if not ok:
        return

    data = encoded.tobytes()
    sock.sendall(struct.pack(">I", len(data)) + data)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--qt-stream", action="store_true")
    args = parser.parse_args()

    if not ensure_model():
        return

    qt_socket = None

    if args.qt_stream:
        qt_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        qt_socket.connect(("127.0.0.1", 5006))

    options = vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.6,
        min_hand_presence_confidence=0.6,
        min_tracking_confidence=0.6,
    )

    cap = cv2.VideoCapture(CAMERA_INDEX)

    if not cap.isOpened():
        print("Cannot open camera")
        return

    with vision.HandLandmarker.create_from_options(options) as landmarker:
        frame_index = 0

        while True:
            ok, frame = cap.read()

            if not ok:
                print("Cannot read camera frame")
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            timestamp_ms = int(time.time() * 1000) + frame_index
            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            finger_count = 0
            handedness = ""

            if result.hand_landmarks:
                landmarks = result.hand_landmarks[0]

                if result.handedness and result.handedness[0]:
                    handedness = result.handedness[0][0].category_name

                finger_count = count_fingers(landmarks, handedness)
                draw_landmarks(frame, landmarks)

            draw_text(frame, finger_count, handedness)

            if qt_socket:
                try:
                    send_frame(qt_socket, frame)
                except OSError:
                    break
            else:
                cv2.imshow("Finger Counter", frame)

                if cv2.waitKey(1) & 0xFF == 27:
                    break

            frame_index += 1

    cap.release()

    if qt_socket:
        qt_socket.close()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
