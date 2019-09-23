package cn.edu.usts.test;

import org.apache.commons.io.FileUtils;
import org.apache.commons.io.IOUtils;
import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.annotation.*;
import org.opencv.core.CvType;

import java.io.File;
import java.io.IOException;
import java.net.URL;
import java.util.Arrays;

import static org.bytedeco.javacpp.opencv_imgcodecs.CV_LOAD_IMAGE_COLOR;

public class Dev_Test {

    public static final String dir = "D:/intellij_workspace/opencv-Java/target/classes/";

    public static void main(String[] args) throws IOException {
        basicOp_calcHist();
    }

    public static void basicOp_calcHist() {
        opencv_core.Mat src = opencv_imgcodecs.imread(dir + "baby.jpg",opencv_imgcodecs.CV_LOAD_IMAGE_GRAYSCALE);

        opencv_core.Mat hist = new opencv_core.Mat();

        opencv_core.Mat mask = new opencv_core.Mat();

        IntPointer channels = new IntPointer(0,1,2);

        IntPointer histSize = new IntPointer( 128,128,128);

        FloatPointer r1 = new FloatPointer(0,180);
        FloatPointer r2 = new FloatPointer(0,180);
        FloatPointer r3 = new FloatPointer(0,180);

        PointerPointer<FloatPointer> pp = new PointerPointer<FloatPointer>(r1,r2,r3);

        System.out.println(r1.address());
        /*@Namespace("cv") public static native void calcHist( @Const Mat images, int nimages,
        @Const IntPointer channels, @ByVal Mat mask,
        @ByVal Mat hist, int dims, @Const IntPointer histSize,
        @Cast("const float**") PointerPointer ranges, @Cast("bool") boolean uniform*//*=true*//*, @Cast("bool") boolean accumulate*//*=false*/
        opencv_imgproc.calcHist(src, 1, channels, mask, hist, 3, histSize, pp,true,false);

        opencv_highgui.namedWindow("src", opencv_highgui.WINDOW_NORMAL);
        opencv_highgui.namedWindow("hist", opencv_highgui.WINDOW_NORMAL);

        opencv_highgui.imshow("src", src);
        opencv_highgui.imshow("hist", hist);

        int key = 0;

        while ((key & 0xFF) != 27)
            key = opencv_highgui.cvWaitKey(20);

        src.release();
        hist.release();

        opencv_highgui.destroyAllWindows();
    }

    public static void basicOp_MatExpre() {
        opencv_core.Mat src = opencv_imgcodecs.imread(dir + "baby.jpg");
        opencv_core.MatExpr t = src.t();

    }

    public static void basicOp_normalize() {
        opencv_core.Mat src = opencv_imgcodecs.imread(dir + "baby.jpg");
        opencv_core.Mat dst = new opencv_core.Mat(src.size(), opencv_core.CV_32FC3);
        opencv_core.normalize(src, dst, 1, 0, opencv_core.NORM_MINMAX, -1, opencv_core.noArray());
        opencv_highgui.namedWindow("src", opencv_highgui.WINDOW_NORMAL);
        opencv_highgui.namedWindow("normalize", opencv_highgui.WINDOW_NORMAL);

        opencv_highgui.imshow("src", src);
        opencv_highgui.imshow("normalize", dst);

        byte[] data = new byte[dst.rows() * dst.cols() * dst.channels()];

        dst.data().get(data);

        System.out.println(Arrays.toString(data));

        int key = 0;

        while ((key & 0xFF) != 27)
            key = opencv_highgui.cvWaitKey(20);

        release_mat(src, dst);

        opencv_highgui.destroyAllWindows();
    }

    public static void basicOp_pyr_down() {
        opencv_core.Mat src = opencv_imgcodecs.imread(dir + "baby.jpg");

        opencv_core.Mat pyr_dst = new opencv_core.Mat();
        opencv_core.Mat resize_dst = new opencv_core.Mat();

        opencv_imgproc.resize(src, resize_dst, new opencv_core.Size(480, 640), 0, 0, opencv_imgproc.INTER_AREA);
        opencv_imgproc.pyrDown(src, pyr_dst);

        System.out.println(pyr_dst.size().width());
        System.out.println(pyr_dst.size().height());
        opencv_highgui.namedWindow("src", opencv_highgui.WINDOW_NORMAL);
        opencv_highgui.namedWindow("pyrDown", opencv_highgui.WINDOW_NORMAL);
        opencv_highgui.namedWindow("resize", opencv_highgui.WINDOW_NORMAL);

        opencv_highgui.imshow("src", src);
        opencv_highgui.imshow("pyrDown", pyr_dst);
        opencv_highgui.imshow("resize", resize_dst);

        int key = 0;

        while ((key & 0xFF) != 27)
            key = opencv_highgui.cvWaitKey(20);

        release_mat(pyr_dst, src, resize_dst);

        opencv_highgui.destroyAllWindows();
    }

    public static void basicOp_data_load() throws IOException {

        byte[] data = FileUtils.readFileToByteArray(new File("data.bin"));

        opencv_core.Mat src = new opencv_core.Mat(data);

        opencv_highgui.namedWindow("data", opencv_highgui.WINDOW_NORMAL);

        opencv_highgui.imshow("data", src);

        int key = 0;

        while ((key & 0xFF) != 27)
            key = opencv_highgui.cvWaitKey(20);

        src.release();

        opencv_highgui.destroyAllWindows();

    }

    public static void basicOp_data_save() throws IOException {
        opencv_core.MatExpr ones = opencv_core.Mat.ones(5, 5, opencv_core.CV_8UC1);

        opencv_core.Mat src = ones.asMat();

        BytePointer ptr = src.ptr();

        byte[] data = new byte[5 * 5 * 1];

        ptr.get(data);

        FileUtils.writeByteArrayToFile(new File("data.bin"), data);

        ones.deallocate();
        src.release();
        ptr.deallocate();
    }

    public static void release_mat(opencv_core.Mat... mats) {
        for (opencv_core.Mat mat : mats) {
            if (!mat.isNull())
                mat.release();
        }
    }

    public static void show_mat_data(opencv_core.Mat mat) {
        BytePointer ptr = mat.ptr();

        byte[] data = new byte[mat.rows() * mat.cols() * mat.channels()];

        ptr.get(data);

        for (byte datum : data) {
            System.out.print(datum);
        }
        System.out.println();
    }

    public static void basicOp_matAndArray() {
        opencv_core.MatExpr ones = opencv_core.Mat.ones(5, 5, opencv_core.CV_8UC1);

        opencv_core.Mat src = ones.asMat();

        show_mat_data(src);

        BytePointer ptr = src.ptr();

        byte[] data = new byte[5 * 5 * 1];

        ptr.get(data);

        BytePointer b_data = new BytePointer(5 * 5 * 1);

        b_data.put(data);

        opencv_core.Mat mat = new opencv_core.Mat(new opencv_core.Size(5, 5), opencv_core.CV_8UC1, ptr);

        opencv_core.Mat decode = opencv_imgcodecs.imdecode(mat, opencv_imgcodecs.IMREAD_UNCHANGED);

        show_mat_data(decode);

        opencv_highgui.namedWindow("src data", opencv_highgui.WINDOW_NORMAL);
        opencv_highgui.namedWindow("new data", opencv_highgui.WINDOW_NORMAL);
//        opencv_highgui.namedWindow("decode data", opencv_highgui.WINDOW_NORMAL);

        opencv_highgui.imshow("src data", src);
        opencv_highgui.imshow("new data", mat);
//        opencv_highgui.imshow("decode data", decode);

        int key = 0;

        while ((key & 0xFF) != 27)
            key = opencv_highgui.cvWaitKey(20);

        decode.release();
        src.release();
        ones.address();
        mat.release();

        opencv_highgui.destroyAllWindows();
    }

    public static void basicOp_decode() throws IOException {
        opencv_core.Mat src = opencv_imgcodecs.imread(dir + "girl.jpg");

        int width = src.cols();

        int height = src.rows();

        int channels = src.channels();

        opencv_core.Mat buf = new opencv_core.Mat(new opencv_core.Size(1, width * height * channels), opencv_core.CV_8UC1);

        opencv_core.Mat decode = opencv_imgcodecs.imdecode(buf, CV_LOAD_IMAGE_COLOR);

        /*opencv_highgui.namedWindow("decode data", opencv_highgui.WINDOW_NORMAL);*/
        opencv_highgui.namedWindow("src data", opencv_highgui.WINDOW_NORMAL);

        /*opencv_highgui.imshow("decode data", buf);*/
        opencv_highgui.imshow("src data", src);

        BytePointer ptr = buf.ptr();

        byte[] data = new byte[width * height * channels];

        ptr.get(data);

        FileUtils.writeByteArrayToFile(new File(dir + "data.bin"), data);

        int key = 0;

        while ((key & 0xFF) != 27)
            key = opencv_highgui.cvWaitKey(20);

        buf.release();
        src.release();

        decode.release();

        opencv_highgui.destroyAllWindows();
    }

    public static void basicOp_floodFill_color() {
        opencv_core.Mat src = opencv_imgcodecs.imread(dir + "girl.jpg");

        opencv_core.Mat mask = new opencv_core.Mat(src.rows() + 2, src.cols() + 2, opencv_core.CV_8UC1);

        opencv_core.Rect rect = new opencv_core.Rect(50, 50, 80, 80);

        opencv_imgproc.floodFill(src, new opencv_core.Point(10, 10), new opencv_core.Scalar(255, 255, 0, 0), rect, new opencv_core.Scalar(40, 40, 40, 0), new opencv_core.Scalar(40, 40, 40, 0), opencv_imgproc.FLOODFILL_FIXED_RANGE);

        opencv_highgui.namedWindow("binary after flood_fill", opencv_highgui.WINDOW_NORMAL);

        opencv_highgui.imshow("binary after flood_fill", src);

        int key = 0;

        while ((key & 0xff) != 27)
            key = opencv_highgui.cvWaitKey(20);

        src.release();
        mask.release();

        rect.close();

        opencv_highgui.destroyAllWindows();
    }

    public static void basicOp_floodFill_binary() {
        opencv_core.Mat src = opencv_imgcodecs.imread(dir + "girl.jpg");

        opencv_core.Mat gray = new opencv_core.Mat();

        opencv_core.Mat binary = new opencv_core.Mat();

        opencv_imgproc.cvtColor(src, gray, opencv_imgproc.COLOR_BGR2GRAY);

        opencv_imgproc.threshold(gray, binary, 127, 255, opencv_imgproc.THRESH_BINARY);

        opencv_imgproc.cvtColor(binary, binary, opencv_imgproc.COLOR_GRAY2BGR);

        opencv_highgui.namedWindow("binary before flood_fill", opencv_highgui.WINDOW_NORMAL);

        opencv_highgui.imshow("binary before flood_fill", binary);

        opencv_core.Mat mask = new opencv_core.Mat(src.rows() + 2, src.cols() + 2, opencv_core.CV_8UC1);

        opencv_imgproc.floodFill(binary, mask, new opencv_core.Point(2, 2), new opencv_core.Scalar(157, 0, 48, 0));
        opencv_imgproc.floodFill(binary, mask, new opencv_core.Point(280, 280), new opencv_core.Scalar(9, 33, 87, 0));

        /*效果同上*/
        /*opencv_imgproc.floodFill(binary, new opencv_core.Point(2, 2), new opencv_core.Scalar(0, 0, 255, 0));*/

        opencv_highgui.namedWindow("binary after flood_fill", opencv_highgui.WINDOW_NORMAL);

        opencv_highgui.imshow("binary after flood_fill", binary);

        int key = 0;

        while ((key & 0xFF) != 27)
            key = opencv_highgui.cvWaitKey(10);

        binary.release();
        gray.release();
        src.release();

        opencv_highgui.destroyAllWindows();
    }

    public static void matExpr() {
        opencv_core.MatExpr expr = opencv_core.Mat.zeros(300, 300, opencv_core.CV_8UC3);

        opencv_core.Mat src = expr.asMat();

        opencv_highgui.namedWindow("src", opencv_highgui.WINDOW_NORMAL);

        opencv_highgui.imshow("src", src);

        int key = 0;

        while ((key & 0xFF) != 27)
            key = opencv_highgui.cvWaitKey(20);

        src.release();
        expr.close();

        opencv_highgui.destroyAllWindows();
    }

    public static void basicOp_copy() {
        opencv_core.Mat src = opencv_imgcodecs.imread(dir + "girl.jpg");

        opencv_core.Mat temp = opencv_imgcodecs.imread(dir + "1.png"); // 34 * 47

        opencv_core.Rect rect = new opencv_core.Rect(80, 80, 34, 47);

        opencv_core.Mat roi = src.apply(rect);

        temp.copyTo(roi);

        opencv_highgui.namedWindow("src", opencv_highgui.WINDOW_NORMAL);

        opencv_highgui.imshow("src", src);

        int key = 0;

        while ((key & 0xFF) != 27)
            key = opencv_highgui.cvWaitKey(20);

        temp.release();
        roi.release();
        src.release();
        rect.close();

        opencv_highgui.destroyAllWindows();
    }

    public static void basicOp_roi() {
        opencv_core.Mat src = opencv_imgcodecs.imread(dir + "girl.jpg");

        opencv_core.Rect rec = new opencv_core.Rect(80, 50, 200, 200);

        /*这三种方式获得的ROI区域均为原图像上的视图,原图像的操作会影响ROI*/
        opencv_core.Mat roi1 = src.apply(rec);
        /*opencv_core.Mat roi1 = new opencv_core.Mat(src, rec);*/
        /*opencv_core.Mat roi1 = src.adjustROI(80, 50, 200, 200);*/

        opencv_core.MatVector mv = new opencv_core.MatVector();

        opencv_core.split(src, mv);

        opencv_core.Mat channel_3 = mv.get(2);

        channel_3.put(new opencv_core.Scalar(0));

        opencv_core.merge(mv, src);

        opencv_highgui.namedWindow("roi1", opencv_highgui.WINDOW_NORMAL);
        opencv_highgui.namedWindow("src", opencv_highgui.WINDOW_NORMAL);

        opencv_highgui.imshow("roi1", roi1);
        opencv_highgui.imshow("src", src);

        int key = 0;

        while ((key & 0xFF) != 27)
            key = opencv_highgui.cvWaitKey(10);

        roi1.release();
        src.release();
        mv.close();

        opencv_highgui.destroyAllWindows();
    }

    public static void basicOp_bitwisze_and() {
        opencv_core.Mat src = opencv_imgcodecs.imread(dir + "girl.jpg");

        opencv_core.Mat mask = new opencv_core.Mat(src.size(), src.type(), new opencv_core.Scalar(0, 0, 0, 0));

        /*opencv_core.Rect rec = new opencv_core.Rect(80, 80, 200, 200);

        opencv_core.Mat roi = new opencv_core.Mat(mask, rec).put(new opencv_core.Scalar(255, 255, 255, 0));*/

        opencv_core.Rect rec = new opencv_core.Rect(80, 50, 200, 200);

        opencv_highgui.namedWindow("mask before rect", opencv_highgui.WINDOW_NORMAL);

        opencv_highgui.imshow("mask before rect", mask);

        opencv_core.Mat roi = mask.apply(rec).put(new opencv_core.Scalar(255, 255, 255, 0));

        opencv_core.Mat dst = new opencv_core.Mat();

        opencv_core.bitwise_and(src, mask, dst);

        opencv_highgui.namedWindow("src", opencv_highgui.WINDOW_NORMAL);
        opencv_highgui.namedWindow("mask after rect", opencv_highgui.WINDOW_NORMAL);
        opencv_highgui.namedWindow("dst", opencv_highgui.WINDOW_NORMAL);

        opencv_highgui.imshow("dst", dst);
        opencv_highgui.imshow("mask after rect", mask);
        opencv_highgui.imshow("src", src);

        int key = 0;

        while ((key & 0xFF) != 27)
            key = opencv_highgui.cvWaitKey(10);

        dst.release();
        mask.release();
        roi.release();
        src.release();
        rec.close();
        /*rec.deallocate();*/

        opencv_highgui.destroyAllWindows();

    }

    public static void basicOp_add() {
        opencv_core.Mat src = opencv_imgcodecs.imread(dir + "girl.jpg");

        opencv_core.Mat bg = new opencv_core.Mat(src.size(), src.type(), new opencv_core.Scalar(30, 30, 30, 0));

        opencv_core.Mat dst = new opencv_core.Mat();

        opencv_core.add(src, bg, dst);

        opencv_highgui.namedWindow("add win", opencv_highgui.WINDOW_NORMAL);
        opencv_highgui.namedWindow("origin win", opencv_highgui.WINDOW_NORMAL);

        opencv_highgui.imshow("add win", dst);
        opencv_highgui.imshow("origin win", src);

        int key = 0;

        while ((key & 0xFF) != 27)
            key = opencv_highgui.cvWaitKey(10);

        dst.release();
        bg.release();
        src.release();

        opencv_highgui.destroyAllWindows();
    }

    public static void inRange() {
        opencv_videoio.VideoCapture capture = new opencv_videoio.VideoCapture(0);

        if (!capture.isOpened())
            return;

        boolean isGrab = capture.grab();

        if (!isGrab)
            return;

        int key = 0;

        opencv_core.Mat lowerb = new opencv_core.Mat(new int[]{0, 43, 46});
        opencv_core.Mat upperb = new opencv_core.Mat(new int[]{34, 255, 255});
        opencv_core.Mat dst = new opencv_core.Mat();
        opencv_highgui.namedWindow("camera", opencv_highgui.WINDOW_NORMAL);

        while ((key & 0xFF) != 27) {
            opencv_core.Mat image = new opencv_core.Mat();

            opencv_core.Mat hsv = new opencv_core.Mat(image.size(), opencv_core.CV_8UC3);

            capture.read(image);

            opencv_imgproc.cvtColor(image, hsv, opencv_imgproc.COLOR_BGR2HSV);

            opencv_core.inRange(hsv, lowerb, upperb, dst);

            opencv_highgui.imshow("camera", dst);

            key = opencv_highgui.cvWaitKey(10);

            hsv.release();
            image.release();

        }

        dst.release();
        upperb.release();
        lowerb.release();

        capture.release();

        opencv_highgui.destroyAllWindows();

    }

    public static void mix_channels() {
        opencv_core.Mat src = opencv_imgcodecs.imread(dir + "girl.jpg");

        opencv_core.MatVector mv = new opencv_core.MatVector();

        opencv_core.split(src, mv);

        opencv_core.MatVector dst = new opencv_core.MatVector();

        dst.put(new opencv_core.Mat(src.size(), opencv_core.CV_8UC1));

        IntPointer fromTo = new IntPointer(2, 0);

        long npairs = 1;

        opencv_core.mixChannels(mv, dst, fromTo, npairs);

        opencv_highgui.namedWindow("mix", opencv_highgui.WINDOW_NORMAL);

        opencv_highgui.imshow("mix", dst.get(0));

        int key = 0;

        while ((key & 0xFF) != 27)
            key = opencv_highgui.cvWaitKey(10);

        src.release();
        dst.get(0).release();

        dst.deallocate();
        mv.deallocate();

        opencv_highgui.destroyAllWindows();
    }

    public static void matMerge() {
        opencv_core.Mat mat = opencv_imgcodecs.imread(dir + "girl.jpg");

        opencv_core.MatVector matVector = new opencv_core.MatVector();

        opencv_core.split(mat, matVector);

        opencv_core.Mat mat_r = matVector.get()[2];
        opencv_core.Mat mat_g = matVector.get()[1];
        opencv_core.Mat mat_b = matVector.get()[0];

        mat_r.put(new opencv_core.Scalar(0, 0, 0, 0));
        mat_g.put(new opencv_core.Scalar(0, 0, 0, 0));
        mat_b.put(new opencv_core.Scalar(0, 0, 0, 0));

        opencv_core.Mat dst = new opencv_core.Mat();

        opencv_core.merge(matVector, dst);

        opencv_highgui.namedWindow("merge", opencv_highgui.WINDOW_NORMAL);

        opencv_highgui.imshow("merge", dst);

        int key = 0;

        while ((key & 0xFF) != 27)
            key = opencv_highgui.cvWaitKey(10);

        dst.release();

        mat.release();

        opencv_highgui.destroyAllWindows();

        matVector.deallocate();
    }

    public static void createMat() {
        opencv_core.Mat m1 = new opencv_core.Mat(300, 300, opencv_core.CV_8UC3, new opencv_core.Scalar(145, 24, 69, 255));

        opencv_highgui.namedWindow("m1", opencv_highgui.WINDOW_NORMAL);

        opencv_highgui.imshow("m1", m1);

        int key = 0;

        while ((key & 0xFF) != 27)
            key = opencv_highgui.cvWaitKey(0);

        m1.release();

        opencv_highgui.destroyAllWindows();

    }

    public static void cv2Color() {
        opencv_core.Mat src = opencv_imgcodecs.imread(dir + "girl.jpg");

        opencv_core.Mat dst = new opencv_core.Mat();

        opencv_imgproc.cvtColor(src, dst, opencv_imgproc.COLOR_BGR2HSV);

        opencv_core.MatVector matVector = new opencv_core.MatVector();

        opencv_core.split(dst, matVector);

        opencv_core.Mat[] mats = matVector.get();

        opencv_highgui.namedWindow("src", opencv_highgui.WINDOW_NORMAL);
        opencv_highgui.namedWindow("channel 1", opencv_highgui.WINDOW_NORMAL);
        opencv_highgui.namedWindow("channel 2", opencv_highgui.WINDOW_NORMAL);
        opencv_highgui.namedWindow("channel 3", opencv_highgui.WINDOW_NORMAL);

        opencv_core.Mat m1 = null, m2 = null, m3 = null;

        m1 = mats[0];
        m2 = mats[1];
        m3 = mats[2];

        opencv_highgui.imshow("src", src);
        opencv_highgui.imshow("channel 1", m1);
        opencv_highgui.imshow("channel 2", m2);
        opencv_highgui.imshow("channel 3", m3);

        int key = 0;

        while ((key & 0xFF) != 27)
            key = opencv_highgui.cvWaitKey(0);

        m1.release();
        m2.release();
        m3.release();
        src.release();

        opencv_highgui.destroyAllWindows();
    }

    public static void pointerOp() {
        opencv_core.Mat src = opencv_imgcodecs.imread("D:/intellij_workspace/opencv-Java/target/classes/pic.jpg");

        opencv_core.Rect rec = new opencv_core.Rect(new opencv_core.Point(100, 100), new opencv_core.Point(400, 400));

        opencv_imgproc.rectangle(src, rec, opencv_core.Scalar.GREEN, 2, opencv_core.LINE_8, 0);

        opencv_highgui.namedWindow("rec", opencv_highgui.WINDOW_NORMAL);

        opencv_highgui.imshow("rec", src);

        int key = 0;

        while ((key & 0xFF) != 27)
            key = opencv_highgui.cvWaitKey(20);

        src.release();
    }

    public static void readPixel() {
        URL url = Dev_Test.class.getResource("/pic.jpg");

        opencv_core.Mat src = opencv_imgcodecs.imread("D:/intellij_workspace/opencv-Java/target/classes/pic.jpg");

        int width = src.cols();

        int height = src.rows();

        int channel = src.channels();

        int type = src.type();

        if (type == CvType.CV_8UC3)
            System.out.println("type : " + "CvType.CV_8SC3");

        BytePointer ptr = src.ptr();

        byte[] data = new byte[width * height * channel];

        BytePointer pointer = ptr.get(data);

        for (int i = 0; i < data.length; i++)
            data[i] = (byte) (255 - data[i]);

        BytePointer bytePointer = ptr.put(data);

        opencv_imgcodecs.imwrite("data.jpg", src);

        opencv_highgui.namedWindow("data", opencv_highgui.WINDOW_NORMAL);

        opencv_highgui.imshow("data", src);

        int key = 0;

        while ((key & 0xFF) != 27)
            key = opencv_highgui.cvWaitKey(0);

        src.release();

    }

    public static void LoadMat() {

        URL url = Dev_Test.class.getResource("/pic.jpg");

        System.out.println(url.getPath());

        opencv_core.Mat src = opencv_imgcodecs.imread("D:/intellij_workspace/opencv-Java/target/classes/pic.jpg");

        opencv_highgui.namedWindow("win", opencv_highgui.WINDOW_NORMAL);

        opencv_highgui.imshow("win", src);

        int key = 0;

        while ((key & 0xFF) != 27)
            key = opencv_highgui.cvWaitKey(20);

        src.release();

    }

    public static void OpenCamera() {

        opencv_videoio.VideoCapture capture = new opencv_videoio.VideoCapture(0);

        if (!capture.isOpened())
            return;

        if (!capture.grab())
            return;

        opencv_core.Mat pic = new opencv_core.Mat();

        opencv_highgui.namedWindow("camera", opencv_highgui.WINDOW_NORMAL);

        int key = 0;

        while ((key & 0xFF) != 27) {

            capture.read(pic);

            opencv_highgui.imshow("camera", pic);

            pic.release();

            key = opencv_highgui.cvWaitKey(20);

        }


    }
}
