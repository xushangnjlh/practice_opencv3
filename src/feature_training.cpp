#include "DBoW3/DBoW3.h"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <iostream>
#include <vector>
#include <string>

using namespace std;
using namespace cv;
int main()
{
  cout << "Reading images..." << endl;
  vector<Mat> images; 
  for(int i=0; i<10; i++)
  {
    string path = "../data/"+to_string(i+1)+".png";
    images.push_back(imread(path));
  }
  
  cout << "Detecting ORB features..." << endl;
  Ptr<Feature2D> detector = ORB::create();
  vector<Mat> descriptors;
  for(Mat& image: images)
  {
    vector<KeyPoint> keyPoints;
    Mat descriptor;
    // 一帧图像对应n个keyPoint, 以及一个n行，64列的描述子矩阵
    detector->detectAndCompute(image, Mat(), keyPoints, descriptor);
    descriptors.push_back(descriptor);
  }
  
  cout << "Creating vocabulary..." << endl;
  DBoW3::Vocabulary vocabulary;
  vocabulary.create(descriptors);
  cout << "Information of the vocabulary: " << vocabulary << endl;
  vocabulary.save("vocabulary.yml.gz");
  cout << "Done." << endl;
  
  return 0;
}